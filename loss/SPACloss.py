import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init 

def prop_topk_loss(cas, labels, mask_cas, is_back=True, topk=8):
    """
    Compute the topk classification loss

    Inputs:
        cas: tensor of size [B, M, C]
        labels: tensor of size [B, C]
        mask_cas: tensor of size [B, M, C]
        is_back: bool
        topk: int

    Outputs:
        loss_mil: tensor
    """
    if is_back:
        labels_with_back = torch.cat((labels, torch.ones_like(labels[:, [0]])), dim=-1)
    else:
        labels_with_back = torch.cat((labels, torch.zeros_like(labels[:, [0]])), dim=-1)
    labels_with_back = labels_with_back / (torch.sum(labels_with_back, dim=-1, keepdim=True) + 1e-4)

    loss_mil = 0
    for b in range(cas.shape[0]):
        cas_b = cas[b][mask_cas[b]].reshape((-1, cas.shape[-1]))
        topk_val, _ = torch.topk(cas_b, k=max(1, int(cas_b.shape[-2] // topk)), dim=-2)
        video_score = torch.mean(topk_val, dim=-2)
        loss_mil += - (labels_with_back[b] * F.log_softmax(video_score, dim=-1)).sum(dim=-1).mean()
    loss_mil /= cas.shape[0]

    return loss_mil

def decay_weight(epoch, total_epochs, k):
    if epoch > total_epochs or total_epochs==0:
        return 0
    return np.exp(-k * epoch)
def SS_weight_decay(epoch, total_epochs,
                        alpha=6, beta=8, lambda_ts=0.1):
    
    t = epoch / total_epochs
    ramp_up = 1 - np.exp(-beta * t)
    ramp_down = np.exp(-alpha * t)
    weight = lambda_ts * ramp_up * ramp_down
    return weight

def segments_iou(segments1, segments2):
    """
    Inputs:
        segments1: tensor of size [M1, 2]
        segments2: tensor of size [M2, 2]

    Outputs:
        iou_temp: tensor of size [M1, M2]
    """
    segments1 = segments1.unsqueeze(1)                          # [M1, 1, 2]
    segments2 = segments2.unsqueeze(0)                          # [1, M2, 2]
    tt1 = torch.maximum(segments1[..., 0], segments2[..., 0])   # [M1, M2]
    tt2 = torch.minimum(segments1[..., 1], segments2[..., 1])   # [M1, M2]
    intersection = tt2 - tt1
    union = (segments1[..., 1] - segments1[..., 0]) + (segments2[..., 1] - segments2[..., 0]) - intersection
    iou = intersection / (union + 1e-6)                         # [M1, M2]
    # Remove negative values
    iou_temp = torch.zeros_like(iou)
    iou_temp[iou > 0] = iou[iou > 0]
    return iou_temp

def PC_loss(proposals, 
            prop_attn, prop_ciou, 
            teacher_prop_ciou, 
            mask, device, args, eps=1e-6):
    """
    Soft-IoU + Teacher-Guided PC Loss
    ---------------------------------
    替代原 PC_loss 内部复杂的中心/归一化计算。
    保留输入输出形状（仍输出一个 scalar）。

    Inputs:
        proposals:   list of [M,2]
        prop_attn:   [B, M, 1]  student attention
        prop_ciou:   [B, M, 1]  student predicted IoU
        teacher_prop_ciou: [B, M, 1]  teacher predicted IoU
        mask:        [B, M, 1] valid proposal mask
        device:      torch device
        args:        arguments

    Output:
        loss: scalar tensor
    """

    B = len(proposals)
    total_loss = 0.0
    valid_batches = 0

    for b in range(B):

        # -------------------------
        # 1. 取有效 proposals
        # -------------------------
        attn = prop_attn[b][mask[b].squeeze(-1)]         # [M]
        pred_iou = prop_ciou[b][mask[b].squeeze(-1)]     # [M]
        teacher_iou = teacher_prop_ciou[b][mask[b].squeeze(-1)]  # [M]

        if attn.numel() == 0:
            continue

        # -------------------------
        # 2. Soft attention → soft weights
        # -------------------------
        soft_w = torch.softmax(attn / args.tau_pc, dim=0)  # [M]

        # -------------------------
        # 3. Soft IoU ground truth （融合 teacher）
        #    soft_IoU_GT = α * teacher + (1-α) * weighted_iou
        # -------------------------
        # teacher IoU 部分
        teacher_iou = torch.sigmoid(teacher_iou)

        # soft IoU 部分
        # （注意：weighted 的是 teacher IoU，而不是 IoU(p,GT)，无须 hard GT proposals）
        soft_iou = (soft_w * teacher_iou.squeeze(-1)).sum(dim=0)

        # final target
        soft_iou_gt = (
            args.alpha_teacher_iou * teacher_iou.squeeze(-1) +
            (1 - args.alpha_teacher_iou) * soft_iou
        ).detach()

        # -------------------------
        # 4. regression loss：Smooth L1
        # -------------------------
        pred_iou = torch.sigmoid(pred_iou.squeeze(-1))

        reg_loss = F.smooth_l1_loss(pred_iou, soft_iou_gt, reduction='mean')

        total_loss += reg_loss
        valid_batches += 1

    if valid_batches > 0:
        total_loss /= valid_batches
    else:
        total_loss = torch.tensor(0.0, device=device)

    return total_loss


def SS_loss(student_outputs, teacher_outputs,
                appearance_descriptors, motion_descriptors, features, mask,
                epoch, args, eps=1e-6):
    """
    Teacher-Student 蒸馏版 SME：
    融合 Teacher 稳定预测、SME 外部先验、运动语义中心和 Proposal 聚合信息
    
    最终伪标签 = α * p_teacher + β * p_sme_rgbflow + δ * p_proposal_aggregate
    """

    # ================================
    # 提取 Student & Teacher proposal 纯类别预测 (不含背景)
    # ================================
    S_cas = F.softmax(student_outputs['prop_fused_cas'][..., :-1], dim=-1)
    T_cas = F.softmax(teacher_outputs['prop_fused_cas'][..., :-1], dim=-1)

    # Attention 权重
    S_attn = torch.sigmoid(student_outputs['prop_fused_attn'])
    T_attn = torch.sigmoid(teacher_outputs['prop_fused_attn'])

    # ================================
    # Proposal → Video 聚合
    # ================================
    def aggregate(cas, attn):
        num = (cas * attn).sum(dim=1)          # [B,C]
        den = attn.sum(dim=1) + eps            # [B,1]
        return (num / den).clamp(min=eps)

    pS_video = aggregate(S_cas, S_attn)        # Student 视频级分布
    pT_video = aggregate(T_cas, T_attn)        # Teacher 视频级分布 (p_teacher)

    # ================================
    # SME-RGB/Flow 语义中心匹配 (p_sme_rgbflow)
    # ================================
    B, T, D = features.shape
    rgb_vecs = []   # [B,1024]
    flow_vecs = []  # [B,1024]
    for i in range(B):
        valid_features = features[i][mask[i].bool()]
        rgb_vecs.append(valid_features[:, :1024].mean(dim=0))
        flow_vecs.append(valid_features[:, 1024:].mean(dim=0))
    rgb_vecs = torch.stack(rgb_vecs, dim=0)    # [B,1024]
    flow_vecs = torch.stack(flow_vecs, dim=0)  # [B,1024]

    # RGB/Flow 与语义中心相似度
    rgb_sim = F.cosine_similarity(rgb_vecs.unsqueeze(1),
                                  appearance_descriptors.unsqueeze(0),
                                  dim=-1)
    flow_sim = F.cosine_similarity(flow_vecs.unsqueeze(1),
                                   motion_descriptors.unsqueeze(0),
                                   dim=-1)
    
    rgb_w = F.softmax(rgb_sim / args.tau_sem, dim=-1)
    flow_w = F.softmax(flow_sim / args.tau_sem, dim=-1)
    
    # 平均 RGB/Flow 相似度作为 p_sme_rgbflow
    p_sme_rgbflow = (rgb_w + flow_w) / 2

    # ================================
    # Proposal 聚合信息 (p_proposal_aggregate)
    # ================================
    p_proposal_aggregate = pS_video  # 这里直接使用 Student 的聚合结果

    # ================================
    # 融合多种伪标签来源
    # pseudo_label = α * p_teacher + β * p_sme_rgbflow  + δ * p_proposal_aggregate
    # ================================
    p_final = (args.alpha_teacher * pT_video + 
               args.alpha_sme * p_sme_rgbflow + 
               args.alpha_proposal * p_proposal_aggregate)
    
    # 归一化得到最终伪标签
    pseudo = p_final / (p_final.sum(dim=-1, keepdim=True) + eps)
    pseudo_entropy = -(pseudo * (pseudo + eps).log()).sum(dim=-1).mean()
    # ================================
    # Student 拟合伪标签（KL散度）
    # ================================
    loss = F.kl_div((pS_video + eps).log(), pseudo.detach(), reduction="batchmean")
    

    return loss, pseudo_entropy.detach()


def SC_loss(prop_fused_cas, prop_cas_supp, 
            prop_fused_feat_fuse,
            teacher_prop_cas, teacher_prop_feat,
            labels, epoch, total_epochs, args, eps=1e-6):
    """
    Teacher–Student Semantic Consistency Loss
    -----------------------------------------
    由三部分组成：
      1. Student 内部语义一致 (原始 SC_loss)
      2. Student-CAS ↔ Teacher-CAS 一致性
      3. Student-feature ↔ Teacher-feature (特征蒸馏)
    """

    device = prop_fused_cas.device

    # ===============================
    # 0. CAS softmax (remove background)
    # ===============================
    S_cas = F.softmax(prop_fused_cas[..., :-1], dim=-1)      # [B,M,C]
    S_supp = F.softmax(prop_cas_supp[..., :-1], dim=-1)      # [B,M,C]

    T_cas = F.softmax(teacher_prop_cas[..., :-1], dim=-1)    # [B,M,C]

    B, M, D = prop_fused_feat_fuse.shape

    # ===============================
    # 1. Student 内部语义一致 (原 SC_loss)
    # ===============================
    proto_list = []
    for i in range(B):
        mask_pos = (torch.argmax(S_cas[i], dim=1) == torch.argmax(S_supp[i], dim=1)) & \
                   (torch.argmax(S_cas[i], dim=1) == torch.argmax(labels[i]))

        if mask_pos.sum() > 0:
            proto = prop_fused_feat_fuse[i][mask_pos].mean(dim=0)
        else:
            proto = torch.zeros(D).to(device)

        proto_list.append(proto)

    student_proto = torch.stack(proto_list)  # [B, D]

    # feature distance to prototype
    sc_loss = 0.0
    count = 0
    for i in range(B):
        mask_cls = (torch.argmax(S_cas[i], dim=1) == torch.argmax(labels[i]))
        if mask_cls.sum() > 0:
            feat_sel = prop_fused_feat_fuse[i][mask_cls]
            dist = torch.norm(feat_sel - student_proto[i].unsqueeze(0), p=2, dim=1)
            sc_loss += dist.mean()
            count += 1

    if count > 0:
        sc_loss = sc_loss / count
    else:
        sc_loss = torch.tensor(0.0, device=device)

    # ===============================
    # 2. CAS distribution consistency：学生模仿教师
    # ===============================
    cas_consistency = F.mse_loss(S_cas, T_cas)

    # ===============================
    # 3. Feature consistency：学生特征模仿教师特征
    # ===============================
    feat_consistency = F.mse_loss(prop_fused_feat_fuse, teacher_prop_feat)

    # ===============================
    # Final loss
    # ===============================
    w = (0.3 + 0.2 * (epoch / args.alpha_2))

    loss = (
        w * sc_loss +
        args.alpha_sc_cas * cas_consistency +
        args.alpha_sc_feat * feat_consistency
    )

    return loss
