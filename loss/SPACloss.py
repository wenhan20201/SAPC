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


