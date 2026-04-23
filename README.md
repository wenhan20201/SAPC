# SPAC: Stable pseudo-label refinement and action completeness learning for weakly supervised temporal action localization

Han Wen, Guangping Zeng, Qingchuan Zhang, Shuo Yang, Yupeng Hou, Qicheng Ma


![](fig/SPAC.jpg)

## Abstract
Weakly Supervised Temporal Action Localization (WTAL) aims to identify and localize action instances in untrimmed videos using only video-level labels. Due to the lack of snippet-level supervision, Existing multiple instance learning (MIL) methods only rely on high-confidence proposals to assign pseudo-labels to snippets. However, limited by single-source pseudo-instance supervision and the inherent instability of MIL, such pseudo-labels are often noisy, which not only interferes with model learning but also weakens its ability to model action completeness, thereby limiting localization performance. To address these issues, we propose a novel weakly supervised temporal action localization method based on stable pseudo-label refinement and action completeness learning (SPAC). The method breaks from conventional MIL paradigms by designing a reliable teacher guidanc module (RTG) to provide low-noise pseudo-instance supervision for the student model. First, we design a self-stabilized pseudo-label refinement module (SPR) to generate multi-source pseudo-labels by integrating video-level, proposal-level and semantic information, which helps suppress noise interference. Second, we introduce an action completeness learning module (ACL) that leverages video-level and proposal-level features to build a smooth and noise-resistant action boundary constraint, promoting the model to focus on complete action instances rather than local discriminative fragment. Extensive experimental on THUMOS14 and ActivityNet v1.3 benchmarks show that SPAC significantly outperforms most existing MIL-based methods, achieving an average mAP of 51.8\% on THUMOS14 and 25.6\% on ActivityNet v1.3.

## Recommended Environment

* Python version: 3.8.0
* Pytorch version: 2.3.1
* CUDA version: 12.1
* Tensorboard version: 2.14.0


## Data Preparation
1. Prepare [THUMOS14](https://www.crcv.ucf.edu/THUMOS14/) dataset.
    * To help you better reproduce our results, we recommend using the [pretrained I3D model](https://github.com/Finspire13/pytorch-i3d-feature-extraction.git) for feature extraction, as we did.
    * You can also get access of it from [Google Drive](https://drive.google.com/drive/folders/1_fGZpPM0PCTAgGQbQpBQEhK2KculypEu?usp=drive_link).

2. Prepare proposals.
    * You can just download the proposals used in our paper from [Google Drive](https://drive.google.com/drive/folders/13iuiiz4xlbAmCMZCwH1xVxPs_meSHoCy?usp=drive_link).

3. Place the features and annotations inside a `data/Thumos14reduced/` folder,  proposals inside a `proposals` folder and descriptors inside a `descriptors` folder. Make sure the data structure is as below.

```
    ├── data
        └── Thumos14reduced
            ├── Thumos14reduced-I3D-JOINTFeatures.npy
            └── Thumos14reduced-Annotations
                ├── Ambiguous_test.txt
                ├── classlist.npy
                ├── duration.npy
                ├── extracted_fps.npy
                ├── labels_all.npy
                ├── labels.npy
                ├── original_fps.npy
                ├── segments.npy
                ├── subset.npy
                └── videoname.npy
    ├── proposals4Thumos14
        ├── Proposals_Thumos14reduced_train.json
        ├── Proposals_Thumos14reduced_test.json
    ├── descriptors
        └── Thumos14reduced
            ├── general_appearance_descriptors.npy
            ├── general_motion_descriptors.npy
```

## Quick Start

### Training

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --run_type train
```

## Acknowledgement

We referred to the following repos when writing our code. We sincerely thank them for their outstanding contributions to the open-source community!

- [W-TALC](https://github.com/sujoyp/wtalc-pytorch)
- [CO2-Net](https://github.com/harlanhong/MM2021-CO2-Net)
- [P-MIL](https://github.com/RenHuan1999/CVPR2023_P-MIL)
- [SEAL_WTAL](https://github.com/Lkydong2020/SEAL_Wtal)
