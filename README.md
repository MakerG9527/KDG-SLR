#KDG-SLR

This repository contains the official PyTorch implementation of the paper:

**Kinematic Description-Guided Vision-Language Alignment for Chinese Sign Language Recognition**
Anonymous Author(s)

> **Abstract:** Isolated Sign Language Recognition (ISLR) classifies individual signs from video and is foundational for accessibility technologies. Existing vision-language methods rely on abstract gloss labels that convey no information about physical execution, leaving a substantial gap between visual and textual representations. To address this, we propose KDG-SLR, a kinematic description-guided framework that leverages detailed execution semantics from sign language dictionaries as a novel text modality, explicitly encoding handshape, orientation, location, and movement to establish execution-grounded cross-modal alignment via contrastive learning. We employ Mamba as the text encoder to fully encode long kinematic descriptions in linear time with higher efficiency than Transformer counterparts, and introduce a LoRA-based decoupled two-stage training strategy for parameter-efficient adaptation of the visual backbone. KDG-SLR achieves state-of-the-art results on three Chinese benchmarks, reaching 67.84\% Top-1 accuracy on the 6,707-class NationalCSL-DP. It further attains 16.52\% Top-1 cross-dataset zero-shot accuracy on unseen signs, which is infeasible for gloss-based methods, and consistent cross-lingual gains on WLASL.
---

## Results

### NationalCSL-DP (6,707 classes)

| Method | Top-1 | Top-5 | Modality |
|--------|-------|-------|----------|
| ST-GCN | 16.02 | 37.84 | Pose |
| NLA-SLR | 55.02 | 86.33 | RGB+Pose+Text |
| **KDG-SLR (Ours)** | **67.84** | **92.68** | RGB+Text |

### SLR-500

| Method | Top-1 | Modality |
|--------|-------|----------|
| NLA-SLR | 97.8 | RGB+Pose+Text |
| **KDG-SLR (Ours)** | **98.27** | RGB+Text |

### NMFs-CSL

| Method | Top-1 | Top-5 |
|--------|-------|-------|
| NLA-SLR | 83.4 | 98.3 |
| **KDG-SLR (Ours)** | **86.5** | **98.7** |

### Cross-Dataset Zero-Shot (NationalCSL-DP → SLR-500, 72 unseen classes)

| Method | Top-1 | Top-5 | Modality |
|--------|-------|-------|----------|
| ST-GCN | 0.28 | 1.68 | Pose |
| NLA-SLR | 0.76 | 3.44 | RGB+Pose+Text |
| **KDG-SLR (Ours)** | **16.52** | **34.08** | RGB+Text |

---

## Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.x
- 4× NVIDIA GPU (A40 or equivalent recommended)

Install dependencies:

```bash
pip install torch torchvision
pip install transformers
pip install causal-conv1d>=1.2.0
pip install mamba-ssm
pip install dotmap pyyaml tqdm
```

---

## Data Preparation

We evaluate on three Chinese sign language datasets:

- **NationalCSL-DP**: 6,707 sign words, 67,070 videos from 10 signers. We construct four subsets (6,707 / 2,000 / 1,000 / 500 words) with a 7:3 train-test split.
- **SLR-500**: 500 words, 12,500 videos from 5 signers.
- **NMFs-CSL**: 1,067 words (610 confusing + 457 normal), 32,010 videos from 10 signers.

Organize the data as follows:

```
data/
├── NationalCSL-DP/
│   └── <signer_id>/<word_id>/img_xxxxx.jpg
├── SLR-500/
│   └── ...
├── NMFs-CSL/
│   └── ...
└── WLASL/
    └── ...
```

Prepare list files (one sample per line: `video_folder label`):

```
lists/
├── NationalCSL-DP/
│   ├── train_rgb_split1.txt
│   └── val_rgb_split1.txt
├── SLR-500/
│   ├── train_rgb_split1.txt
│   └── val_rgb_split1.txt
├── NMFs-CSL/
│   ├── train_rgb_split1.txt
│   └── val_rgb_split1.txt
└── WLASL/
    ├── train_rgb_split1.txt
    └── val_rgb_split1.txt
```

Prepare the kinematic description CSV (label list):

```
lists/csl_labels.csv   # columns: gloss, kinematic_description; for NationalCSL-DL/SLR500/NMFs-CSL
lists/asl_labels.csv   # columns: gloss, kinematic_description; for WLASL
```

---

## Pretrained Models

| Model | Dataset | Top-1 |
|-------|---------|-------|
| KDG-SLR | NationalCSL-DP (6,707) | 67.84% |
| KDG-SLR | SLR-500 | 98.27% |
| KDG-SLR | NMFs-CSL | 86.50% |
| KDG-SLR | WLASL$\ddagger$ | 82.46% |

The visual backbone uses **CLIP ViT-B/16** pretrained weights (downloaded automatically via the `clip` package). The Mamba text encoder uses the `state-spaces/mamba-130M` architecture.

---

## Configuration

Edit the YAML config files under `configs/` before running. Key fields:

```yaml
pretrain: '/path/to/checkpoint.pt'   # path to saved model
data:
    dataset: nationalcsl             # nationalcsl / slr500 / nmfscsl
    num_segments: 64                 # number of sampled frames
    num_classes: 6707
    val_list: '/path/to/val_list.txt'
    label_list: '/path/to/csl_labels.csv'
network:
    arch: ViT-B/16
    sim_header: "Transf"             # temporal fusion module
solver:
    evaluate: True                   # True for test-only
```

---

## Training

**Stage 1** — Train Mamba text encoder and temporal fusion module (ViT frozen):

```bash
bash ./scripts/run_train.sh ./configs/NationalCSL-DP/csl_train.yaml
```

**Stage 2** — Apply LoRA to fine-tune the visual encoder:

Update the config to set `pretrain` to the Stage 1 checkpoint and enable LoRA (`lora_rank: 16`), then run the same command.

> All experiments in the paper are conducted on 4 NVIDIA A40 GPUs.
> Training hyperparameters: AdamW (β₁=0.9, β₂=0.98, weight decay=0.2), lr=8×10⁻⁵, batch size=32, 200 epochs with cosine annealing for NationalCSL-DP.

---

## Testing

```bash
bash ./scripts/run_test.sh ./configs/NationalCSL-DP/csl_test.yaml
```

Set the `pretrain` field in the config to the path of your checkpoint. The evaluation reports Top-1 and Top-5 accuracy using cosine similarity as the default distance metric.

---


## Citation

If you find this work useful, please cite:

```bibtex
None
```

---

## License

This project is released for research and accessibility purposes. See [LICENSE](LICENSE) for details.
