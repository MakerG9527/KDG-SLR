# KDG-SLR

This repository contains the official PyTorch implementation of the paper:

**Kinematic Description-Guided Vision-Language Alignment for Chinese Sign Language Recognition**
Anonymous Author(s)

> **Abstract:** Isolated Sign Language Recognition (ISLR) classifies individual signs from video and is foundational for accessibility technologies. Existing vision-language methods rely on abstract gloss labels that convey no information about physical execution, leaving a substantial gap between visual and textual representations. To address this, we propose KDG-SLR, a kinematic description-guided framework that leverages detailed execution semantics from sign language dictionaries as a novel text modality, explicitly encoding handshape, orientation, location, and movement to establish execution-grounded cross-modal alignment via contrastive learning. We employ Mamba as the text encoder to fully encode long kinematic descriptions in linear time with higher efficiency than Transformer counterparts, and introduce a LoRA-based decoupled two-stage training strategy for parameter-efficient adaptation of the visual backbone. KDG-SLR achieves state-of-the-art results on three Chinese benchmarks, reaching 67.84% Top-1 accuracy on the 6,707-class NationalCSL-DP. It further attains 16.52% Top-1 cross-dataset zero-shot accuracy on unseen signs, which is infeasible for gloss-based methods, and competitive cross-lingual results on the ASL benchmark WLASL.

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
| **KDG-SLR (Ours)** | **98.30** | RGB+Text |

### NMFs-CSL

| Method | Top-1 | Top-5 |
|--------|-------|-------|
| NLA-SLR | 83.4 | 98.3 |
| **KDG-SLR (Ours)** | **86.5** | **98.7** |

### Cross-Dataset Zero-Shot (NationalCSL-DP → SLR-500, 72 unseen classes)

Models trained on NationalCSL-DP (6,707 words) are applied to SLR-500 **without fine-tuning**. Evaluation is restricted to the 72 SLR-500 words entirely absent from the NationalCSL-DP training vocabulary, while retaining all 500 kinematic descriptions as the retrieval gallery.

| Method | Top-1 | Top-5 | Modality |
|--------|-------|-------|----------|
| ST-GCN | 0.28 | 1.68 | Pose |
| NLA-SLR | 0.76 | 3.44 | RGB+Pose+Text |
| **KDG-SLR (Ours)** | **16.52** | **34.08** | RGB+Text |

### Cross-Lingual Generalization on WLASL (ASL)

Supervised cross-lingual transfer: the model is trained and tested on the same WLASL vocabulary. Per-instance (P-I) Top-1 / Top-5 accuracy is shown below; full per-instance and per-class results are reported in the paper. WLASL1000 is restricted to the 581 signs with curated descriptions, with all methods evaluated on this same subset under identical splits.

| Method | WLASL100 (P-I T1/T5) | WLASL300 (P-I T1/T5) | WLASL1000 (P-I T1/T5) |
|--------|----------------------|----------------------|------------------------|
| I3D | 65.89 / 84.11 | 56.14 / 79.94 | 47.33 / 76.44 |
| BEST | 81.01 / 94.19 | 75.60 / 92.81 | – |
| SignBERT | 82.56 / 94.96 | 74.40 / 91.32 | – |
| NLA-SLR | 91.47 / 96.90 | 86.23 / 97.60 | 80.34 / 94.62 |
| NLA-SLR (3-crop) | 92.64 / 96.90 | 86.98 / 97.60 | 80.92 / 94.85 |
| **KDG-SLR (Ours)** | **93.45 / 98.06** | **87.96 / 98.15** | **82.46 / 95.13** |

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

We evaluate on three Chinese sign language datasets, plus the ASL benchmark WLASL for cross-lingual evaluation:

- **NationalCSL-DP**: 6,707 sign words, 67,070 videos from 10 signers. We construct four subsets (6,707 / 2,000 / 1,000 / 500 words) with a 7:3 train-test split.
- **SLR-500**: 500 words, 12,500 videos from 5 signers.
- **NMFs-CSL**: 1,067 words (610 confusing + 457 normal), 32,010 videos.
- **WLASL** (cross-lingual): subsets WLASL100 / 300 / 1000. For WLASL1000, evaluation is restricted to the 581 signs with curated kinematic descriptions, and all compared methods use this same 581-class subset under identical splits.

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
lists/csl_labels.csv   # columns: gloss, kinematic_description; for NationalCSL-DP / SLR-500 / NMFs-CSL
lists/asl_labels.csv   # columns: gloss, kinematic_description; for WLASL
```

The Chinese kinematic descriptions are drawn from the *National Common Sign Language Dictionary*; the corpus contains 6,707 entries averaging 50.2 Chinese characters (range 6–174). The curated English descriptions for WLASL average 23.6 words per entry, annotating the same four phonological components (handshape, palm orientation, location, movement trajectory).

---

## Pretrained Models

| Model | Dataset | Top-1 |
|-------|---------|-------|
| KDG-SLR | NationalCSL-DP (6,707) | 67.84% |
| KDG-SLR | SLR-500 | 98.30% |
| KDG-SLR | NMFs-CSL | 86.50% |
| KDG-SLR | WLASL1000 (581-class subset) | 82.46% |

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
    lora_rank: 16                    # LoRA rank for Stage-2 visual adaptation
```

---

## Training

KDG-SLR uses a **decoupled two-stage** training strategy.

**Stage 1** — Train the Mamba text encoder and temporal fusion module (ViT frozen):

```bash
bash ./scripts/run_train.sh ./configs/NationalCSL-DP/csl_train.yaml
```

**Stage 2** — Apply LoRA to fine-tune the visual encoder:

Update the config to set `pretrain` to the Stage-1 checkpoint and enable LoRA (`lora_rank: 16`), then run the same command.

**Dataset-specific settings.**

- **NationalCSL-DP** is trained from scratch: AdamW (β₁=0.9, β₂=0.98, weight decay=0.2), lr=8×10⁻⁵, batch size=32, 200 epochs with cosine annealing.
- **SLR-500** and **NMFs-CSL** are transferred from the 6,707-word NationalCSL-DP model with the **visual encoder kept frozen** (only the Mamba text encoder and temporal module are trained): lr=8×10⁻⁴, batch size=128, 500 epochs.
- **WLASL** freezes the visual encoder (pretrained on NationalCSL-DP) and trains the Mamba text encoder from scratch per subset, as the Chinese-pretrained text encoder does not transfer to ASL.

> All experiments in the paper are conducted on 4 NVIDIA A40 GPUs.

---

## Testing

```bash
bash ./scripts/run_test.sh ./configs/NationalCSL-DP/csl_test.yaml
```

Set the `pretrain` field in the config to the path of your checkpoint. The evaluation reports Top-1 and Top-5 accuracy, using **cosine similarity** as the default distance metric (best accuracy–efficiency trade-off; see the paper for a comparison with Euclidean and Manhattan distances).

---

## Citation

If you find this work useful, please cite our paper. *(Citation will be added after the anonymous review period.)*

```bibtex
% To be released upon acceptance.
```

---

## License

This project is released for research and accessibility purposes. See [LICENSE](LICENSE) for details.
