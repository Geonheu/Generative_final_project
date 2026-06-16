# Generative Virtual Multi-View Augmentation for Action Recognition via c-MAS

This project explores the use of c-MAS (Cross-view Motion Aware Synthesis), a diffusion-based generative model, to synthesize virtual multi-view skeleton data from a single-camera NTU60 input. Various fusion strategies are applied on top of the MotionBERT backbone to improve action recognition accuracy.

---

## Overview

### Problem Statement

In real-world settings, actions are often captured from a single camera, limiting viewpoint diversity. Installing additional cameras is costly, but the c-MAS diffusion model can generate virtual multi-view skeletons from a single-view input, addressing this limitation without extra hardware.

### Pipeline Comparison

Two pipelines are compared to evaluate the effectiveness and limitations of generative virtual multi-view synthesis.

| | GT Pipeline (upper bound) | c-MAS Pipeline |
|---|---|---|
| Input | Kinect-25 3D | Kinect-25 3D |
| Conversion path | Kinect-25 → COCO-17 3D → rotate → 2D | Kinect-25 → NBA-16 2D → **c-MAS** → NBA-16 3D → COCO-17 → rotate → 2D |
| Characteristics | Ideal geometric ground truth, noise-free | Two domain conversions + stochastic generation, error accumulation possible |
| Data scale | Train 40,091 / Val 16,487 samples | Val 480 samples (Zero-shot) |

---

## Experimental Results

### Experiment 2: Final Fusion Results (GT-based and c-MAS-based)

**Setup:** NTU60 `xsub_val`, MotionBERT backbone frozen, MLP meta-classifier trained on GT (40K) and evaluated

#### GT Data (16,487 val samples)

| Combo | Views | Raw Avg | Logit-Avg | Logit-Concat | Feat-Avg | Feat-Concat |
|---|---|---|---|---|---|---|
| C1 | 0° (Baseline) | 50.77% | 68.84% | 68.54% | 73.47% | 73.23% |
| C2 | 0°, ±30° | 51.83% | 71.04% | 71.64% | 75.43% | 75.93% |
| C3 | 0°, ±45° | 51.87% | 71.81% | 72.60% | 76.12% | 76.44% |
| C4 | 0°, ±90° | 50.98% | 71.72% | 73.45% | 76.18% | 77.27% |
| C8 | All 7 Views | 51.90% | 72.52% | 73.87% | 76.82% | **77.84%** |

#### c-MAS Data (480 val samples, GT-trained models, zero-shot)

| Combo | Views | Raw Avg | Logit-Avg | Logit-Concat | Feat-Avg | Feat-Concat |
|---|---|---|---|---|---|---|
| C1 | 0° (Baseline) | 40.21% | 57.92% | 56.46% | 59.58% | 60.21% |
| C2 | 0°, ±30° | 39.79% | 57.50% | 56.67% | 60.42% | 56.88% |
| C3 | 0°, ±45° | 40.00% | 56.67% | 53.96% | 59.38% | 55.42% |
| C4 | 0°, ±90° | 37.71% | 50.42% | 52.71% | 50.00% | 51.04% |
| C8 | All 7 Views | 40.21% | 53.75% | 53.33% | 55.83% | 53.96% |

> Raw 0° accuracy gap: GT 50.77% vs c-MAS 40.21% (-10.56%p)

---

### Additional Experiment 1: GT-trained MLP + c-MAS Views (Zero-shot Inference)

**Setup:** MLP trained only on GT distribution; c-MAS views are combined only at inference time.

| Combo | Views | Raw Avg | Logit-Avg | Logit-Concat | Feat-Avg | Feat-Concat |
|---|---|---|---|---|---|---|
| C1 | GT 0° (Baseline) | 49.38% | 70.21% | 70.21% | **74.79%** | 73.75% |
| C2 | GT 0° + c-MAS ±30° | 45.21% | 67.50% | 56.67% | 74.38% | 62.71% |
| C3 | GT 0° + c-MAS ±45° | 45.00% | 66.67% | 58.75% | 72.08% | 63.75% |
| C4 | GT 0° + c-MAS ±90° | 43.12% | 57.50% | 54.58% | 62.50% | 62.92% |
| C8 | GT 0° + c-MAS All | 41.88% | 58.13% | 53.96% | 60.62% | 58.33% |

> Performance degrades as more c-MAS views are added — the GT-trained MLP does not adapt to the noisy c-MAS distribution.

---

### Additional Experiment 2: Mixed Training (GT + c-MAS)

**Setup:** MLP is exposed to c-MAS data during training to induce noise adaptation (small-scale sampling condition).

| Combo | Views | Logit-Avg | Logit-Concat | Feat-Avg | Feat-Concat |
|---|---|---|---|---|---|
| C1 | GT 0° (Baseline) | 44.17% | 43.12% | 68.12% | 69.17% |
| C2 | GT 0° + c-MAS ±30° | 28.33% | 33.75% | 64.58% | 60.62% |
| C3 | GT 0° + c-MAS ±45° | 28.12% | 31.46% | 62.29% | 55.83% |
| C4 | GT 0° + c-MAS ±90° | 22.29% | 26.88% | 58.13% | 55.21% |
| C8 | GT 0° + c-MAS All | 14.79% | 32.71% | 50.62% | 39.58% |

> Small-scale mixed training leads to further degradation — insufficient training data combined with domain mismatch.

---

## Key Findings

1. **Feature-level fusion > Logit-level fusion**: Fusing high-dimensional features (2048-dim) consistently outperforms fusing low-dimensional logits (60-dim).
2. **GT multi-view is effective**: C8 Feat-Concat reaches 77.84%, a +4.61%p gain over single-view baseline.
3. **c-MAS domain gap exists**: -10.56%p gap in raw accuracy; zero-shot application of GT-trained models recovers only a limited portion.
4. **Adding c-MAS views hurts**: Error accumulation from double domain conversion (Kinect→NBA→COCO) is the primary cause of quality degradation.

---

## Project Structure

```
final_project/
├── c-MAS/                          # Generative multi-view pipeline
│   ├── pipeline.py                 # Multi-view generation (--split val/train, --stage 1/2/3/all)
│   ├── eval.py                     # c-MAS inference evaluation (--mode full/quick/time)
│   ├── model/                      # MDM diffusion model
│   ├── diffusion/                  # Gaussian diffusion implementation
│   ├── data_loaders/nba/           # NBA-16 format data loader
│   ├── sample/                     # MAS sampling (mas.py, sampler.py)
│   ├── utils/                      # Shared utilities
│   ├── save/yoga_diffusion_model/  # c-MAS pretrained checkpoint (200K steps)
│   ├── pipeline_out/               # Val generation outputs (npy, pkls, vis)
│   └── pipeline_train_out/         # Train generation outputs (npy, pkls)
│
└── MotionBERT/                     # Action recognition backbone + fusion classifier
    ├── run_full_eval.py            # Reproduce all four result tables at once
    ├── train.py                    # MLP training (--mode logit/feat/mixed)
    ├── eval.py                     # Evaluation (--mode gt_single/gt_combo/cmas/gt0_cmas)
    ├── make_table.py               # Result table output (--mode cmas/final)
    ├── proj_nba2coco.py            # NBA→COCO coordinate conversion (preprocessing)
    ├── visualize_projection.py     # GT skeleton 2D projection visualization
    ├── lib/                        # MotionBERT model / data / utils
    ├── configs/action/             # Training config YAMLs
    ├── data/action/                # Preprocessed pkl data
    │   ├── ntu60_gt_*.pkl          # GT projection data (symlink, ~1.1GB each)
    │   └── ntu60_cmas_*.pkl        # c-MAS generated data (~8MB each)
    ├── save/MB_ft_NTU60_xsub/      # MotionBERT pretrained checkpoint
    └── view_attn_out/
        ├── checkpoints_logit_combo/  # Logit-level MLP checkpoints
        ├── checkpoints_feat/         # Feature-level MLP checkpoints
        ├── checkpoints_mixed/        # Mixed GT+c-MAS MLP checkpoints
        ├── features/                 # Extracted 2048-dim features (symlink, 3.1GB)
        ├── logits/                   # Extracted 60-dim logits
        └── vis/                      # Result visualization images
```

---

## Reproducing Results

### Prerequisites

- c-MAS environment: `conda activate cMAS`
- MotionBERT environment: `conda activate POSCO`

The following files must be present before running inference:
- `MotionBERT/view_attn_out/logits/` — pre-extracted GT and c-MAS logits (`.npz`)
- `MotionBERT/view_attn_out/features/` — pre-extracted GT and c-MAS features (`.npz`)
- `MotionBERT/view_attn_out/checkpoints_logit_combo/` — trained logit-level MLP weights
- `MotionBERT/view_attn_out/checkpoints_feat/` — trained feature-level MLP weights
- `MotionBERT/view_attn_out/checkpoints_mixed/` — trained mixed MLP weights

### Reproduce All Four Tables (Recommended)

Run a single script that evaluates all experiments and prints results to stdout:

```bash
cd MotionBERT
conda run -n POSCO python run_full_eval.py
```

This script covers:
- **Experiment 2 GT**: Raw Avg / Logit-Avg / Logit-Concat / Feat-Avg / Feat-Concat on GT val (16,487 samples)
- **Experiment 2 c-MAS**: Same models applied zero-shot on c-MAS val (480 samples)
- **Additional Exp. 1**: GT 0° + c-MAS views, GT-trained MLP, zero-shot inference
- **Additional Exp. 2**: Mixed GT+c-MAS training MLP evaluation

---

## Step-by-Step Usage

### 1. Generate Virtual Multi-View Data (c-MAS)

```bash
cd c-MAS

# Full pipeline for val subset: infer → visualize → build pkls
python pipeline.py --split val --n_samples 500 --diffusion_steps 20 --stage all

# Train subset only: infer → build pkls (no visualization)
python pipeline.py --split train --per_class 8 --diffusion_steps 20 --stage 13

# Run individual stages
python pipeline.py --split val --stage 1   # inference only
python pipeline.py --split val --stage 2   # visualization only
python pipeline.py --split val --stage 3   # build pkls only
```

### 2. Evaluate c-MAS Output

```bash
cd c-MAS

# End-to-end evaluation including MotionBERT inference
python eval.py --mode full --n_eval 200

# Quick visualization test on a few samples
python eval.py --mode quick --num_samples 3

# Benchmark inference speed
python eval.py --mode time
```

### 3. Train MLP Fusion Classifiers

```bash
cd MotionBERT

python train.py --mode logit   # Logit-level MLP (Experiment 2)
python train.py --mode feat    # Feature-level MLP (Experiment 2)
python train.py --mode mixed   # Mixed GT+c-MAS MLP (Additional Exp. 2)
```

### 4. Evaluate Individual Modes

```bash
cd MotionBERT

python eval.py --mode gt_single    # GT single-view baseline
python eval.py --mode gt_combo     # GT multi-view combo (Experiment 2)
python eval.py --mode cmas         # c-MAS zero-shot full eval (Experiment 2)
python eval.py --mode gt0_cmas     # GT 0° + c-MAS views (Additional Exp. 1)
```

### 5. Print Result Tables

```bash
cd MotionBERT

python make_table.py --mode cmas             # c-MAS result comparison table
python make_table.py --mode final            # GT vs c-MAS final comparison table
python make_table.py --mode final --no_plot  # Text output only (no figure saved)
```

---

## Data and Models

| Item | Path | Note |
|---|---|---|
| NTU60 3D raw data | `/data/ntu60_3danno.pkl` | Shared |
| GT projection data (val, ×8 angles) | `MotionBERT/data/action/ntu60_gt_*.pkl` | Symlink (~1.1GB each) |
| c-MAS generated data (val, ×7 angles) | `MotionBERT/data/action/ntu60_cmas_*.pkl` | Included (~8MB each) |
| c-MAS generated data (train, ×7 angles) | `MotionBERT/data/action/ntu60_cmas_train_*.pkl` | Included (~8MB each) |
| c-MAS pretrained checkpoint | `c-MAS/save/yoga_diffusion_model/checkpoint_200000.pth` | Included (209MB) |
| MotionBERT pretrained checkpoint | `MotionBERT/save/MB_ft_NTU60_xsub/best_epoch.bin` | Included (231MB) |
| Extracted GT features | `MotionBERT/view_attn_out/features/` | Symlink (3.1GB) |
| Logit-level MLP checkpoints | `MotionBERT/view_attn_out/checkpoints_logit_combo/` | Included |
| Feature-level MLP checkpoints | `MotionBERT/view_attn_out/checkpoints_feat/` | Included |
| Mixed MLP checkpoints | `MotionBERT/view_attn_out/checkpoints_mixed/` | Included |
