# Visual Reconstruction of Incomplete Classical Sculptures Using Artificial Intelligence

UAB Final Degree Project in Artificial Intelligence, Escola d'Enginyeria, 2025/26.

This repository contains the implementation of an end-to-end automatic pipeline for the reconstruction of missing body parts in photographs of classical Greco-Roman sculptures, using a combination of semantic segmentation, dense human-body analysis, anatomically-shaped inpainting masks, three generative inpainting paradigms (LaMa, MAT, Stable Diffusion with Multi-ControlNet) and a PatchGAN evaluator trained from scratch as a domain-specific quality metric.

---

## Project overview

The system takes a photograph of a damaged classical sculpture (a marble figure with a missing arm, leg, hand, etc.) and produces a visual reconstruction of the missing region while preserving the rest of the figure. The reconstruction is intended as a proof of concept for the technical feasibility of automated sculpture restoration; it does not aim to replace professional restoration or to produce historically accurate results.

The pipeline is organised into four phases (one folder per phase in this repository):

| Phase | Folder | What it does |
|---|---|---|
| 1 | `phase1_classification/` | Builds the synthetic marble-style training set, fine-tunes DeepLabv3+, runs DensePose, removes backgrounds with rembg+SAM, and classifies the sculpture corpus into `whole_body`, `broken_body`, `head_only` and `no_human`. |
| 2 | `phase2_mask_generation/` | Generates the anatomically-shaped v8 mask for each `broken_body` sculpture by mirroring the contralateral preserved limb or by PCA extrapolation. |
| 3 | `phase3_inpainting/` | Runs the three inpainting paradigms (LaMa, MAT, Stable Diffusion with Multi-ControlNet) on the masked sculptures, both off-the-shelf (baseline) and with the full pipeline (DensePose conditioning + bg removal + adversarial fine-tuning), and recomposes the reconstructed region onto the original background. |
| 4 | `phase4_evaluation/` | Trains the global and per-generator PatchGAN evaluators and computes the four families of metrics (FID, chroma a/b, absolute delta-L, PatchGAN scores) on every variant. |

The `utils/` folder contains auxiliary scripts (e.g. synthetic dataset preparation).

Each `phaseX_*` folder also contains a `previous_versions/` subfolder that documents the iteration history of the corresponding component (for transparency and reproducibility).

---

## Pipeline at a glance

```
┌──────────────┐    ┌──────────────┐    ┌────────────────┐    ┌──────────────┐
│   Phase 1    │    │   Phase 2    │    │    Phase 3     │    │   Phase 4    │
│ Classify the │ -> │ Generate v8  │ -> │   Inpainting   │ -> │  Evaluation  │
│   sculpture  │    │ anatomical   │    │ (LaMa, MAT, SD │    │   (FID,      │
│   corpus     │    │    masks     │    │  + ControlNet) │    │ PatchGAN,    │
│              │    │              │    │  + recompose   │    │ chroma, |ΔL|)│
└──────────────┘    └──────────────┘    └────────────────┘    └──────────────┘
```

The two-by-three experimental matrix (three generative paradigms x two configurations: baseline / with full pipeline) is the core comparison reported in the project memoir. **The same v8 anatomical mask is used in both columns of the matrix**, so any difference between baseline and with-pipeline cells is attributable to the three components that change together (background removal, anatomical conditioning, and, for LaMa and MAT, adversarial fine-tuning).

---

## Models used

The project evaluates the same three generative paradigms in two configurations (baseline and with full pipeline). All six configurations receive the v8 anatomical mask of Phase 2 as the region to reconstruct; what changes between the baseline and the with-pipeline cell is the rest of the inputs and the model itself.

### Generative inpainting models (the reconstruction backbone)

| Paradigm | Baseline (without pipeline) | With full pipeline |
|---|---|---|
| **LaMa** (FFC, GAN-based, Places2 pre-trained) | Off-the-shelf weights, v8 anatomical mask on the original photograph with its museum background. No DensePose conditioning, no bg removal, no adversarial fine-tuning. | Adversarially fine-tuned on the synthetic marble-style dataset of Phase 1 with three extra DensePose (I, U, V) input channels. Receives the v8 mask on the background-removed input, and the output is recomposed onto the original background. |
| **MAT** (Mask-Aware Transformer, Places2 pre-trained) | Off-the-shelf weights, v8 anatomical mask on the original photograph with its museum background. No DensePose conditioning, no bg removal, no adversarial fine-tuning. | Same treatment as LaMa: adversarially fine-tuned on the synthetic marble-style dataset with (I, U, V) channels, v8 mask on the background-removed input, recomposition on the original background. |
| **Stable Diffusion Inpainting** (latent diffusion, LAION-5B pre-trained) | Off-the-shelf weights with a generic prompt, v8 anatomical mask on the original photograph with its museum background. No bg removal, no ControlNet branch. | Off-the-shelf weights (no fine-tuning) with the same prompt, v8 anatomical mask on the background-removed input, and Multi-ControlNet conditioning (one ControlNet on the DensePose segmentation map, one on an OpenPose-style skeleton derived from the DensePose regions). Output is recomposed onto the original background. |

The two-phase adversarial schedule for LaMa and MAT uses an adversarial weight of 0.01 for 8 epochs followed by 0.5 for 6 more epochs, applied identically to both models for fairness. SD is never adversarially fine-tuned; the only thing that changes between SD baseline and SD with-pipeline is the bg removal and the ControlNet conditioning.

### Auxiliary models (not generative)

- **DeepLabv3+** fine-tuned on a synthetic marble-style dataset (Reinhard color transfer + histogram matching on COCO 2014 + DensePose annotations, around 26,437 image-mask pairs). Used in Phase 1 as a coarse classifier.
- **DensePose** (pre-trained, no fine-tuning) for dense per-pixel anatomical analysis (24 fine-grained SMPL parts plus UV coordinates).
- **rembg + U2-Net + SAM** for background removal.
- **PatchGAN** trained from scratch as a domain-specific adversarial evaluator (global and per-generator variants).

---

## Repository structure

```
.
├── .gitignore
├── phase1_classification/
│   ├── style_transfer.py
│   ├── data_augmentation.py
│   ├── finetune_deeplabv3_full.py
│   ├── img_classification.py
│   ├── extract_densepose.py
│   ├── classify_with_densepose.py
│   ├── delete_background.py
│   └── previous_versions/
├── phase2_mask_generation/
│   ├── compute_mask_from_densepose_v8.py
│   └── previous_versions/
├── phase3_inpainting/
│   ├── lama/
│   │   ├── lama_inpainting.py            # baseline (off-the-shelf, v8 mask, raw image)
│   │   ├── finetune_lama_v9_adversarial.py
│   │   ├── lama_v9_adversarial.py        # with full pipeline (v9 = v8 + adversarial fine-tuning + bg removal)
│   │   └── previous_versions/
│   ├── mat/
│   │   ├── mat_v1_real.py                # baseline (off-the-shelf, v8 mask, raw image)
│   │   ├── finetune_mat_v9_adversarial.py
│   │   ├── mat_v9_adversarial.py         # with full pipeline (same recipe as LaMa)
│   │   └── previous_versions/
│   ├── stable_diffusion/
│   │   ├── sd_baseline_sin_pipeline.py   # baseline (raw + v8 mask, no bg removal, no ControlNet)
│   │   ├── sd_controlnet_densepose.py    # with full pipeline (bg-removed + v8 mask + Multi-ControlNet)
│   │   └── previous_versions/
│   ├── recomponer_con_fondo.py           # recomposition onto the original background
│   └── previous_versions/
├── phase4_evaluation/
│   ├── train_patchgan_evaluator.py       # global evaluator
│   ├── train_patchgan_per_gen.py         # per-generator evaluators
│   ├── score_patchgan_evaluator.py
│   ├── score_patchgan_per_gen.py
│   ├── metricas_finales.py               # FID, chroma a/b, |ΔL|
│   └── previous_versions/
└── utils/
    └── crear_synthetic_no_bg.py
```

---

## How to run

Each phase is meant to be run sequentially, and each script expects the outputs of the previous phase as inputs. The execution is designed for a GPU cluster (the project was developed on the UAB Escola d'Enginyeria GPU cluster). The typical order is:

```bash
# Phase 1: build the synthetic dataset, fine-tune DeepLabv3+, classify the corpus
python phase1_classification/style_transfer.py
python phase1_classification/finetune_deeplabv3_full.py
python phase1_classification/img_classification.py
python phase1_classification/delete_background.py
python phase1_classification/extract_densepose.py
python phase1_classification/classify_with_densepose.py

# Phase 2: generate the v8 anatomical masks
python phase2_mask_generation/compute_mask_from_densepose_v8.py

# Phase 3: run the three paradigms in both configurations on the v8 mask
# LaMa
python phase3_inpainting/lama/lama_inpainting.py                   # baseline
python phase3_inpainting/lama/finetune_lama_v9_adversarial.py
python phase3_inpainting/lama/lama_v9_adversarial.py               # with pipeline
# MAT
python phase3_inpainting/mat/mat_v1_real.py                        # baseline
python phase3_inpainting/mat/finetune_mat_v9_adversarial.py
python phase3_inpainting/mat/mat_v9_adversarial.py                 # with pipeline
# Stable Diffusion
python phase3_inpainting/stable_diffusion/sd_baseline_sin_pipeline.py    # baseline
python phase3_inpainting/stable_diffusion/sd_controlnet_densepose.py     # with pipeline
# Recomposition onto the original background (for the with-pipeline outputs)
python phase3_inpainting/recomponer_con_fondo.py

# Phase 4: train the evaluators and compute the metrics
python phase4_evaluation/train_patchgan_evaluator.py
python phase4_evaluation/train_patchgan_per_gen.py
python phase4_evaluation/score_patchgan_evaluator.py
python phase4_evaluation/score_patchgan_per_gen.py
python phase4_evaluation/metricas_finales.py
```

Paths to the input dataset, the pre-trained model checkpoints (LaMa Places2 weights, MAT Places2 weights, Stable Diffusion Inpainting checkpoint, DensePose weights) and the output directories are configured at the top of each script.

---

## Dataset

The project uses two complementary datasets:

- **COCO 2014 + DensePose annotations** (approximately 26,437 image-mask pairs), used through a marble-style transfer to build the synthetic training set for DeepLabv3+.
- **A curated collection of 3,315 sculpture photographs** (1,755 `whole_body`, 667 `broken_body`, 893 `head_only`) compiled from Wikimedia Commons, the Metropolitan Museum of Art API, and two public Kaggle datasets. Of these, 661 broken-body sculptures with a non-empty mask are used as the evaluation corpus.

The dataset itself is not redistributed in this repository (due to size and licensing); the collection process is documented in the project memoir.

---

## Notes on reproducibility

- The inpainting models go through a sequence of iterations (`v1` to `v9` for LaMa and MAT, plus the baseline / with-ControlNet split for SD), all documented in the `previous_versions/` subfolders. The variants used in the main results are: `lama_inpainting.py` (baseline) and `lama_v9_adversarial.py` (with pipeline) for LaMa; `mat_v1_real.py` (baseline) and `mat_v9_adversarial.py` (with pipeline) for MAT; `sd_baseline_sin_pipeline.py` (baseline) and `sd_controlnet_densepose.py` (with pipeline) for SD.
- The mask generator also went through eight iterations (`v1` to `v8`), documented in `phase2_mask_generation/previous_versions/`. The final mask generator used in **both columns** of the experimental matrix is `compute_mask_from_densepose_v8.py`.
- The PatchGAN evaluator was trained for 15 epochs without explicit early stopping (best validation accuracy checkpoint kept). The LaMa and MAT adversarial fine-tunings use early stopping on validation loss within each phase of the two-phase schedule. DeepLabv3+ is trained for the full 50 epochs without explicit early stopping (best validation mIoU checkpoint kept).

---

## Author

**Clara Escuder Inhiesto**
Final Degree Project in Artificial Intelligence
Escola d'Enginyeria, Universitat Autònoma de Barcelona
Academic year 2025/26

Supervised by Álvaro Wong González (Area of Architecture and Computer Technology, UAB).

Contact: claraesc.04@gmail.com
