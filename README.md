# Visual Reconstruction of Incomplete Classical Sculptures Using Artificial Intelligence

UAB Final Degree Project in Artificial Intelligence, Escola d'Enginyeria, 2025/26.

This repository contains the implementation of an end-to-end automatic pipeline for the reconstruction of missing body parts in photographs of classical Greco-Roman sculptures, using a combination of semantic segmentation, dense human-body analysis, anatomically-shaped inpainting masks, three generative inpainting paradigms (LaMa, MAT, Stable Diffusion with Multi-ControlNet) and a PatchGAN evaluator trained from scratch as a domain-specific quality metric.

---

## Project overview

The system takes a photograph of a damaged classical sculpture (a marble figure with a missing arm, leg, hand, etc.) and produces a visual reconstruction of the missing region while preserving the rest of the figure. The reconstruction is intended as a proof of concept for the technical feasibility of automated sculpture restoration; it does not aim to replace professional restoration or to produce historically accurate results.

The pipeline is organised into four sequential phases, each one in its own folder:

| Phase | Folder | What it does |
|---|---|---|
| 1 | `phase1_classification/` | Builds the synthetic marble-style training set, fine-tunes DeepLabv3+, runs DensePose, removes backgrounds with rembg+SAM, and classifies the sculpture corpus into `whole_body`, `broken_body`, `head_only` and `no_human`. |
| 2 | `phase2_mask_generation/` | Generates the anatomically-shaped mask for each `broken_body` sculpture by mirroring the contralateral preserved limb or by PCA extrapolation. |
| 3 | `phase3_inpainting/` | Runs the three inpainting paradigms (LaMa, MAT, Stable Diffusion with Multi-ControlNet) on the masked sculptures, both off-the-shelf (baseline) and with the full pipeline (DensePose conditioning + background removal + adversarial fine-tuning), and recomposes the reconstructed region onto the original background. |
| 4 | `phase4_evaluation/` | Trains the global and per-generator PatchGAN evaluators and computes the four families of metrics (FID, chroma a/b, absolute delta-L, PatchGAN scores) on every variant. |

The `utils/` folder contains auxiliary scripts (e.g. synthetic dataset preparation). Each phase folder also has a `previous_versions/` subfolder that keeps the iteration history of that component, so that someone reading the code can trace the design decisions back to the experiments that motivated them.

The full experimental design is a two-by-three matrix: three generative paradigms (LaMa, MAT, SD) evaluated in two configurations each (baseline / with the full pipeline). The same anatomical mask is used in both columns, so any difference between a baseline cell and its with-pipeline counterpart is attributable to the three components that change together: background removal, anatomical conditioning, and (for LaMa and MAT) adversarial fine-tuning. The detailed comparison is reported in the project memoir.

---

## Models used

All six configurations receive the same anatomical mask of Phase 2 as the region to reconstruct; what changes between the baseline and the with-pipeline cell is the rest of the inputs and the model itself.

### Generative inpainting models (the reconstruction backbone)

| Paradigm | Baseline (without pipeline) | With full pipeline |
|---|---|---|
| **LaMa** (FFC, GAN-based, Places2 pre-trained) | Off-the-shelf weights, anatomical mask on the original photograph with its museum background. No DensePose conditioning, no background removal, no adversarial fine-tuning. | Adversarially fine-tuned on the synthetic marble-style dataset of Phase 1 with three extra DensePose (I, U, V) input channels. Receives the anatomical mask on the background-removed input, and the output is intended to be recomposed onto the original background. |
| **MAT** (Mask-Aware Transformer, Places2 pre-trained) | Off-the-shelf weights, anatomical mask on the original photograph with its museum background. No DensePose conditioning, no background removal, no adversarial fine-tuning. | Same treatment as LaMa: adversarially fine-tuned on the synthetic marble-style dataset with (I, U, V) channels, anatomical mask on the background-removed input, recomposition on the original background. |
| **Stable Diffusion Inpainting** (latent diffusion, LAION-5B pre-trained) | Off-the-shelf weights with a generic prompt, anatomical mask on the original photograph with its museum background. No background removal, no ControlNet branch. | Off-the-shelf weights (no fine-tuning) with the same prompt, anatomical mask on the background-removed input, and Multi-ControlNet conditioning (one ControlNet on the DensePose segmentation map, one on an OpenPose-style skeleton derived from the DensePose regions). Output is recomposed onto the original background. |

The two-phase adversarial schedule for LaMa and MAT uses an adversarial weight of 0.01 for 8 epochs followed by 0.5 for 6 more epochs, applied identically to both models for fairness. SD is never adversarially fine-tuned; the only thing that changes between SD baseline and SD with-pipeline is the background removal and the ControlNet conditioning.

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
│   │   ├── ffc_standalone.py             # FFC architecture, see note below
│   │   ├── lama_inpainting.py            # baseline (off-the-shelf, anatomical mask, raw image)
│   │   ├── finetune_lama_v9_adversarial.py
│   │   ├── lama_v9_adversarial.py        # with full pipeline
│   │   └── previous_versions/
│   ├── mat/
│   │   ├── mat_v1_real.py                # baseline (off-the-shelf, anatomical mask, raw image)
│   │   ├── finetune_mat_v9_adversarial.py
│   │   ├── mat_v9_adversarial.py         # with full pipeline (same recipe as LaMa)
│   │   └── previous_versions/
│   ├── stable_diffusion/
│   │   ├── sd_baseline_sin_pipeline.py   # baseline (raw + anatomical mask, no bg removal, no ControlNet)
│   │   ├── sd_controlnet_densepose.py    # with full pipeline (bg-removed + anatomical mask + Multi-ControlNet)
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

### A note on `ffc_standalone.py`

`phase3_inpainting/lama/ffc_standalone.py` is not an inpainting iteration; it is the FFCResNetGenerator architecture (the network used by LaMa) extracted from the official `advimman/lama` repository and repackaged so that it does not depend on the `saicinpainting` Python package. The original package brings in `albumentations 0.5`, `pytorch_lightning 1.6` and a few other dependencies that are broken in modern environments, which made it impractical to install on the UAB cluster. The standalone file only needs `torch` and `kornia`, and it is what `lama_v9_adversarial.py` and `finetune_lama_v9_adversarial.py` import to load the official `big-lama` checkpoint. So even though it is just a helper file, it is part of the current pipeline.

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

# Phase 2: generate the anatomical masks
python phase2_mask_generation/compute_mask_from_densepose_v8.py

# Phase 3: run the three paradigms in both configurations on the anatomical mask
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

## Results at a glance

The qualitative ranking, confirmed by visual inspection across the 661 broken-body sculptures of the corpus, is **SD > MAT > LaMa**:

- **SD with ControlNet** is the only paradigm that generates a reconstruction whose shape and details (proportions, pose, marble texture, shading) are its own and not bound to the silhouette of the mask. It is the most realistic of the three on well-masked cases. On edge cases where the mask is misplaced, it can fall back to non-arm body parts (wings, halos) or non-body objects (furniture) — on those cases the bottleneck is the mask, not the generator.

- **MAT with pipeline** commits to a reconstruction inside the mask: it fills the masked region with marble of coherent colour and reasonable texture, but the shape of the reconstruction is the silhouette of the anatomical mask itself rather than an independent anatomical commitment, and it consistently leaves a visible boundary gap at the joint between the surviving body and the reconstructed limb.

- **LaMa with pipeline** does not produce a visible reconstruction (the model converges to leaving the input essentially unmodified). The intermediate v7 variant of LaMa, with DensePose conditioning but without the adversarial fine-tuning, isolates the adversarial fine-tuning as the specific component that breaks LaMa under a discriminator that learned the background as a shortcut.

The pipeline also generalises across materials beyond white marble (limestone, oxidised green bronze) and across moderate deviations from realistic anatomy, with the same qualitative behaviour as on the standard classical-sculpture cases.

### A note on the quantitative numbers

The recomposition step of the pipeline did not behave as intended at evaluation time: a colour-threshold heuristic assumed a black background while rembg+SAM exports a white one, so the with-pipeline outputs were evaluated on a uniform white background instead of the museum background that the recomposition was designed to restore. The bias goes against the pipeline (the with-pipeline outputs are compared against a museum-background reference, so a uniform-background output looks artificially further from it), so the FID and PatchGAN improvements reported in the project memoir are conservative; the chroma a/b and absolute delta-L metrics are computed only inside the body region and are not affected. The qualitative findings above are also unaffected and stand independently of any re-evaluation. A re-evaluation with a corrected recomposition is listed as a future-work item. The full numerical tables and their interpretation (including the trace-term mechanism that explains why LaMa's FID drops despite producing no visible reconstruction) are discussed in the project memoir.

---

## Dataset

The project uses two complementary datasets:

- **COCO 2014 + DensePose annotations** (approximately 26,437 image-mask pairs), used through a marble-style transfer to build the synthetic training set for DeepLabv3+.
- **A curated collection of 3,315 sculpture photographs** (1,755 `whole_body`, 667 `broken_body`, 893 `head_only`) compiled from Wikimedia Commons, the Metropolitan Museum of Art API, and two public Kaggle datasets. Of these, 661 broken-body sculptures with a non-empty mask are used as the evaluation corpus.

The dataset itself is not redistributed in this repository (due to size and licensing); the collection process is documented in the project memoir.

---

## Notes on reproducibility

- The inpainting models go through a sequence of iterations, all documented in the `previous_versions/` subfolders. The variants used in the main results are: `lama_inpainting.py` (baseline) and `lama_v9_adversarial.py` (with pipeline) for LaMa; `mat_v1_real.py` (baseline) and `mat_v9_adversarial.py` (with pipeline) for MAT; `sd_baseline_sin_pipeline.py` (baseline) and `sd_controlnet_densepose.py` (with pipeline) for SD.
- The mask generator also went through several iterations, documented in `phase2_mask_generation/previous_versions/`. The final mask generator used in both columns of the experimental matrix is `compute_mask_from_densepose_v8.py`.
- The PatchGAN evaluator was trained for 15 epochs without explicit early stopping (best validation accuracy checkpoint kept). The LaMa and MAT adversarial fine-tunings use early stopping on validation loss within each phase of the two-phase schedule. DeepLabv3+ is trained for the full 50 epochs without explicit early stopping (best validation mIoU checkpoint kept).

---

## Author

**Clara Escuder Inhiesto**
Final Degree Project in Artificial Intelligence
Escola d'Enginyeria, Universitat Autònoma de Barcelona
Academic year 2025/26

Supervised by Álvaro Wong González (Area of Architecture and Computer Technology, UAB).

Contact: claraesc.04@gmail.com
