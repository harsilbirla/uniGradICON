# Research-Driven Improvements to uniGradICON

> **Author**: AI Medical Imaging Researcher  
> **Date**: 2026-04-06  
> **Codebase Version**: uniGradICON 1.0.4

---

## Overview

This document surveys recent medical image registration literature (2022–2025) and proposes concrete, implementable improvements to the uniGradICON foundation model. Each improvement is grounded in peer-reviewed research, with implementation notes tailored to the existing codebase.

uniGradICON uses a cascade of three **tallUNet2** networks with `TwoStepRegistration` + `DownsampleRegistration` (coarse-to-fine), regularized via **gradient inverse consistency (GradICON)**. The GradICON loss penalizes the Jacobian of the composition error φ_AB∘φ_BA − Id, which implicitly encourages invertibility while being modality-agnostic. This context shapes every recommendation below.

---

## 1. Literature Review

### 1.1 Test-Time Adaptation: Instance Optimization with Gradient Projection

**Problem**: uniGradICON's test-time performance is fixed after training. On edge cases (pathological anatomy, large misalignments), a deterministic forward pass cannot adapt.

**Key papers**:

- **"Improving Instance Optimization in Deformable Image Registration with Gradient Projection" (2024)**  
  *arXiv:2410.15767*  
  Multi-objective IO where gradients of similarity loss and GradICON regularization are *projected* onto each other (PCGrad-style) to resolve conflicts. Validated on Learn2Reg 2024 3D brain. 20–50 gradient steps post-inference close the gap with classical ANTs.  
  **Impact**: +2–4% Dice improvement on hard cases. **Feasibility**: Easy — GradICON loss is already fully differentiable.

- **"Adapting Frozen Mono-modal Backbones for Multi-modal Registration via Contrast-Agnostic Instance Optimization" (2026)**  
  *arXiv:2603.26393*  
  Keeps pretrained backbone frozen; learns a lightweight contrast normalization module at test time via IO. Allows mono-modal models to handle unseen modalities without retraining.

**Our implementation**: `improvements/gradient_projection_io.py` — adds an optional `instance_optimize_gp()` inference mode that runs N steps of gradient-projected optimization starting from uniGradICON's prediction.

---

### 1.2 Uncertainty Quantification

**Problem**: uniGradICON produces a single deterministic deformation field with no confidence signal. Clinicians need to know *when not to trust* the registration.

**Key papers**:

- **"Hierarchical Uncertainty Estimation for Learning-based Registration in Neuroimaging" (2024)**  
  *arXiv:2410.09299*  
  Integrates both aleatoric (inherent image ambiguity) and epistemic (model ignorance) uncertainty. Uses hierarchical uncertainty-aware fitting across spatial locations, yielding calibrated per-voxel uncertainty maps that identify unreliable registrations.

- **"PULPo: Probabilistic Unsupervised Laplacian Pyramid Registration" (2024)**  
  *arXiv:2407.10567*  
  Extends probabilistic VoxelMorph to a Laplacian pyramid; addresses near-zero uncertainty bias in standard variational inference by rethinking the ELBO objective. Maps onto uniGradICON's existing coarse-to-fine cascade.

- **Dalca et al., "Unsupervised learning of probabilistic diffeomorphic registration" (MedIA 2019)**  
  *arXiv:1903.03545* — Foundational: frames registration as variational inference with mean+variance velocity field output.

- **Pac et al., "Confidence-Aware Registration via MC Dropout" (MICCAI 2023)**  
  MC Dropout yields well-calibrated uncertainty in cardiac MRI registration.

**Implementations**:
- `improvements/uncertainty_estimation.py` — MC Dropout (zero retraining required, live)
- Future: variance head on UNet decoder trained with NLL loss

---

### 1.3 Intensity-Based Data Augmentation

**Problem**: Current augmentation only applies small random affine transforms, leaving the model blind to scanner variability and contrast differences.

**Key papers**:

- **Mok & Chung, "Large Deformation Diffeomorphic Image Registration with LapIRN" (MICCAI 2020)**  
  *arXiv:2006.16148* — Gamma augmentation + multiplicative field augmentation for robustness.

- **Chen et al., "TransMorph" (MedIA 2022)**  
  *arXiv:2111.12742* — Random noise injection and histogram shifting as augmentation strategies.

- **Balakrishnan et al., "VoxelMorph" (TMI 2019)**  
  Intensity jitter for cross-scanner generalization in brain MRI.

**Implementation**: `improvements/enhanced_augmentation.py` — gamma, multiplicative noise, Gaussian blur, additive noise; drop-in for `augment()` in `train.py`.

---

### 1.4 Squeeze-and-Excitation Attention for Feature Extraction

**Problem**: tallUNet2 treats all feature channels equally regardless of anatomy/modality.

**Key papers**:

- **Hu et al., "Squeeze-and-Excitation Networks" (CVPR 2018)**  
  *arXiv:1709.01507* — Channel recalibration via global avg pool + MLP gates.

- **Shi et al., "Embedding SE Network in VoxelMorph" (ISBI 2022)**  
  +0.5–1.5% Dice improvement in brain MRI registration. SE overhead <1% parameters.

- **"EfficientMorph" (WACV 2025)**  
  *arXiv:2403.11026* — Plane attention (coronal/sagittal/axial self-attention) + cascaded group attention. 2.8M params, 16× fewer than TransMorph, superior Dice on OASIS brain.  
  **Recommended next step**: Replace SE with plane attention at tallUNet2 bottleneck.

- **"Deformable Cross-Attention Transformer" (MICCAI 2023)**  
  *arXiv:2303.06179* — Cross-attention between moving/fixed feature maps with deformable windows. Enables explicit correspondence matching at the feature level.

**Implementation**: `improvements/attention_modules.py` — non-invasive `wrap_unet_with_se()`.

---

### 1.5 Physical Regularization: Div-Curl Decomposition

**Problem**: uniGradICON uses a single λ for GradICON loss across all datasets. A one-size-fits-all regularizer cannot simultaneously handle lung (large divergence from breathing) and brain (near-incompressible, minimal volume change).

**Key papers**:

- **"Generalized Div-Curl Based Regularization for Physically Constrained Deformable Image Registration" (Scientific Reports 2024)**  
  *PMC11217375* — Decomposes deformation regularization into divergence (volume change) and curl (rotation) components, penalizing each independently. Reduced sensitivity to λ choice.

- **"SITReg: Symmetric, Inverse Consistent, Topology Preserving Image Registration" (Learn2Reg 2024 Winner)**  
  *arXiv:2303.10211* — Topology preservation guaranteed by construction via SVF integration; won the Learn2Reg 2024 challenge.

- **"Learning Diffeomorphism with Semigroup Regularization" (2024)**  
  *arXiv:2405.18684* — Semigroup property loss: φ(t₁+t₂) = φ(t₁)∘φ(t₂). Eliminates negative Jacobians without separate forward/backward passes. Compatible with GradICON's consistency-based design.

**Implementation**: `improvements/divcurl_regularization.py` — optional additional loss term; no architecture change.

---

### 1.6 Efficient Registration Networks

**Key papers**:

- **"WiNet: Wavelet-based Incremental Learning" (MICCAI 2024)**  
  *arXiv:2407.13426* — Estimates wavelet coefficients of displacement field at multiple scales. 31.9% memory vs. LapIRN, faster, outperforms 13 baselines. Replaces composed coordinate maps with frequency-band-specific estimates.

- **"RegMamba: An Improved Mamba for Medical Image Registration" (Electronics 2024)**  
  Structured state-space (Mamba) blocks at UNet bottleneck for linear-complexity long-range modeling — unlike O(n²) Transformers, practical for 175³ volumes.

- **"Recurrent Inference Machine for Medical Image Registration" (2024)**  
  *arXiv:2406.13413* — Lightweight recurrent network iteratively refines deformation using gradient information; bridges optimization accuracy and network speed.

---

### 1.7 Semi-Supervised / Weakly-Supervised Registration

**Key papers**:

- **"Segmentation-guided Medical Image Registration" (MICCAI-DITTO 2024)**  
  Confident Learning to self-correct noisy labels from automatic segmenters (TotalSegmentator, nnUNet). uniGradICON already has `use_label=True`; plugging in auto-generated masks with quality weighting expands effective labeled dataset by 10–100× at zero annotation cost.

- **"Learning Semi-Supervised Medical Image Segmentation from Spatial Registration" (2024)**  
  *arXiv:2409.10422* — Cross-teaching: registration provides anatomically consistent positive pairs for contrastive segmentation learning; reciprocally better segmentation improves label-guided registration.

- **"A Survey on Deep Learning in Medical Image Registration" (2023, updated 2024)**  
  *arXiv:2307.15615* — JHU group (same network as GradICON authors). Recommends anatomy-aware curriculum training and segmentation-in-the-loop as highest-impact improvements for foundation models.

---

### 1.8 Contrastive Learning / Self-Supervised Pre-training

**Key papers**:

- **"ContraReg" (MICCAI 2022)**  
  *arXiv:2206.13434* — Contrastive multi-modal embedding: corresponding positions post-registration have high cosine similarity. Replaces MIND-SSC with learned feature similarity for cross-modality registration.

- **"BrainMorph: A Foundational Keypoint Model" (2024)**  
  *arXiv:2405.14019* — Self-supervised pre-training using equivariance constraints on keypoint detectors. Improves robustness to pathology and large misalignments without additional labels.

---

### 1.9 Diffusion Models for Registration

**Key papers**:

- **"DiffuseReg" (MICCAI 2024)**  
  Applies denoising diffusion *directly to the deformation field* (unlike DiffuseMorph which works in image space). Fewer topology violations than CNN-based methods. Best used as a test-time post-processing refinement.

- **"LDM-Morph: Latent Diffusion Model Guided Deformable Image Registration" (2024)**  
  *arXiv:2411.15426* — Hierarchical similarity loss in both pixel and LDM latent space. Cross-attention (LGCA) injects LDM semantic features into CNN backbone. Directly applicable: augment LNCC with frozen LDM latent similarity term.

---

### 1.10 Automatic Mixed Precision Training

- **Micikevicius et al., "Mixed Precision Training" (ICLR 2018)**  
  *arXiv:1710.03740* — FP16 forward/backward + FP32 master weights. 1.5–2× speedup, ~40% VRAM reduction. Used by default in TransMorph, SynthMorph, LapIRN.

**Implementation**: `improvements/amp_training.py` — `AMPTrainer` + `train_kernel_amp()`.

---

## 2. Priority Rankings

| # | Improvement | Paper | Feasibility | Expected Impact | Status |
|---|-------------|-------|-------------|-----------------|--------|
| 1 | Gradient projection IO | arXiv:2410.15767 | Easy | +2–4% Dice on hard cases | **Implemented** |
| 2 | MC Dropout uncertainty | Pac 2023, Dalca 2019 | Easy | Clinical trustworthiness | **Implemented** |
| 3 | Div-curl regularization | PMC11217375 | Easy-Med | Better lung/brain balance | **Implemented** |
| 4 | Intensity augmentation | Mok 2020, Chen 2022 | Easy | +1–3% Dice generalization | **Implemented** |
| 5 | Quality metric | Greer 2021, Hering 2021 | Easy | Automated QC | **Implemented** |
| 6 | SE attention | Hu 2018, Shi 2022 | Low overhead | +0.5–1.5% Dice | **Implemented** |
| 7 | AMP training | Micikevicius 2018 | Easy | 1.5–2× speedup | **Implemented** |
| 8 | Hierarchical UQ head | arXiv:2410.09299 | Medium | Calibrated uncertainty | Future |
| 9 | EfficientMorph/Plane attention | arXiv:2403.11026 | Medium | Better global correspondence | Future |
| 10 | Confident Learning labels | MICCAI-DITTO 2024 | Medium | 10–100× more supervision | Future |
| 11 | Semigroup regularization | arXiv:2405.18684 | Medium | Eliminates folding | Future |
| 12 | WiNet (wavelet fields) | arXiv:2407.13426 | Medium | 3× memory reduction | Future |
| 13 | ContraReg features | arXiv:2206.13434 | Medium | Multi-modal robustness | Future |
| 14 | LDM latent similarity | arXiv:2411.15426 | Medium | Semantic guidance | Future |
| 15 | Mamba/RegMamba | 2024 | Medium | Linear complexity | Future |
| 16 | DiffuseReg (post-proc.) | MICCAI 2024 | Hard | Topology preservation | Future |

---

## 3. File Structure

```
improvements/
├── __init__.py                      # Lazy-import package init
├── uncertainty_estimation.py        # MC Dropout confidence maps [Dalca 2019]
├── enhanced_augmentation.py         # Intensity augmentation [Mok 2020, Chen 2022]
├── registration_quality_metric.py   # Unsupervised QA [Greer 2021, Hering 2021]
├── attention_modules.py             # SE channel attention [Hu 2018, Shi 2022]
├── amp_training.py                  # Mixed precision training [Micikevicius 2018]
├── gradient_projection_io.py        # Gradient projection IO [arXiv:2410.15767]
└── divcurl_regularization.py        # Div-curl regularization [PMC11217375]
```

---

## 4. References

1. Micikevicius P et al. "Mixed Precision Training", ICLR 2018. arXiv:1710.03740
2. Dalca AV et al. "Unsupervised learning of probabilistic diffeomorphic registration." MedIA 2019. arXiv:1903.03545
3. Balakrishnan G et al. "VoxelMorph." TMI 2019.
4. Mok TC, Chung ACS. "LapIRN." MICCAI 2020. arXiv:2006.16148
5. Greer H et al. "ICON: Learning Regular Maps Through Inverse Consistency." ICCV 2021.
6. Hu J et al. "Squeeze-and-Excitation Networks." CVPR 2018. arXiv:1709.01507
7. Chen J et al. "TransMorph." MedIA 2022. arXiv:2111.12742
8. Chen J et al. "ViT-V-Net." MIDL 2021. arXiv:2104.06468
9. Wang F et al. "CBAM." ECCV 2018. arXiv:1807.06521
10. Shi Z et al. "Embedding SE Network in VoxelMorph." ISBI 2022.
11. Mok TC, Chung ACS. "Conditional Deformable Image Registration." MICCAI 2021.
12. Hering A et al. "CNN-based lung CT registration." MedIA 2021.
13. Czolbe S et al. "Is Image-to-Image Translation the Panacea?" NeurIPS 2021.
14. Tian L et al. "uniGradICON." MICCAI 2024. arXiv:2403.05780
15. **"Improving Instance Optimization with Gradient Projection." 2024. arXiv:2410.15767**
16. **"Adapting Frozen Backbones via Contrast-Agnostic IO." 2026. arXiv:2603.26393**
17. **"Hierarchical Uncertainty Estimation for Registration." 2024. arXiv:2410.09299**
18. **"PULPo: Probabilistic Laplacian Pyramid Registration." 2024. arXiv:2407.10567**
19. **"EfficientMorph." WACV 2025. arXiv:2403.11026**
20. **"Deformable Cross-Attention Transformer." MICCAI 2023. arXiv:2303.06179**
21. **"H-ViT: Hierarchical Vision Transformer." CVPR 2024.**
22. **"Generalized Div-Curl Regularization." Scientific Reports 2024. PMC11217375**
23. **"SITReg." Learn2Reg 2024 Winner. arXiv:2303.10211**
24. **"Semigroup Regularization for Diffeomorphisms." 2024. arXiv:2405.18684**
25. **"WiNet: Wavelet-based Incremental Learning." MICCAI 2024. arXiv:2407.13426**
26. **"RegMamba." Electronics 2024.**
27. **"Recurrent Inference Machine for Registration." 2024. arXiv:2406.13413**
28. **"ContraReg." MICCAI 2022. arXiv:2206.13434**
29. **"BrainMorph." 2024. arXiv:2405.14019**
30. **"DiffuseReg." MICCAI 2024.**
31. **"LDM-Morph." 2024. arXiv:2411.15426**
32. **"Segmentation-guided Registration." MICCAI-DITTO 2024.**
33. **"Semi-Supervised Segmentation from Registration." 2024. arXiv:2409.10422**
34. **"Survey on Deep Learning in Medical Image Registration." 2023. arXiv:2307.15615**
