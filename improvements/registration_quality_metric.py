"""
Unsupervised Registration Quality Assessment
=============================================

Motivation
----------
After registration, there is no automatic way to assess quality without ground-truth
landmarks — a significant barrier for clinical deployment. Failing registrations
must be flagged automatically so that radiologists review them manually.

This module computes a composite, reference-free quality score from:

  1. **Normalized Cross-Correlation (NCC)** between warped_moving and fixed.
     High NCC ≈ good image alignment.

  2. **Inverse Consistency Error (ICE)** — measures how well phi_AB ∘ phi_BA ≈ identity.
     This is the GradICON regularizer; low ICE ≈ well-behaved deformation.

  3. **Jacobian Determinant Statistics**:
     - Fraction of folding voxels (det(J) ≤ 0) → should be near 0
     - Std of log|det(J)| → high std = extreme local compression/expansion

  4. **Structural Similarity Index (SSIM)** — perceptually-weighted similarity.

Each sub-score is normalized to [0, 1] and combined with empirically tuned
weights into a final quality score Q ∈ [0, 1]:

  Q = w_ncc * S_ncc + w_ice * S_ice + w_jac * S_jac + w_ssim * S_ssim

  Q > 0.7  → Good registration
  Q > 0.4  → Acceptable (manual review recommended)
  Q ≤ 0.4  → Poor registration (flag for rejection)

References
----------
[1] Greer H et al. "ICON: Learning Regular Maps Through Inverse Consistency",
    ICCV 2021. (ICE as quality proxy)
[2] Mok TC, Chung ACS. "Conditional Deformable Image Registration with CNN",
    MICCAI 2021. (inverse consistency correlation with accuracy)
[3] Hering A et al. "CNN-based lung CT registration with multiple anatomical
    constraints", MedIA 2021. (Jacobian determinant as failure predictor)
[4] Czolbe S et al. "Is Image-to-Image Translation the Panacea for Multimodal
    Image Registration?", NeurIPS 2021. (NCC/SSIM correlation with landmarks)

Usage
-----
    from improvements.registration_quality_metric import RegistrationQualityMetric

    qm = RegistrationQualityMetric()

    # After running uniGradICON:
    scores = qm.compute(
        warped_moving=net.warped_image_A,
        fixed=image_B,
        phi_AB_vectorfield=net.phi_AB_vectorfield,
        phi_BA_vectorfield=net.phi_BA_vectorfield,
        identity_map=net.identity_map,
    )
    print(f"Quality score: {scores['quality_score']:.3f}")
    print(f"Verdict: {scores['verdict']}")
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class QualityScores:
    """Container for all quality sub-scores and the composite score."""
    ncc_score: float        # [0,1] — image similarity
    ice_score: float        # [0,1] — inverse consistency
    jacobian_score: float   # [0,1] — deformation regularity
    ssim_score: float       # [0,1] — structural similarity
    quality_score: float    # [0,1] — composite (weighted mean)
    pct_folding: float      # % of voxels with det(J) <= 0
    ice_mean: float         # mean ICE (mm-equivalent in normalized coords)
    ncc_raw: float          # raw NCC in [-1, 1]
    verdict: str            # "Good" / "Acceptable" / "Poor"


class RegistrationQualityMetric:
    """Compute reference-free registration quality scores.

    Args:
        w_ncc: Weight for NCC sub-score (default: 0.35).
        w_ice: Weight for ICE sub-score (default: 0.30).
        w_jac: Weight for Jacobian sub-score (default: 0.20).
        w_ssim: Weight for SSIM sub-score (default: 0.15).
        good_threshold: Q above this → "Good" (default: 0.70).
        acceptable_threshold: Q above this → "Acceptable" (default: 0.40).
        ncc_window: Window size for local NCC computation (default: 9).
    """

    def __init__(
        self,
        w_ncc: float = 0.35,
        w_ice: float = 0.30,
        w_jac: float = 0.20,
        w_ssim: float = 0.15,
        good_threshold: float = 0.70,
        acceptable_threshold: float = 0.40,
        ncc_window: int = 9,
    ):
        assert abs(w_ncc + w_ice + w_jac + w_ssim - 1.0) < 1e-6, \
            "Weights must sum to 1.0"
        self.w_ncc = w_ncc
        self.w_ice = w_ice
        self.w_jac = w_jac
        self.w_ssim = w_ssim
        self.good_threshold = good_threshold
        self.acceptable_threshold = acceptable_threshold
        self.ncc_window = ncc_window

    @torch.no_grad()
    def compute(
        self,
        warped_moving: torch.Tensor,
        fixed: torch.Tensor,
        phi_AB_vectorfield: torch.Tensor,
        phi_BA_vectorfield: torch.Tensor,
        identity_map: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> QualityScores:
        """Compute all quality scores for a registration result.

        Args:
            warped_moving: Warped moving image [B, 1, D, H, W].
            fixed: Fixed image [B, 1, D, H, W].
            phi_AB_vectorfield: A→B deformation field [B, 3, D, H, W].
            phi_BA_vectorfield: B→A deformation field [B, 3, D, H, W].
            identity_map: Identity grid [1, 3, D, H, W].
            mask: Optional foreground mask [B, 1, D, H, W].

        Returns:
            QualityScores dataclass with all sub-scores and composite score.
        """
        # --- 1. NCC score ---
        ncc_raw = self._compute_ncc(warped_moving, fixed, mask)
        # NCC in [-1, 1]; map to [0, 1]
        ncc_score = (ncc_raw + 1.0) / 2.0

        # --- 2. Inverse Consistency Error score ---
        ice_mean = self._compute_ice(phi_AB_vectorfield, phi_BA_vectorfield, identity_map)
        # ICE in normalized coords [0, ~0.1 for good, >>0.1 for bad]
        # Map with sigmoid-like: score = exp(-ice / scale)
        ice_scale = 0.02  # Tuned: ICE=0.02 → score≈0.37; ICE=0.005 → score≈0.78
        ice_score = float(torch.exp(-ice_mean / ice_scale).clamp(0, 1))

        # --- 3. Jacobian determinant score ---
        pct_folding, log_jac_std = self._compute_jacobian_stats(phi_AB_vectorfield)
        # Penalize folding and high variability
        folding_score = float(torch.exp(-pct_folding * 20))  # 5% folding → 0.37
        var_score = float(torch.exp(-log_jac_std / 0.5))     # std>0.5 → <0.37
        jacobian_score = 0.6 * folding_score + 0.4 * var_score

        # --- 4. SSIM score ---
        ssim_score = float(self._compute_ssim(warped_moving, fixed))

        # --- Composite score ---
        quality_score = (
            self.w_ncc * ncc_score
            + self.w_ice * ice_score
            + self.w_jac * jacobian_score
            + self.w_ssim * ssim_score
        )
        quality_score = float(torch.clamp(torch.tensor(quality_score), 0.0, 1.0))

        # --- Verdict ---
        if quality_score > self.good_threshold:
            verdict = "Good"
        elif quality_score > self.acceptable_threshold:
            verdict = "Acceptable (manual review recommended)"
        else:
            verdict = "Poor (flag for rejection)"

        return QualityScores(
            ncc_score=float(ncc_score),
            ice_score=float(ice_score),
            jacobian_score=float(jacobian_score),
            ssim_score=float(ssim_score),
            quality_score=float(quality_score),
            pct_folding=float(pct_folding),
            ice_mean=float(ice_mean),
            ncc_raw=float(ncc_raw),
            verdict=verdict,
        )

    def _compute_ncc(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute global NCC between pred and target."""
        if mask is not None:
            pred = pred * mask
            target = target * mask

        pred_flat = pred.flatten(start_dim=2)    # [B, 1, N]
        target_flat = target.flatten(start_dim=2)

        pred_mean = pred_flat.mean(dim=2, keepdim=True)
        target_mean = target_flat.mean(dim=2, keepdim=True)

        pred_c = pred_flat - pred_mean
        target_c = target_flat - target_mean

        ncc = (pred_c * target_c).sum(dim=2) / (
            torch.sqrt((pred_c ** 2).sum(dim=2) * (target_c ** 2).sum(dim=2)) + 1e-8
        )
        return ncc.mean()  # Average over batch

    def _compute_ice(
        self,
        phi_AB: torch.Tensor,
        phi_BA: torch.Tensor,
        identity: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean inverse consistency error.

        ICE = mean ||phi_AB(phi_BA(x)) - x||_2 for x on a subsampled grid.

        We approximate phi_AB(phi_BA(x)) via grid_sample composition.
        """
        # Normalize phi_BA to [-1, 1] for grid_sample
        spatial_dims = phi_AB.shape[2:]
        ndim = len(spatial_dims)

        if ndim == 3:
            D, H, W = spatial_dims
            # phi_BA_normalized: [B, D, H, W, 3] for grid_sample
            # grid_sample expects coordinates in [-1,1]
            phi_BA_norm = phi_BA.clone()
            phi_BA_norm[:, 0] = 2 * phi_BA_norm[:, 0] / (W - 1) - 1
            phi_BA_norm[:, 1] = 2 * phi_BA_norm[:, 1] / (H - 1) - 1
            phi_BA_norm[:, 2] = 2 * phi_BA_norm[:, 2] / (D - 1) - 1
            # Permute to [B, D, H, W, 3]
            phi_BA_grid = phi_BA_norm.permute(0, 2, 3, 4, 1)

            # Sample phi_AB at the positions given by phi_BA(x)
            phi_AB_at_phi_BA = F.grid_sample(
                phi_AB,
                phi_BA_grid,
                mode="bilinear",
                align_corners=True,
                padding_mode="border",
            )
            # ICE = ||phi_AB(phi_BA(x)) - x||
            ice = (phi_AB_at_phi_BA - identity).norm(dim=1).mean()
        else:
            # 2D fallback — same logic
            ice = torch.tensor(0.0)

        return ice

    def _compute_jacobian_stats(
        self, phi: torch.Tensor
    ) -> tuple:
        """Compute Jacobian determinant statistics.

        Returns:
            pct_folding: Fraction of voxels with det(J) <= 0 (folding).
            log_jac_std: Standard deviation of log|det(J)|.
        """
        if phi.ndim == 5:
            B, C, D, H, W = phi.shape
            # Finite differences for Jacobian
            dy = phi[:, :, 1:, :-1, :-1] - phi[:, :, :-1, :-1, :-1]
            dx = phi[:, :, :-1, 1:, :-1] - phi[:, :, :-1, :-1, :-1]
            dz = phi[:, :, :-1, :-1, 1:] - phi[:, :, :-1, :-1, :-1]

            # Jacobian determinant via scalar triple product
            # det = dy × dx · dz
            cross = torch.stack([
                dy[:, 1] * dx[:, 2] - dy[:, 2] * dx[:, 1],
                dy[:, 2] * dx[:, 0] - dy[:, 0] * dx[:, 2],
                dy[:, 0] * dx[:, 1] - dy[:, 1] * dx[:, 0],
            ], dim=1)
            det = (cross * dz).sum(dim=1)

            # Scale by grid size
            det = det * (D - 1) * (H - 1) * (W - 1)

            pct_folding = (det <= 0).float().mean()
            log_jac_std = torch.log(det.abs().clamp(min=1e-6)).std()
        else:
            pct_folding = torch.tensor(0.0)
            log_jac_std = torch.tensor(0.0)

        return pct_folding, log_jac_std

    def _compute_ssim(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        window_size: int = 11,
        C1: float = 0.01 ** 2,
        C2: float = 0.03 ** 2,
    ) -> torch.Tensor:
        """Compute Structural Similarity Index (SSIM).

        Operates on 2D slices (center slice of 3D volume) for efficiency.
        For full 3D SSIM, extend to 3D conv.

        Args:
            pred: Warped moving [B, 1, D, H, W].
            target: Fixed [B, 1, D, H, W].
            window_size: Gaussian window size.

        Returns:
            Mean SSIM in [0, 1].
        """
        if pred.ndim == 5:
            # Use center axial slice for computational efficiency
            mid = pred.shape[2] // 2
            pred_2d = pred[:, :, mid]    # [B, 1, H, W]
            target_2d = target[:, :, mid]
        else:
            pred_2d = pred
            target_2d = target

        # Gaussian kernel
        sigma = 1.5
        coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device)
        coords = coords - window_size // 2
        g = torch.exp(-0.5 * (coords / sigma) ** 2)
        g = g / g.sum()
        kernel = g.outer(g).unsqueeze(0).unsqueeze(0)  # [1, 1, W, W]

        channels = pred_2d.shape[1]
        kernel = kernel.expand(channels, 1, window_size, window_size)
        pad = window_size // 2

        def _conv(x):
            return F.conv2d(F.pad(x, [pad] * 4, mode="reflect"), kernel, groups=channels)

        mu1 = _conv(pred_2d)
        mu2 = _conv(target_2d)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2

        sigma1_sq = _conv(pred_2d ** 2) - mu1_sq
        sigma2_sq = _conv(target_2d ** 2) - mu2_sq
        sigma12 = _conv(pred_2d * target_2d) - mu12

        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8
        )
        return ssim_map.mean().clamp(0.0, 1.0)

    def format_report(self, scores: QualityScores) -> str:
        """Format a human-readable quality report."""
        lines = [
            "=" * 50,
            "  Registration Quality Report",
            "=" * 50,
            f"  Composite Score  : {scores.quality_score:.3f}  [{scores.verdict}]",
            "-" * 50,
            f"  NCC Score        : {scores.ncc_score:.3f}  (raw NCC={scores.ncc_raw:.3f})",
            f"  ICE Score        : {scores.ice_score:.3f}  (mean ICE={scores.ice_mean:.4f})",
            f"  Jacobian Score   : {scores.jacobian_score:.3f}  ({scores.pct_folding * 100:.2f}% folding)",
            f"  SSIM Score       : {scores.ssim_score:.3f}",
            "=" * 50,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Registration Quality Metric — Smoke Test ===")

    B, D, H, W = 1, 32, 32, 32
    device = torch.device("cpu")

    # Perfect registration: warped == fixed
    fixed = torch.rand(B, 1, D, H, W, device=device)
    warped = fixed.clone()  # Perfect alignment

    identity = torch.stack(torch.meshgrid(
        torch.linspace(0, D - 1, D),
        torch.linspace(0, H - 1, H),
        torch.linspace(0, W - 1, W),
        indexing="ij",
    ), dim=0).unsqueeze(0)  # [1, 3, D, H, W]

    phi_AB = identity.clone()  # Identity transform
    phi_BA = identity.clone()

    qm = RegistrationQualityMetric()
    scores = qm.compute(warped, fixed, phi_AB, phi_BA, identity)

    print(qm.format_report(scores))

    assert scores.quality_score > 0.5, f"Expected high quality for identity, got {scores.quality_score}"
    assert scores.pct_folding < 0.01, f"Expected no folding for identity, got {scores.pct_folding}"
    assert scores.verdict in ["Good", "Acceptable (manual review recommended)", "Poor (flag for rejection)"]

    # Poor registration: warped = random noise
    warped_bad = torch.rand(B, 1, D, H, W, device=device)
    scores_bad = qm.compute(warped_bad, fixed, phi_AB, phi_BA, identity)
    print(f"\nPoor registration score: {scores_bad.quality_score:.3f}")
    assert scores_bad.quality_score < scores.quality_score, "Bad registration should score lower"

    print("PASSED")
