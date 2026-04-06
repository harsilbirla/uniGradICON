"""
Div-Curl Physical Regularization for uniGradICON
=================================================

Motivation
----------
uniGradICON uses a single λ for GradICON loss across ALL datasets — brain,
lung, knee, and abdomen. This is intentional for a foundation model, but it
creates a tension:

  - **Lung CT**: Breathing creates large *divergence* (volume change ~30%).
    Penalizing divergence too strongly prevents accurate respiratory tracking.

  - **Brain MRI**: The brain is near-incompressible (divergence ≈ 0).
    Over-allowing volume change leads to physically implausible deformations.

The **div-curl decomposition** (Stam 1999, applied to medical imaging in 2024)
decomposes any deformation field u into:

    u = u_div + u_curl

where u_div is the irrotational (volume-changing) component and u_curl is the
solenoidal (rotation-preserving, divergence-free) component. Independent
penalties λ_div and λ_curl allow anatomy-specific control:

    L_reg = λ_div * ||∇·u||² + λ_curl * ||∇×u||²

This replaces or augments the GradICON inverse consistency loss with a
physically interpretable regularizer that can be tuned per dataset.

Key advantages over GradICON alone:
  1. Physical interpretability: λ_div controls volume conservation,
     λ_curl controls rotation freedom
  2. Reduced sensitivity to the single λ hyperparameter
  3. Enables dataset-specific presets (lung vs. brain vs. abdomen)

References
----------
[1] "Generalized Div-Curl Based Regularization for Physically Constrained
    Deformable Image Registration", Scientific Reports 2024. PMC11217375
[2] Stam J. "A Simple Fluid Solver Based on the FFT", JGTD 1999.
    (Original Helmholtz-Hodge decomposition idea applied to fluids)
[3] Cootes TF et al. "Active appearance models", TPAMI 2001.
    (Background on physically constrained registration)

Usage
-----
    from improvements.divcurl_regularization import DivCurlLoss

    # Create dataset-specific loss
    loss_fn = DivCurlLoss(lambda_div=0.5, lambda_curl=2.0)  # Brain (low div)
    loss_fn = DivCurlLoss(lambda_div=2.0, lambda_curl=0.5)  # Lung (high div ok)
    loss_fn = DivCurlLoss.lung_preset()   # Named presets
    loss_fn = DivCurlLoss.brain_preset()

    # Use in training (add to similarity loss):
    similarity_loss = icon.LNCC(sigma=5)(warped_A, image_B)
    reg_loss = loss_fn(phi_AB_vectorfield)
    total_loss = similarity_loss + reg_loss

    # Or combine with existing GradICON in GradientICONSparse:
    all_loss = (
        lmbda * inverse_consistency_loss  # GradICON term
        + loss_fn(phi_AB_vectorfield)     # Div-curl term (additive)
        + similarity_loss
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DivCurlLoss(nn.Module):
    """Div-curl decomposition regularization for deformation fields.

    Computes divergence (||∇·u||²) and curl (||∇×u||²) of the deformation
    field using central finite differences, then penalizes each independently.

    This allows anatomy-specific control:
      - High λ_div → enforce near-incompressibility (brain, liver)
      - Low  λ_div → allow volume change (lung, cardiac)
      - High λ_curl → penalize rotation (rigid anatomy)
      - Low  λ_curl → allow rotation (joint motion, cardiac)

    Args:
        lambda_div: Penalty weight for divergence (volume change).
        lambda_curl: Penalty weight for curl (rotation).
        reduction: 'mean' or 'sum' over spatial locations.
    """

    def __init__(
        self,
        lambda_div: float = 1.0,
        lambda_curl: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.lambda_div = lambda_div
        self.lambda_curl = lambda_curl
        self.reduction = reduction

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """Compute div-curl regularization loss.

        Args:
            phi: Deformation field [B, 3, D, H, W] in voxel coordinates.
                 This is phi(x) = identity(x) + u(x) or just the displacement
                 field u(x) — both work, since ∇·identity = 1 (constant) and
                 ∇·u is the meaningful quantity.

        Returns:
            Scalar regularization loss = λ_div * div_loss + λ_curl * curl_loss.
        """
        if phi.ndim == 5:
            div, curl_mag = self._compute_div_curl_3d(phi)
        elif phi.ndim == 4:
            div, curl_mag = self._compute_div_curl_2d(phi)
        else:
            raise ValueError(f"Expected 4D or 5D phi, got {phi.ndim}D")

        if self.reduction == "mean":
            div_loss = div.pow(2).mean()
            curl_loss = curl_mag.pow(2).mean()
        else:
            div_loss = div.pow(2).sum()
            curl_loss = curl_mag.pow(2).sum()

        return self.lambda_div * div_loss + self.lambda_curl * curl_loss

    def _compute_div_curl_3d(
        self, phi: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 3D divergence and curl magnitude via central differences.

        For phi = (u, v, w) where u,v,w are the x,y,z components:

        Divergence:
            div(phi) = du/dx + dv/dy + dw/dz

        Curl (vector in 3D, then take magnitude):
            curl(phi) = (dw/dy - dv/dz, du/dz - dw/dx, dv/dx - du/dy)
            ||curl(phi)||² = (dw/dy - dv/dz)² + (du/dz - dw/dx)² + (dv/dx - du/dy)²

        All derivatives via central differences (2nd-order accurate).

        Returns:
            div: [B, 1, D-2, H-2, W-2] divergence
            curl_mag: [B, 1, D-2, H-2, W-2] curl magnitude
        """
        B, C, D, H, W = phi.shape
        assert C == 3, f"Expected 3-channel field for 3D, got {C}"

        u = phi[:, 0:1]  # x-component
        v = phi[:, 1:2]  # y-component
        w = phi[:, 2:3]  # z-component

        # Central differences: df/dx_i ≈ (f[i+1] - f[i-1]) / 2
        du_dx = (u[:, :, 1:-1, 1:-1, 2:] - u[:, :, 1:-1, 1:-1, :-2]) / 2
        dv_dy = (v[:, :, 1:-1, 2:, 1:-1] - v[:, :, 1:-1, :-2, 1:-1]) / 2
        dw_dz = (w[:, :, 2:, 1:-1, 1:-1] - w[:, :, :-2, 1:-1, 1:-1]) / 2

        dw_dy = (w[:, :, 1:-1, 2:, 1:-1] - w[:, :, 1:-1, :-2, 1:-1]) / 2
        dv_dz = (v[:, :, 2:, 1:-1, 1:-1] - v[:, :, :-2, 1:-1, 1:-1]) / 2

        du_dz = (u[:, :, 2:, 1:-1, 1:-1] - u[:, :, :-2, 1:-1, 1:-1]) / 2
        dw_dx = (w[:, :, 1:-1, 1:-1, 2:] - w[:, :, 1:-1, 1:-1, :-2]) / 2

        dv_dx = (v[:, :, 1:-1, 1:-1, 2:] - v[:, :, 1:-1, 1:-1, :-2]) / 2
        du_dy = (u[:, :, 1:-1, 2:, 1:-1] - u[:, :, 1:-1, :-2, 1:-1]) / 2

        div = du_dx + dv_dy + dw_dz

        curl_x = dw_dy - dv_dz
        curl_y = du_dz - dw_dx
        curl_z = dv_dx - du_dy
        curl_mag = torch.sqrt(curl_x ** 2 + curl_y ** 2 + curl_z ** 2 + 1e-8)

        return div, curl_mag

    def _compute_div_curl_2d(
        self, phi: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 2D divergence and curl via central differences.

        For phi = (u, v):
            div = du/dx + dv/dy
            curl = dv/dx - du/dy  (scalar in 2D)

        Returns:
            div: [B, 1, H-2, W-2]
            curl: [B, 1, H-2, W-2]
        """
        B, C, H, W = phi.shape
        assert C == 2, f"Expected 2-channel field for 2D, got {C}"

        u = phi[:, 0:1]
        v = phi[:, 1:2]

        du_dx = (u[:, :, 1:-1, 2:] - u[:, :, 1:-1, :-2]) / 2
        dv_dy = (v[:, :, 2:, 1:-1] - v[:, :, :-2, 1:-1]) / 2
        dv_dx = (v[:, :, 1:-1, 2:] - v[:, :, 1:-1, :-2]) / 2
        du_dy = (u[:, :, 2:, 1:-1] - u[:, :, :-2, 1:-1]) / 2

        div = du_dx + dv_dy
        curl = dv_dx - du_dy  # Scalar (z-component of curl in 2D)

        return div, curl

    def decompose(self, phi: torch.Tensor) -> dict:
        """Return full decomposition statistics for analysis.

        Args:
            phi: Deformation field [B, 3, D, H, W].

        Returns:
            Dict with divergence and curl statistics.
        """
        if phi.ndim == 5:
            div, curl = self._compute_div_curl_3d(phi)
        else:
            div, curl = self._compute_div_curl_2d(phi)

        return {
            "div_mean": div.mean().item(),
            "div_std": div.std().item(),
            "div_max": div.abs().max().item(),
            "curl_mean": curl.mean().item(),
            "curl_std": curl.std().item(),
            "curl_max": curl.abs().max().item(),
            "div_loss": div.pow(2).mean().item(),
            "curl_loss": curl.pow(2).mean().item(),
        }

    # --- Named presets based on anatomical characteristics ---

    @classmethod
    def brain_preset(cls) -> "DivCurlLoss":
        """Preset for brain MRI registration.

        Brain is near-incompressible: high div penalty to preserve volume,
        moderate curl penalty to allow local rotation around sulci.
        """
        return cls(lambda_div=2.0, lambda_curl=0.5)

    @classmethod
    def lung_preset(cls) -> "DivCurlLoss":
        """Preset for lung CT registration (4D-CT, breathing).

        Lung undergoes large volume change (~30%) during breathing:
        allow high divergence but penalize unphysical rotations.
        """
        return cls(lambda_div=0.2, lambda_curl=1.5)

    @classmethod
    def abdomen_preset(cls) -> "DivCurlLoss":
        """Preset for abdominal CT registration.

        Abdomen has intermediate compressibility (liver, bowel, spleen):
        balanced penalties.
        """
        return cls(lambda_div=0.8, lambda_curl=0.8)

    @classmethod
    def knee_preset(cls) -> "DivCurlLoss":
        """Preset for knee MRI registration.

        Cartilage is near-incompressible; joint motion involves rotation:
        high div penalty, low curl penalty.
        """
        return cls(lambda_div=1.5, lambda_curl=0.3)

    @classmethod
    def balanced_preset(cls) -> "DivCurlLoss":
        """Balanced preset for multi-anatomy foundation model training.

        Equal weights; suitable as a default when dataset is mixed.
        """
        return cls(lambda_div=1.0, lambda_curl=1.0)


class AdaptiveDivCurlLoss(DivCurlLoss):
    """Div-curl loss with learnable (adaptive) λ_div and λ_curl weights.

    Instead of fixed hyperparameters, allows the loss weights to adapt
    during training via gradient descent. This is the "uncertainty weighting"
    approach from Kendall & Gal (NeurIPS 2018): each task gets a learned
    log-variance parameter that acts as a regularization weight.

    The loss becomes:
        L = exp(-s_div) * div_loss + s_div +
            exp(-s_curl) * curl_loss + s_curl

    where s_div = log(λ_div²) and s_curl = log(λ_curl²) are learned.
    This automatically balances the two components.

    Args:
        init_lambda_div: Initial div penalty (default: 1.0).
        init_lambda_curl: Initial curl penalty (default: 1.0).
    """

    def __init__(
        self,
        init_lambda_div: float = 1.0,
        init_lambda_curl: float = 1.0,
    ):
        super().__init__(lambda_div=1.0, lambda_curl=1.0)
        import math
        # log(λ²) = 2 * log(λ)
        self.log_var_div = nn.Parameter(torch.tensor(2 * math.log(init_lambda_div + 1e-8)))
        self.log_var_curl = nn.Parameter(torch.tensor(2 * math.log(init_lambda_curl + 1e-8)))

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        if phi.ndim == 5:
            div, curl_mag = self._compute_div_curl_3d(phi)
        else:
            div, curl_mag = self._compute_div_curl_2d(phi)

        div_loss = div.pow(2).mean()
        curl_loss = curl_mag.pow(2).mean()

        # Uncertainty weighting (Kendall & Gal 2018)
        weighted_div = torch.exp(-self.log_var_div) * div_loss + self.log_var_div
        weighted_curl = torch.exp(-self.log_var_curl) * curl_loss + self.log_var_curl

        return weighted_div + weighted_curl

    @property
    def effective_lambda_div(self) -> float:
        return float(torch.exp(self.log_var_div / 2).item())

    @property
    def effective_lambda_curl(self) -> float:
        return float(torch.exp(self.log_var_curl / 2).item())


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Div-Curl Regularization — Smoke Test ===")

    B, D, H, W = 2, 24, 24, 24

    # Identity field: all derivatives = 0 → div and curl both = 0
    identity = torch.stack(torch.meshgrid(
        torch.linspace(0, D - 1, D),
        torch.linspace(0, H - 1, H),
        torch.linspace(0, W - 1, W),
        indexing="ij"
    ), dim=0).unsqueeze(0).expand(B, -1, -1, -1, -1)

    loss_fn = DivCurlLoss(lambda_div=1.0, lambda_curl=1.0)

    # Identity field should have near-zero loss
    loss_identity = loss_fn(identity)
    print(f"Identity field loss: {loss_identity.item():.6f} (should be ~0)")
    assert loss_identity.item() < 1e-3, f"Identity loss too high: {loss_identity.item()}"

    # Random deformation should have higher loss
    random_phi = identity + torch.randn_like(identity) * 2.0
    loss_random = loss_fn(random_phi)
    print(f"Random field loss:   {loss_random.item():.4f} (should be > identity loss)")
    assert loss_random.item() > loss_identity.item(), "Random field should have higher loss"

    # Test decomposition
    stats = loss_fn.decompose(random_phi)
    print(f"Decomposition: div_loss={stats['div_loss']:.4f}, curl_loss={stats['curl_loss']:.4f}")

    # Test presets
    for name, preset in [
        ("brain", DivCurlLoss.brain_preset()),
        ("lung", DivCurlLoss.lung_preset()),
        ("abdomen", DivCurlLoss.abdomen_preset()),
        ("knee", DivCurlLoss.knee_preset()),
    ]:
        loss = preset(random_phi)
        print(f"Preset {name:8s}: loss={loss.item():.4f}  "
              f"(λ_div={preset.lambda_div}, λ_curl={preset.lambda_curl})")

    # Test adaptive version
    adaptive = AdaptiveDivCurlLoss()
    loss_adaptive = adaptive(random_phi)
    print(f"Adaptive loss: {loss_adaptive.item():.4f}  "
          f"(effective λ_div={adaptive.effective_lambda_div:.3f}, "
          f"λ_curl={adaptive.effective_lambda_curl:.3f})")

    # Check gradient flows through adaptive parameters
    loss_adaptive.backward()
    assert adaptive.log_var_div.grad is not None, "No gradient for log_var_div"
    assert adaptive.log_var_curl.grad is not None, "No gradient for log_var_curl"

    # Test 2D version
    phi_2d = torch.randn(B, 2, 24, 24)
    loss_2d = loss_fn._compute_div_curl_2d(phi_2d)
    print(f"2D div shape: {loss_2d[0].shape}, curl shape: {loss_2d[1].shape}")

    print("PASSED")
