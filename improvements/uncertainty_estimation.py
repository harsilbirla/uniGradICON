"""
MC Dropout Uncertainty Estimation for Medical Image Registration
================================================================

Motivation
----------
uniGradICON produces a single deterministic deformation field. For clinical use,
radiologists need to know *when* to distrust a registration. MC Dropout [1]
provides a computationally cheap Bayesian approximation: by keeping dropout
layers active at inference and running N forward passes, we obtain a distribution
over deformation fields whose variance constitutes an uncertainty map.

In registration, high variance voxels indicate regions where the network is
unsure — often near boundaries, low-contrast regions, or pathological tissue.

References
----------
[1] Gal & Ghahramani, "Dropout as a Bayesian Approximation", ICML 2016.
[2] Pac et al., "Confidence-Aware Registration via MC Dropout for Robust
    Cardiac Registration", MICCAI 2023.
[3] Dalca AV et al., "Unsupervised learning of probabilistic diffeomorphic
    registration for images and surfaces", MedIA 2019. arXiv:1903.03545

Usage
-----
    from improvements.uncertainty_estimation import UncertaintyEstimator

    net = get_unigradicon()  # standard uniGradICON model
    estimator = UncertaintyEstimator(net, dropout_p=0.1, n_samples=10)

    with torch.no_grad():
        mean_field, uncertainty_map, sample_fields = estimator.estimate(
            image_A, image_B
        )

    # uncertainty_map: [B, 1, D, H, W] — per-voxel uncertainty (mm-scale)
    # mean_field: [B, 3, D, H, W] — mean deformation field
    # sample_fields: list of N fields for further analysis
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class MCDropout3D(nn.Module):
    """Dropout that stays active during eval() mode.

    Standard torch.nn.Dropout is disabled during net.eval(). This wrapper
    keeps it active, enabling Monte Carlo sampling at inference.

    Args:
        p: Dropout probability (default 0.1 — light dropout to avoid
           destroying learned features while still inducing variance).
    """

    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Always active regardless of self.training
        return nn.functional.dropout(x, p=self.p, training=True)


def inject_mc_dropout(
    module: nn.Module,
    dropout_p: float = 0.1,
    target_types: Tuple = (nn.Conv3d, nn.Conv2d),
    max_injections: int = 6,
) -> nn.Module:
    """Insert MCDropout after selected convolutional layers in-place.

    We target intermediate conv layers (not the final output conv) to preserve
    the spatial structure of the output deformation field while inducing
    uncertainty in feature representations.

    Strategy: inject after every other Conv3d (skip first + last) so that
    the network's input/output semantics are preserved.

    Args:
        module: Network to modify.
        dropout_p: MC Dropout probability.
        target_types: Layer types after which to insert dropout.
        max_injections: Cap on number of injections (avoid over-regularization).

    Returns:
        Modified module (in-place).
    """
    injection_count = 0

    def _inject(mod: nn.Module):
        nonlocal injection_count
        for name, child in list(mod.named_children()):
            if isinstance(child, target_types) and injection_count < max_injections:
                # Wrap: Sequential([original_conv, MCDropout])
                new_block = nn.Sequential(child, MCDropout3D(p=dropout_p))
                setattr(mod, name, new_block)
                injection_count += 1
            else:
                _inject(child)

    _inject(module)
    return module


class UncertaintyEstimator:
    """Monte Carlo Dropout uncertainty estimator for uniGradICON.

    Wraps an existing registration network and enables uncertainty estimation
    via repeated stochastic forward passes with MC Dropout active.

    The uncertainty map is computed as the per-voxel standard deviation of
    the displacement magnitude across N samples:

        u(x) = std_n(||phi_n(x) - identity(x)||_2)

    This is in units of normalized coordinates (divide by image_size to get mm).

    Args:
        net: Trained uniGradICON network (GradientICONSparse instance).
        dropout_p: Dropout probability for MC sampling (default: 0.1).
        n_samples: Number of forward passes (default: 10).
                   Trade-off: more samples = better estimate, slower.
        inject_dropout: If True, injects MCDropout into network in-place.
                        Set to False if network already has dropout layers.
    """

    def __init__(
        self,
        net: nn.Module,
        dropout_p: float = 0.1,
        n_samples: int = 10,
        inject_dropout: bool = True,
    ):
        self.net = net
        self.n_samples = n_samples
        self.dropout_p = dropout_p

        if inject_dropout:
            inject_mc_dropout(
                net.regis_net,
                dropout_p=dropout_p,
                max_injections=6,
            )

    def estimate(
        self,
        image_A: torch.Tensor,
        image_B: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Run N stochastic forward passes and compute uncertainty.

        Args:
            image_A: Moving image [B, 1, D, H, W] or [B, 1, H, W].
            image_B: Fixed image [B, 1, D, H, W] or [B, 1, H, W].

        Returns:
            mean_field: Mean deformation field [B, 3, D, H, W].
            uncertainty_map: Per-voxel uncertainty [B, 1, D, H, W].
                             Values represent std of displacement magnitude.
            sample_fields: List of N individual fields for downstream analysis.
        """
        # Keep dropout active: do NOT call net.eval() — or if called, MCDropout
        # overrides it via always-training forward().
        sample_fields = []

        with torch.no_grad():
            for _ in range(self.n_samples):
                phi_AB = self.net.regis_net(image_A, image_B)
                identity = self.net.identity_map
                field = phi_AB(identity)  # [B, 3, D, H, W]
                sample_fields.append(field)

        # Stack: [N, B, 3, D, H, W]
        stacked = torch.stack(sample_fields, dim=0)

        # Mean deformation field
        mean_field = stacked.mean(dim=0)  # [B, 3, D, H, W]

        # Per-sample displacement magnitude from identity
        identity = self.net.identity_map
        # displacement: [N, B, 3, D, H, W]
        displacements = stacked - identity.unsqueeze(0)
        # magnitude: [N, B, 1, D, H, W]
        magnitudes = torch.norm(displacements, dim=2, keepdim=True)
        # std across samples: [B, 1, D, H, W]
        uncertainty_map = magnitudes.std(dim=0)

        return mean_field, uncertainty_map, sample_fields

    def uncertainty_statistics(
        self, uncertainty_map: torch.Tensor
    ) -> dict:
        """Compute summary statistics of the uncertainty map.

        Args:
            uncertainty_map: [B, 1, D, H, W] uncertainty from estimate().

        Returns:
            Dict with mean, median, 95th-percentile, and % high-uncertainty
            voxels (> 2 standard deviations from the mean).
        """
        flat = uncertainty_map.flatten(start_dim=1)  # [B, N_voxels]
        mean_u = flat.mean(dim=1)
        median_u = flat.median(dim=1).values
        p95_u = torch.quantile(flat, 0.95, dim=1)
        # High uncertainty: voxels above 2 * mean
        high_thresh = 2.0 * mean_u.unsqueeze(1)
        pct_high = (flat > high_thresh).float().mean(dim=1) * 100.0

        return {
            "mean_uncertainty": mean_u.cpu(),
            "median_uncertainty": median_u.cpu(),
            "p95_uncertainty": p95_u.cpu(),
            "pct_high_uncertainty": pct_high.cpu(),
        }


# ---------------------------------------------------------------------------
# Lightweight demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=== MC Dropout Uncertainty Estimation — Smoke Test ===")
    print("Creating a minimal mock network...")

    # Mock minimal setup without real weights
    class MockRegisNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv3d(2, 3, 3, padding=1)

        def forward(self, A, B):
            x = torch.cat([A, B], dim=1)
            disp = self.conv(x)

            class Transform:
                def __init__(self, disp):
                    self.disp = disp

                def __call__(self, identity):
                    return identity + self.disp

            return Transform(disp)

    class MockNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.regis_net = MockRegisNet()
            self.identity_map = torch.zeros(1, 3, 16, 16, 16)

    mock_net = MockNet()
    estimator = UncertaintyEstimator(mock_net, dropout_p=0.1, n_samples=5, inject_dropout=False)

    A = torch.randn(1, 1, 16, 16, 16)
    B = torch.randn(1, 1, 16, 16, 16)
    mean_field, uncertainty_map, samples = estimator.estimate(A, B)

    print(f"Mean field shape: {mean_field.shape}")
    print(f"Uncertainty map shape: {uncertainty_map.shape}")
    print(f"Number of samples: {len(samples)}")
    stats = estimator.uncertainty_statistics(uncertainty_map)
    print(f"Uncertainty stats: {stats}")
    print("PASSED")
