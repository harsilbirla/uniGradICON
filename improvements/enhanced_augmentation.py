"""
Enhanced Intensity-Based Data Augmentation for uniGradICON
============================================================

Motivation
----------
The current augmentation in train.py only applies small random *spatial* affine
transforms (±5% noise). This is blind to intensity-domain variability, leaving
the model vulnerable to:

  - Scanner differences (different field strengths, coils, vendors)
  - Contrast agent presence or absence (T1 vs T1ce)
  - Pathological intensity distributions (tumors, edema)
  - Acquisition noise and reconstruction kernels

Intensity augmentation is widely used in state-of-the-art registration networks:
  - TransMorph [Chen 2022] uses gamma + additive noise
  - LapIRN [Mok 2020] uses multiplicative noise
  - VoxelMorph [Balakrishnan 2019] uses contrast jitter for cross-scanner transfer

References
----------
[1] Mok TC, Chung ACS. "Large Deformation Diffeomorphic Image Registration
    with Laplacian Pyramid Networks", MICCAI 2020. arXiv:2006.16148
[2] Chen J et al. "TransMorph: Transformer for Unsupervised Medical Image
    Registration", MedIA 2022. arXiv:2111.12742
[3] Balakrishnan G et al. "VoxelMorph: A Learning Framework for Deformable
    Medical Image Registration", TMI 2019.
[4] Zhao S et al. "Recursive Cascaded Networks for Unsupervised Medical Image
    Registration", ICCV 2019.

Usage
-----
    # Drop-in replacement for the existing augment() in train.py:
    from improvements.enhanced_augmentation import augment_with_intensity

    moving_aug, fixed_aug = augment_with_intensity(moving_image, fixed_image)

    # Or use the class-based interface with full control:
    augmenter = IntensityAugmenter(
        gamma_range=(0.7, 1.5),
        noise_std=0.05,
        blur_sigma_range=(0.0, 1.0),
        contrast_range=(0.8, 1.2),
        p_apply=0.8,
    )
    moving_aug, fixed_aug = augmenter(moving_image, fixed_image)
"""

import random
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class IntensityAugmenter:
    """Apply stochastic intensity transforms to paired medical images.

    Each augmentation is applied *independently* to moving and fixed image to
    simulate real-world scanner variability between acquisitions.

    Transforms applied (each with probability p_apply):
      1. Gamma correction  — simulates brightness/contrast curve changes
      2. Multiplicative noise — simulates gain field / bias field variation
      3. Gaussian blur      — simulates PSF differences between scanners
      4. Additive Gaussian noise — simulates acquisition noise

    All transforms are image-preserving: they do not alter the underlying
    anatomy or the ground-truth deformation.

    Args:
        gamma_range: (min, max) exponent for gamma correction.
                     γ < 1 brightens, γ > 1 darkens midtones.
        noise_std: Standard deviation for additive Gaussian noise.
                   Images are in [0,1], so 0.02-0.05 is realistic.
        blur_sigma_range: (min, max) sigma for Gaussian blur.
                          0 = no blur, 1.5 = moderate smoothing.
        contrast_range: (min, max) multiplicative contrast factor.
        p_apply: Probability of applying each individual transform.
    """

    def __init__(
        self,
        gamma_range: Tuple[float, float] = (0.7, 1.5),
        noise_std: float = 0.04,
        blur_sigma_range: Tuple[float, float] = (0.0, 1.2),
        contrast_range: Tuple[float, float] = (0.85, 1.15),
        p_apply: float = 0.7,
    ):
        self.gamma_range = gamma_range
        self.noise_std = noise_std
        self.blur_sigma_range = blur_sigma_range
        self.contrast_range = contrast_range
        self.p_apply = p_apply

    def __call__(
        self,
        image_A: torch.Tensor,
        image_B: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply independent intensity augmentation to image pair.

        Args:
            image_A: Moving image [B, C, ...] in [0, 1].
            image_B: Fixed image [B, C, ...] in [0, 1].

        Returns:
            Augmented (image_A, image_B) still in [0, 1].
        """
        image_A = self._augment_single(image_A)
        image_B = self._augment_single(image_B)
        return image_A, image_B

    def _augment_single(self, image: torch.Tensor) -> torch.Tensor:
        """Apply all transforms to a single image."""
        # Only augment the intensity channel (channel 0). If image has
        # segmentation channels (shape[1] > 1), preserve them unchanged.
        if image.shape[1] > 1:
            img_intensity = image[:, :1]
            img_seg = image[:, 1:]
            img_intensity = self._apply_transforms(img_intensity)
            return torch.cat([img_intensity, img_seg], dim=1)
        return self._apply_transforms(image)

    def _apply_transforms(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stochastic transform pipeline."""
        if random.random() < self.p_apply:
            x = self._gamma_correction(x)
        if random.random() < self.p_apply:
            x = self._multiplicative_noise(x)
        if random.random() < self.p_apply:
            x = self._gaussian_blur(x)
        if random.random() < self.p_apply:
            x = self._additive_noise(x)
        return x

    def _gamma_correction(self, x: torch.Tensor) -> torch.Tensor:
        """Random gamma correction: x -> x^gamma.

        Operates in [0,1] so gamma correction is well-defined and
        output stays in [0,1].
        """
        gamma = random.uniform(*self.gamma_range)
        # Clamp to avoid numerical issues with fractional powers near 0
        return torch.clamp(x, min=1e-6).pow(gamma)

    def _multiplicative_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Multiplicative noise: x -> x * N(1, sigma^2).

        Models MRI bias field and gain field artifacts.
        Noise is spatially smooth (use scale-space approach for real bias
        field simulation — here we use per-voxel for simplicity).
        """
        noise = torch.randn_like(x) * 0.05 + 1.0  # N(1, 0.05^2)
        return torch.clamp(x * noise, 0.0, 1.0)

    def _gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur with random sigma.

        Uses separable 1D convolutions for efficiency in 3D.
        Sigma drawn from blur_sigma_range.
        """
        sigma = random.uniform(*self.blur_sigma_range)
        if sigma < 0.1:  # Skip negligible blur
            return x
        return _gaussian_blur_nd(x, sigma)

    def _additive_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Additive Gaussian noise: x -> x + N(0, noise_std^2).

        Models thermal noise and Rician noise in MRI magnitude images.
        """
        noise = torch.randn_like(x) * self.noise_std
        return torch.clamp(x + noise, 0.0, 1.0)


def _gaussian_blur_nd(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur for 3D or 2D tensors.

    Uses 1D Gaussian kernels along each spatial dimension independently.
    This is much faster than a full 3D convolution for large kernels.

    Args:
        x: Input tensor [B, C, D, H, W] (3D) or [B, C, H, W] (2D).
        sigma: Gaussian sigma in pixels.

    Returns:
        Blurred tensor, same shape as input.
    """
    ndim = x.ndim - 2  # Spatial dimensions (2 or 3)
    kernel_size = max(3, int(2 * round(3 * sigma) + 1))  # 3-sigma rule
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Build 1D Gaussian kernel
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device)
    coords = coords - kernel_size // 2
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    channels = x.shape[1]

    if ndim == 3:
        # Apply along D, H, W sequentially
        for dim in range(3):
            # Shape kernel for depthwise conv along each spatial dim
            k_shape = [1] * 5
            k_shape[dim + 2] = kernel_size
            k = kernel_1d.view(k_shape).expand(channels, 1, *([1] * 3))
            k_shape2 = [channels, 1] + [1] * 3
            k_shape2[dim + 2] = kernel_size
            k = kernel_1d.view(k_shape2).expand(channels, 1, *[kernel_size if i == dim else 1 for i in range(3)])
            pad = [0] * 6
            pad[2 * (2 - dim)] = kernel_size // 2
            pad[2 * (2 - dim) + 1] = kernel_size // 2
            x_padded = F.pad(x, pad, mode="reflect")
            x = F.conv3d(x_padded, k, groups=channels)
    elif ndim == 2:
        for dim in range(2):
            k_shape = [channels, 1, 1, 1]
            k_shape[dim + 2] = kernel_size
            k = kernel_1d.view(k_shape)
            pad = [0] * 4
            pad[2 * (1 - dim)] = kernel_size // 2
            pad[2 * (1 - dim) + 1] = kernel_size // 2
            x_padded = F.pad(x, pad, mode="reflect")
            x = F.conv2d(x_padded, k, groups=channels)

    return x


def augment_with_intensity(
    image_A: torch.Tensor,
    image_B: torch.Tensor,
    augmenter: Optional[IntensityAugmenter] = None,
    also_apply_spatial: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop-in replacement for augment() in train.py.

    Applies both the original spatial augmentation AND the new intensity
    augmentation. Can be used as a direct replacement:

        # Old:
        moving_image, fixed_image = augment(moving_image, fixed_image)

        # New:
        moving_image, fixed_image = augment_with_intensity(
            moving_image, fixed_image
        )

    Args:
        image_A: Moving image [B, C, ...] in [0, 1].
        image_B: Fixed image [B, C, ...] in [0, 1].
        augmenter: IntensityAugmenter instance (creates default if None).
        also_apply_spatial: If True, also apply the original spatial
                            augmentation from train.py.

    Returns:
        Augmented (image_A, image_B).
    """
    if augmenter is None:
        augmenter = IntensityAugmenter()

    if also_apply_spatial:
        image_A, image_B = _spatial_augment(image_A, image_B)

    image_A, image_B = augmenter(image_A, image_B)
    return image_A, image_B


def _spatial_augment(
    image_A: torch.Tensor,
    image_B: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reproduce the original spatial augmentation from train.py.

    This keeps the spatial augmentation intact when using augment_with_intensity
    as a drop-in replacement.
    """
    device = image_A.device
    identity_list = []
    for i in range(image_A.shape[0]):
        identity = torch.tensor(
            [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], dtype=torch.float32, device=device
        )
        idxs = set((0, 1, 2))
        for j in range(3):
            k = random.choice(list(idxs))
            idxs.remove(k)
            identity[0, j, k] = 1
        identity = identity * (torch.randint_like(identity, 0, 2, device=device) * 2 - 1)
        identity_list.append(identity)

    identity = torch.cat(identity_list)
    noise = torch.randn((image_A.shape[0], 3, 4), device=device)
    forward = identity + 0.05 * noise

    grid_shape = list(image_A.shape)
    grid_shape[1] = 3
    forward_grid = F.affine_grid(forward, grid_shape, align_corners=False)

    def _warp(img):
        if img.shape[1] > 1:
            warped = F.grid_sample(img[:, :1], forward_grid, padding_mode="border", align_corners=False)
            warped_seg = F.grid_sample(img[:, 1:], forward_grid, mode="nearest", padding_mode="border", align_corners=False)
            return torch.cat([warped, warped_seg], dim=1)
        return F.grid_sample(img, forward_grid, padding_mode="border", align_corners=False)

    warped_A = _warp(image_A)

    noise = torch.randn((image_A.shape[0], 3, 4), device=device)
    forward = identity + 0.05 * noise
    forward_grid = F.affine_grid(forward, grid_shape, align_corners=False)
    warped_B = _warp(image_B)

    return warped_A, warped_B


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Enhanced Augmentation — Smoke Test ===")

    augmenter = IntensityAugmenter(p_apply=1.0)  # Force all transforms

    A = torch.rand(2, 1, 32, 32, 32)
    B = torch.rand(2, 1, 32, 32, 32)

    A_aug, B_aug = augmenter(A, B)

    assert A_aug.shape == A.shape, f"Shape mismatch: {A_aug.shape} vs {A.shape}"
    assert A_aug.min() >= 0.0 and A_aug.max() <= 1.0, "Output out of [0,1] range"
    assert B_aug.min() >= 0.0 and B_aug.max() <= 1.0, "Output out of [0,1] range"

    # Verify augmentation actually changes pixel values
    assert not torch.allclose(A_aug, A), "Augmentation had no effect (A)"
    assert not torch.allclose(B_aug, B), "Augmentation had no effect (B)"

    # Test with segmentation channels
    A_seg = torch.rand(2, 2, 32, 32, 32)
    A_seg[:, 1:] = (A_seg[:, 1:] > 0.5).float()
    A_seg_aug, _ = augmenter(A_seg, B)
    assert A_seg_aug.shape == A_seg.shape
    # Segmentation channel should be unchanged by intensity augmentation
    assert torch.allclose(A_seg_aug[:, 1:], A_seg[:, 1:]), "Seg channels should not change"

    # Test drop-in replacement
    A_full, B_full = augment_with_intensity(A, B, also_apply_spatial=True)
    assert A_full.shape == A.shape

    print(f"Original range: [{A.min():.3f}, {A.max():.3f}]")
    print(f"Augmented range: [{A_aug.min():.3f}, {A_aug.max():.3f}]")
    print(f"Mean change: {(A_aug - A).abs().mean():.4f}")
    print("PASSED")
