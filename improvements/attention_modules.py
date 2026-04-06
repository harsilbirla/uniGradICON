"""
Squeeze-and-Excitation (SE) Attention for uniGradICON
=======================================================

Motivation
----------
The tallUNet2 backbone treats all feature channels equally at every layer.
SE blocks [Hu 2018] add a lightweight channel-attention mechanism:

  1. **Squeeze**: Global average pool each feature map → scalar descriptor per channel
  2. **Excitation**: Two FC layers learn to re-weight channels based on global context
  3. **Scale**: Multiply original features by learned weights

For a foundation model that handles brain, lung, knee, and abdomen across
modalities, channel attention allows the shared backbone to dynamically
emphasize modality/anatomy-specific features at inference — without per-task
heads or any change to the registration framework.

Empirical evidence:
  - Shi et al. (ISBI 2022): +0.5–1.5% Dice in brain MRI registration
    by adding SE blocks to VoxelMorph's U-Net encoder.
  - SE blocks add <0.1% parameters relative to the base network.

References
----------
[1] Hu J et al. "Squeeze-and-Excitation Networks", CVPR 2018. arXiv:1709.01507
[2] Woo S et al. "CBAM: Convolutional Block Attention Module", ECCV 2018.
    arXiv:1807.06521
[3] Shi Z et al. "Embedding Squeeze and Excitation Network in VoxelMorph for
    Brain MRI Registration", ISBI 2022.

Usage
-----
    from improvements.attention_modules import SEBlock3D, wrap_unet_with_se

    # Option A: Wrap entire tallUNet2 network to add SE attention everywhere
    from icon_registration.networks import tallUNet2
    unet = tallUNet2(dimension=3)
    unet_with_se = wrap_unet_with_se(unet, reduction=8, verbose=True)

    # Option B: Use SEBlock3D as a building block in custom networks
    se = SEBlock3D(channels=64, reduction=8)
    features = se(features)  # channel-recalibrated features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation block for 3D feature maps.

    Implements the channel recalibration from Hu et al. 2018 for 3D volumes.
    Added after convolutional layers to dynamically re-weight feature channels.

    Architecture:
        Input [B, C, D, H, W]
          → GlobalAvgPool → [B, C]
          → FC(C, C//reduction) + ReLU
          → FC(C//reduction, C) + Sigmoid
          → Reshape [B, C, 1, 1, 1]
          → Multiply with input
          → Output [B, C, D, H, W]

    Args:
        channels: Number of input/output feature channels.
        reduction: Reduction ratio for the bottleneck FC layers.
                   Higher = fewer parameters, less expressivity.
                   Recommended: 4–16 (default: 8).
        activation: Activation for the excitation FC layers (default: ReLU).
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 8,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        if channels < reduction:
            reduction = max(1, channels // 2)
        bottleneck = max(1, channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, bottleneck, bias=False),
            activation or nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[:2]
        # Squeeze: global average pool → [B, C]
        y = self.avg_pool(x).view(B, C)
        # Excitation: channel-wise weights in [0, 1]
        y = self.fc(y).view(B, C, 1, 1, 1)
        # Scale
        return x * y


class SEBlock2D(nn.Module):
    """Squeeze-and-Excitation block for 2D feature maps.

    Same architecture as SEBlock3D but for 2D inputs [B, C, H, W].
    Useful when uniGradICON is run in 2D mode.

    Args:
        channels: Number of input/output feature channels.
        reduction: Reduction ratio (default: 8).
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        if channels < reduction:
            reduction = max(1, channels // 2)
        bottleneck = max(1, channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[:2]
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y


class SEConv3D(nn.Module):
    """Conv3d + BatchNorm + ReLU + SE attention in one block.

    A drop-in replacement for a standard conv-bn-relu block that adds
    channel attention. Used to upgrade existing tallUNet2 blocks.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Convolution kernel size (default: 3).
        stride: Convolution stride (default: 1).
        padding: Convolution padding (default: 1).
        se_reduction: SE reduction ratio (default: 8).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        se_reduction: int = 8,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock3D(out_channels, reduction=se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.se(x)
        return x


class _SEWrapper(nn.Module):
    """Wraps an existing nn.Module to add SE attention after its output.

    Used by wrap_unet_with_se() to non-invasively add SE to pre-built networks.

    Args:
        module: The module to wrap.
        channels: Output channel count of the wrapped module.
        reduction: SE reduction ratio.
        ndim: Spatial dimensionality (2 or 3).
    """

    def __init__(self, module: nn.Module, channels: int, reduction: int, ndim: int):
        super().__init__()
        self.module = module
        self.se = SEBlock3D(channels, reduction) if ndim == 3 else SEBlock2D(channels, reduction)

    def forward(self, *args, **kwargs):
        out = self.module(*args, **kwargs)
        # Only apply SE to tensor outputs; pass through non-tensor outputs unchanged
        if isinstance(out, torch.Tensor):
            return self.se(out)
        return out


def wrap_unet_with_se(
    unet: nn.Module,
    reduction: int = 8,
    min_channels: int = 8,
    max_injections: int = 8,
    verbose: bool = False,
) -> nn.Module:
    """Add SE attention blocks to an existing U-Net non-invasively.

    Traverses the network and wraps Conv3d layers (that are followed by
    activation functions in tallUNet2) with SE attention. Only wraps layers
    with enough output channels (min_channels) to make SE meaningful.

    This is a **non-invasive** modification: the original layer's weights
    are preserved; only new SE parameters are added.

    Args:
        unet: The tallUNet2 or similar U-Net network.
        reduction: SE reduction ratio (default: 8).
        min_channels: Minimum output channels to add SE (default: 8).
                      Avoids adding SE to very small layers.
        max_injections: Maximum number of SE blocks to inject.
        verbose: Print injection locations.

    Returns:
        Modified network with SE blocks.
    """
    injection_count = 0

    def _wrap(module: nn.Module, depth: int = 0) -> None:
        nonlocal injection_count
        for name, child in list(module.named_children()):
            if injection_count >= max_injections:
                return
            if isinstance(child, nn.Conv3d) and child.out_channels >= min_channels:
                wrapped = _SEWrapper(child, child.out_channels, reduction, ndim=3)
                setattr(module, name, wrapped)
                injection_count += 1
                if verbose:
                    print(
                        f"  [SE] Injected after {name} "
                        f"(out_channels={child.out_channels}, depth={depth})"
                    )
            elif isinstance(child, nn.Conv2d) and child.out_channels >= min_channels:
                wrapped = _SEWrapper(child, child.out_channels, reduction, ndim=2)
                setattr(module, name, wrapped)
                injection_count += 1
                if verbose:
                    print(
                        f"  [SE] Injected after {name} "
                        f"(out_channels={child.out_channels}, depth={depth})"
                    )
            else:
                _wrap(child, depth + 1)

    _wrap(unet)

    if verbose:
        print(f"  Total SE blocks injected: {injection_count}")
        n_se_params = sum(
            p.numel()
            for m in unet.modules()
            if isinstance(m, (SEBlock3D, SEBlock2D))
            for p in m.parameters()
        )
        total_params = sum(p.numel() for p in unet.parameters())
        print(f"  SE parameters: {n_se_params:,} / {total_params:,} "
              f"({100 * n_se_params / total_params:.2f}%)")

    return unet


def count_se_parameters(model: nn.Module) -> dict:
    """Count SE vs. total parameters in a model.

    Returns:
        Dict with 'se_params', 'total_params', 'se_fraction'.
    """
    se_params = sum(
        p.numel()
        for m in model.modules()
        if isinstance(m, (SEBlock3D, SEBlock2D))
        for p in m.parameters()
    )
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "se_params": se_params,
        "total_params": total_params,
        "se_fraction": se_params / max(total_params, 1),
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== SE Attention Modules — Smoke Test ===")

    # Test SEBlock3D
    se3d = SEBlock3D(64, reduction=8)
    x = torch.randn(2, 64, 8, 8, 8)
    out = se3d(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
    # SE should rescale (not zero out entirely)
    assert not torch.allclose(out, x), "SE block had no effect"
    print(f"SEBlock3D: input {x.shape} → output {out.shape}")

    # Test SEBlock2D
    se2d = SEBlock2D(32, reduction=4)
    x2 = torch.randn(2, 32, 16, 16)
    out2 = se2d(x2)
    assert out2.shape == x2.shape
    print(f"SEBlock2D: input {x2.shape} → output {out2.shape}")

    # Test wrap_unet_with_se on a minimal network
    class MinimalUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = nn.Conv3d(1, 16, 3, padding=1)
            self.enc2 = nn.Conv3d(16, 32, 3, padding=1)
            self.dec1 = nn.Conv3d(32, 16, 3, padding=1)
            self.out = nn.Conv3d(16, 3, 1)

        def forward(self, x):
            x = F.relu(self.enc1(x))
            x = F.relu(self.enc2(x))
            x = F.relu(self.dec1(x))
            return self.out(x)

    unet = MinimalUNet()
    total_before = sum(p.numel() for p in unet.parameters())

    unet_se = wrap_unet_with_se(unet, reduction=4, verbose=True)

    total_after = sum(p.numel() for p in unet_se.parameters())
    param_stats = count_se_parameters(unet_se)

    print(f"Parameters before SE: {total_before:,}")
    print(f"Parameters after SE:  {total_after:,}")
    print(f"SE overhead: {param_stats['se_fraction'] * 100:.2f}%")

    # Forward pass still works
    dummy_input = torch.randn(1, 1, 8, 8, 8)
    output = unet_se(dummy_input)
    assert output.shape == (1, 3, 8, 8, 8), f"Unexpected output shape: {output.shape}"
    print(f"Forward pass: input {dummy_input.shape} → output {output.shape}")

    assert param_stats["se_fraction"] < 0.05, "SE overhead should be <5%"
    print("PASSED")
