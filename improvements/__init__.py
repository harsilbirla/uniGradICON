"""
uniGradICON Research Improvements
===================================
Research-driven enhancements to the uniGradICON foundation model.

Modules
-------
uncertainty_estimation        : MC Dropout-based uncertainty quantification
enhanced_augmentation         : Intensity-based data augmentation for robustness
registration_quality_metric   : Unsupervised registration quality assessment
attention_modules             : Squeeze-and-Excitation blocks for tallUNet2
amp_training                  : Automatic Mixed Precision training utilities

References
----------
See RESEARCH_IMPROVEMENTS.md for the full literature review and implementation plan.

Quick start
-----------
    from improvements.enhanced_augmentation import augment_with_intensity
    from improvements.uncertainty_estimation import UncertaintyEstimator
    from improvements.registration_quality_metric import RegistrationQualityMetric
    from improvements.attention_modules import wrap_unet_with_se
    from improvements.amp_training import AMPTrainer
"""

__version__ = "0.1.0"

# Lazy imports — torch not required at import time
def _lazy_import(module_name, class_name):
    """Import class lazily to avoid torch dependency at import time."""
    def _getter():
        import importlib
        mod = importlib.import_module(f".{module_name}", package=__package__)
        return getattr(mod, class_name)
    return _getter


__all__ = [
    "UncertaintyEstimator",
    "IntensityAugmenter",
    "augment_with_intensity",
    "RegistrationQualityMetric",
    "SEBlock3D",
    "SEBlock2D",
    "wrap_unet_with_se",
    "AMPTrainer",
    "train_kernel_amp",
    "GPIORegistrar",
    "instance_optimize_gp",
    "DivCurlLoss",
    "AdaptiveDivCurlLoss",
]


def __getattr__(name):
    _registry = {
        "UncertaintyEstimator": ("uncertainty_estimation", "UncertaintyEstimator"),
        "IntensityAugmenter": ("enhanced_augmentation", "IntensityAugmenter"),
        "augment_with_intensity": ("enhanced_augmentation", "augment_with_intensity"),
        "RegistrationQualityMetric": ("registration_quality_metric", "RegistrationQualityMetric"),
        "QualityScores": ("registration_quality_metric", "QualityScores"),
        "SEBlock3D": ("attention_modules", "SEBlock3D"),
        "SEBlock2D": ("attention_modules", "SEBlock2D"),
        "wrap_unet_with_se": ("attention_modules", "wrap_unet_with_se"),
        "AMPTrainer": ("amp_training", "AMPTrainer"),
        "train_kernel_amp": ("amp_training", "train_kernel_amp"),
        "make_scaler": ("amp_training", "make_scaler"),
        "GPIORegistrar": ("gradient_projection_io", "GPIORegistrar"),
        "instance_optimize_gp": ("gradient_projection_io", "instance_optimize_gp"),
        "gradient_projection": ("gradient_projection_io", "gradient_projection"),
        "DivCurlLoss": ("divcurl_regularization", "DivCurlLoss"),
        "AdaptiveDivCurlLoss": ("divcurl_regularization", "AdaptiveDivCurlLoss"),
    }
    if name in _registry:
        module_name, class_name = _registry[name]
        import importlib
        mod = importlib.import_module(f".{module_name}", package=__package__)
        return getattr(mod, class_name)
    raise AttributeError(f"module 'improvements' has no attribute {name!r}")
