"""Lag kernels (implemented in the Rust engine)."""

from ._core import ExponentialDecay, GammaKernel, Kernel, LogNormalKernel, WindowKernel, kernel_from_dict

KERNEL_TYPES = {
    ExponentialDecay.name: ExponentialDecay,
    WindowKernel.name: WindowKernel,
    LogNormalKernel.name: LogNormalKernel,
    GammaKernel.name: GammaKernel,
}

__all__ = ["ExponentialDecay", "GammaKernel", "KERNEL_TYPES", "Kernel", "LogNormalKernel",
           "WindowKernel", "kernel_from_dict"]
