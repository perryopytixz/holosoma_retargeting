"""Configuration types for sandbox retargeting entry points."""

from __future__ import annotations

from dataclasses import dataclass, field

from holosoma_retargeting.config_types.retargeter import RetargeterConfig
from holosoma_retargeting.config_types.retargeting import RetargetingConfig


@dataclass(frozen=True)
class SandboxRetargeterConfig(RetargeterConfig):
    """Sandbox-only retargeter parameters."""

    hessian_record_enabled: bool = True
    """Whether to save Hessian component diagnostics sidecars."""

    hessian_record_frame_stride: int = 1
    """Record Hessian diagnostics only for frames whose index is divisible by this stride."""

    hessian_record_inner_stride: int = 1
    """Record Hessian diagnostics only for inner iterations whose index is divisible by this stride."""


@dataclass
class SandboxRetargetingConfig(RetargetingConfig):
    """Top-level configuration for sandbox retargeting entry points."""

    retargeter: SandboxRetargeterConfig = field(default_factory=SandboxRetargeterConfig)
    """Sandbox retargeter configuration with Hessian diagnostic recording controls."""
