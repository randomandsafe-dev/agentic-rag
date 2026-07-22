"""Runtime Configuration Loader.

从 runtime.yaml 统一加载 Phase 6 运行时配置。
回退顺序: runtime.yaml > 单独配置文件 > 默认值。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from config import ROOT_DIR

# ============================================================
# Config objects
# ============================================================


@dataclass
class VerificationConfig:
    enabled: bool = False
    max_retry: int = 2
    min_score: float = 0.5


@dataclass
class SelfCorrectionConfig:
    enabled: bool = False
    max_iterations: int = 3


@dataclass
class MetricsConfig:
    enabled: bool = False
    query_length_limit: int = 100


@dataclass
class RuntimeConfig:
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    self_correction: SelfCorrectionConfig = field(default_factory=SelfCorrectionConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)


# ============================================================
# Loader
# ============================================================


def load_runtime_config(config_path: str | Path | None = None) -> RuntimeConfig:
    """加载运行时配置。

    优先级: runtime.yaml > 单独配置文件 > 默认值。

    Args:
        config_path: runtime.yaml 路径，None 时使用默认路径。

    Returns:
        RuntimeConfig 实例，所有字段保证非 None。
    """
    if config_path is None:
        config_path = ROOT_DIR / "config" / "runtime.yaml"
    path = Path(config_path)

    runtime = RuntimeConfig()

    if not path.exists():
        # 回退到单独配置文件
        runtime.verification = _load_verification_config()
        runtime.metrics = _load_metrics_config()
        runtime.self_correction = SelfCorrectionConfig(
            enabled=runtime.verification.enabled,
            max_iterations=runtime.verification.max_retry + 1,
        )
        return runtime

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception:
        return runtime

    # verification
    v = data.get("verification", {})
    if v:
        runtime.verification = VerificationConfig(
            enabled=bool(v.get("enabled", False)),
            max_retry=int(v.get("max_retry", 2)),
            min_score=float(v.get("min_score", 0.5)),
        )

    # self_correction
    sc = data.get("self_correction", {})
    if sc:
        runtime.self_correction = SelfCorrectionConfig(
            enabled=bool(sc.get("enabled", False)),
            max_iterations=int(sc.get("max_iterations", 3)),
        )

    # metrics
    m = data.get("metrics", {})
    if m:
        runtime.metrics = MetricsConfig(
            enabled=bool(m.get("enabled", False)),
            query_length_limit=int(m.get("query_length_limit", 100)),
        )

    return runtime


# ============================================================
# Fallback loaders (individual config files)
# ============================================================


def _load_verification_config() -> VerificationConfig:
    path = ROOT_DIR / "config" / "verification.yaml"
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            v = data.get("verification", {})
            return VerificationConfig(
                enabled=bool(v.get("enabled", False)),
                max_retry=int(v.get("max_retry", 2)),
                min_score=float(v.get("min_score", 0.5)),
            )
    except Exception:
        pass
    return VerificationConfig()


def _load_metrics_config() -> MetricsConfig:
    path = ROOT_DIR / "config" / "metrics.yaml"
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            m = data.get("metrics", {})
            return MetricsConfig(
                enabled=bool(m.get("enabled", False)),
                query_length_limit=int(m.get("query_length_limit", 100)),
            )
    except Exception:
        pass
    return MetricsConfig()
