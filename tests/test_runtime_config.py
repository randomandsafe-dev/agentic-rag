"""RuntimeConfig 加载器测试。

验证默认值、runtime.yaml 覆盖、回退、优先级。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from runtime_loader import (
    RuntimeConfig,
    VerificationConfig,
    SelfCorrectionConfig,
    MetricsConfig,
    load_runtime_config,
)


# ============================================================
# Helpers
# ============================================================


def _yaml(content: str) -> Path:
    p = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    p.write(content)
    p.close()
    return Path(p.name)


# ============================================================
# Defaults
# ============================================================


def test_default_verification_config():
    """默认值：全部关闭。"""
    v = VerificationConfig()
    assert v.enabled is False
    assert v.max_retry == 2
    assert v.min_score == 0.5


def test_default_runtime_config():
    """默认 RuntimeConfig 全部 disabled。"""
    rt = RuntimeConfig()
    assert rt.verification.enabled is False
    assert rt.self_correction.enabled is False
    assert rt.metrics.enabled is False


def test_load_missing_file_returns_defaults():
    """runtime.yaml 不存在 → 回退到默认值。"""
    rt = load_runtime_config("/nonexistent/runtime.yaml")
    assert rt.verification.enabled is False
    assert rt.metrics.enabled is False


# ============================================================
# runtime.yaml override
# ============================================================


def test_runtime_overrides_defaults():
    """runtime.yaml 覆盖默认值。"""
    path = _yaml("""
verification:
  enabled: true
  max_retry: 3
  min_score: 0.7
self_correction:
  enabled: true
  max_iterations: 4
metrics:
  enabled: true
  query_length_limit: 200
""")
    try:
        rt = load_runtime_config(path)
        assert rt.verification.enabled is True
        assert rt.verification.max_retry == 3
        assert rt.verification.min_score == 0.7
        assert rt.self_correction.enabled is True
        assert rt.self_correction.max_iterations == 4
        assert rt.metrics.enabled is True
        assert rt.metrics.query_length_limit == 200
    finally:
        path.unlink()


def test_runtime_partial_override():
    """部分字段覆盖，其余保持默认。"""
    path = _yaml("""
verification:
  enabled: true
""")
    try:
        rt = load_runtime_config(path)
        assert rt.verification.enabled is True
        assert rt.verification.max_retry == 2  # 默认
        assert rt.self_correction.enabled is False  # 默认
    finally:
        path.unlink()


def test_invalid_yaml_fallback():
    """格式错误的 YAML → 回退默认值。"""
    path = _yaml("invalid: [")
    try:
        rt = load_runtime_config(path)
        assert rt.verification.enabled is False
    finally:
        path.unlink()


def test_empty_yaml_defaults():
    """空 YAML → 默认值。"""
    path = _yaml("")
    try:
        rt = load_runtime_config(path)
        assert rt.verification.enabled is False
    finally:
        path.unlink()


# ============================================================
# Typed access
# ============================================================


def test_config_is_typed():
    """返回值是 typed dataclass。"""
    rt = load_runtime_config()
    assert isinstance(rt, RuntimeConfig)
    assert isinstance(rt.verification, VerificationConfig)
    assert isinstance(rt.self_correction, SelfCorrectionConfig)
    assert isinstance(rt.metrics, MetricsConfig)
