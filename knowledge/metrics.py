"""Metrics Layer — 检索工作流可观测性。

记录每次 search 的关键指标：延迟、重试、验证分数、异常。
支持 No-op 模式（默认），零开销。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================
# RetrievalMetrics
# ============================================================


@dataclass
class RetrievalMetrics:
    """单次检索的完整指标快照。

    Attributes:
        request_id: 唯一请求标识。
        query: 原始查询（截断至 100 字符）。
        user_id: 可选用户 ID。
        domain_ids: 命中的 domain ID 列表。
        retrieval_count: 检索返回的文档数。
        retry_count: 自纠正重试次数。
        verification_score: 最终验证分数（未启用时为 None）。
        verification_passed: 最终验证是否通过（未启用时为 None）。
        latency_ms: 端到端延迟（毫秒）。
        error: 异常信息（无异常时为 None）。
    """

    request_id: str = ""
    query: str = ""
    user_id: str | None = None
    domain_ids: list[str] = field(default_factory=list)
    retrieval_count: int = 0
    retry_count: int = 0
    verification_score: float | None = None
    verification_passed: bool | None = None
    latency_ms: float = 0.0
    error: str | None = None


# ============================================================
# MetricsCollector
# ============================================================


class MetricsCollector:
    """指标收集器接口。

    默认实现仅记录结构化日志。替换为自定义子类可接入外部系统。
    """

    def record(self, metrics: RetrievalMetrics) -> None:
        """记录一次检索的完整指标。

        Args:
            metrics: 检索指标快照。
        """
        self._log(metrics)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    @staticmethod
    def _log(m: RetrievalMetrics) -> None:
        """结构化日志输出。"""
        parts = [
            f"request_id={m.request_id}",
            f"query={m.query[:100]}",
            f"domains={m.domain_ids}",
            f"docs={m.retrieval_count}",
            f"retries={m.retry_count}",
            f"verify_score={m.verification_score}",
            f"verify_passed={m.verification_passed}",
            f"latency_ms={m.latency_ms:.0f}",
        ]
        if m.user_id:
            parts.append(f"user={m.user_id}")
        if m.error:
            parts.append(f"error={m.error}")
            logger.warning("retrieval_completed " + " ".join(parts))
        else:
            logger.info("retrieval_completed " + " ".join(parts))


# ============================================================
# No-op Collector
# ============================================================


class NoopMetricsCollector(MetricsCollector):
    """空操作收集器 —— 未启用 metrics 时的默认实现，零开销。"""

    def record(self, metrics: RetrievalMetrics) -> None:
        pass
