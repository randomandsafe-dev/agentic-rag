"""Metrics Layer 测试。

验证 collector 记录、No-op 零开销、latency 统计。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from knowledge.metrics import (
    MetricsCollector,
    NoopMetricsCollector,
    RetrievalMetrics,
)
from knowledge.registry import KnowledgeBaseRegistry


# ============================================================
# RetrievalMetrics
# ============================================================


def test_metrics_defaults():
    """默认值正确。"""
    m = RetrievalMetrics()
    assert m.retrieval_count == 0
    assert m.retry_count == 0
    assert m.verification_score is None
    assert m.latency_ms == 0.0
    assert m.error is None


def test_metrics_populated():
    """所有字段正确填充。"""
    m = RetrievalMetrics(
        request_id="abc",
        query="test query",
        user_id="alice",
        domain_ids=["default"],
        retrieval_count=5,
        retry_count=1,
        verification_score=0.85,
        verification_passed=True,
        latency_ms=250.0,
    )
    assert m.retrieval_count == 5
    assert m.retry_count == 1
    assert m.verification_passed is True
    assert m.user_id == "alice"


# ============================================================
# MetricsCollector
# ============================================================


def test_collector_records_without_error():
    """正常记录不抛异常。"""
    collector = MetricsCollector()
    m = RetrievalMetrics(request_id="x", query="q", retrieval_count=3, latency_ms=100)
    collector.record(m)  # 不应抛异常


def test_collector_records_with_error():
    """含 error 的记录。"""
    collector = MetricsCollector()
    m = RetrievalMetrics(request_id="x", query="q", error="timeout")
    collector.record(m)


# ============================================================
# NoopMetricsCollector
# ============================================================


def test_noop_collector_no_side_effects():
    """No-op 收集器不执行任何操作。"""
    collector = NoopMetricsCollector()
    m = RetrievalMetrics(request_id="x")
    collector.record(m)  # 不记录，不抛异常


# ============================================================
# KnowledgeService integration
# ============================================================


def test_service_search_records_metrics():
    """search() 调用后 metrics 被记录。"""
    mock_llm = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [Document(page_content="result")]

    with patch("knowledge.service.create_llm", return_value=mock_llm):
        from knowledge.service import KnowledgeService
        svc = KnowledgeService()
        svc._registry.get_retriever = MagicMock(return_value=mock_retriever)
        svc._registry.list_domains = MagicMock(return_value=[
            MagicMock(id="default", default=True),
        ])

        # 替换为真实 collector
        mock_collector = MagicMock(wraps=MetricsCollector())
        svc._metrics = mock_collector

        docs = svc.search("test query")
        assert len(docs) == 1

        # 验证 record 被调用
        mock_collector.record.assert_called_once()
        recorded = mock_collector.record.call_args[0][0]
        assert isinstance(recorded, RetrievalMetrics)
        assert recorded.query == "test query"
        assert recorded.retrieval_count == 1
        assert recorded.latency_ms >= 0


def test_service_with_noop_collector():
    """No-op collector 不影响 search 正常执行。"""
    mock_llm = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [Document(page_content="ok")]

    with patch("knowledge.service.create_llm", return_value=mock_llm):
        from knowledge.service import KnowledgeService
        svc = KnowledgeService()
        svc._registry.get_retriever = MagicMock(return_value=mock_retriever)
        svc._registry.list_domains = MagicMock(return_value=[
            MagicMock(id="default", default=True),
        ])
        svc._metrics = NoopMetricsCollector()

        docs = svc.search("test")
        assert len(docs) == 1


def test_service_metrics_on_exception():
    """search() 异常时仍记录 metrics（含 error）。"""
    mock_llm = MagicMock()

    with patch("knowledge.service.create_llm", return_value=mock_llm):
        from knowledge.service import KnowledgeService
        svc = KnowledgeService()
        svc._registry.list_domains = MagicMock(side_effect=RuntimeError("boom"))

        mock_collector = MagicMock(wraps=MetricsCollector())
        svc._metrics = mock_collector

        with pytest.raises(RuntimeError):
            svc.search("test")

        mock_collector.record.assert_called_once()
        recorded = mock_collector.record.call_args[0][0]
        assert "boom" in recorded.error
        assert recorded.latency_ms >= 0


def test_metrics_includes_user_id():
    """user 不为 None 时 metrics 记录 user_id。"""
    mock_llm = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [Document(page_content="x")]

    with patch("knowledge.service.create_llm", return_value=mock_llm):
        from knowledge.service import KnowledgeService
        from knowledge.access import UserContext

        svc = KnowledgeService()
        svc._registry.get_retriever = MagicMock(return_value=mock_retriever)
        svc._registry.list_domains = MagicMock(return_value=[
            MagicMock(id="default", default=True),
        ])

        mock_collector = MagicMock(wraps=MetricsCollector())
        svc._metrics = mock_collector

        svc.search("q", user=UserContext(user_id="alice"))
        recorded = mock_collector.record.call_args[0][0]
        assert recorded.user_id == "alice"


def test_metrics_user_none():
    """user=None 时 metrics.user_id 为 None。"""
    mock_llm = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [Document(page_content="x")]

    with patch("knowledge.service.create_llm", return_value=mock_llm):
        from knowledge.service import KnowledgeService
        svc = KnowledgeService()
        svc._registry.get_retriever = MagicMock(return_value=mock_retriever)
        svc._registry.list_domains = MagicMock(return_value=[
            MagicMock(id="default", default=True),
        ])
        mock_collector = MagicMock(wraps=MetricsCollector())
        svc._metrics = mock_collector

        svc.search("q", user=None)
        recorded = mock_collector.record.call_args[0][0]
        assert recorded.user_id is None


def test_metrics_latency_positive():
    """latency_ms 必须为正数。"""
    mock_llm = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [Document(page_content="x")]

    with patch("knowledge.service.create_llm", return_value=mock_llm):
        from knowledge.service import KnowledgeService
        svc = KnowledgeService()
        svc._registry.get_retriever = MagicMock(return_value=mock_retriever)
        svc._registry.list_domains = MagicMock(return_value=[
            MagicMock(id="default", default=True),
        ])
        mock_collector = MagicMock(wraps=MetricsCollector())
        svc._metrics = mock_collector

        svc.search("q")
        recorded = mock_collector.record.call_args[0][0]
        assert recorded.latency_ms >= 0
