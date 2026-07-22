"""Concurrent Multi-KB Search 测试。

验证并行检索、结果合并、权限隔离、单 KB 兼容。
全部使用 Mock Retriever。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from knowledge.access import AccessGuard, AccessPolicy, UserContext
from knowledge.concurrent import ConcurrentRetriever
from knowledge.domain import KnowledgeDomain
from knowledge.registry import KnowledgeBaseRegistry
from knowledge.router import KnowledgeRouter, KeywordRouter, RoutingDecision


# ============================================================
# Helpers
# ============================================================


def _domain(id: str, default: bool = False) -> KnowledgeDomain:
    return KnowledgeDomain(
        id=id, name=id, description="",
        data_dir=Path(".") / id, persist_dir=Path(".") / id,
        collection_name=f"kb_{id}", default=default,
    )


def _policy(content: str) -> Path:
    p = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    p.write(content)
    p.close()
    return Path(p.name)


# ============================================================
# ConcurrentRetriever unit
# ============================================================


def test_concurrent_single_kb_zero_overhead():
    """单 KB → 直接返回，无并发开销。"""
    cr = ConcurrentRetriever()
    mock = MagicMock()
    mock.search.return_value = [Document(page_content="a")]
    docs = cr.search("q", {"default": mock})
    assert len(docs) == 1
    mock.search.assert_called_once()


def test_concurrent_multi_kb_parallel():
    """多 KB → 并发检索，合并结果。"""
    cr = ConcurrentRetriever()
    r1 = MagicMock()
    r1.search.return_value = [Document(page_content="from_a")]
    r2 = MagicMock()
    r2.search.return_value = [Document(page_content="from_b")]

    docs = cr.search("q", {"a": r1, "b": r2})
    assert len(docs) == 2
    contents = {d.page_content for d in docs}
    assert contents == {"from_a", "from_b"}


def test_concurrent_tags_domain_in_metadata():
    """每个文档标记来源 _domain。"""
    cr = ConcurrentRetriever()
    r = MagicMock()
    r.search.return_value = [Document(page_content="x")]
    docs = cr.search("q", {"tech": r})
    assert docs[0].metadata["_domain"] == "tech"


def test_concurrent_one_kb_fails_others_succeed():
    """一个 KB 检索异常 → 其他 KB 正常返回。"""
    cr = ConcurrentRetriever()
    good = MagicMock()
    good.search.return_value = [Document(page_content="ok")]
    bad = MagicMock()
    bad.search.side_effect = RuntimeError("boom")

    docs = cr.search("q", {"good": good, "bad": bad})
    assert len(docs) == 1
    assert docs[0].page_content == "ok"


def test_concurrent_deduplicates():
    """相同内容 → 去重，仅保留首次出现的。"""
    cr = ConcurrentRetriever()
    r1 = MagicMock()
    r1.search.return_value = [Document(page_content="same")]
    r2 = MagicMock()
    r2.search.return_value = [Document(page_content="same")]

    docs = cr.search("q", {"a": r1, "b": r2})
    assert len(docs) == 1


def test_concurrent_empty_retrievers():
    """无 retriever → 返回空。"""
    cr = ConcurrentRetriever()
    assert cr.search("q", {}) == []


# ============================================================
# RoutingDecision backward compat
# ============================================================


def test_routing_decision_backward_compat():
    """domain_id 字段仍然可用，旧代码不受影响。"""
    rd = RoutingDecision(domain_id="tech", domain_ids=["tech", "product"],
                         confidence=0.85, strategy="llm")
    assert rd.domain_id == "tech"
    assert rd.domain_ids == ["tech", "product"]


def test_knowledge_router_populates_domain_ids():
    """KnowledgeRouter 所有路径都填充 domain_ids。"""
    router = KnowledgeRouter(KeywordRouter())
    domains = [_domain("default", default=True)]

    # 单 KB
    rd = router.route("q", domains)
    assert rd.domain_id == "default"
    assert rd.domain_ids == ["default"]

    # 空 domains
    rd = router.route("q", [])
    assert rd.domain_id == ""
    assert rd.domain_ids == []


# ============================================================
# KnowledgeService integration
# ============================================================


def test_service_single_kb_unchanged():
    """单 KB → 仍然走 Pipeline 路径（行为不变）。"""
    mock_llm = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [Document(page_content="result")]

    with patch("knowledge.service.create_llm", return_value=mock_llm):
        from knowledge.service import KnowledgeService
        svc = KnowledgeService()
        svc._registry.get_retriever = MagicMock(return_value=mock_retriever)
        svc._registry.list_domains = MagicMock(return_value=[_domain("default", True)])

        docs = svc.search("q")
        mock_retriever.search.assert_called_once()
        assert len(docs) == 1


def test_service_multi_kb_uses_concurrent():
    """多 domain_ids → 走并发路径。"""
    mock_llm = MagicMock()
    r1 = MagicMock()
    r1.search.return_value = [Document(page_content="a")]
    r2 = MagicMock()
    r2.search.return_value = [Document(page_content="b")]

    with patch("knowledge.service.create_llm", return_value=mock_llm):
        from knowledge.service import KnowledgeService
        svc = KnowledgeService()

        # Router 返回多 domain
        svc._router.route = MagicMock(return_value=RoutingDecision(
            domain_id="a", domain_ids=["a", "b"],
            confidence=0.8, strategy="llm",
        ))
        svc._registry.get_retriever = MagicMock(side_effect=lambda did: {"a": r1, "b": r2}[did])
        svc._registry.list_domains = MagicMock(return_value=[_domain("a"), _domain("b")])

        docs = svc.search("q")
        assert len(docs) == 2


def test_service_with_permission_guard():
    """权限过滤 → Router 只看授权 domains → 并发仅搜索授权 KB。"""
    policy_path = _policy("""
users:
  restricted:
    role: viewer
    allowed_kbs: ["a"]
default:
  role: viewer
  allowed_kbs: []
""")
    try:
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [Document(page_content="x")]

        with patch("knowledge.service.create_llm", return_value=mock_llm):
            from knowledge.service import KnowledgeService
            svc = KnowledgeService()
            svc._access_guard = AccessGuard(AccessPolicy(policy_path))
            svc._registry.list_domains = MagicMock(return_value=[
                _domain("a"), _domain("b"),
            ])
            svc._registry.get_retriever = MagicMock(return_value=mock_retriever)

            user = UserContext(user_id="restricted", role="viewer")
            with patch.object(svc._router, "route", wraps=svc._router.route) as spy:
                svc.search("q", user=user)
                domains_passed = spy.call_args[0][1]
                assert len(domains_passed) == 1
                assert domains_passed[0].id == "a"
    finally:
        Path(policy_path).unlink()
