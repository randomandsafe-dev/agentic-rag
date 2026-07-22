"""ConcurrentPipeline 集成测试。

验证多 KB 合并、去重、Pipeline 后处理、domain metadata 保留。
"""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.documents import Document

from knowledge.concurrent_pipeline import ConcurrentPipeline


# ============================================================
# Helpers
# ============================================================


def _docs(*contents: str) -> list[Document]:
    return [Document(page_content=c) for c in contents]


def _retriever(docs: list[Document]) -> MagicMock:
    r = MagicMock()
    r.search.return_value = docs
    return r


def _mock_pipeline() -> MagicMock:
    """Mock SearchPipeline: retrieve 原样返回 docs。"""
    p = MagicMock()
    p.retrieve = MagicMock(side_effect=lambda q, r: r.search(q))
    return p


# ============================================================
# Multi KB merge
# ============================================================


def test_multi_kb_merge():
    """多 KB 检索 → 合并结果。"""
    cp = ConcurrentPipeline()
    pipeline = _mock_pipeline()

    docs = cp.retrieve("q", {
        "a": _retriever(_docs("from_a")),
        "b": _retriever(_docs("from_b")),
    }, pipeline)

    assert len(docs) == 2
    contents = {d.page_content for d in docs}
    assert contents == {"from_a", "from_b"}


def test_duplicate_removal():
    """相同内容 → 去重。"""
    cp = ConcurrentPipeline()
    pipeline = _mock_pipeline()

    docs = cp.retrieve("q", {
        "a": _retriever(_docs("same")),
        "b": _retriever(_docs("same")),
    }, pipeline)

    assert len(docs) == 1


def test_domain_metadata_preserved():
    """_domain metadata 在 Pipeline 处理后保留。"""
    cp = ConcurrentPipeline()
    pipeline = _mock_pipeline()

    r1 = _retriever(_docs("content_a"))
    r2 = _retriever(_docs("content_b"))

    docs = cp.retrieve("q", {"tech": r1, "product": r2}, pipeline)

    domains = {d.metadata.get("_domain") for d in docs}
    assert domains == {"tech", "product"}


def test_pipeline_called():
    """Pipeline.retrieve 被调用。"""
    cp = ConcurrentPipeline()
    pipeline = _mock_pipeline()

    cp.retrieve("test query", {"a": _retriever(_docs("x"))}, pipeline)

    pipeline.retrieve.assert_called_once()
    call_query = pipeline.retrieve.call_args[0][0]
    assert call_query == "test query"


def test_empty_concurrent_result():
    """并发检索无结果 → 返回空。"""
    cp = ConcurrentPipeline()
    pipeline = _mock_pipeline()

    r = _retriever([])
    docs = cp.retrieve("q", {"a": r}, pipeline)
    assert docs == []


def test_single_kb_through_pipeline():
    """单 KB 也通过 ConcurrentPipeline（作为对比基准）。"""
    cp = ConcurrentPipeline()
    pipeline = _mock_pipeline()

    docs = cp.retrieve("q", {"default": _retriever(_docs("only"))}, pipeline)
    assert len(docs) == 1
    assert docs[0].page_content == "only"


# ============================================================
# Single KB regression (KnowledgeService)
# ============================================================


def test_single_kb_service_unchanged():
    """单 KB KnowledgeService.search() 行为不变。"""
    from unittest.mock import patch, MagicMock
    from knowledge.registry import KnowledgeBaseRegistry

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

        docs = svc.search("test")
        assert len(docs) == 1
        assert docs[0].page_content == "result"
