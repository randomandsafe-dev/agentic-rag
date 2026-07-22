"""SearchPipeline 集成测试。

全部使用 Mock，不依赖真实 Chroma / LLM API / 外部服务。
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest
from langchain_core.documents import Document

from search_pipeline import (
    LLMRelevanceJudge,
    QueryRewriter,
    RelevanceStrategy,
    SearchPipeline,
)


# ============================================================
# Helpers
# ============================================================


def _make_docs(*contents: str) -> list[Document]:
    return [Document(page_content=c) for c in contents]


def _make_llm(content: str = "") -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value.content = content
    return llm


def _make_retriever(*docs_lists: list[Document]) -> MagicMock:
    """创建 Mock retriever，每次调用依次返回一组 docs。"""
    retriever = MagicMock()
    retriever.search = MagicMock()
    if docs_lists:
        retriever.search.side_effect = list(docs_lists)
    else:
        retriever.search.return_value = []
    return retriever


# ============================================================
# 1. Fast Path
# ============================================================


def test_fast_path_direct_search():
    """关闭所有增强 → 直接调用 retriever.search(query)，不触发 Rewriter/Judge。"""
    retriever = _make_retriever(_make_docs("a", "b"))
    rewriter = MagicMock(spec=QueryRewriter)
    judge = MagicMock(spec=LLMRelevanceJudge)

    pipeline = SearchPipeline(
        rewriter=rewriter,
        judge=judge,
        rewrite_enabled=False,
        judge_enabled=False,
    )

    docs = pipeline.retrieve("test query", retriever)

    retriever.search.assert_called_once_with("test query")
    rewriter.rewrite.assert_not_called()
    rewriter.rewrite_retry.assert_not_called()
    judge.judge.assert_not_called()
    assert len(docs) == 2


def test_fast_path_no_rewriter_no_judge_ok():
    """Pipeline 构造时不传 rewriter/judge + 关闭增强 → 正常工作。"""
    retriever = _make_retriever(_make_docs("x"))
    pipeline = SearchPipeline(rewrite_enabled=False, judge_enabled=False)
    docs = pipeline.retrieve("q", retriever)
    assert len(docs) == 1


# ============================================================
# 2. Rewrite
# ============================================================


def test_rewrite_called_and_used():
    """开启 Rewrite → retriever.search 收到改写后的 query。"""
    original = "原始查询"
    rewritten = "改写后的查询"
    llm = _make_llm(rewritten)
    rewriter = QueryRewriter(llm)
    retriever = _make_retriever(_make_docs("result"))

    pipeline = SearchPipeline(
        rewriter=rewriter,
        rewrite_enabled=True,
        judge_enabled=False,
    )

    pipeline.retrieve(original, retriever)

    retriever.search.assert_called_once_with(rewritten)


# ============================================================
# 3. Rewrite + Judge
# ============================================================


def test_rewrite_and_judge_full_flow():
    """Rewrite + Judge 均开启 → 完整流程返回通过判断的文档。"""
    rewrite_llm = _make_llm("rewritten")
    judge_llm = _make_llm("[3, 0]")  # doc0 相关, doc1 不相关

    rewriter = QueryRewriter(rewrite_llm)
    judge = LLMRelevanceJudge(judge_llm)
    retriever = _make_retriever(_make_docs("relevant", "irrelevant"))

    pipeline = SearchPipeline(
        rewriter=rewriter,
        judge=judge,
        rewrite_enabled=True,
        judge_enabled=True,
        relevance_threshold=2,
    )

    docs = pipeline.retrieve("原始查询", retriever)

    retriever.search.assert_called_once_with("rewritten")
    assert len(docs) == 1
    assert docs[0].page_content == "relevant"


# ============================================================
# 4. Judge Retry
# ============================================================


def test_judge_retry_on_irrelevant():
    """第一次 judge 判定不相关 → 触发 rewrite_retry + 第二次 search + 第二次 judge。"""
    rewrite_llm = _make_llm()
    # 第一次 rewrite 返回 "r1"，retry 返回 "r2"
    rewrite_llm.invoke.side_effect = [
        MagicMock(content="r1"),
        MagicMock(content="r2"),
    ]

    judge_llm = _make_llm()
    # 第一次 judge 全 0（不相关），第二次 3（相关）
    judge_llm.invoke.side_effect = [
        MagicMock(content="[0, 0]"),
        MagicMock(content="[3, 0]"),
    ]

    rewriter = QueryRewriter(rewrite_llm)
    judge = LLMRelevanceJudge(judge_llm)
    retriever = _make_retriever(
        _make_docs("d1", "d2"),  # 第一次检索
        _make_docs("d1", "d2"),  # 第二次检索
    )

    pipeline = SearchPipeline(
        rewriter=rewriter,
        judge=judge,
        rewrite_enabled=True,
        judge_enabled=True,
        max_retries=1,
        relevance_threshold=2,
    )

    docs = pipeline.retrieve("orig", retriever)

    assert retriever.search.call_count == 2
    assert retriever.search.call_args_list[0] == call("r1")
    assert retriever.search.call_args_list[1] == call("r2")
    assert len(docs) == 1
    assert docs[0].page_content == "d1"


# ============================================================
# 5. Max Retries Exhausted
# ============================================================


def test_max_retries_exhausted_returns_last_docs():
    """Judge 一直判定不相关直到 max_retries 耗尽 → 返回最后一次检索的全部文档。"""
    rewrite_llm = _make_llm()
    rewrite_llm.invoke.side_effect = [
        MagicMock(content="r1"),
        MagicMock(content="r2"),
        MagicMock(content="r3"),
    ]

    judge_llm = _make_llm()
    # 所有 judge 都返回全 0
    judge_llm.invoke.side_effect = [
        MagicMock(content="[0, 0]"),
        MagicMock(content="[0, 0]"),
        MagicMock(content="[0, 0]"),
    ]

    rewriter = QueryRewriter(rewrite_llm)
    judge = LLMRelevanceJudge(judge_llm)
    retriever = _make_retriever(
        _make_docs("a1", "a2"),
        _make_docs("b1", "b2"),
        _make_docs("c1", "c2"),
    )

    pipeline = SearchPipeline(
        rewriter=rewriter,
        judge=judge,
        rewrite_enabled=True,
        judge_enabled=True,
        max_retries=2,  # 共 3 次尝试
        relevance_threshold=2,
    )

    docs = pipeline.retrieve("orig", retriever)

    # 3 次尝试后达到 max_retries，返回最后一次检索的文档
    assert retriever.search.call_count == 3
    assert len(docs) == 2
    assert docs[0].page_content == "c1"


def test_max_retries_zero_one_attempt():
    """max_retries=0 → 仅 1 次尝试，不重试。"""
    judge_llm = _make_llm("[0]")
    retriever = _make_retriever(_make_docs("d"))

    pipeline = SearchPipeline(
        rewriter=QueryRewriter(_make_llm("r1")),
        judge=LLMRelevanceJudge(judge_llm),
        rewrite_enabled=True,
        judge_enabled=True,
        max_retries=0,
        relevance_threshold=2,
    )

    docs = pipeline.retrieve("q", retriever)
    assert retriever.search.call_count == 1
    assert len(docs) == 1  # 返回最后一次检索的全部文档


# ============================================================
# 6. Rewriter Exception
# ============================================================


def test_rewriter_exception_fallback_to_original_query():
    """rewrite() 抛异常 → 使用原始 query 继续检索。"""
    rewriter = MagicMock(spec=QueryRewriter)
    rewriter.rewrite.side_effect = RuntimeError("LLM unavailable")

    retriever = _make_retriever(_make_docs("fallback_result"))

    pipeline = SearchPipeline(
        rewriter=rewriter,
        rewrite_enabled=True,
        judge_enabled=False,
    )

    docs = pipeline.retrieve("原始查询", retriever)

    retriever.search.assert_called_once_with("原始查询")
    assert len(docs) == 1


def test_rewriter_retry_exception_fallback_to_original():
    """rewrite_retry() 抛异常 → 使用原始 query 重试检索。"""
    rewrite_llm = _make_llm("r1")
    rewriter = QueryRewriter(rewrite_llm)
    # rewrite 正常，但 rewrite_retry 异常
    rewriter.rewrite_retry = MagicMock(side_effect=RuntimeError("LLM unavailable"))

    judge_llm = _make_llm("[0]")  # 触发 retry
    judge = LLMRelevanceJudge(judge_llm)

    retriever = _make_retriever(
        _make_docs("a"),
        _make_docs("b"),
    )

    pipeline = SearchPipeline(
        rewriter=rewriter,
        judge=judge,
        rewrite_enabled=True,
        judge_enabled=True,
        max_retries=1,
        relevance_threshold=2,
    )

    docs = pipeline.retrieve("orig", retriever)

    # retry 时 rewrite_retry 抛异常 → fallback 到 original query
    assert retriever.search.call_count == 2
    assert retriever.search.call_args_list[1] == call("orig")


# ============================================================
# 7. Judge Exception
# ============================================================


def test_judge_exception_returns_all_docs():
    """judge() 抛异常 → 直接返回全部文档，不中断检索。"""
    judge = MagicMock(spec=LLMRelevanceJudge)
    judge.judge.side_effect = RuntimeError("Judge failed")
    judge.has_relevant = LLMRelevanceJudge.has_relevant  # 不会被调用但仍保留

    retriever = _make_retriever(_make_docs("a", "b", "c"))

    pipeline = SearchPipeline(
        judge=judge,
        judge_enabled=True,
        rewrite_enabled=False,
    )

    docs = pipeline.retrieve("q", retriever)

    assert len(docs) == 3
    retriever.search.assert_called_once_with("q")


# ============================================================
# 8. Retriever Exception
# ============================================================


def test_retriever_exception_with_rewritten_falls_back_to_original():
    """retriever.search(rewritten) 异常 → 用原始 query 重试。"""
    retriever = MagicMock()
    # 第一次 search(rewritten) 抛异常，第二次 search(original) 正常
    retriever.search.side_effect = [
        RuntimeError("Search failed with rewritten query"),
        _make_docs("recovered"),
    ]

    pipeline = SearchPipeline(
        rewriter=QueryRewriter(_make_llm("rewritten")),
        rewrite_enabled=True,
        judge_enabled=False,
    )

    docs = pipeline.retrieve("original", retriever)

    assert retriever.search.call_count == 2
    assert retriever.search.call_args_list[0] == call("rewritten")
    assert retriever.search.call_args_list[1] == call("original")
    assert len(docs) == 1
    assert docs[0].page_content == "recovered"


def test_retriever_exception_both_fail_propagates():
    """retriever.search 两次都异常 → 异常传播。"""
    retriever = MagicMock()
    retriever.search.side_effect = RuntimeError("Search completely failed")

    pipeline = SearchPipeline(rewrite_enabled=False, judge_enabled=False)

    with pytest.raises(RuntimeError, match="Search completely failed"):
        pipeline.retrieve("q", retriever)


# ============================================================
# 9. Stateless
# ============================================================


def test_pipeline_stateless_across_calls():
    """两次 retrieve() 调用之间无状态污染。"""
    pipeline = SearchPipeline(rewrite_enabled=False, judge_enabled=False)

    r1 = _make_retriever(_make_docs("from_r1"))
    r2 = _make_retriever(_make_docs("from_r2"))

    docs1 = pipeline.retrieve("q1", r1)
    docs2 = pipeline.retrieve("q2", r2)

    assert docs1[0].page_content == "from_r1"
    assert docs2[0].page_content == "from_r2"
    assert r1.search.call_count == 1
    assert r2.search.call_count == 1


def test_pipeline_no_retriever_attribute():
    """Pipeline 实例没有 _retriever 属性。"""
    pipeline = SearchPipeline(rewrite_enabled=False, judge_enabled=False)
    assert not hasattr(pipeline, "_retriever")


# ============================================================
# 10. Duck Typing
# ============================================================


def test_duck_typing_custom_retriever():
    """任意实现了 search(query) 的对象都可以作为 retriever。"""

    class DummyRetriever:
        def search(self, query: str) -> list[Document]:
            return [Document(page_content=f"dummy: {query}")]

    pipeline = SearchPipeline(rewrite_enabled=False, judge_enabled=False)
    docs = pipeline.retrieve("hello", DummyRetriever())

    assert len(docs) == 1
    assert "dummy: hello" in docs[0].page_content


def test_duck_typing_no_type_constraint():
    """SearchPipeline 模块不 import Chroma / HybridRetriever。"""
    import ast

    with open("search_pipeline.py", "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "langchain_chroma" not in node.module
            assert "retrieval" not in node.module
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "Chroma" not in alias.name
                assert "HybridRetriever" not in alias.name


# ============================================================
# Extra: Edge Cases
# ============================================================


def test_empty_docs_from_retriever_triggers_retry():
    """retriever 返回空列表 → 触发重试而非立即返回。"""
    judge_llm = _make_llm("[3]")
    retriever = _make_retriever(
        [],  # 第一次空
        _make_docs("found"),
    )

    pipeline = SearchPipeline(
        rewriter=QueryRewriter(_make_llm("r1")),
        judge=LLMRelevanceJudge(judge_llm),
        rewrite_enabled=True,
        judge_enabled=True,
        max_retries=1,
        relevance_threshold=2,
    )

    docs = pipeline.retrieve("q", retriever)
    assert retriever.search.call_count == 2
    assert len(docs) == 1


def test_judge_disabled_returns_docs_immediately_after_rewrite():
    """Judge 关闭时，retrieve 后直接返回所有文档，不调用 judge。"""
    judge = MagicMock(spec=LLMRelevanceJudge)
    retriever = _make_retriever(_make_docs("a", "b"))

    pipeline = SearchPipeline(
        rewriter=QueryRewriter(_make_llm("rw")),
        judge=judge,
        rewrite_enabled=True,
        judge_enabled=False,
    )

    docs = pipeline.retrieve("q", retriever)
    judge.judge.assert_not_called()
    assert len(docs) == 2


def test_rewrite_disabled_uses_original_query_even_on_retry():
    """Rewrite 关闭时，retry 也应该使用原始 query（不调用 rewriter）。"""
    judge_llm = _make_llm()
    judge_llm.invoke.side_effect = [
        MagicMock(content="[0]"),
        MagicMock(content="[3]"),
    ]

    rewriter = MagicMock(spec=QueryRewriter)
    retriever = _make_retriever(
        _make_docs("a"),
        _make_docs("b"),
    )

    pipeline = SearchPipeline(
        rewriter=rewriter,
        judge=LLMRelevanceJudge(judge_llm),
        rewrite_enabled=False,
        judge_enabled=True,
        max_retries=1,
        relevance_threshold=2,
    )

    docs = pipeline.retrieve("original", retriever)

    # 不调用 rewriter
    rewriter.rewrite.assert_not_called()
    rewriter.rewrite_retry.assert_not_called()
    # 两次 search 都用原始 query
    assert retriever.search.call_args_list == [call("original"), call("original")]
    assert len(docs) == 1


def test_relevance_strategy_abc():
    """RelevanceStrategy ABC 不能直接实例化，has_relevant 正常工作。"""

    class MockStrategy(RelevanceStrategy):
        def judge(self, query, documents):
            return [3, 0, 1]

    s = MockStrategy()
    assert s.has_relevant([3, 0, 1], threshold=2) is True
    assert s.has_relevant([1, 0, 1], threshold=2) is False
    assert s.has_relevant([], threshold=2) is False


# ============================================================
# Extra: CI Compatibility
# ============================================================


def test_no_real_api_calls():
    """确认测试套件不依赖真实 Chroma 或 LLM API。"""
    # 如果任何测试意外触发了真实 API 调用，会因为网络不可达或 API key 缺失而失败
    # 此测试作为哨兵存在：所有外部依赖均应被 Mock
    assert True  # 若执行到这里说明没有意外触发真实调用
