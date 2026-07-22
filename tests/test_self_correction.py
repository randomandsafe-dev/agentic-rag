"""SelfCorrectionController 测试。

验证闭环：pass不retry / fail触发修正 / 迭代限制 / 异常安全。
全部 Mock LLM + Mock Retriever。
"""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.documents import Document

from agent.self_correction.controller import SelfCorrectionController
from agent.verifier import RetrievalVerifier, VerificationResult


# ============================================================
# Helpers
# ============================================================


def _docs(*contents: str) -> list[Document]:
    return [Document(page_content=c) for c in contents]


def _verifier(results: list[VerificationResult]) -> RetrievalVerifier:
    v = MagicMock(spec=RetrievalVerifier)
    v.verify.side_effect = results
    return v


def _passed(score: float = 0.9) -> VerificationResult:
    return VerificationResult(passed=True, score=score, reason="ok", missing_topics=[])


def _failed(score: float = 0.2, missing: list | None = None) -> VerificationResult:
    return VerificationResult(passed=False, score=score, reason="fail",
                              missing_topics=missing or ["missing_topic"])


# ============================================================
# pass — no retry
# ============================================================


def test_pass_no_retry():
    """验证通过 → 不触发重试，直接返回。"""
    verifier = _verifier([_passed(0.9)])
    ctrl = SelfCorrectionController(verifier, enabled=True, max_iterations=2, min_score=0.5)

    retry_called = []
    docs, result = ctrl.run("q", _docs("a"), lambda q: retry_called.append(q) or _docs("b"))

    assert retry_called == []
    assert result.passed is True
    assert result.score == 0.9


def test_pass_metadata_tagged():
    """通过时 metadata 包含验证信息。"""
    verifier = _verifier([_passed(0.95)])
    ctrl = SelfCorrectionController(verifier, enabled=True)

    docs, _ = ctrl.run("q", _docs("content"), lambda q: _docs("x"))
    assert docs[0].metadata["_verification_passed"] is True
    assert docs[0].metadata["_verification_score"] == 0.95


# ============================================================
# fail → retry
# ============================================================


def test_fail_triggers_retry():
    """验证失败 → 触发修正查询 + 重新检索。"""
    verifier = _verifier([_failed(0.2, ["docker"]), _passed(0.9)])
    ctrl = SelfCorrectionController(verifier, enabled=True, max_iterations=2, min_score=0.5)

    retry_queries = []
    def retriever_fn(q: str):
        retry_queries.append(q)
        return _docs(f"result_for_{q[:20]}")

    docs, result = ctrl.run("kubernetes", _docs("old"), retriever_fn)

    assert len(retry_queries) == 1
    assert "docker" in retry_queries[0]
    assert "kubernetes" in retry_queries[0]
    assert result.passed is True


def test_fail_then_pass_returns_retry_docs():
    """第一次 fail → retry → 第二次 pass → 返回重试结果。"""
    verifier = _verifier([_failed(0.3), _passed(0.85)])
    ctrl = SelfCorrectionController(verifier, enabled=True, max_iterations=2, min_score=0.5)

    retry_docs = _docs("better result")
    docs, result = ctrl.run("q", _docs("bad"), lambda q: retry_docs)

    assert docs == retry_docs
    assert result.passed is True


# ============================================================
# iteration limit
# ============================================================


def test_max_iterations_respected():
    """达到 max_iterations → 返回最后结果，不无限循环。"""
    # 全部 fail
    verifier = _verifier([_failed(0.2), _failed(0.3), _failed(0.1)])
    ctrl = SelfCorrectionController(verifier, enabled=True, max_iterations=2, min_score=0.5)

    retry_count = []
    docs, result = ctrl.run("q", _docs("a"), lambda q: retry_count.append(1) or _docs("b"))

    assert verifier.verify.call_count == 2
    assert len(retry_count) == 1  # max_iterations=2 → 1 retry after initial
    assert result is not None


def test_max_iterations_one_no_retry():
    """max_iterations=1 → 仅验证一次，不重试。"""
    verifier = _verifier([_failed(0.2)])
    ctrl = SelfCorrectionController(verifier, enabled=True, max_iterations=1, min_score=0.5)

    retry_called = []
    docs, result = ctrl.run("q", _docs("a"), lambda q: retry_called.append(1) or _docs("b"))

    assert retry_called == []
    assert verifier.verify.call_count == 1


# ============================================================
# disabled
# ============================================================


def test_disabled_returns_unchanged():
    """enabled=False → 不验证，原样返回。"""
    verifier = _verifier([_passed()])
    ctrl = SelfCorrectionController(verifier, enabled=False)

    docs = _docs("original")
    result_docs, result = ctrl.run("q", docs, lambda q: _docs("never"))

    assert result_docs == docs
    assert result is None
    verifier.verify.assert_not_called()


# ============================================================
# exception safety
# ============================================================


def test_verifier_exception_returns_docs():
    """verifier 异常 → 放行当前 docs。"""
    verifier = MagicMock(spec=RetrievalVerifier)
    verifier.verify.side_effect = RuntimeError("crash")
    ctrl = SelfCorrectionController(verifier, enabled=True)

    docs = _docs("safe")
    result_docs, result = ctrl.run("q", docs, lambda q: _docs("x"))

    assert result_docs == docs
    assert result is None


def test_retriever_exception_returns_current():
    """重试时 retriever_fn 异常 → 放行当前 docs。"""
    verifier = _verifier([_failed(0.2, ["x"])])
    ctrl = SelfCorrectionController(verifier, enabled=True, max_iterations=2, min_score=0.5)

    docs = _docs("current")
    result_docs, result = ctrl.run("q", docs, lambda q: (_ for _ in ()).throw(RuntimeError("boom")))

    assert result_docs == docs


def test_retriever_returns_empty_keeps_current():
    """重试返回空 docs → 保持当前结果。"""
    verifier = _verifier([_failed(0.2, ["x"])])
    ctrl = SelfCorrectionController(verifier, enabled=True, max_iterations=2, min_score=0.5)

    docs = _docs("keep")
    result_docs, result = ctrl.run("q", docs, lambda q: [])

    assert result_docs == docs


# ============================================================
# UserContext compatibility
# ============================================================


def test_user_none_not_affected():
    """user=None 时自纠正正常工作（权限与纠正独立）。"""
    verifier = _verifier([_passed(0.9)])
    ctrl = SelfCorrectionController(verifier, enabled=True)

    docs, result = ctrl.run("q", _docs("x"), lambda q: _docs("y"))
    assert result.passed is True
