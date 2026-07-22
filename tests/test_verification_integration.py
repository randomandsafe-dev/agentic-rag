"""VerificationController 集成测试。

验证 verification 启停、pass/fail、retry、异常 fallback。
全部使用 Mock LLM + Mock Retriever。
"""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.documents import Document

from agent.verifier import RetrievalVerifier, VerificationResult
from agent.verification.controller import VerificationController


# ============================================================
# Helpers
# ============================================================


def _docs(*contents: str) -> list[Document]:
    return [Document(page_content=c) for c in contents]


def _mock_verifier(passed: bool, score: float = 0.9, missing: list | None = None) -> RetrievalVerifier:
    verifier = MagicMock(spec=RetrievalVerifier)
    verifier.verify.return_value = VerificationResult(
        passed=passed, score=score, reason="mock",
        missing_topics=missing or [],
    )
    return verifier


# ============================================================
# Disabled — backward compatibility
# ============================================================


def test_disabled_returns_docs_unchanged():
    """verification disabled → 原样返回 docs，不调用 verifier。"""
    verifier = _mock_verifier(True)
    ctrl = VerificationController(verifier, enabled=False)
    docs = _docs("a", "b")

    result_docs, vresult = ctrl.verify_and_retry("q", docs, lambda q: _docs("c"))

    assert result_docs == docs
    assert vresult is None
    verifier.verify.assert_not_called()


def test_disabled_no_metadata_injection():
    """verification disabled → 不注入 _verification metadata。"""
    verifier = _mock_verifier(True)
    ctrl = VerificationController(verifier, enabled=False)
    docs = _docs("x")

    result_docs, _ = ctrl.verify_and_retry("q", docs, lambda q: _docs("y"))
    assert "_verification_passed" not in result_docs[0].metadata


# ============================================================
# Enabled — pass
# ============================================================


def test_verifier_pass_returns_docs_with_metadata():
    """验证通过 → 返回 docs + _verification metadata。"""
    verifier = _mock_verifier(True, score=0.9)
    ctrl = VerificationController(verifier, enabled=True, min_score=0.5)
    docs = _docs("relevant")

    result_docs, vresult = ctrl.verify_and_retry("q", docs, lambda q: _docs("new"))

    assert vresult is not None
    assert vresult.passed is True
    assert result_docs[0].metadata["_verification_passed"] is True
    assert result_docs[0].metadata["_verification_score"] == 0.9


# ============================================================
# Enabled — fail + retry
# ============================================================


def test_verifier_fail_triggers_retry():
    """验证失败 → 触发 retriever_fn 重新检索。"""
    verifier = _mock_verifier(False, score=0.2, missing=["deployment"])
    ctrl = VerificationController(verifier, enabled=True, max_retry=1, min_score=0.5)

    retry_called = []
    def retriever_fn(q: str):
        retry_called.append(q)
        return _docs("retry result")

    result_docs, vresult = ctrl.verify_and_retry("Docker", _docs("old"), retriever_fn)

    assert len(retry_called) == 1
    assert "deployment" in retry_called[0]
    assert "retry result" in result_docs[0].page_content


def test_verifier_pass_on_retry():
    """第一次 fail → retry → 第二次 pass → 返回第二次结果。"""
    verifier = MagicMock(spec=RetrievalVerifier)
    verifier.verify.side_effect = [
        VerificationResult(passed=False, score=0.3, reason="fail", missing_topics=["docker"]),
        VerificationResult(passed=True, score=0.85, reason="pass", missing_topics=[]),
    ]
    ctrl = VerificationController(verifier, enabled=True, max_retry=1, min_score=0.5)

    retry_docs = _docs("retry doc")
    result_docs, vresult = ctrl.verify_and_retry("q", _docs("old"), lambda q: retry_docs)

    assert vresult.passed is True
    assert verifier.verify.call_count == 2


# ============================================================
# Retry limit
# ============================================================


def test_retry_limit_respected():
    """max_retry=1 → 最多 2 次 verify（初次 + 1 次重试）。"""
    verifier = MagicMock(spec=RetrievalVerifier)
    verifier.verify.return_value = VerificationResult(
        passed=False, score=0.2, reason="fail", missing_topics=["x"],
    )
    ctrl = VerificationController(verifier, enabled=True, max_retry=1, min_score=0.5)

    result_docs, vresult = ctrl.verify_and_retry("q", _docs("a"), lambda q: _docs("b"))

    assert verifier.verify.call_count == 2  # initial + 1 retry
    assert vresult is not None


def test_max_retry_zero_no_retry():
    """max_retry=0 → 仅验证一次，不重试。"""
    verifier = _mock_verifier(False, score=0.2)
    ctrl = VerificationController(verifier, enabled=True, max_retry=0, min_score=0.5)

    retry_called = []
    result_docs, vresult = ctrl.verify_and_retry("q", _docs("a"), lambda q: retry_called.append(q) or _docs("b"))

    assert retry_called == []
    assert vresult is not None
    assert verifier.verify.call_count == 1


# ============================================================
# Exception fallback
# ============================================================


def test_verifier_exception_returns_docs():
    """verifier 抛异常 → 放行原 docs，不中断检索。"""
    verifier = MagicMock(spec=RetrievalVerifier)
    verifier.verify.side_effect = RuntimeError("timeout")
    ctrl = VerificationController(verifier, enabled=True)

    docs = _docs("safe")
    result_docs, vresult = ctrl.verify_and_retry("q", docs, lambda q: _docs("never"))

    assert result_docs == docs
    assert vresult is None


def test_retry_retriever_exception_falls_back():
    """retry 时 retriever_fn 异常 → 放行当前 docs。"""
    verifier = _mock_verifier(False, score=0.2, missing=["x"])
    ctrl = VerificationController(verifier, enabled=True, max_retry=1, min_score=0.5)

    docs = _docs("current")
    result_docs, vresult = ctrl.verify_and_retry("q", docs, lambda q: (_ for _ in ()).throw(RuntimeError("boom")))

    assert result_docs == docs


# ============================================================
# user=None compatibility
# ============================================================


def test_user_none_does_not_affect_verification():
    """user=None 时 verification 仍正常工作（权限与验证独立）。"""
    verifier = _mock_verifier(True, score=0.95)
    ctrl = VerificationController(verifier, enabled=True)
    docs = _docs("content")

    result_docs, vresult = ctrl.verify_and_retry("q", docs, lambda q: _docs("new"))

    assert vresult.passed is True
    assert result_docs[0].metadata["_verification_score"] == 0.95
