"""RetrievalVerifier 单元测试。

全部使用 Mock LLM，不依赖真实 API。
"""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.documents import Document

from agent.verifier import RetrievalVerifier, VerificationResult


# ============================================================
# Helpers
# ============================================================


def _docs(*contents: str) -> list[Document]:
    return [Document(page_content=c) for c in contents]


def _mock_llm(json_response: str) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value.content = json_response
    return llm


# ============================================================
# VerificationResult
# ============================================================


def test_verification_result_defaults():
    """默认值正确。"""
    r = VerificationResult()
    assert r.passed is False
    assert r.score == 0.0
    assert r.reason == ""
    assert r.missing_topics == []


def test_verification_result_passed():
    """通过验证的结果。"""
    r = VerificationResult(
        passed=True, score=0.9, reason="文档充分覆盖问题。",
        missing_topics=[],
    )
    assert r.passed is True
    assert r.score == 0.9


# ============================================================
# RetrievalVerifier
# ============================================================


def test_verifier_passes_relevant_docs():
    """相关文档 → 验证通过。"""
    llm = _mock_llm('{"passed": true, "score": 0.9, "reason": "充分覆盖", "missing_topics": []}')
    verifier = RetrievalVerifier(llm)

    result = verifier.verify(
        question="Python 是什么？",
        retrieved_docs=_docs("Python 是一种编程语言。"),
    )

    assert result.passed is True
    assert result.score == 0.9
    assert result.reason == "充分覆盖"


def test_verifier_fails_empty_docs():
    """空文档 → 验证失败。"""
    llm = _mock_llm("{}")
    verifier = RetrievalVerifier(llm)

    result = verifier.verify(
        question="Python 是什么？",
        retrieved_docs=[],
    )

    assert result.passed is False
    assert result.score == 0.0
    assert "未检索到" in result.reason


def test_verifier_fails_low_relevance():
    """低相关性 → 验证不通过。"""
    llm = _mock_llm('{"passed": false, "score": 0.2, "reason": "文档不相关", "missing_topics": ["Python 定义"]}')
    verifier = RetrievalVerifier(llm)

    result = verifier.verify(
        question="Python 是什么？",
        retrieved_docs=_docs("Java 是一种编程语言。"),
    )

    assert result.passed is False
    assert result.score == 0.2
    assert "Python 定义" in result.missing_topics


def test_verifier_with_draft_answer():
    """传入 draft_answer → prompt 包含。"""
    llm = _mock_llm('{"passed": true, "score": 0.8, "reason": "OK", "missing_topics": []}')
    verifier = RetrievalVerifier(llm)

    result = verifier.verify(
        question="Docker 怎么用？",
        retrieved_docs=_docs("Docker 是一个容器平台。"),
        draft_answer="Docker 用于容器化部署。",
    )

    assert result.passed is True
    # 验证 prompt 包含 draft_answer
    call_args = llm.invoke.call_args[0][0]
    user_msg = call_args[1]["content"]
    assert "Docker 用于容器化部署" in user_msg


def test_verifier_handles_llm_error():
    """LLM 调用异常 → 默认放行（不阻塞检索）。"""
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("API timeout")
    verifier = RetrievalVerifier(llm)

    result = verifier.verify(
        question="test",
        retrieved_docs=_docs("content"),
    )

    assert result.passed is True  # 默认放行
    assert result.score == 0.5
    assert "默认放行" in result.reason


def test_verifier_handles_malformed_json():
    """LLM 返回非法 JSON → 容错处理，默认放行。"""
    llm = _mock_llm("not valid json at all")
    verifier = RetrievalVerifier(llm)

    result = verifier.verify(
        question="test",
        retrieved_docs=_docs("content"),
    )

    assert result.passed is True
    assert result.score == 0.5


def test_verifier_handles_markdown_json_block():
    """LLM 返回 ```json ... ``` 代码块 → 正确解析。"""
    llm = _mock_llm('```json\n{"passed": true, "score": 0.95, "reason": "完美覆盖", "missing_topics": []}\n```')
    verifier = RetrievalVerifier(llm)

    result = verifier.verify(
        question="test",
        retrieved_docs=_docs("content"),
    )

    assert result.passed is True
    assert result.score == 0.95


def test_verifier_di_requires_llm():
    """RetrievalVerifier 必须通过 DI 传入 llm。"""
    llm = _mock_llm("{}")
    verifier = RetrievalVerifier(llm)
    assert verifier._llm is llm


def test_verifier_does_not_access_guard_router_registry():
    """Verifier 不导入任何 Knowledge Layer 模块。"""
    import ast
    with open("agent/verifier/verifier.py", "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "knowledge" not in node.module, f"Verifier imports {node.module}"
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "knowledge" not in alias.name, f"Verifier imports {alias.name}"
