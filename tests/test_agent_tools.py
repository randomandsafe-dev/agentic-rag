"""Agent Retrieval Tools 测试。

验证 list_knowledge_bases / verify_retrieval_result 行为。
全部使用 Mock，不依赖真实 LLM / Chroma。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from knowledge.registry import KnowledgeBaseRegistry


# ============================================================
# list_knowledge_bases
# ============================================================


def _mock_service_for_tools(domains=None):
    """Mock KnowledgeService 的 list_domains + search。"""
    mock_llm = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []

    patches = [
        patch("knowledge.service.create_llm", return_value=mock_llm),
        patch.object(KnowledgeBaseRegistry, "get_retriever", return_value=mock_retriever),
    ]
    for p in patches:
        p.start()
    return patches


def _stop(patches):
    for p in reversed(patches):
        p.stop()


def test_list_knowledge_bases_returns_domains():
    """list_knowledge_bases 返回当前可用 KB 列表。"""
    patches = _mock_service_for_tools()
    try:
        from knowledge.service import get_knowledge_service

        svc = get_knowledge_service()
        svc.list_domains = MagicMock(return_value=[
            {"id": "default", "name": "Default KB", "description": "通用知识库", "default": True},
            {"id": "tech_docs", "name": "Tech Docs", "description": "技术文档", "default": False},
        ])

        from agent.tools.knowledge_tools import list_knowledge_bases
        result = list_knowledge_bases.invoke({"dummy": ""})

        assert "Default KB" in result
        assert "Tech Docs" in result
        assert "通用知识库" in result
        assert "技术文档" in result
    finally:
        _stop(patches)


def test_list_knowledge_bases_empty():
    """无可用 KB 时返回提示信息。"""
    patches = _mock_service_for_tools()
    try:
        from knowledge.service import get_knowledge_service

        svc = get_knowledge_service()
        svc.list_domains = MagicMock(return_value=[])

        from agent.tools.knowledge_tools import list_knowledge_bases
        result = list_knowledge_bases.invoke({"dummy": ""})

        assert "没有可用的知识库" in result
    finally:
        _stop(patches)


def test_list_knowledge_bases_passes_user():
    """list_knowledge_bases 将 _current_user 传递给 KnowledgeService。"""
    patches = _mock_service_for_tools()
    try:
        from knowledge.service import get_knowledge_service
        from knowledge.access import UserContext
        from rag_agent import set_agent_user

        svc = get_knowledge_service()
        mock_list = MagicMock(return_value=[])
        svc.list_domains = mock_list

        admin = UserContext(user_id="admin", role="admin")
        set_agent_user(admin)

        from agent.tools.knowledge_tools import list_knowledge_bases
        list_knowledge_bases.invoke({"dummy": ""})

        mock_list.assert_called_once_with(user=admin)
        set_agent_user(None)
    finally:
        _stop(patches)


# ============================================================
# verify_retrieval_result
# ============================================================


def test_verify_with_sources_passes():
    """有有效来源时验证通过。"""
    from agent.tools.knowledge_tools import verify_retrieval_result
    result = verify_retrieval_result.invoke({
        "verification_input": "回答: Python 是一种编程语言 | 来源: [来源 1: docs/python.md]"
    })
    assert "验证通过" in result


def test_verify_without_sources_fails():
    """无来源时验证失败。"""
    from agent.tools.knowledge_tools import verify_retrieval_result
    result = verify_retrieval_result.invoke({
        "verification_input": "回答: Python 是一种编程语言 | 来源: "
    })
    assert "验证失败" in result


def test_verify_with_default_message_fails():
    """来源为默认"未检索到相关内容"时验证失败。"""
    from agent.tools.knowledge_tools import verify_retrieval_result
    result = verify_retrieval_result.invoke({
        "verification_input": "回答: 不知道 | 来源: 未检索到相关内容。"
    })
    assert "验证失败" in result


def test_verify_weak_sources_warns():
    """来源信息不完整时警告。"""
    from agent.tools.knowledge_tools import verify_retrieval_result
    result = verify_retrieval_result.invoke({
        "verification_input": "回答: test | 来源: 未知来源"
    })
    assert "警告" in result or "失败" in result


# ============================================================
# search_knowledge_base (existing tool, verify unchanged)
# ============================================================


def test_search_knowledge_base_still_works():
    """现有 search_knowledge_base tool 行为不变。"""
    patches = _mock_service_for_tools()
    try:
        from rag_agent import search_knowledge_base, set_agent_user
        set_agent_user(None)

        result = search_knowledge_base.invoke({"query": "test"})
        assert isinstance(result, str)
    finally:
        _stop(patches)


def test_search_knowledge_base_passes_user():
    """search_knowledge_base 正确传递 UserContext。"""
    patches = _mock_service_for_tools()
    try:
        from knowledge.access import UserContext
        from knowledge.service import get_knowledge_service
        from rag_agent import search_knowledge_base, set_agent_user

        svc = get_knowledge_service()
        mock_search = MagicMock(return_value=[])
        svc.search = mock_search

        admin = UserContext(user_id="admin", role="admin")
        set_agent_user(admin)

        search_knowledge_base.invoke({"query": "test"})

        mock_search.assert_called_once_with("test", user=admin)
        set_agent_user(None)
    finally:
        _stop(patches)
