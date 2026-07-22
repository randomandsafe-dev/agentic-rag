"""Agent 层 UserContext 透传测试。

验证 search_knowledge_base 正确将 user 传递给 KnowledgeService。
全部使用 Mock，不依赖真实 LLM / Chroma。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from knowledge.access import UserContext
from knowledge.registry import KnowledgeBaseRegistry


# ============================================================
# Helpers
# ============================================================


def _mock_agent_env():
    """Mock KnowledgeService 的 search 方法，返回固定文档。"""
    mock_llm = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [Document(page_content="test doc")]

    patches = [
        patch("knowledge.service.create_llm", return_value=mock_llm),
        patch.object(KnowledgeBaseRegistry, "get_retriever", return_value=mock_retriever),
    ]
    for p in patches:
        p.start()

    return patches


def _stop_patches(patches):
    for p in patches:
        p.stop()


# ============================================================
# Tests
# ============================================================


def test_agent_passes_user_to_service():
    """Agent 调用 search_knowledge_base 时，user 参数正确传递给 KnowledgeService。"""
    patches = _mock_agent_env()
    try:
        from rag_agent import search_knowledge_base, set_agent_user

        admin = UserContext(user_id="admin", role="admin")
        set_agent_user(admin)

        with patch("knowledge.service.KnowledgeService.search") as mock_search:
            mock_search.return_value = [Document(page_content="admin doc")]
            search_knowledge_base.invoke({"query": "test"})
            mock_search.assert_called_once()
            _, kwargs = mock_search.call_args
            assert kwargs.get("user") is admin
    finally:
        _stop_patches(patches)


def test_agent_passes_none_user_when_not_set():
    """未调用 set_agent_user → _current_user=None → 传递给 service 的 user=None。"""
    patches = _mock_agent_env()
    try:
        from rag_agent import search_knowledge_base, set_agent_user

        set_agent_user(None)

        with patch("knowledge.service.KnowledgeService.search") as mock_search:
            mock_search.return_value = [Document(page_content="public doc")]
            search_knowledge_base.invoke({"query": "test"})
            mock_search.assert_called_once()
            _, kwargs = mock_search.call_args
            assert kwargs.get("user") is None
    finally:
        _stop_patches(patches)


def test_agent_without_set_agent_user_still_works():
    """模块加载后未调用 set_agent_user → _current_user 初始为 None → 正常检索。"""
    patches = _mock_agent_env()
    try:
        # 重新导入以获取初始状态
        import importlib
        import rag_agent

        importlib.reload(rag_agent)

        with patch("knowledge.service.KnowledgeService.search") as mock_search:
            mock_search.return_value = [Document(page_content="doc")]
            rag_agent.search_knowledge_base.invoke({"query": "test"})
            mock_search.assert_called_once()
            _, kwargs = mock_search.call_args
            assert kwargs.get("user") is None
    finally:
        _stop_patches(patches)


def test_set_agent_user_updates_global():
    """set_agent_user 正确更新模块级 _current_user。"""
    from rag_agent import set_agent_user

    viewer = UserContext(user_id="viewer", role="viewer")
    set_agent_user(viewer)

    import rag_agent
    assert rag_agent._current_user is viewer

    set_agent_user(None)
    assert rag_agent._current_user is None


def test_multiple_users_sequential():
    """连续切换用户 → 每次检索使用当前用户。"""
    patches = _mock_agent_env()
    try:
        from rag_agent import search_knowledge_base, set_agent_user

        with patch("knowledge.service.KnowledgeService.search") as mock_search:
            mock_search.return_value = [Document(page_content="doc")]

            # 用户 A
            alice = UserContext(user_id="alice", role="admin")
            set_agent_user(alice)
            search_knowledge_base.invoke({"query": "q"})
            assert mock_search.call_args[1]["user"] is alice

            # 用户 B
            bob = UserContext(user_id="bob", role="viewer")
            set_agent_user(bob)
            search_knowledge_base.invoke({"query": "q"})
            assert mock_search.call_args[1]["user"] is bob

            # 清除
            set_agent_user(None)
            search_knowledge_base.invoke({"query": "q"})
            assert mock_search.call_args[1]["user"] is None
    finally:
        _stop_patches(patches)
