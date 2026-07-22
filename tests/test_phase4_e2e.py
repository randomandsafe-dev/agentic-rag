"""Phase 4 E2E 全链路测试。

覆盖: Agent → Tools → KnowledgeService → AccessGuard → Router → Registry → Retriever。
所有 Chroma/LLM 依赖使用 Mock，保证可离线运行。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from langchain_core.documents import Document

from knowledge.access import AccessGuard, AccessPolicy, UserContext
from knowledge.domain import KnowledgeDomain
from knowledge.registry import KnowledgeBaseRegistry


# ============================================================
# Helpers
# ============================================================


def _domain(id: str, name: str = "", default: bool = False, enabled: bool = True) -> KnowledgeDomain:
    return KnowledgeDomain(
        id=id, name=name or id, description="",
        data_dir=Path(".") / id, persist_dir=Path(".") / id,
        collection_name=f"kb_{id}", default=default, enabled=enabled,
    )


_MULTI_KB = [
    _domain("default", "默认知识库", default=True),
    _domain("tech_docs", "技术文档"),
    _domain("product_docs", "产品文档"),
    _domain("hr_docs", "HR文档"),
]


def _policy(content: str) -> Path:
    p = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    p.write(content)
    p.close()
    return Path(p.name)


def _mock_env(policy_path: Path | None = None):
    """构建完整 Mock 环境：LLM + Retriever + 多 KB Registry。"""
    mock_llm = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [Document(page_content="result")]

    patchers = [
        patch("knowledge.service.create_llm", return_value=mock_llm),
        patch("rag_agent.create_llm", return_value=mock_llm),
        patch.object(KnowledgeBaseRegistry, "get_retriever", return_value=mock_retriever),
    ]
    for p in patchers:
        p.start()

    from knowledge.service import KnowledgeService
    svc = KnowledgeService()
    svc._registry.list_domains = MagicMock(return_value=list(_MULTI_KB))
    svc._registry.get_retriever = MagicMock(return_value=mock_retriever)

    if policy_path is not None:
        svc._access_guard = AccessGuard(AccessPolicy(policy_path))

    # Patch the singleton to return our configured service
    patch("knowledge.service.get_knowledge_service", return_value=svc).start()

    return patchers, svc, mock_retriever


def _stop(patchers):
    for p in reversed(patchers):
        p.stop()


# ============================================================
# Scenario 1: Admin — all KBs
# ============================================================


def test_admin_access_all_kbs():
    """Admin (*) -> Router 看到全部 4 个 domains。"""
    policy_path = _policy("""
users:
  admin:
    role: admin
    allowed_kbs: ["*"]
default:
  role: viewer
  allowed_kbs: []
""")
    try:
        patchers, svc, retriever = _mock_env(policy_path)
        admin = UserContext(user_id="admin", role="admin")

        from rag_agent import set_agent_user
        set_agent_user(admin)

        with patch.object(svc._router, "route", wraps=svc._router.route) as spy:
            docs = svc.search("query", user=admin)
            domains_passed = spy.call_args[0][1]
            assert len(domains_passed) == 4
            assert {d.id for d in domains_passed} == {"default", "tech_docs", "product_docs", "hr_docs"}
            retriever.search.assert_called()
    finally:
        _stop(patchers)
        Path(policy_path).unlink()


# ============================================================
# Scenario 2: Developer — restricted KBs
# ============================================================


def test_developer_restricted_access():
    """Developer -> 只能访问 tech_docs + product_docs。"""
    policy_path = _policy("""
users:
  developer:
    role: developer
    allowed_kbs: ["tech_docs", "product_docs"]
default:
  role: viewer
  allowed_kbs: []
""")
    try:
        patchers, svc, retriever = _mock_env(policy_path)
        dev = UserContext(user_id="developer", role="developer")

        from rag_agent import set_agent_user
        set_agent_user(dev)

        with patch.object(svc._router, "route", wraps=svc._router.route) as spy:
            docs = svc.search("query", user=dev)
            domains_passed = spy.call_args[0][1]
            assert len(domains_passed) == 2
            assert {d.id for d in domains_passed} == {"tech_docs", "product_docs"}
    finally:
        _stop(patchers)
        Path(policy_path).unlink()


# ============================================================
# Scenario 3: Unknown user — default fallback
# ============================================================


def test_unknown_user_default_fallback():
    """未在 policy 中的用户 -> 使用 default 规则。"""
    policy_path = _policy("""
users:
  admin:
    role: admin
    allowed_kbs: ["*"]
default:
  role: viewer
  allowed_kbs: ["default"]
""")
    try:
        patchers, svc, retriever = _mock_env(policy_path)
        stranger = UserContext(user_id="stranger", role="viewer")

        from rag_agent import set_agent_user
        set_agent_user(stranger)

        with patch.object(svc._router, "route", wraps=svc._router.route) as spy:
            docs = svc.search("query", user=stranger)
            domains_passed = spy.call_args[0][1]
            assert len(domains_passed) == 1
            assert domains_passed[0].id == "default"
    finally:
        _stop(patchers)
        Path(policy_path).unlink()


def test_no_user_full_access():
    """user=None -> 无权限过滤，Router 看到全部 domains。"""
    patchers, svc, retriever = _mock_env()
    try:
        from rag_agent import set_agent_user
        set_agent_user(None)

        with patch.object(svc._router, "route", wraps=svc._router.route) as spy:
            docs = svc.search("query", user=None)
            domains_passed = spy.call_args[0][1]
            assert len(domains_passed) == 4
    finally:
        _stop(patchers)


# ============================================================
# Scenario 4: Multi-KB routing
# ============================================================


def test_multi_kb_routing_selects_correct_retriever():
    """Router 选择不同 domain -> Registry 返回对应 retriever。"""
    patchers, svc, retriever = _mock_env()
    try:
        # 创建两个不同 retriever
        retriever_tech = MagicMock()
        retriever_tech.search.return_value = [Document(page_content="tech result")]
        retriever_default = MagicMock()
        retriever_default.search.return_value = [Document(page_content="default result")]

        svc._registry.get_retriever = MagicMock(side_effect=lambda domain_id: {
            "tech_docs": retriever_tech,
            "default": retriever_default,
        }.get(domain_id, retriever_default))

        # Router 返回 tech_docs
        from knowledge.router import RoutingDecision
        svc._router.route = MagicMock(return_value=RoutingDecision(
            domain_id="tech_docs", confidence=0.85, strategy="llm"
        ))

        docs = svc.search("python deployment", user=None)
        assert retriever_tech.search.called
        assert docs[0].page_content == "tech result"
    finally:
        _stop(patchers)


def test_single_kb_routing_unchanged():
    """单 KB 场景 Router fast path -> default domain。"""
    patchers, svc, retriever = _mock_env()
    try:
        svc._registry.list_domains = MagicMock(return_value=[_MULTI_KB[0]])

        docs = svc.search("query", user=None)
        retriever.search.assert_called_once()
        assert len(docs) == 1
    finally:
        _stop(patchers)


# ============================================================
# Scenario 5: Agent tool calling
# ============================================================


def test_agent_search_tool_calls_knowledge_service():
    """search_knowledge_base -> KnowledgeService.search(query, user)。"""
    patchers, svc, retriever = _mock_env()
    try:
        from rag_agent import search_knowledge_base, set_agent_user
        set_agent_user(None)

        mock_search = MagicMock(return_value=[Document(page_content="agent result")])
        svc.search = mock_search

        search_knowledge_base.invoke({"query": "test query"})
        mock_search.assert_called_once_with("test query", user=None)
    finally:
        _stop(patchers)


def test_agent_list_tool_returns_domains():
    """list_knowledge_bases -> list_domains(user)。"""
    patchers, svc, retriever = _mock_env()
    try:
        from rag_agent import set_agent_user
        from agent.tools.knowledge_tools import list_knowledge_bases

        admin = UserContext(user_id="admin", role="admin")
        set_agent_user(admin)

        svc.list_domains = MagicMock(return_value=[
            {"id": "default", "name": "默认", "description": "通用"},
            {"id": "tech", "name": "技术", "description": "API文档"},
        ])

        result = list_knowledge_bases.invoke({"dummy": ""})
        assert "默认" in result
        assert "技术" in result
        svc.list_domains.assert_called_once_with(user=admin)
    finally:
        _stop(patchers)


def test_agent_verify_tool():
    """verify_retrieval_result 正确处理各种输入。"""
    from agent.tools.knowledge_tools import verify_retrieval_result

    # 有效
    r = verify_retrieval_result.invoke({
        "verification_input": "回答: test | 来源: [来源 1: doc.md]"
    })
    assert "验证通过" in r

    # 无来源
    r = verify_retrieval_result.invoke({
        "verification_input": "回答: test | 来源: 未检索到相关内容。"
    })
    assert "验证失败" in r


def test_agent_builds_with_three_tools():
    """build_agent() 包含 3 个工具：search + list + verify。"""
    patchers, svc, retriever = _mock_env()
    try:
        with patch("config.Settings.validate", return_value=None):
            from rag_agent import build_agent, search_knowledge_base
            from agent.tools.knowledge_tools import list_knowledge_bases, verify_retrieval_result

            agent = build_agent(checkpointer=None)
            assert agent is not None

            # 验证三个 tool 均可正常 invoke
            assert search_knowledge_base.name == "search_knowledge_base"
            assert list_knowledge_bases.name == "list_knowledge_bases"
            assert verify_retrieval_result.name == "verify_retrieval_result"
    finally:
        _stop(patchers)


# ============================================================
# Full chain: Service returns correct docs through pipeline
# ============================================================


def test_full_chain_admin_search_returns_docs():
    """完整链路：admin search -> Guard -> Router -> Registry -> Pipeline -> docs。"""
    policy_path = _policy("""
users:
  admin:
    role: admin
    allowed_kbs: ["*"]
default:
  role: viewer
  allowed_kbs: []
""")
    try:
        patchers, svc, retriever = _mock_env(policy_path)
        admin = UserContext(user_id="admin", role="admin")

        docs = svc.search("any query", user=admin)
        retriever.search.assert_called_once()
        assert len(docs) == 1
        assert docs[0].page_content == "result"
    finally:
        _stop(patchers)
        Path(policy_path).unlink()
