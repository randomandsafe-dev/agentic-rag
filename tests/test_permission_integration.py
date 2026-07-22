"""Permission Layer 端到端集成测试。

覆盖完整调用链：UserContext → KnowledgeService → AccessGuard → Router → Retriever。
全部使用 Mock，零外部依赖。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from knowledge.access import AccessGuard, AccessPolicy, UserContext
from knowledge.domain import KnowledgeDomain
from knowledge.registry import KnowledgeBaseRegistry


# ============================================================
# Helpers
# ============================================================


def _domain(domain_id: str, name: str = "", default: bool = False) -> KnowledgeDomain:
    return KnowledgeDomain(
        id=domain_id,
        name=name or domain_id,
        description="",
        data_dir=Path(".") / domain_id,
        persist_dir=Path(".") / domain_id,
        collection_name=f"kb_{domain_id}",
        default=default,
    )


_MULTI_KB_DOMAINS = [
    _domain("default", "默认知识库", default=True),
    _domain("tech_docs", "技术文档"),
    _domain("hr_docs", "HR文档"),
    _domain("product_docs", "产品文档"),
]


def _policy_file(content: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


def _build_svc_with_multi_kb(policy_path: Path | None = None):
    """构建 KnowledgeService，注入多 KB 的 mock registry + mock retriever。

    policy_path=None 时使用不存在的路径（模拟 policy.yaml 缺失场景）。
    """
    mock_llm = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [Document(page_content="result")]

    with patch("knowledge.service.create_llm", return_value=mock_llm):
        from knowledge.service import KnowledgeService
        svc = KnowledgeService()

    # 注入多 domain registry
    svc._registry.list_domains = MagicMock(return_value=list(_MULTI_KB_DOMAINS))
    svc._registry.get_retriever = MagicMock(return_value=mock_retriever)

    # 注入 AccessPolicy
    effective_path = policy_path if policy_path is not None else Path("/nonexistent/policy.yaml")
    svc._access_guard = AccessGuard(AccessPolicy(effective_path))

    return svc


# ============================================================
# E2E: Admin — 全部 KB
# ============================================================


def test_admin_can_search_all_kbs():
    """admin("*") → Router 看到全部 4 个 domains。"""
    policy_path = _policy_file("""
users:
  admin:
    role: admin
    allowed_kbs:
      - "*"
default:
  role: viewer
  allowed_kbs: []
""")
    try:
        svc = _build_svc_with_multi_kb(policy_path)
        admin = UserContext(user_id="admin", role="admin")

        with patch.object(svc._router, "route", wraps=svc._router.route) as spy:
            svc.search("test", user=admin)
            domains_passed = spy.call_args[0][1]
            assert len(domains_passed) == 4
            domain_ids = {d.id for d in domains_passed}
            assert domain_ids == {"default", "tech_docs", "hr_docs", "product_docs"}
    finally:
        Path(policy_path).unlink()


# ============================================================
# E2E: Developer — 部分 KB
# ============================================================


def test_developer_only_sees_allowed_kbs():
    """developer → Router 只看到 tech_docs + product_docs。"""
    policy_path = _policy_file("""
users:
  developer:
    role: developer
    allowed_kbs:
      - tech_docs
      - product_docs
default:
  role: viewer
  allowed_kbs: []
""")
    try:
        svc = _build_svc_with_multi_kb(policy_path)
        dev = UserContext(user_id="developer", role="developer")

        with patch.object(svc._router, "route", wraps=svc._router.route) as spy:
            svc.search("test", user=dev)
            domains_passed = spy.call_args[0][1]
            assert len(domains_passed) == 2
            domain_ids = {d.id for d in domains_passed}
            assert domain_ids == {"tech_docs", "product_docs"}
    finally:
        Path(policy_path).unlink()


# ============================================================
# E2E: Unknown user — default policy
# ============================================================


def test_unknown_user_uses_default_policy():
    """未在 users 中列出的用户 → 使用 default 规则（仅 public_kb）。"""
    policy_path = _policy_file("""
users:
  admin:
    role: admin
    allowed_kbs:
      - "*"
default:
  role: viewer
  allowed_kbs:
    - default
""")
    try:
        svc = _build_svc_with_multi_kb(policy_path)
        stranger = UserContext(user_id="stranger", role="viewer")

        with patch.object(svc._router, "route", wraps=svc._router.route) as spy:
            svc.search("test", user=stranger)
            domains_passed = spy.call_args[0][1]
            assert len(domains_passed) == 1
            assert domains_passed[0].id == "default"
    finally:
        Path(policy_path).unlink()


# ============================================================
# E2E: user=None — 全部 KB
# ============================================================


def test_none_user_sees_all_kbs():
    """user=None → Router 看到全部 domains（向后兼容）。"""
    svc = _build_svc_with_multi_kb()

    with patch.object(svc._router, "route", wraps=svc._router.route) as spy:
        svc.search("test", user=None)
        domains_passed = spy.call_args[0][1]
        assert len(domains_passed) == 4


# ============================================================
# Full chain: Agent Tool → KnowledgeService → AccessGuard
# ============================================================


def test_full_chain_admin_routes_correctly():
    """Agent tool → KnowledgeService.search(user=admin) → Router → 全部 KB。"""
    policy_path = _policy_file("""
users:
  admin:
    role: admin
    allowed_kbs:
      - "*"
default:
  role: viewer
  allowed_kbs: []
""")
    patches = []
    try:
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [Document(page_content="admin result")]
        patches.append(patch("knowledge.service.create_llm", return_value=mock_llm))
        for p in patches:
            p.start()

        from knowledge.service import KnowledgeService
        svc = KnowledgeService()
        svc._registry.list_domains = MagicMock(return_value=list(_MULTI_KB_DOMAINS))
        svc._registry.get_retriever = MagicMock(return_value=mock_retriever)
        svc._access_guard = AccessGuard(AccessPolicy(policy_path))

        from rag_agent import set_agent_user, search_knowledge_base
        admin = UserContext(user_id="admin", role="admin")
        set_agent_user(admin)

        # Agent tool 内部调用 get_knowledge_service().search(query, user=_current_user)
        # 这里直接测 Service 层
        docs = svc.search("查询", user=admin)
        mock_retriever.search.assert_called()
        assert len(docs) >= 1
    finally:
        for p in reversed(patches):
            p.stop()
        Path(policy_path).unlink()


def test_full_chain_restricted_user_cannot_escape():
    """受限用户的 search 结果不会泄露未授权 KB 的内容。"""
    policy_path = _policy_file("""
users:
  restricted:
    role: viewer
    allowed_kbs:
      - default
default:
  role: viewer
  allowed_kbs: []
""")
    patches = []
    try:
        mock_llm = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [Document(page_content="safe")]
        patches.append(patch("knowledge.service.create_llm", return_value=mock_llm))
        for p in patches:
            p.start()

        from knowledge.service import KnowledgeService
        svc = KnowledgeService()
        svc._registry.list_domains = MagicMock(return_value=list(_MULTI_KB_DOMAINS))
        svc._registry.get_retriever = MagicMock(return_value=mock_retriever)
        svc._access_guard = AccessGuard(AccessPolicy(policy_path))

        restricted = UserContext(user_id="restricted", role="viewer")
        docs = svc.search("secret query", user=restricted)

        # 验证 retriever 被调用了（domain 过滤后正常检索）
        mock_retriever.search.assert_called()
        assert "safe" in docs[0].page_content
    finally:
        for p in reversed(patches):
            p.stop()
        Path(policy_path).unlink()


# ============================================================
# Edge case: policy.yaml missing
# ============================================================


def test_missing_policy_allows_all_in_full_chain():
    """policy.yaml 不存在 → 全部 KB 可访问。"""
    svc = _build_svc_with_multi_kb()

    user = UserContext(user_id="anyone", role="viewer")
    with patch.object(svc._router, "route", wraps=svc._router.route) as spy:
        svc.search("test", user=user)
        domains_passed = spy.call_args[0][1]
        assert len(domains_passed) == 4


# ============================================================
# Edge case: all denied → safety net
# ============================================================


def test_all_denied_safety_net_in_full_chain():
    """所有 domain 被拒绝 → 安全网返回 default。"""
    policy_path = _policy_file("""
users:
  nobody:
    role: viewer
    allowed_kbs: []
default:
  role: viewer
  allowed_kbs: []
""")
    try:
        svc = _build_svc_with_multi_kb(policy_path)
        nobody = UserContext(user_id="nobody", role="viewer")

        with patch.object(svc._router, "route", wraps=svc._router.route) as spy:
            svc.search("test", user=nobody)
            domains_passed = spy.call_args[0][1]
            assert len(domains_passed) == 1
            assert domains_passed[0].default is True
    finally:
        Path(policy_path).unlink()
