"""KnowledgeService + AccessGuard 集成测试。

验证 search() / list_domains() 的权限过滤行为。
全部使用 Mock Retriever，不依赖真实 Chroma / LLM。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from knowledge.access import AccessGuard, AccessPolicy, UserContext
from knowledge.domain import KnowledgeDomain
from knowledge.registry import KnowledgeBaseRegistry
from knowledge.router import RoutingDecision


# ============================================================
# Helpers
# ============================================================


def _make_domain(
    domain_id: str,
    name: str = "",
    default: bool = False,
) -> KnowledgeDomain:
    return KnowledgeDomain(
        id=domain_id,
        name=name or domain_id,
        description="",
        data_dir=Path("."),
        persist_dir=Path("."),
        collection_name=f"kb_{domain_id}",
        default=default,
    )


def _policy_yaml(content: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


def _mock_service(policy_path: Path | None = None):
    """创建 KnowledgeService，Mock 掉 LLM + Retriever 以绕过 API key 检查。"""
    mock_llm = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = []

    with patch("knowledge.service.create_llm", return_value=mock_llm):
        from knowledge.service import KnowledgeService
        svc = KnowledgeService()

    # 实例级 patch：替换 registry 的 get_retriever 避免真实 Chroma 调用
    svc._registry.get_retriever = MagicMock(return_value=mock_retriever)

    if policy_path is not None:
        from knowledge.access import AccessPolicy
        svc._access_guard = AccessGuard(AccessPolicy(policy_path))

    return svc


# ============================================================
# search() — user=None (backward compatibility)
# ============================================================


def test_search_without_user_unchanged():
    """search(query) 不传 user → 行为与 Phase 2.5 完全一致。"""
    svc = _mock_service()
    docs = svc.search("test query")
    assert isinstance(docs, list)


def test_search_with_none_user_unchanged():
    """search(query, user=None) 等价于 search(query)。"""
    svc = _mock_service()
    docs = svc.search("test query", user=None)
    assert isinstance(docs, list)


# ============================================================
# search() — admin (full access)
# ============================================================


def test_admin_searches_all_domains():
    """admin 用户 → Router 看到全部 domains。"""
    policy_path = _policy_yaml("""
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
        svc = _mock_service(policy_path)
        admin = UserContext(user_id="admin", role="admin")

        # 验证 Router 收到全部 domains
        with patch.object(svc._router, "route", wraps=svc._router.route) as spy:
            svc.search("test", user=admin)
            domains_passed_to_router = spy.call_args[0][1]
            assert len(domains_passed_to_router) == len(svc._registry.list_domains())
    finally:
        Path(policy_path).unlink()


# ============================================================
# search() — restricted user
# ============================================================


def test_restricted_user_only_allowed_domains():
    """受限用户 → Router 只看到授权的 domains。"""
    policy_path = _policy_yaml("""
users:
  restricted:
    role: viewer
    allowed_kbs:
      - default
default:
  role: viewer
  allowed_kbs: []
""")
    try:
        svc = _mock_service(policy_path)
        restricted = UserContext(user_id="restricted", role="viewer")

        with patch.object(svc._router, "route", wraps=svc._router.route) as spy:
            svc.search("test", user=restricted)
            domains_passed = spy.call_args[0][1]
            assert len(domains_passed) == 1
            assert domains_passed[0].id == "default"
    finally:
        Path(policy_path).unlink()


# ============================================================
# list_domains()
# ============================================================


def test_list_domains_without_user_returns_all():
    """list_domains() 不传 user → 返回全部。"""
    svc = _mock_service()
    domains = svc.list_domains()
    assert len(domains) >= 1
    assert domains[0]["id"] == "default"


def test_list_domains_with_admin_returns_all():
    """list_domains(user=admin) → 返回全部。"""
    policy_path = _policy_yaml("""
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
        svc = _mock_service(policy_path)
        admin = UserContext(user_id="admin", role="admin")
        domains = svc.list_domains(user=admin)
        assert len(domains) >= 1
    finally:
        Path(policy_path).unlink()


def test_list_domains_with_restricted_user_returns_filtered():
    """list_domains(user=restricted) → 仅返回授权 domains。"""
    policy_path = _policy_yaml("""
users:
  restricted:
    role: viewer
    allowed_kbs:
      - default
default:
  role: viewer
  allowed_kbs: []
""")
    try:
        svc = _mock_service(policy_path)
        restricted = UserContext(user_id="restricted", role="viewer")
        domains = svc.list_domains(user=restricted)
        assert len(domains) == 1
        assert domains[0]["id"] == "default"
    finally:
        Path(policy_path).unlink()


# ============================================================
# Edge cases
# ============================================================


def test_user_denied_all_falls_back_to_default():
    """全部 domain 被拒绝 → search() 仍正常工作（安全网返回 default）。"""
    policy_path = _policy_yaml("""
users:
  nobody:
    role: viewer
    allowed_kbs: []
default:
  role: viewer
  allowed_kbs: []
""")
    try:
        svc = _mock_service(policy_path)
        nobody = UserContext(user_id="nobody", role="viewer")

        with patch.object(svc._router, "route", wraps=svc._router.route) as spy:
            docs = svc.search("test", user=nobody)
            domains_passed = spy.call_args[0][1]
            # 安全网保证至少返回 default domain
            assert len(domains_passed) == 1
            assert domains_passed[0].default is True
    finally:
        Path(policy_path).unlink()
