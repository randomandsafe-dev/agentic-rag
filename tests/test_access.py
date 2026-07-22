"""Permission Layer 单元测试。

全部使用 Mock / 临时文件，不依赖真实 Chroma / LLM。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from knowledge.access import AccessGuard, AccessPolicy, UserContext
from knowledge.domain import KnowledgeDomain


# ============================================================
# Helpers
# ============================================================


def _make_domain(
    domain_id: str,
    name: str = "",
    description: str = "",
    default: bool = False,
) -> KnowledgeDomain:
    return KnowledgeDomain(
        id=domain_id,
        name=name or domain_id,
        description=description,
        data_dir=Path("."),
        persist_dir=Path("."),
        collection_name=f"kb_{domain_id}",
        default=default,
    )


def _make_policy_yaml(content: str) -> Path:
    """创建临时 policy.yaml 文件并返回路径。"""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


# ============================================================
# UserContext
# ============================================================


def test_user_context_creation():
    """UserContext 默认值和自定义字段。"""
    u = UserContext(user_id="alice")
    assert u.user_id == "alice"
    assert u.role == "viewer"
    assert u.metadata == {}

    u2 = UserContext(user_id="bob", role="admin", metadata={"tenant": "acme"})
    assert u2.user_id == "bob"
    assert u2.role == "admin"
    assert u2.metadata["tenant"] == "acme"

    # 可变默认值安全
    u3 = UserContext(user_id="carol")
    u3.metadata["key"] = "val"
    u4 = UserContext(user_id="dave")
    assert u4.metadata == {}


# ============================================================
# AccessPolicy
# ============================================================


def test_admin_access_all():
    """admin 用户 allowed_kbs=["*"] → 可访问任意 domain。"""
    policy_yaml = """
users:
  admin:
    role: admin
    allowed_kbs:
      - "*"
default:
  role: viewer
  allowed_kbs: []
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        policy = AccessPolicy(path)
        admin = UserContext(user_id="admin", role="admin")
        assert policy.is_allowed(admin, "any_domain") is True
        assert policy.is_allowed(admin, "tech_docs") is True
        assert policy.is_allowed(admin, "") is True  # wildcard means any
    finally:
        Path(path).unlink()


def test_user_allowed_domain():
    """developer 可访问 allowed_kbs 中的 domain。"""
    policy_yaml = """
users:
  developer:
    role: developer
    allowed_kbs:
      - tech_docs
      - api_docs
default:
  role: viewer
  allowed_kbs: []
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        policy = AccessPolicy(path)
        dev = UserContext(user_id="developer", role="developer")
        assert policy.is_allowed(dev, "tech_docs") is True
        assert policy.is_allowed(dev, "api_docs") is True
    finally:
        Path(path).unlink()


def test_user_denied_domain():
    """developer 不可访问 allowed_kbs 之外的 domain。"""
    policy_yaml = """
users:
  developer:
    role: developer
    allowed_kbs:
      - tech_docs
default:
  role: viewer
  allowed_kbs: []
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        policy = AccessPolicy(path)
        dev = UserContext(user_id="developer", role="developer")
        assert policy.is_allowed(dev, "hr_docs") is False
        assert policy.is_allowed(dev, "admin_panel") is False
    finally:
        Path(path).unlink()


def test_unknown_user_uses_default():
    """未在 users 中列出的用户 → 使用 default 规则。"""
    policy_yaml = """
users:
  admin:
    role: admin
    allowed_kbs:
      - "*"
default:
  role: viewer
  allowed_kbs:
    - public_kb
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        policy = AccessPolicy(path)
        unknown = UserContext(user_id="stranger", role="viewer")
        assert policy.is_allowed(unknown, "public_kb") is True
        assert policy.is_allowed(unknown, "tech_docs") is False
    finally:
        Path(path).unlink()


def test_missing_policy_file_allows_all():
    """policy.yaml 不存在 → 全部允许（向后兼容）。"""
    policy = AccessPolicy("/nonexistent/policy.yaml")
    user = UserContext(user_id="anyone", role="viewer")
    assert policy.is_allowed(user, "anything") is True


def test_invalid_policy_handling():
    """格式错误的 policy.yaml → is_allowed 时抛出 RuntimeError。"""
    path = _make_policy_yaml("this: is: not: valid: yaml: [}")
    try:
        policy = AccessPolicy(path)
        user = UserContext(user_id="test")
        with pytest.raises(RuntimeError, match="policy.yaml 格式错误"):
            policy.is_allowed(user, "anything")
    finally:
        Path(path).unlink()


def test_default_policy_allows_default_kb():
    """没有 policy.yaml 时，默认策略允许 default domain。"""
    policy_yaml = """
default:
  role: viewer
  allowed_kbs:
    - default
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        policy = AccessPolicy(path)
        viewer = UserContext(user_id="viewer", role="viewer")
        assert policy.is_allowed(viewer, "default") is True
        assert policy.is_allowed(viewer, "secret") is False
    finally:
        Path(path).unlink()


# ============================================================
# AccessGuard
# ============================================================


def test_none_user_backward_compatibility():
    """user=None → 返回全部 domains（向后兼容）。"""
    guard = AccessGuard()
    domains = [
        _make_domain("default", default=True),
        _make_domain("tech_docs"),
        _make_domain("hr_docs"),
    ]
    result = guard.filter_domains(None, domains)
    assert len(result) == 3
    assert {d.id for d in result} == {"default", "tech_docs", "hr_docs"}


def test_admin_sees_all_domains():
    """admin ("*") → 返回全部 domains。"""
    policy_yaml = """
users:
  admin:
    role: admin
    allowed_kbs:
      - "*"
default:
  role: viewer
  allowed_kbs: []
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        guard = AccessGuard(AccessPolicy(path))
        admin = UserContext(user_id="admin", role="admin")
        domains = [
            _make_domain("default"),
            _make_domain("tech_docs"),
            _make_domain("hr_docs"),
        ]
        result = guard.filter_domains(admin, domains)
        assert len(result) == 3
    finally:
        Path(path).unlink()


def test_developer_sees_only_allowed():
    """developer 只能看到 allowed_kbs 中的 domain。"""
    policy_yaml = """
users:
  developer:
    role: developer
    allowed_kbs:
      - tech_docs
      - api_docs
default:
  role: viewer
  allowed_kbs: []
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        guard = AccessGuard(AccessPolicy(path))
        dev = UserContext(user_id="developer", role="developer")
        domains = [
            _make_domain("default"),
            _make_domain("tech_docs"),
            _make_domain("hr_docs"),
        ]
        result = guard.filter_domains(dev, domains)
        assert len(result) == 1
        assert result[0].id == "tech_docs"
    finally:
        Path(path).unlink()


def test_all_denied_falls_back_to_default():
    """全部 domain 被拒绝 → 安全网返回 default domain。"""
    policy_yaml = """
users:
  restricted:
    role: viewer
    allowed_kbs: []
default:
  role: viewer
  allowed_kbs: []
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        guard = AccessGuard(AccessPolicy(path))
        user = UserContext(user_id="restricted", role="viewer")
        domains = [
            _make_domain("default", default=True),
            _make_domain("tech_docs"),
        ]
        result = guard.filter_domains(user, domains)
        assert len(result) == 1
        assert result[0].id == "default"
    finally:
        Path(path).unlink()


def test_empty_domains_with_none_user():
    """domain 列表为空 + user=None → 返回空列表。"""
    guard = AccessGuard()
    result = guard.filter_domains(None, [])
    assert result == []


def test_empty_domains_with_user():
    """domain 列表为空 + 有效 user → 返回空列表（安全网也兜不住）。"""
    guard = AccessGuard()
    user = UserContext(user_id="anyone")
    result = guard.filter_domains(user, [])
    assert result == []
