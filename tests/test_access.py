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


# ============================================================
# Role-based matching (Commit 1)
# ============================================================


def test_role_match_when_user_not_in_users():
    """用户不在 users 中 → 按其 role 匹配 roles 节。"""
    policy_yaml = """
users:
  admin:
    role: admin
    allowed_kbs:
      - "*"
roles:
  developer:
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
        dev = UserContext(user_id="alice_new", role="developer")
        assert policy.is_allowed(dev, "tech_docs") is True
        assert policy.is_allowed(dev, "api_docs") is True
        assert policy.is_allowed(dev, "hr_docs") is False
    finally:
        Path(path).unlink()


def test_user_priority_over_role():
    """user_id 显式匹配优先于 role 匹配。"""
    policy_yaml = """
users:
  bob:
    role: developer
    allowed_kbs:
      - hr_docs
roles:
  developer:
    allowed_kbs:
      - tech_docs
default:
  role: viewer
  allowed_kbs: []
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        policy = AccessPolicy(path)
        bob = UserContext(user_id="bob", role="developer")
        # user 显式策略优先
        assert policy.is_allowed(bob, "hr_docs") is True
        # role 策略被跳过
        assert policy.is_allowed(bob, "tech_docs") is False
    finally:
        Path(path).unlink()


def test_role_fallback_to_default():
    """role 不在 roles 中 → 回退到 default 规则。"""
    policy_yaml = """
users: {}
roles:
  developer:
    allowed_kbs:
      - tech_docs
default:
  role: viewer
  allowed_kbs:
    - default
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        policy = AccessPolicy(path)
        viewer = UserContext(user_id="stranger", role="viewer")
        assert policy.is_allowed(viewer, "default") is True
        assert policy.is_allowed(viewer, "tech_docs") is False
    finally:
        Path(path).unlink()


def test_role_wildcard_match():
    """role 有 "*" 通配符 → 全部允许。"""
    policy_yaml = """
roles:
  admin:
    allowed_kbs:
      - "*"
default:
  role: viewer
  allowed_kbs: []
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        policy = AccessPolicy(path)
        admin = UserContext(user_id="admin_new", role="admin")
        assert policy.is_allowed(admin, "anything") is True
        assert policy.is_allowed(admin, "everything") is True
    finally:
        Path(path).unlink()


def test_no_roles_section_falls_back():
    """policy.yaml 无 roles 节 → 旧格式兼容，user 不匹配时直接走 default。"""
    policy_yaml = """
users:
  admin:
    role: admin
    allowed_kbs:
      - "*"
default:
  role: viewer
  allowed_kbs:
    - default
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        policy = AccessPolicy(path)
        unknown = UserContext(user_id="new_user", role="developer")
        # 无 roles 节，回退到 default
        assert policy.is_allowed(unknown, "default") is True
        assert policy.is_allowed(unknown, "tech_docs") is False
    finally:
        Path(path).unlink()


# ============================================================
# Explicit deny list (Commit 2)
# ============================================================


def test_deny_overrides_allow():
    """denied_kbs 优先于 allowed_kbs。"""
    policy_yaml = """
users:
  alice:
    role: developer
    allowed_kbs:
      - "*"
    denied_kbs:
      - hr_docs
default:
  role: viewer
  allowed_kbs: []
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        policy = AccessPolicy(path)
        alice = UserContext(user_id="alice", role="developer")
        # allow all except hr_docs
        assert policy.is_allowed(alice, "tech_docs") is True
        assert policy.is_allowed(alice, "product_docs") is True
        assert policy.is_allowed(alice, "hr_docs") is False
    finally:
        Path(path).unlink()


def test_deny_wildcard_blocks_all():
    """denied_kbs=["*"] → 全部拒绝。"""
    policy_yaml = """
users:
  blocked:
    role: viewer
    allowed_kbs:
      - "*"
    denied_kbs:
      - "*"
default:
  role: viewer
  allowed_kbs: []
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        policy = AccessPolicy(path)
        blocked = UserContext(user_id="blocked", role="viewer")
        assert policy.is_allowed(blocked, "anything") is False
        assert policy.is_allowed(blocked, "default") is False
    finally:
        Path(path).unlink()


def test_deny_in_role_policy():
    """role 策略中的 denied_kbs 生效。"""
    policy_yaml = """
roles:
  developer:
    allowed_kbs:
      - "*"
    denied_kbs:
      - admin_panel
default:
  role: viewer
  allowed_kbs: []
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        policy = AccessPolicy(path)
        dev = UserContext(user_id="new_dev", role="developer")
        assert policy.is_allowed(dev, "tech_docs") is True
        assert policy.is_allowed(dev, "admin_panel") is False
    finally:
        Path(path).unlink()


def test_user_deny_overrides_role():
    """user 和 role 都有 deny → user 的策略优先。"""
    policy_yaml = """
users:
  bob:
    role: developer
    allowed_kbs:
      - "*"
    denied_kbs:
      - hr_docs
      - finance
roles:
  developer:
    allowed_kbs:
      - tech_docs
default:
  role: viewer
  allowed_kbs: []
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        policy = AccessPolicy(path)
        bob = UserContext(user_id="bob", role="developer")
        # user 策略: allow all except hr_docs + finance
        assert policy.is_allowed(bob, "tech_docs") is True
        assert policy.is_allowed(bob, "hr_docs") is False
        assert policy.is_allowed(bob, "finance") is False
    finally:
        Path(path).unlink()


def test_old_yaml_no_denied_kbs_still_works():
    """旧 policy.yaml 无 denied_kbs 字段 → 正常 allow。"""
    policy_yaml = """
users:
  charlie:
    role: viewer
    allowed_kbs:
      - default
      - public_kb
default:
  role: viewer
  allowed_kbs: []
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        policy = AccessPolicy(path)
        charlie = UserContext(user_id="charlie", role="viewer")
        assert policy.is_allowed(charlie, "default") is True
        assert policy.is_allowed(charlie, "public_kb") is True
        assert policy.is_allowed(charlie, "secret") is False
    finally:
        Path(path).unlink()


# ============================================================
# Schema normalization & validation (Commit 3)
# ============================================================


def test_normalized_schema_all_sections():
    """users + roles + default 三段完整加载。"""
    policy_yaml = """
users:
  admin:
    role: admin
    allowed_kbs:
      - "*"
  developer:
    role: developer
    allowed_kbs:
      - "*"
    denied_kbs:
      - hr_docs
roles:
  admin:
    allowed_kbs:
      - "*"
  developer:
    allowed_kbs:
      - tech_docs
  viewer:
    allowed_kbs:
      - default
default:
  allowed_kbs:
    - default
  denied_kbs: []
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        policy = AccessPolicy(path)
        # admin via user
        assert policy.is_allowed(UserContext(user_id="admin"), "anything") is True
        # developer via user (allow all except hr_docs)
        assert policy.is_allowed(UserContext(user_id="developer"), "tech_docs") is True
        assert policy.is_allowed(UserContext(user_id="developer"), "hr_docs") is False
        # unknown user → role match (viewer)
        assert policy.is_allowed(UserContext(user_id="new", role="viewer"), "default") is True
        # unknown user + unknown role → default
        assert policy.is_allowed(UserContext(user_id="x", role="unknown"), "default") is True
    finally:
        Path(path).unlink()


def test_default_denied_kbs_works():
    """default.denied_kbs 生效。"""
    policy_yaml = """
users: {}
roles: {}
default:
  allowed_kbs:
    - "*"
  denied_kbs:
    - secret_kb
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        policy = AccessPolicy(path)
        u = UserContext(user_id="anyone", role="viewer")
        assert policy.is_allowed(u, "public") is True
        assert policy.is_allowed(u, "secret_kb") is False
    finally:
        Path(path).unlink()


def test_allowed_kbs_not_list_fallback():
    """allowed_kbs 不是 list → warning + 重置为 []。"""
    policy_yaml = """
users:
  bad_user:
    allowed_kbs: not_a_list
default:
  allowed_kbs:
    - default
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        policy = AccessPolicy(path)
        u = UserContext(user_id="bad_user")
        # allowed_kbs 被重置为 []，不匹配任何 KB
        assert policy.is_allowed(u, "default") is False
    finally:
        Path(path).unlink()


def test_users_section_not_dict_fallback():
    """users 不是 dict → 警告 + 重置为空。"""
    policy_yaml = """
users:
  - this_is_a_list
default:
  allowed_kbs:
    - default
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        policy = AccessPolicy(path)
        # users 被重置，回退到 default
        u = UserContext(user_id="anyone")
        assert policy.is_allowed(u, "default") is True
    finally:
        Path(path).unlink()


def test_old_yaml_user_role_field_preserved():
    """users.<id>.role 字段保留兼容（不报错）。"""
    policy_yaml = """
users:
  alice:
    role: admin
    allowed_kbs:
      - "*"
default:
  role: viewer
  allowed_kbs:
    - default
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        policy = AccessPolicy(path)
        alice = UserContext(user_id="alice", role="admin")
        assert policy.is_allowed(alice, "anything") is True
    finally:
        Path(path).unlink()


def test_old_yaml_no_roles_still_works():
    """旧 YAML 无 roles 节 + 无 denied_kbs → 兼容。"""
    policy_yaml = """
users:
  admin:
    role: admin
    allowed_kbs:
      - "*"
default:
  role: viewer
  allowed_kbs:
    - default
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        policy = AccessPolicy(path)
        assert policy.is_allowed(UserContext(user_id="admin"), "x") is True
        assert policy.is_allowed(UserContext(user_id="new", role="viewer"), "default") is True
    finally:
        Path(path).unlink()


# ============================================================
# access_level filtering (Commit 4)
# ============================================================


def test_effective_merge_field_level():
    """user 声明 allowed_kbs，继承 role 的 allowed_levels。"""
    policy_yaml = """
users:
  alice:
    allowed_kbs:
      - tech_docs
roles:
  developer:
    allowed_levels:
      - public
      - internal
default:
  allowed_kbs: []
  allowed_levels:
    - public
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        policy = AccessPolicy(path)
        effective = policy.get_effective_policy(UserContext(user_id="alice", role="developer"))
        assert effective["allowed_kbs"] == ["tech_docs"]       # user 覆盖
        assert effective["allowed_levels"] == ["public", "internal"]  # 继承 role
        assert effective["denied_kbs"] == []                    # 继承 default
    finally:
        Path(path).unlink()


def test_user_field_overrides_role_field():
    """user 声明 allowed_levels → 覆盖 role。"""
    policy_yaml = """
users:
  bob:
    allowed_levels:
      - public
roles:
  developer:
    allowed_levels:
      - public
      - internal
default:
  allowed_kbs: []
  allowed_levels:
    - public
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        policy = AccessPolicy(path)
        effective = policy.get_effective_policy(UserContext(user_id="bob", role="developer"))
        assert effective["allowed_levels"] == ["public"]  # user 覆盖了 role
    finally:
        Path(path).unlink()


def test_admin_access_all_levels():
    """admin 有 "*" allowed_levels → 全部 access_level 通过。"""
    policy_yaml = """
roles:
  admin:
    allowed_kbs:
      - "*"
    allowed_levels:
      - "*"
default:
  allowed_kbs: []
  allowed_levels:
    - public
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        policy = AccessPolicy(path)
        admin = UserContext(user_id="admin_new", role="admin")
        assert policy.is_level_allowed(admin, "public") is True
        assert policy.is_level_allowed(admin, "internal") is True
        assert policy.is_level_allowed(admin, "restricted") is True
    finally:
        Path(path).unlink()


def test_developer_restricted_level():
    """developer → internal 可访问，restricted 被过滤。"""
    policy_yaml = """
roles:
  developer:
    allowed_kbs:
      - "*"
    allowed_levels:
      - public
      - internal
default:
  allowed_kbs: []
  allowed_levels:
    - public
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        policy = AccessPolicy(path)
        dev = UserContext(user_id="dev", role="developer")
        assert policy.is_level_allowed(dev, "public") is True
        assert policy.is_level_allowed(dev, "internal") is True
        assert policy.is_level_allowed(dev, "restricted") is False
    finally:
        Path(path).unlink()


def test_viewer_public_only():
    """viewer 仅 public → internal/restricted 被过滤。"""
    policy_yaml = """
roles:
  viewer:
    allowed_kbs:
      - default
    allowed_levels:
      - public
default:
  allowed_kbs: []
  allowed_levels:
    - public
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        policy = AccessPolicy(path)
        viewer = UserContext(user_id="v", role="viewer")
        assert policy.is_level_allowed(viewer, "public") is True
        assert policy.is_level_allowed(viewer, "internal") is False
    finally:
        Path(path).unlink()


def test_filter_domains_with_access_level():
    """filter_domains 同时过滤 domain_id + access_level。"""
    policy_yaml = """
users:
  dev:
    allowed_kbs:
      - "*"
    allowed_levels:
      - public
      - internal
default:
  allowed_kbs: []
  allowed_levels:
    - public
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        guard = AccessGuard(AccessPolicy(path))
        dev = UserContext(user_id="dev", role="developer")
        domains = [
            KnowledgeDomain(id="a", name="a", description="", data_dir=Path("."), persist_dir=Path("."),
                           collection_name="kb_a", access_level="public"),
            KnowledgeDomain(id="b", name="b", description="", data_dir=Path("."), persist_dir=Path("."),
                           collection_name="kb_b", access_level="internal"),
            KnowledgeDomain(id="c", name="c", description="", data_dir=Path("."), persist_dir=Path("."),
                           collection_name="kb_c", access_level="restricted"),
        ]
        result = guard.filter_domains(dev, domains)
        assert len(result) == 2
        assert {d.id for d in result} == {"a", "b"}
    finally:
        Path(path).unlink()


def test_domain_no_access_level_defaults_public():
    """domain 无 access_level → 默认 public，viewer 可访问。"""
    policy_yaml = """
roles:
  viewer:
    allowed_kbs:
      - "*"
    allowed_levels:
      - public
default:
  allowed_kbs: []
  allowed_levels:
    - public
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        guard = AccessGuard(AccessPolicy(path))
        viewer = UserContext(user_id="v", role="viewer")
        domains = [
            KnowledgeDomain(id="x", name="x", description="", data_dir=Path("."), persist_dir=Path("."),
                           collection_name="kb_x"),  # no access_level → "public"
        ]
        result = guard.filter_domains(viewer, domains)
        assert len(result) == 1
    finally:
        Path(path).unlink()


def test_user_none_skips_level_check():
    """user=None → 不过滤 access_level。"""
    guard = AccessGuard()
    domains = [
        KnowledgeDomain(id="x", name="x", description="", data_dir=Path("."), persist_dir=Path("."),
                       collection_name="kb_x", access_level="restricted"),
    ]
    result = guard.filter_domains(None, domains)
    assert len(result) == 1


def test_old_yaml_no_allowed_levels_compatible():
    """旧 YAML 无 allowed_levels → 默认仅 ["public"]。"""
    policy_yaml = """
users:
  alice:
    allowed_kbs:
      - "*"
default:
  allowed_kbs: []
"""
    path = _make_policy_yaml(policy_yaml)
    try:
        policy = AccessPolicy(path)
        alice = UserContext(user_id="alice")
        effective = policy.get_effective_policy(alice)
        assert effective["allowed_levels"] == ["public"]
    finally:
        Path(path).unlink()
