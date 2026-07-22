"""Multi-KB Permission Integration Test。

完整链路: KnowledgeService → AccessGuard → Router → Registry。
验证多 KB 场景下权限过滤发生在 Router 之前。
全部使用 Mock Retriever + Mock LLM + 临时 YAML。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml
from langchain_core.documents import Document

from knowledge.access import AccessPolicy, AccessGuard, UserContext
from knowledge.domain import KnowledgeDomain
from knowledge.registry import KnowledgeBaseRegistry


# ============================================================
# Helpers
# ============================================================


def _kb_yaml(domains: list[dict]) -> Path:
    """创建临时 knowledge_bases.yaml。"""
    p = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.safe_dump({"domains": domains}, p, allow_unicode=True)
    p.close()
    return Path(p.name)


def _policy_yaml(content: str) -> Path:
    p = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    p.write(content)
    p.close()
    return Path(p.name)


def _four_kb_yaml() -> Path:
    """4 个 domain，不同 access_level。"""
    return _kb_yaml([
        {"id": "default", "name": "默认", "data_dir": "data", "persist_dir": "c1",
         "collection_name": "kb_default", "default": True, "access_level": "public"},
        {"id": "tech_docs", "name": "技术", "data_dir": "data/t", "persist_dir": "c2",
         "collection_name": "kb_tech", "access_level": "internal"},
        {"id": "product_docs", "name": "产品", "data_dir": "data/p", "persist_dir": "c3",
         "collection_name": "kb_product", "access_level": "internal"},
        {"id": "hr_docs", "name": "HR", "data_dir": "data/h", "persist_dir": "c4",
         "collection_name": "kb_hr", "access_level": "restricted"},
    ])


def _mock_env(kb_path: Path, policy_path: Path | None = None):
    """构建 KnowledgeService，注入临时 KB + Policy。"""
    mock_llm = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = [Document(page_content="result")]

    patchers = [
        patch("knowledge.service.create_llm", return_value=mock_llm),
        patch.object(KnowledgeBaseRegistry, "get_retriever", return_value=mock_retriever),
    ]
    for p in patchers:
        p.start()

    from knowledge.service import KnowledgeService
    svc = KnowledgeService()
    svc._registry = KnowledgeBaseRegistry(config_path=kb_path)
    svc._registry.get_retriever = MagicMock(return_value=mock_retriever)

    if policy_path is not None:
        svc._access_guard = AccessGuard(AccessPolicy(policy_path))

    return patchers, svc, mock_retriever


def _stop(patchers):
    for p in reversed(patchers):
        p.stop()


# ============================================================
# 1. user=None — all KBs
# ============================================================


def test_user_none_access_all_kbs():
    """user=None → Router 看到全部 4 个 domains。"""
    kb_path = _four_kb_yaml()
    try:
        patchers, svc, retriever = _mock_env(kb_path)

        with patch.object(svc._router, "route", wraps=svc._router.route) as spy:
            svc.search("q", user=None)
            domains_passed = spy.call_args[0][1]
            assert len(domains_passed) == 4
    finally:
        _stop(patchers)
        kb_path.unlink()


# ============================================================
# 2. admin — all KBs
# ============================================================


def test_admin_access_all_kbs():
    """admin(*) + all levels → Router 看到全部 4 个 domains。"""
    kb_path = _four_kb_yaml()
    policy_path = _policy_yaml("""
users:
  admin:
    role: admin
    allowed_kbs: ["*"]
    allowed_levels: ["*"]
default:
  allowed_kbs: []
  allowed_levels: ["public"]
""")
    try:
        patchers, svc, retriever = _mock_env(kb_path, policy_path)
        admin = UserContext(user_id="admin", role="admin")

        with patch.object(svc._router, "route", wraps=svc._router.route) as spy:
            svc.search("q", user=admin)
            domains_passed = spy.call_args[0][1]
            assert len(domains_passed) == 4
            assert {d.id for d in domains_passed} == {"default", "tech_docs", "product_docs", "hr_docs"}
    finally:
        _stop(patchers)
        kb_path.unlink()
        policy_path.unlink()


# ============================================================
# 3. developer — filtered domains
# ============================================================


def test_developer_filtered_domains():
    """developer → 只能看到 tech_docs + product_docs（internal level）。"""
    kb_path = _four_kb_yaml()
    policy_path = _policy_yaml("""
users:
  dev:
    role: developer
    allowed_kbs: ["tech_docs", "product_docs"]
    allowed_levels: ["public", "internal"]
default:
  allowed_kbs: []
  allowed_levels: ["public"]
""")
    try:
        patchers, svc, retriever = _mock_env(kb_path, policy_path)
        dev = UserContext(user_id="dev", role="developer")

        with patch.object(svc._router, "route", wraps=svc._router.route) as spy:
            svc.search("q", user=dev)
            domains_passed = spy.call_args[0][1]
            assert len(domains_passed) == 2
            assert {d.id for d in domains_passed} == {"tech_docs", "product_docs"}
    finally:
        _stop(patchers)
        kb_path.unlink()
        policy_path.unlink()


# ============================================================
# 4. viewer — public only
# ============================================================


def test_viewer_public_only():
    """viewer → 仅 default（public level）。"""
    kb_path = _four_kb_yaml()
    policy_path = _policy_yaml("""
users:
  v:
    role: viewer
    allowed_kbs: ["default", "tech_docs"]
    allowed_levels: ["public"]
default:
  allowed_kbs: []
  allowed_levels: ["public"]
""")
    try:
        patchers, svc, retriever = _mock_env(kb_path, policy_path)
        viewer = UserContext(user_id="v", role="viewer")

        with patch.object(svc._router, "route", wraps=svc._router.route) as spy:
            svc.search("q", user=viewer)
            domains_passed = spy.call_args[0][1]
            # tech_docs 是 internal level，viewer 只有 public
            assert len(domains_passed) == 1
            assert domains_passed[0].id == "default"
    finally:
        _stop(patchers)
        kb_path.unlink()
        policy_path.unlink()


# ============================================================
# 5. denied_kbs
# ============================================================


def test_deny_overrides_allow_in_multi_kb():
    """allowed_kbs="*" + denied_kbs=[hr_docs] → hr_docs 被排除。"""
    kb_path = _four_kb_yaml()
    policy_path = _policy_yaml("""
users:
  dev:
    role: developer
    allowed_kbs: ["*"]
    denied_kbs: ["hr_docs"]
    allowed_levels: ["*"]
default:
  allowed_kbs: []
  allowed_levels: ["public"]
""")
    try:
        patchers, svc, retriever = _mock_env(kb_path, policy_path)
        dev = UserContext(user_id="dev", role="developer")

        with patch.object(svc._router, "route", wraps=svc._router.route) as spy:
            svc.search("q", user=dev)
            domains_passed = spy.call_args[0][1]
            assert len(domains_passed) == 3
            assert {d.id for d in domains_passed} == {"default", "tech_docs", "product_docs"}
            assert "hr_docs" not in {d.id for d in domains_passed}
    finally:
        _stop(patchers)
        kb_path.unlink()
        policy_path.unlink()


# ============================================================
# 6. access_level filtering
# ============================================================


def test_access_level_filtering():
    """public/internal/restricted 三级过滤验证。"""
    kb_path = _four_kb_yaml()
    policy_path = _policy_yaml("""
users:
  dev:
    role: developer
    allowed_kbs: ["*"]
    allowed_levels: ["public", "internal"]
default:
  allowed_kbs: []
  allowed_levels: ["public"]
""")
    try:
        patchers, svc, retriever = _mock_env(kb_path, policy_path)
        dev = UserContext(user_id="dev", role="developer")

        with patch.object(svc._router, "route", wraps=svc._router.route) as spy:
            svc.search("q", user=dev)
            domains_passed = spy.call_args[0][1]
            # default(public) + tech_docs(internal) + product_docs(internal) = 3
            # hr_docs(restricted) excluded
            assert len(domains_passed) == 3
            assert "hr_docs" not in {d.id for d in domains_passed}
    finally:
        _stop(patchers)
        kb_path.unlink()
        policy_path.unlink()


# ============================================================
# 7. filter before Router
# ============================================================


def test_filter_happens_before_router():
    """验证 Router 收到的 domains 已经过 AccessGuard 过滤。"""
    kb_path = _four_kb_yaml()
    policy_path = _policy_yaml("""
users:
  restricted:
    role: viewer
    allowed_kbs: ["default"]
    allowed_levels: ["public"]
default:
  allowed_kbs: []
  allowed_levels: ["public"]
""")
    try:
        patchers, svc, retriever = _mock_env(kb_path, policy_path)
        user = UserContext(user_id="restricted", role="viewer")

        # 记录 registry.list_domains() 原始数量 vs Router 收到的数量
        all_domains = svc._registry.list_domains()
        assert len(all_domains) == 4  # 原始 4 个

        with patch.object(svc._router, "route", wraps=svc._router.route) as spy:
            svc.search("q", user=user)
            domains_passed = spy.call_args[0][1]
            assert len(domains_passed) == 1  # Router 只看到 1 个
            assert domains_passed[0].id == "default"
    finally:
        _stop(patchers)
        kb_path.unlink()
        policy_path.unlink()


# ============================================================
# 8. list_domains(user)
# ============================================================


def test_list_domains_with_user():
    """list_domains(user=dev) → 仅返回授权 domain。"""
    kb_path = _four_kb_yaml()
    policy_path = _policy_yaml("""
users:
  dev:
    role: developer
    allowed_kbs: ["tech_docs", "product_docs"]
    allowed_levels: ["public", "internal"]
default:
  allowed_kbs: []
  allowed_levels: ["public"]
""")
    try:
        patchers, svc, retriever = _mock_env(kb_path, policy_path)
        dev = UserContext(user_id="dev", role="developer")

        result = svc.list_domains(user=dev)
        assert len(result) == 2
        ids = {d["id"] for d in result}
        assert ids == {"tech_docs", "product_docs"}
    finally:
        _stop(patchers)
        kb_path.unlink()
        policy_path.unlink()


# ============================================================
# 9. list_domains() — no user
# ============================================================


def test_list_domains_without_user():
    """list_domains() → 全部 KB（user=None 兼容）。"""
    kb_path = _four_kb_yaml()
    try:
        patchers, svc, retriever = _mock_env(kb_path)
        result = svc.list_domains()
        assert len(result) == 4
    finally:
        _stop(patchers)
        kb_path.unlink()


# ============================================================
# 10. unauthorized user — safety fallback
# ============================================================


def test_unauthorized_user_no_privilege_escalation():
    """allowed_kbs=[] → 安全网返回 default domain（不越权到未授权 KB）。"""
    kb_path = _four_kb_yaml()
    policy_path = _policy_yaml("""
users:
  nobody:
    role: viewer
    allowed_kbs: []
    allowed_levels: ["public"]
default:
  allowed_kbs: []
  allowed_levels: ["public"]
""")
    try:
        patchers, svc, retriever = _mock_env(kb_path, policy_path)
        nobody = UserContext(user_id="nobody", role="viewer")

        with patch.object(svc._router, "route", wraps=svc._router.route) as spy:
            svc.search("q", user=nobody)
            domains_passed = spy.call_args[0][1]
            # 安全网：至少 default domain
            assert len(domains_passed) == 1
            assert domains_passed[0].id == "default"
            # 不应越权到其他 KB
            assert "hr_docs" not in {d.id for d in domains_passed}
            assert "tech_docs" not in {d.id for d in domains_passed}
    finally:
        _stop(patchers)
        kb_path.unlink()
        policy_path.unlink()
