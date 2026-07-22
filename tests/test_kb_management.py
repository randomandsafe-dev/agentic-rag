"""KnowledgeManager 单元测试。

验证 KB 查询、状态、enable/disable、配置持久化。
使用临时 YAML 文件。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from knowledge.management import KnowledgeManager


# ============================================================
# Helpers
# ============================================================


def _config_file(content: str) -> Path:
    p = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    p.write(content)
    p.close()
    return Path(p.name)


def _read_yaml(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ============================================================
# list_knowledge_bases
# ============================================================


def test_list_all_kbs_including_disabled():
    """list_knowledge_bases 返回全部 KB（含 disabled）。"""
    path = _config_file("""
knowledge_bases:
  - id: a
    name: A
    data_dir: data/a
    persist_dir: chroma_db/a
    collection: kb_a
    enabled: true
  - id: b
    name: B
    data_dir: data/b
    persist_dir: chroma_db/b
    collection: kb_b
    enabled: false
""")
    try:
        mgr = KnowledgeManager(path)
        kbs = mgr.list_knowledge_bases()
        assert len(kbs) == 2
        assert kbs[0]["enabled"] is True
        assert kbs[1]["enabled"] is False
    finally:
        path.unlink()


# ============================================================
# get_status
# ============================================================


def test_get_status_returns_kb_info():
    """get_status 返回 KB 详细信息。"""
    path = _config_file("""
knowledge_bases:
  - id: tech
    name: Tech Docs
    description: API documentation
    data_dir: data/tech
    persist_dir: chroma_db/tech
    collection: kb_tech
    enabled: true
    metadata:
      owner: alice
""")
    try:
        mgr = KnowledgeManager(path)
        status = mgr.get_status("tech")
        assert status is not None
        assert status["name"] == "Tech Docs"
        assert status["metadata"] == {"owner": "alice"}
    finally:
        path.unlink()


def test_get_status_invalid_domain():
    """不存在的 domain → None。"""
    path = _config_file("""
knowledge_bases:
  - id: a
    name: A
    data_dir: data/a
    persist_dir: chroma_db/a
    collection: kb_a
    enabled: true
""")
    try:
        mgr = KnowledgeManager(path)
        assert mgr.get_status("nonexistent") is None
    finally:
        path.unlink()


# ============================================================
# enable / disable
# ============================================================


def test_enable_kb():
    """enable → YAML 中 enabled 变为 true。"""
    path = _config_file("""
knowledge_bases:
  - id: test
    name: Test
    data_dir: data/test
    persist_dir: chroma_db/test
    collection: kb_test
    enabled: false
""")
    try:
        mgr = KnowledgeManager(path)
        assert mgr.enable("test") is True

        config = _read_yaml(path)
        entries = config["knowledge_bases"]
        assert entries[0]["enabled"] is True
    finally:
        path.unlink()


def test_disable_kb():
    """disable → YAML 中 enabled 变为 false。"""
    path = _config_file("""
knowledge_bases:
  - id: test
    name: Test
    data_dir: data/test
    persist_dir: chroma_db/test
    collection: kb_test
    enabled: true
""")
    try:
        mgr = KnowledgeManager(path)
        assert mgr.disable("test") is True

        config = _read_yaml(path)
        entries = config["knowledge_bases"]
        assert entries[0]["enabled"] is False
    finally:
        path.unlink()


def test_cannot_disable_default_kb():
    """不允许禁用 default KB。"""
    path = _config_file("""
knowledge_bases:
  - id: default
    name: Default
    data_dir: data
    persist_dir: chroma_db
    collection: kb_default
    default: true
    enabled: true
""")
    try:
        mgr = KnowledgeManager(path)
        assert mgr.disable("default") is False

        # 确认 YAML 未被修改
        config = _read_yaml(path)
        assert config["knowledge_bases"][0]["enabled"] is True
    finally:
        path.unlink()


def test_enable_invalid_domain():
    """不存在的 domain → enable 返回 False。"""
    path = _config_file("""
knowledge_bases:
  - id: a
    name: A
    data_dir: data/a
    persist_dir: chroma_db/a
    collection: kb_a
    enabled: true
""")
    try:
        mgr = KnowledgeManager(path)
        assert mgr.enable("nonexistent") is False
    finally:
        path.unlink()


# ============================================================
# Config persistence
# ============================================================


def test_config_persistence_preserves_structure():
    """enable/disable 后 YAML 结构保留（不丢失其他字段）。"""
    path = _config_file("""
knowledge_bases:
  - id: a
    name: Alpha
    description: First KB
    data_dir: data/a
    persist_dir: chroma_db/a
    collection: kb_a
    default: true
    enabled: true
    keywords:
      - alpha
      - test
    metadata:
      version: "1.0"
  - id: b
    name: Beta
    data_dir: data/b
    persist_dir: chroma_db/b
    collection: kb_b
    enabled: false
""")
    try:
        mgr = KnowledgeManager(path)
        # 禁用 b (非 default)，a 保持不变
        mgr.disable("b")

        config = _read_yaml(path)
        entries = config["knowledge_bases"]
        assert entries[0]["enabled"] is True   # a unchanged
        assert entries[0]["name"] == "Alpha"
        assert entries[0]["keywords"] == ["alpha", "test"]
        assert entries[0]["metadata"] == {"version": "1.0"}
        # b 本来就是 false，再次 disable 仍然是 false
        assert entries[1]["enabled"] is False
        assert entries[1]["name"] == "Beta"
    finally:
        path.unlink()


def test_old_domains_format_preserved():
    """旧格式 (domains key) enable/disable 后保留原有 key。"""
    path = _config_file("""
domains:
  - id: legacy
    name: Legacy
    data_dir: data/legacy
    persist_dir: chroma_db/legacy
    collection_name: kb_legacy
    enabled: true
""")
    try:
        mgr = KnowledgeManager(path)
        assert mgr.disable("legacy") is True

        config = _read_yaml(path)
        assert "domains" in config
        assert config["domains"][0]["enabled"] is False
    finally:
        path.unlink()
