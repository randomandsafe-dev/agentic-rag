"""Multi-KB Registry 基础设施测试。

验证多 KB 加载、enabled 过滤、配置兼容性。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from knowledge.domain import KnowledgeDomain
from knowledge.registry import KnowledgeBaseRegistry


# ============================================================
# Helpers
# ============================================================


def _config_file(content: str, use_new_key: bool = True) -> Path:
    """创建临时 YAML 配置文件。"""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


# ============================================================
# KnowledgeDomain
# ============================================================


def test_domain_default_enabled():
    """新建 KnowledgeDomain 默认 enabled=True。"""
    d = KnowledgeDomain(
        id="test", name="Test", description="",
        data_dir=Path("."), persist_dir=Path("."),
        collection_name="kb_test",
    )
    assert d.enabled is True


def test_domain_explicit_disabled():
    """显式设置 enabled=False。"""
    d = KnowledgeDomain(
        id="test", name="Test", description="",
        data_dir=Path("."), persist_dir=Path("."),
        collection_name="kb_test",
        enabled=False,
    )
    assert d.enabled is False


# ============================================================
# Multi-KB loading (new format: knowledge_bases key)
# ============================================================


def test_load_multiple_kbs_new_format():
    """新格式 (knowledge_bases key) 加载多个 KB。"""
    yaml_content = """
knowledge_bases:
  - id: default
    name: Default KB
    data_dir: data
    persist_dir: chroma_db
    collection: kb_default
    default: true
    enabled: true

  - id: tech_docs
    name: Tech Docs
    data_dir: data/tech
    persist_dir: chroma_db/tech
    collection: kb_tech
    enabled: true

  - id: hr_docs
    name: HR Docs
    data_dir: data/hr
    persist_dir: chroma_db/hr
    collection: kb_hr
    enabled: false
"""
    path = _config_file(yaml_content)
    try:
        reg = KnowledgeBaseRegistry(config_path=path)
        # list_domains 只返回 enabled 的
        domains = reg.list_domains()
        assert len(domains) == 2
        ids = {d.id for d in domains}
        assert ids == {"default", "tech_docs"}
    finally:
        path.unlink()


def test_list_all_domains_includes_disabled():
    """list_all_domains 返回 disabled KB 在内全部。"""
    yaml_content = """
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
"""
    path = _config_file(yaml_content)
    try:
        reg = KnowledgeBaseRegistry(config_path=path)
        all_domains = reg.list_all_domains()
        assert len(all_domains) == 2
    finally:
        path.unlink()


# ============================================================
# Backward compatibility (old format: domains key)
# ============================================================


def test_load_old_format_domains_key():
    """旧格式 (domains key) 仍然可加载。"""
    yaml_content = """
domains:
  - id: default
    name: Default
    data_dir: data
    persist_dir: chroma_db
    collection_name: knowledge_base
    default: true
    enabled: true
"""
    path = _config_file(yaml_content)
    try:
        reg = KnowledgeBaseRegistry(config_path=path)
        domains = reg.list_domains()
        assert len(domains) == 1
        assert domains[0].id == "default"
    finally:
        path.unlink()


def test_load_collection_alias():
    """collection 字段作为 collection_name 别名。"""
    yaml_content = """
knowledge_bases:
  - id: test
    name: Test
    data_dir: data/test
    persist_dir: chroma_db/test
    collection: my_collection
    enabled: true
"""
    path = _config_file(yaml_content)
    try:
        reg = KnowledgeBaseRegistry(config_path=path)
        domain = reg.get_domain("test")
        assert domain.collection_name == "my_collection"
    finally:
        path.unlink()


def test_load_collection_name_fallback():
    """collection_name 优先于 collection。两者都没填时自动生成。"""
    yaml_content = """
knowledge_bases:
  - id: test
    name: Test
    data_dir: data/test
    persist_dir: chroma_db/test
    enabled: true
"""
    path = _config_file(yaml_content)
    try:
        reg = KnowledgeBaseRegistry(config_path=path)
        domain = reg.get_domain("test")
        assert domain.collection_name == "kb_test"
    finally:
        path.unlink()


# ============================================================
# Single-KB compatibility
# ============================================================


def test_single_kb_unchanged_behavior():
    """单 KB 配置 → list_domains 返回那个 KB。"""
    yaml_content = """
knowledge_bases:
  - id: default
    name: Default
    data_dir: data
    persist_dir: chroma_db
    collection: knowledge_base
    default: true
    enabled: true
"""
    path = _config_file(yaml_content)
    try:
        reg = KnowledgeBaseRegistry(config_path=path)
        domains = reg.list_domains()
        assert len(domains) == 1
        assert domains[0].id == "default"
        assert domains[0].default is True

        # get_domain 正常工作
        d = reg.get_domain("default")
        assert d.name == "Default"
    finally:
        path.unlink()


def test_missing_config_creates_default():
    """配置文件不存在 → 自动创建 default domain。"""
    reg = KnowledgeBaseRegistry(config_path="/nonexistent/kb.yaml")
    domains = reg.list_domains()
    assert len(domains) == 1
    assert domains[0].id == "default"
    assert domains[0].enabled is True


# ============================================================
# Invalid config
# ============================================================


def test_invalid_yaml_raises():
    """格式错误的 YAML → 抛出异常。"""
    path = _config_file("this: is: not: valid: [")
    try:
        with pytest.raises(Exception):
            KnowledgeBaseRegistry(config_path=path)
    finally:
        path.unlink()


def test_empty_domains_list():
    """domains 列表为空 → 返回空。"""
    yaml_content = """
knowledge_bases: []
"""
    path = _config_file(yaml_content)
    try:
        reg = KnowledgeBaseRegistry(config_path=path)
        domains = reg.list_domains()
        assert domains == []
    finally:
        path.unlink()


# ============================================================
# Keywords field
# ============================================================


def test_keywords_parsed():
    """keywords 字段正确解析。"""
    yaml_content = """
knowledge_bases:
  - id: tech
    name: Tech
    data_dir: data/tech
    persist_dir: chroma_db/tech
    collection: kb_tech
    enabled: true
    keywords:
      - python
      - api
"""
    path = _config_file(yaml_content)
    try:
        reg = KnowledgeBaseRegistry(config_path=path)
        domain = reg.get_domain("tech")
        assert domain.keywords == ["python", "api"]
    finally:
        path.unlink()
