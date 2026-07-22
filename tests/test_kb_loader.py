"""KB Loader 单元测试。

验证 YAML 加载、配置校验、回退逻辑。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from knowledge.kb_loader import load_domains


# ============================================================
# Helpers
# ============================================================


def _yaml_file(content: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


# ============================================================
# Basic loading
# ============================================================


def test_load_single_domain():
    """加载单个 domain。"""
    yaml_content = """
knowledge_bases:
  - id: default
    name: Default KB
    data_dir: data
    persist_dir: chroma_db
    collection: kb_default
    default: true
    enabled: true
"""
    path = _yaml_file(yaml_content)
    try:
        domains = load_domains(path)
        assert len(domains) == 1
        assert "default" in domains
        d = domains["default"]
        assert d.name == "Default KB"
        assert d.enabled is True
        assert d.default is True
    finally:
        path.unlink()


def test_load_multiple_domains():
    """加载多个 domain，含 disabled。"""
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
  - id: c
    name: C
    data_dir: data/c
    persist_dir: chroma_db/c
    collection: kb_c
    enabled: true
"""
    path = _yaml_file(yaml_content)
    try:
        domains = load_domains(path)
        assert len(domains) == 3
        assert domains["a"].enabled is True
        assert domains["b"].enabled is False
        assert domains["c"].enabled is True
    finally:
        path.unlink()


# ============================================================
# Metadata
# ============================================================


def test_metadata_preserved():
    """metadata 字段正确解析并保留。"""
    yaml_content = """
knowledge_bases:
  - id: test
    name: Test
    data_dir: data/test
    persist_dir: chroma_db/test
    collection: kb_test
    enabled: true
    metadata:
      owner: alice
      version: "1.0"
      embedding_model: BAAI/bge-small-zh-v1.5
"""
    path = _yaml_file(yaml_content)
    try:
        domains = load_domains(path)
        d = domains["test"]
        assert d.metadata == {
            "owner": "alice",
            "version": "1.0",
            "embedding_model": "BAAI/bge-small-zh-v1.5",
        }
    finally:
        path.unlink()


def test_metadata_defaults_to_empty():
    """未填写 metadata 时默认为空 dict。"""
    yaml_content = """
knowledge_bases:
  - id: test
    name: Test
    data_dir: data/test
    persist_dir: chroma_db/test
    collection: kb_test
    enabled: true
"""
    path = _yaml_file(yaml_content)
    try:
        domains = load_domains(path)
        assert domains["test"].metadata == {}
    finally:
        path.unlink()


# ============================================================
# Config validation & fallback
# ============================================================


def test_missing_config_creates_default():
    """配置文件不存在 → 自动创建 default domain。"""
    domains = load_domains("/nonexistent/path/kb.yaml")
    assert len(domains) == 1
    d = domains["default"]
    assert d.default is True
    assert d.enabled is True


def test_old_format_domains_key():
    """旧格式 (domains key) 仍可加载。"""
    yaml_content = """
domains:
  - id: default
    name: Legacy
    data_dir: data
    persist_dir: chroma_db
    collection_name: knowledge_base
    default: true
    enabled: true
"""
    path = _yaml_file(yaml_content)
    try:
        domains = load_domains(path)
        assert len(domains) == 1
        assert domains["default"].name == "Legacy"
    finally:
        path.unlink()


def test_empty_knowledge_bases_list():
    """knowledge_bases 列表为空 → 返回空 dict。"""
    yaml_content = """
knowledge_bases: []
"""
    path = _yaml_file(yaml_content)
    try:
        domains = load_domains(path)
        assert domains == {}
    finally:
        path.unlink()


def test_invalid_yaml_raises():
    """格式错误的 YAML → 抛出异常。"""
    path = _yaml_file("this: is: not: valid: [")
    try:
        with pytest.raises(Exception):
            load_domains(path)
    finally:
        path.unlink()


def test_keywords_parsed():
    """keywords 字段正确解析为 list。"""
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
      - fastapi
      - docker
"""
    path = _yaml_file(yaml_content)
    try:
        domains = load_domains(path)
        assert domains["tech"].keywords == ["python", "fastapi", "docker"]
    finally:
        path.unlink()


def test_keywords_defaults_to_empty():
    """未填写 keywords 时默认为空 list。"""
    yaml_content = """
knowledge_bases:
  - id: test
    name: Test
    data_dir: data/test
    persist_dir: chroma_db/test
    collection: kb_test
    enabled: true
"""
    path = _yaml_file(yaml_content)
    try:
        domains = load_domains(path)
        assert domains["test"].keywords == []
    finally:
        path.unlink()
