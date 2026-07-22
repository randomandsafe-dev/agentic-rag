"""Knowledge Base 配置加载器。

从 YAML 加载 domain 定义，与 Registry 解耦。
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from config import ROOT_DIR
from knowledge.domain import KnowledgeDomain


def load_domains(
    config_path: str | Path | None = None,
) -> dict[str, KnowledgeDomain]:
    """从 YAML 配置文件加载 KnowledgeDomain 定义。

    回退顺序：
    1. config_path（显式传入）
    2. KB_CONFIG_PATH 环境变量
    3. config/knowledge_bases.yaml（新默认路径）
    4. knowledge_bases.yaml（旧路径回退）
    5. settings 构造单 domain（最终回退）

    Args:
        config_path: 配置文件路径；None 时使用默认路径 + 回退逻辑。

    Returns:
        domain_id -> KnowledgeDomain 的映射字典。
    """
    if config_path is None:
        config_path = os.getenv(
            "KB_CONFIG_PATH", str(ROOT_DIR / "config" / "knowledge_bases.yaml")
        )
    path = Path(config_path)

    # 新路径不存在时尝试旧路径
    if not path.exists():
        legacy = ROOT_DIR / "knowledge_bases.yaml"
        if legacy.exists():
            path = legacy
        else:
            return _fallback_default()

    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    # 支持两种 key：knowledge_bases（新）或 domains（旧）
    entries = config.get("knowledge_bases") or config.get("domains") or []

    domains: dict[str, KnowledgeDomain] = {}
    for entry in entries:
        domain = KnowledgeDomain(
            id=entry["id"],
            name=entry.get("name", entry["id"]),
            description=entry.get("description", ""),
            data_dir=ROOT_DIR / entry.get("data_dir", f"data/{entry['id']}"),
            persist_dir=ROOT_DIR
            / entry.get("persist_dir", f"chroma_db/{entry['id']}"),
            collection_name=entry.get(
                "collection_name",
                entry.get("collection", f"kb_{entry['id']}"),
            ),
            default=bool(entry.get("default", False)),
            enabled=bool(entry.get("enabled", True)),
            keywords=entry.get("keywords", []),
            metadata=entry.get("metadata", {}),
            access_level=entry.get("access_level", "public"),
        )
        domains[domain.id] = domain

    return domains


def _fallback_default() -> dict[str, KnowledgeDomain]:
    """从 settings 构造默认单 domain。"""
    from config import settings

    default = KnowledgeDomain(
        id="default",
        name="默认知识库",
        description="通用知识库",
        data_dir=settings.data_dir,
        persist_dir=settings.persist_dir,
        collection_name=settings.collection_name,
        default=True,
        enabled=True,
    )
    return {"default": default}
