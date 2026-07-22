"""KnowledgeManager — 知识库管理服务层。

UI / CLI 通过此类管理 KB，不直接操作 Registry。
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from config import ROOT_DIR


class KnowledgeManager:
    """知识库生命周期管理。

    负责 KB 查询、状态管理、enable/disable 等操作。
    配置变更通过读写 YAML 持久化。
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        if config_path is None:
            config_path = os.getenv(
                "KB_CONFIG_PATH", str(ROOT_DIR / "config" / "knowledge_bases.yaml")
            )
        self._config_path = Path(config_path)
        # 回退
        if not self._config_path.exists():
            legacy = ROOT_DIR / "knowledge_bases.yaml"
            if legacy.exists():
                self._config_path = legacy

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def list_knowledge_bases(self) -> list[dict[str, object]]:
        """列出所有 KB（含 disabled）的摘要信息。"""
        from knowledge.registry import KnowledgeBaseRegistry

        registry = KnowledgeBaseRegistry(config_path=self._config_path)
        domains = registry.list_all_domains()
        return [
            {
                "id": d.id,
                "name": d.name,
                "description": d.description,
                "enabled": d.enabled,
                "default": d.default,
                "collection": d.collection_name,
                "metadata": d.metadata,
            }
            for d in domains
        ]

    def get_status(self, domain_id: str) -> dict[str, object] | None:
        """获取单个 KB 的详细状态。"""
        for kb in self.list_knowledge_bases():
            if kb["id"] == domain_id:
                # 检查索引是否存在
                from knowledge.registry import KnowledgeBaseRegistry
                registry = KnowledgeBaseRegistry(config_path=self._config_path)
                domain = registry.get_domain(domain_id)
                kb["index_exists"] = domain.persist_dir.exists()
                return kb
        return None

    # ------------------------------------------------------------------
    # 管理
    # ------------------------------------------------------------------

    def enable(self, domain_id: str) -> bool:
        """启用指定 KB。返回 True 表示成功。"""
        return self._set_enabled(domain_id, True)

    def disable(self, domain_id: str) -> bool:
        """禁用指定 KB。不允许禁用 default KB。返回 True 表示成功。"""
        # 检查是否为 default
        status = self.get_status(domain_id)
        if status and status.get("default"):
            return False
        return self._set_enabled(domain_id, False)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _set_enabled(self, domain_id: str, enabled: bool) -> bool:
        """修改 YAML 中指定 domain 的 enabled 字段。"""
        if not self._config_path.exists():
            return False

        with open(self._config_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}

        entries = config.get("knowledge_bases") or config.get("domains") or []
        found = False
        for entry in entries:
            if entry.get("id") == domain_id:
                entry["enabled"] = enabled
                found = True
                break

        if not found:
            return False

        # 保留原有顶层 key
        key = "knowledge_bases" if "knowledge_bases" in config else "domains"
        with open(self._config_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump({key: entries}, fh, allow_unicode=True, default_flow_style=False)

        return True
