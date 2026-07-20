"""KnowledgeBaseRegistry —— 加载 YAML 配置，管理每个 domain 的 Chroma + HybridRetriever。

仅管理检索器生命周期，不做路由（Router 留给 Phase 2）。
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from langchain_chroma import Chroma

from config import ROOT_DIR
from knowledge.domain import KnowledgeDomain
from rag_agent import get_embeddings
from retrieval import HybridRetriever


class KnowledgeBaseRegistry:
    """管理所有 KnowledgeDomain 及其 HybridRetriever 实例。

    - 从 knowledge_bases.yaml 加载 domain 定义
    - 若配置文件不存在，回退为单 domain（从 settings 构建）
    - 惰性初始化：首次 search 时才创建 Chroma + HybridRetriever
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        if config_path is None:
            config_path = os.getenv("KB_CONFIG_PATH", str(ROOT_DIR / "knowledge_bases.yaml"))
        self._config_path = Path(config_path)
        self._domains: dict[str, KnowledgeDomain] = {}
        self._retrievers: dict[str, HybridRetriever] = {}
        self._load()

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """加载 domain 定义；YAML 不存在时从 settings 构造默认 domain。"""
        if not self._config_path.exists():
            from config import settings

            self._domains["default"] = KnowledgeDomain(
                id="default",
                name="默认知识库",
                description="通用知识库",
                data_dir=settings.data_dir,
                persist_dir=settings.persist_dir,
                collection_name=settings.collection_name,
                default=True,
            )
            return

        with open(self._config_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)

        for entry in config.get("domains", []):
            domain = KnowledgeDomain(
                id=entry["id"],
                name=entry.get("name", entry["id"]),
                description=entry.get("description", ""),
                data_dir=ROOT_DIR / entry["data_dir"],
                persist_dir=ROOT_DIR / entry.get("persist_dir", f"chroma_db/{entry['id']}"),
                collection_name=entry.get("collection_name", f"kb_{entry['id']}"),
                default=bool(entry.get("default", False)),
            )
            self._domains[domain.id] = domain

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get_domain(self, domain_id: str) -> KnowledgeDomain:
        """按 ID 获取 domain 元数据。"""
        if domain_id not in self._domains:
            raise KeyError(f"未知知识库: {domain_id}")
        return self._domains[domain_id]

    def get_default_domain(self) -> KnowledgeDomain:
        """获取默认 domain。"""
        for domain in self._domains.values():
            if domain.default:
                return domain
        # 没有标记 default 时回退到第一个
        return next(iter(self._domains.values()))

    def list_domains(self) -> list[KnowledgeDomain]:
        """列出全部已注册 domain。"""
        return list(self._domains.values())

    # ------------------------------------------------------------------
    # 检索器
    # ------------------------------------------------------------------

    def get_retriever(self, domain_id: str) -> HybridRetriever:
        """惰性获取 domain 对应的 HybridRetriever。

        首次调用时创建 Chroma + HybridRetriever 并缓存。
        Phase 1：HybridRetriever 内部通过 settings.persist_dir 读取 BM25，
        因此 domain.persist_dir 须与 settings.persist_dir 一致。
        """
        if domain_id not in self._retrievers:
            domain = self.get_domain(domain_id)
            if not domain.persist_dir.exists():
                raise RuntimeError(
                    f"知识库「{domain.name}」尚未创建索引。请先运行: python ingest.py"
                )
            vector_store = Chroma(
                collection_name=domain.collection_name,
                persist_directory=str(domain.persist_dir),
                embedding_function=get_embeddings(),
            )
            self._retrievers[domain_id] = HybridRetriever(vector_store)
        return self._retrievers[domain_id]

    def get_default_retriever(self) -> HybridRetriever:
        """获取默认 domain 的检索器。"""
        return self.get_retriever(self.get_default_domain().id)

    # ------------------------------------------------------------------
    # 缓存管理
    # ------------------------------------------------------------------

    def invalidate(self) -> None:
        """清空所有缓存的检索器与 embedding（重建索引后调用）。"""
        self._retrievers.clear()
        get_embeddings.cache_clear()
