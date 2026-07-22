"""KnowledgeBaseRegistry —— 管理每个 domain 的 Chroma + HybridRetriever。

仅管理检索器生命周期。Domain 加载委托给 kb_loader。
"""

from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma

from embeddings import get_embeddings
from knowledge.domain import KnowledgeDomain
from knowledge.kb_loader import load_domains
from retrieval import HybridRetriever


class KnowledgeBaseRegistry:
    """管理所有 KnowledgeDomain 及其 HybridRetriever 实例。

    - Domain 定义由 kb_loader.load_domains() 加载
    - 惰性初始化：首次 search 时才创建 Chroma + HybridRetriever
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        self._domains: dict[str, KnowledgeDomain] = load_domains(config_path)
        self._retrievers: dict[str, HybridRetriever] = {}

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
        return next(iter(self._domains.values()))

    def list_domains(self) -> list[KnowledgeDomain]:
        """列出全部已启用 domain。"""
        return [d for d in self._domains.values() if d.enabled]

    def list_all_domains(self) -> list[KnowledgeDomain]:
        """列出全部 domain（含 disabled）。"""
        return list(self._domains.values())

    # ------------------------------------------------------------------
    # 检索器
    # ------------------------------------------------------------------

    def get_retriever(self, domain_id: str) -> HybridRetriever:
        """惰性获取 domain 对应的 HybridRetriever。

        首次调用时创建 Chroma + HybridRetriever 并缓存。
        Phase 1.5：persist_dir 从 domain 显式传入，不再依赖 settings 全局。
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
            self._retrievers[domain_id] = HybridRetriever(
                vector_store, persist_dir=domain.persist_dir
            )
        return self._retrievers[domain_id]

    # ------------------------------------------------------------------
    # 缓存管理
    # ------------------------------------------------------------------

    def invalidate(self) -> None:
        """清空所有缓存的检索器与 embedding（重建索引后调用）。"""
        self._retrievers.clear()
        get_embeddings.cache_clear()
