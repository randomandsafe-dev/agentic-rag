"""KnowledgeService —— Agent 层的统一检索入口。

Agent 只通过此类访问知识库，不直接感知 Collection / Retriever 等底层细节。
Phase 1：仅单 KB 代理；Phase 2 接入 Router。
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document

from knowledge.registry import KnowledgeBaseRegistry


class KnowledgeService:
    """知识库检索的统一门面。

    Phase 1：直接委托给默认 domain 的 HybridRetriever。
    Phase 2：接入 KnowledgeRouter 实现多 KB 自动路由。
    Phase 3：接入 AccessGuard 实现权限过滤。
    """

    def __init__(self) -> None:
        self._registry = KnowledgeBaseRegistry()

    def search(self, query: str) -> list[Document]:
        """检索与查询最相关的文档。

        Phase 1：始终使用默认 domain。
        Phase 2：Router 根据 query 选择 domain(s)。
        """
        return self._registry.get_default_retriever().search(query)

    def list_domains(self) -> list[dict[str, object]]:
        """列出所有可用知识库的摘要信息（供 Agent 或 UI 展示）。"""
        return [
            {
                "id": d.id,
                "name": d.name,
                "description": d.description,
                "default": d.default,
            }
            for d in self._registry.list_domains()
        ]

    def invalidate(self) -> None:
        """清空内部缓存（重建索引后调用）。"""
        self._registry.invalidate()


# ------------------------------------------------------------------
# 模块级单例（替代 rag_agent 中旧的 get_hybrid_retriever 缓存）
# ------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_knowledge_service() -> KnowledgeService:
    """获取全局唯一的 KnowledgeService 实例。"""
    return KnowledgeService()
