"""KnowledgeService —— Agent 层的统一检索入口。

Agent 只通过此类访问知识库，不直接感知 Collection / Retriever / Router 等底层细节。
Phase 3：接入 AccessGuard 实现权限过滤。
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document

from config import settings
from knowledge.access import AccessGuard, UserContext
from knowledge.concurrent import ConcurrentRetriever
from knowledge.registry import KnowledgeBaseRegistry
from knowledge.router import KnowledgeRouter, KeywordRouter, LLMRouter
from llm_factory import create_llm
from search_pipeline import SearchPipeline, QueryRewriter, LLMRelevanceJudge


class KnowledgeService:
    """知识库检索的统一门面。

    调用链：AccessGuard → Router → Registry → SearchPipeline → Retriever。
    """

    def __init__(self) -> None:
        self._registry = KnowledgeBaseRegistry()
        self._llm = create_llm(temperature=0)
        self._router = self._build_router()
        self._pipeline = SearchPipeline(
            rewriter=QueryRewriter(self._llm),
            judge=LLMRelevanceJudge(self._llm),
        )
        self._access_guard = AccessGuard()
        self._concurrent = ConcurrentRetriever()

    def _build_router(self) -> KnowledgeRouter:
        """根据 settings.router_strategy 构造路由策略与 KnowledgeRouter。"""
        if settings.router_strategy == "keyword":
            strategy = KeywordRouter()
        else:
            strategy = LLMRouter(self._llm)

        return KnowledgeRouter(strategy)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        user: UserContext | None = None,
    ) -> list[Document]:
        """检索与查询最相关的文档。

        Args:
            query: 用户查询。
            user: 可选的用户上下文；None 时跳过权限过滤。

        调用链：AccessGuard → Router → Registry → SearchPipeline → Retriever。
        多 domain 时自动使用并发检索。
        """
        domains = self._registry.list_domains()

        if user is not None:
            domains = self._access_guard.filter_domains(user, domains)

        decision = self._router.route(query, domains)

        # 并发多 KB 检索
        if len(decision.domain_ids) > 1:
            retrievers = {
                did: self._registry.get_retriever(did)
                for did in decision.domain_ids
            }
            docs = self._concurrent.search(query, retrievers)
            # 单文档也走 pipeline 的 reranker 逻辑？
            # 并发结果已去重，直接返回
            return docs

        # 单 KB 路径（现有行为，完全不变）
        retriever = self._registry.get_retriever(decision.domain_id)
        return self._pipeline.retrieve(query, retriever)

    # ------------------------------------------------------------------
    # 元数据
    # ------------------------------------------------------------------

    def list_domains(
        self,
        user: UserContext | None = None,
    ) -> list[dict[str, object]]:
        """列出可用知识库的摘要信息。

        Args:
            user: 可选的用户上下文；None 时返回全部 domain。
        """
        domains = self._registry.list_domains()

        if user is not None:
            domains = self._access_guard.filter_domains(user, domains)

        return [
            {
                "id": d.id,
                "name": d.name,
                "description": d.description,
                "default": d.default,
            }
            for d in domains
        ]

    def invalidate(self) -> None:
        """清空内部缓存（重建索引后调用）。"""
        self._registry.invalidate()


# ------------------------------------------------------------------
# 模块级单例
# ------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_knowledge_service() -> KnowledgeService:
    """获取全局唯一的 KnowledgeService 实例。"""
    return KnowledgeService()
