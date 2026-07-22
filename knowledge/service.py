"""KnowledgeService —— Agent 层的统一检索入口。

Agent 只通过此类访问知识库，不直接感知 Collection / Retriever / Router 等底层细节。
Phase 2.5：接入 SearchPipeline，形成统一的搜索编排链路。
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document

from config import settings
from knowledge.registry import KnowledgeBaseRegistry
from knowledge.router import KnowledgeRouter, KeywordRouter, LLMRouter
from llm_factory import create_llm
from search_pipeline import SearchPipeline, QueryRewriter, LLMRelevanceJudge


class KnowledgeService:
    """知识库检索的统一门面。

    Phase 2.5：Router → Registry → SearchPipeline → Retriever 完整链路。
    Phase 3：接入 AccessGuard 实现权限过滤。
    """

    def __init__(self) -> None:
        self._registry = KnowledgeBaseRegistry()
        self._llm = create_llm(temperature=0)
        self._router = self._build_router()
        self._pipeline = SearchPipeline(
            rewriter=QueryRewriter(self._llm),
            judge=LLMRelevanceJudge(self._llm),
        )

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

    def search(self, query: str) -> list[Document]:
        """检索与查询最相关的文档。

        调用链：Router → Registry → SearchPipeline → Retriever。
        """
        domains = self._registry.list_domains()
        # Phase 3: domains = self._access_guard.filter(domains, user)
        decision = self._router.route(query, domains)
        retriever = self._registry.get_retriever(decision.domain_id)
        return self._pipeline.retrieve(query, retriever)

    # ------------------------------------------------------------------
    # 元数据
    # ------------------------------------------------------------------

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
# 模块级单例
# ------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_knowledge_service() -> KnowledgeService:
    """获取全局唯一的 KnowledgeService 实例。"""
    return KnowledgeService()
