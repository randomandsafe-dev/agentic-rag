"""KnowledgeService —— Agent 层的统一检索入口。

Agent 只通过此类访问知识库，不直接感知 Collection / Retriever / Router 等底层细节。
Phase 3：接入 AccessGuard 实现权限过滤。
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document

from config import settings
import time

from runtime_loader import load_runtime_config
from knowledge.access import AccessGuard, UserContext
from knowledge.concurrent import ConcurrentRetriever
from knowledge.concurrent_pipeline import ConcurrentPipeline
from knowledge.metrics import MetricsCollector, NoopMetricsCollector, RetrievalMetrics
from knowledge.registry import KnowledgeBaseRegistry
from knowledge.router import KnowledgeRouter, KeywordRouter, LLMRouter, RoutingDecision
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
        self._concurrent_pipeline = ConcurrentPipeline()

        # Phase 6: runtime config
        rt = load_runtime_config()
        self._metrics = (
            MetricsCollector() if rt.metrics.enabled else NoopMetricsCollector()
        )

        # Phase 6: self-correction loop
        from agent.verifier import RetrievalVerifier
        from agent.self_correction.controller import SelfCorrectionController
        self._self_correction = SelfCorrectionController(
            RetrievalVerifier(self._llm),
            enabled=rt.self_correction.enabled,
            max_iterations=rt.self_correction.max_iterations,
            min_score=rt.verification.min_score,
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

    def search(
        self,
        query: str,
        user: UserContext | None = None,
    ) -> list[Document]:
        """检索与查询最相关的文档。

        Args:
            query: 用户查询。
            user: 可选的用户上下文；None 时跳过权限过滤。

        调用链：AccessGuard → Router → Registry → SearchPipeline → Retriever → Verifier。
        """
        import uuid
        started = time.time()
        m = RetrievalMetrics(
            request_id=uuid.uuid4().hex[:12],
            query=query[:100],
            user_id=user.user_id if user else None,
        )

        try:
            domains = self._registry.list_domains()

            if user is not None:
                domains = self._access_guard.filter_domains(user, domains)

            decision = self._router.route(query, domains)
            m.domain_ids = decision.domain_ids

            # 初次检索
            docs = self._retrieve_docs(query, decision)

            # Phase 6: 自纠正闭环
            docs, vresult = self._self_correction.run(
                query, docs,
                retriever_fn=lambda q: self._retrieve_docs(q, decision),
            )

            m.retrieval_count = len(docs)
            if vresult is not None:
                m.verification_score = vresult.score
                m.verification_passed = vresult.passed

            return docs
        except Exception as exc:
            m.error = str(exc)
            raise
        finally:
            m.latency_ms = (time.time() - started) * 1000
            self._metrics.record(m)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _retrieve_docs(
        self,
        query: str,
        decision: RoutingDecision,
    ) -> list[Document]:
        """根据路由决策执行检索。"""
        if len(decision.domain_ids) > 1:
            retrievers = {
                did: self._registry.get_retriever(did)
                for did in decision.domain_ids
            }
            return self._concurrent_pipeline.retrieve(
                query, retrievers, self._pipeline,
            )

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
# 配置加载
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# 模块级单例
# ------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_knowledge_service() -> KnowledgeService:
    """获取全局唯一的 KnowledgeService 实例。"""
    return KnowledgeService()
