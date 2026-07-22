"""KnowledgeService —— Agent 层的统一检索入口。

Agent 只通过此类访问知识库，不直接感知 Collection / Retriever / Router 等底层细节。
Phase 3：接入 AccessGuard 实现权限过滤。
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document

from config import settings
import os
from pathlib import Path

import yaml

from config import ROOT_DIR
from knowledge.access import AccessGuard, UserContext
from knowledge.concurrent import ConcurrentRetriever
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

        # Phase 6: verification controller
        from agent.verifier import RetrievalVerifier
        from agent.verification.controller import VerificationController
        vconfig = _load_verification_config()
        self._verification = VerificationController(
            RetrievalVerifier(self._llm),
            enabled=vconfig.get("enabled", False),
            max_retry=vconfig.get("max_retry", 2),
            min_score=vconfig.get("min_score", 0.5),
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
        domains = self._registry.list_domains()

        if user is not None:
            domains = self._access_guard.filter_domains(user, domains)

        decision = self._router.route(query, domains)

        # 初次检索
        docs = self._retrieve_docs(query, decision)

        # Phase 6: 检索验证 + 按需重试
        docs, _ = self._verification.verify_and_retry(
            query, docs,
            retriever_fn=lambda q: self._retrieve_docs(q, decision),
        )
        return docs

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
            return self._concurrent.search(query, retrievers)

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


def _load_verification_config() -> dict:
    """加载 config/verification.yaml。文件不存在时返回默认值。"""
    config_path = ROOT_DIR / "config" / "verification.yaml"
    if not config_path.exists():
        return {"enabled": False, "max_retry": 2, "min_score": 0.5}
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
        return config.get("verification", {})
    except Exception:
        return {"enabled": False, "max_retry": 2, "min_score": 0.5}


# ------------------------------------------------------------------
# 模块级单例
# ------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_knowledge_service() -> KnowledgeService:
    """获取全局唯一的 KnowledgeService 实例。"""
    return KnowledgeService()
