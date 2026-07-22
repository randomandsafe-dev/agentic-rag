"""ConcurrentPipeline — 多 KB 并发检索 + 统一 Pipeline 后处理。

将 ConcurrentRetriever 的合并结果重新接入 SearchPipeline，
保证多 KB 检索也经过 reranker / relevance judge。
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document

from knowledge.concurrent import ConcurrentRetriever

logger = logging.getLogger(__name__)


# ============================================================
# PassthroughRetriever
# ============================================================


class _PassthroughRetriever:
    """适配器：将预检索的文档伪装为 retriever.search(query) 的返回值。

    Duck Typing：实现 search(query) -> list[Document] 接口，
    供 SearchPipeline.retrieve() 消费。
    """

    def __init__(self, documents: list[Document]) -> None:
        self._docs = documents

    def search(self, query: str) -> list[Document]:
        return list(self._docs)


# ============================================================
# ConcurrentPipeline
# ============================================================


class ConcurrentPipeline:
    """多 KB 并发检索 + Pipeline 统一后处理。

    不接触 AccessGuard / Router / Registry。
    """

    def __init__(self) -> None:
        self._concurrent = ConcurrentRetriever()

    def retrieve(
        self,
        query: str,
        retrievers: dict[str, object],
        pipeline,
    ) -> list[Document]:
        """并发检索多 KB → 合并去重 → 通过 SearchPipeline 后处理。

        Args:
            query: 检索查询。
            retrievers: {domain_id: retriever} 映射。
            pipeline: SearchPipeline 实例，提供 retrieve(query, retriever) 方法。

        Returns:
            经过 Pipeline 处理后的最终文档列表。
        """
        # 1. 并发检索 + 合并去重
        raw_docs = self._concurrent.search(query, retrievers)

        if not raw_docs:
            return []

        # 2. 通过 SearchPipeline 后处理（reranker + judge）
        passthrough = _PassthroughRetriever(raw_docs)
        processed = pipeline.retrieve(query, passthrough)

        # 3. 保留 _domain metadata（Pipeline 可能返回新 Document 对象）
        self._restore_domain_metadata(raw_docs, processed)

        return processed

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    @staticmethod
    def _restore_domain_metadata(
        raw: list[Document],
        processed: list[Document],
    ) -> None:
        """将并发检索的 _domain 标记恢复到 Pipeline 处理后的文档上。"""
        # 按内容前 200 字符建立映射
        domain_map: dict[str, str] = {}
        for doc in raw:
            key = doc.page_content[:200]
            if "_domain" in doc.metadata:
                domain_map[key] = doc.metadata["_domain"]

        for doc in processed:
            key = doc.page_content[:200]
            if key in domain_map:
                doc.metadata.setdefault("_domain", domain_map[key])
