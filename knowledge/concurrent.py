"""ConcurrentRetriever — 多知识库并行检索。

接收多个 domain_id → retriever 映射，使用线程池并发执行检索并合并结果。
不接触 AccessGuard / Router / Registry。
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class ConcurrentRetriever:
    """多 KB 并行检索器。

    职责单一：接收 retrievers 字典，并发搜索，合并去重结果。
    不参与权限判断、路由选择、pipeline 编排。
    """

    def search(
        self,
        query: str,
        retrievers: dict[str, object],
        top_k: int = 5,
    ) -> list[Document]:
        """对多个 retriever 并发执行 search(query)，合并结果。

        Args:
            query: 检索查询。
            retrievers: {domain_id: retriever} 映射，retriever 须实现 search(query)。
            top_k: 最终返回的文档数量上限。

        Returns:
            合并去重后的文档列表，每个文档的 metadata 中增加 _domain 标记来源。
        """
        if not retrievers:
            return []

        if len(retrievers) == 1:
            # 单 KB → 直接返回，零并发开销
            _, retriever = next(iter(retrievers.items()))
            docs = retriever.search(query)
            for d in docs:
                d.metadata.setdefault("_domain", list(retrievers.keys())[0])
            return docs[:top_k]

        # 多 KB → 线程池并发
        all_docs: list[Document] = []
        max_workers = min(len(retrievers), 8)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for domain_id, retriever in retrievers.items():
                future = executor.submit(self._safe_search, query, retriever, domain_id)
                futures[future] = domain_id

            for future in as_completed(futures):
                domain_id = futures[future]
                try:
                    docs = future.result()
                    all_docs.extend(docs)
                except Exception as exc:
                    logger.warning("KB '%s' 检索失败: %s", domain_id, exc)

        # 去重 + 截断
        return self._deduplicate(all_docs)[:top_k]

    @staticmethod
    def _safe_search(
        query: str,
        retriever,
        domain_id: str,
    ) -> list[Document]:
        """安全检索：捕获异常，标记来源 domain。"""
        docs = retriever.search(query)
        for d in docs:
            d.metadata["_domain"] = domain_id
        return docs

    @staticmethod
    def _deduplicate(docs: list[Document]) -> list[Document]:
        """按内容去重，保留首次出现的文档。"""
        seen: set[str] = set()
        unique: list[Document] = []
        for doc in docs:
            key = doc.page_content[:200]
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        return unique
