"""SelfCorrectionController — 检索自纠正闭环。

验证 → 失败 → 查询修正 → 重新检索 → 再验证。
不接触 AccessGuard / Router / Registry / Retriever 内部。
"""

from __future__ import annotations

import logging
from typing import Callable

from langchain_core.documents import Document

from agent.verifier import RetrievalVerifier, VerificationResult

logger = logging.getLogger(__name__)


class SelfCorrectionController:
    """检索自纠正控制器。

    将单次验证升级为闭环：
    1. 验证检索结果
    2. 如果未通过 → 根据 missing_topics 修正查询 → 重新检索 → 再验证
    3. 循环直到通过或达到最大迭代次数

    异常安全：任何环节失败都放行当前结果，不阻塞检索。
    """

    def __init__(
        self,
        verifier: RetrievalVerifier,
        *,
        enabled: bool = False,
        max_iterations: int = 2,
        min_score: float = 0.5,
    ) -> None:
        """Args:
            verifier: RetrievalVerifier 实例。
            enabled: 是否启用自纠正。
            max_iterations: 最大迭代次数（含初次验证）。
            min_score: 低于此分数触发修正。
        """
        self._verifier = verifier
        self._enabled = enabled
        self._max_iterations = max_iterations
        self._min_score = min_score

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        initial_docs: list[Document],
        retriever_fn: Callable[[str], list[Document]],
    ) -> tuple[list[Document], VerificationResult | None]:
        """执行自纠正检索闭环。

        Args:
            query: 用户原始查询。
            initial_docs: 初次检索的文档。
            retriever_fn: 重新检索函数，签名为 (revised_query) -> list[Document]。

        Returns:
            (final_docs, result): 最终文档和最后一次验证结果。
            result 为 None 表示未启用自纠正。
        """
        if not self._enabled:
            return initial_docs, None

        docs = initial_docs
        last_result: VerificationResult | None = None

        for iteration in range(self._max_iterations):
            # 1. 验证
            try:
                result = self._verifier.verify(query, docs)
                last_result = result
            except Exception as exc:
                logger.warning("验证异常，放行当前结果: %s", exc)
                self._tag_docs(docs, VerificationResult(
                    passed=True, score=0.5,
                    reason=f"验证异常: {exc}",
                ))
                return docs, None

            # 2. 通过 → 返回
            if result.passed and result.score >= self._min_score:
                self._tag_docs(docs, result)
                return docs, result

            # 3. 未通过 + 已达上限 → 返回当前
            if iteration >= self._max_iterations - 1:
                self._tag_docs(docs, result)
                return docs, result

            # 4. 未通过 → 修正查询 → 重新检索
            revised = self._rewrite_query(query, result)
            logger.info("自纠正第 %d 次: %s", iteration + 1, revised)
            try:
                new_docs = retriever_fn(revised)
            except Exception as exc:
                logger.warning("重新检索失败，返回当前结果: %s", exc)
                self._tag_docs(docs, result)
                return docs, result

            if not new_docs:
                self._tag_docs(docs, result)
                return docs, result

            docs = new_docs

        # 循环结束（不应到达这里）
        if last_result:
            self._tag_docs(docs, last_result)
        return docs, last_result

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    @staticmethod
    def _rewrite_query(original: str, result: VerificationResult) -> str:
        """根据 missing_topics 生成修正后的查询。"""
        if result.missing_topics:
            topics = " ".join(result.missing_topics)
            return f"{original} {topics}"
        return f"{original}（请从不同角度检索）"

    @staticmethod
    def _tag_docs(
        docs: list[Document],
        result: VerificationResult,
    ) -> None:
        """将验证结果注入文档 metadata。"""
        meta = {
            "_verification_passed": result.passed,
            "_verification_score": result.score,
            "_verification_missing": result.missing_topics,
            "_verification_reason": result.reason,
        }
        for doc in docs:
            doc.metadata.update(meta)
