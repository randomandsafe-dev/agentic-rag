"""VerificationController — 检索验证控制层。

编排 RetrievalVerifier 的调用、retry 控制、结果元数据注入。
不接触 AccessGuard / Router / Registry / Retriever 内部逻辑。
"""

from __future__ import annotations

import logging
from typing import Callable

from langchain_core.documents import Document

from agent.verifier import RetrievalVerifier, VerificationResult

logger = logging.getLogger(__name__)


class VerificationController:
    """检索验证控制器。

    职责：
    - 判断是否启用验证（配置驱动）
    - 调用 RetrievalVerifier
    - 控制重试次数
    - 验证失败时触发重新检索
    - 异常时安全放行
    """

    def __init__(
        self,
        verifier: RetrievalVerifier,
        *,
        enabled: bool = False,
        max_retry: int = 2,
        min_score: float = 0.5,
    ) -> None:
        """Args:
            verifier: RetrievalVerifier 实例。
            enabled: 是否启用验证。
            max_retry: 最大重试次数（0 = 仅验证一次，不重试）。
            min_score: 低于此分数触发重试。
        """
        self._verifier = verifier
        self._enabled = enabled
        self._max_retry = max_retry
        self._min_score = min_score

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def verify_and_retry(
        self,
        query: str,
        initial_docs: list[Document],
        retriever_fn: Callable[[str], list[Document]],
    ) -> tuple[list[Document], VerificationResult | None]:
        """验证检索结果，按需重试。

        Args:
            query: 原始查询。
            initial_docs: 初次检索到的文档。
            retriever_fn: 重新检索的函数，签名为 (revised_query) -> list[Document]。

        Returns:
            (final_docs, result): 最终文档和验证结果。result 为 None 表示未执行验证。
        """
        if not self._enabled:
            return initial_docs, None

        docs = initial_docs
        result: VerificationResult | None = None

        for attempt in range(self._max_retry + 1):
            try:
                result = self._verifier.verify(query, docs)
            except Exception as exc:
                logger.warning("验证调用异常，放行原结果: %s", exc)
                self._attach_metadata(docs, VerificationResult(
                    passed=True, score=0.5,
                    reason=f"验证异常，默认放行: {exc}",
                ))
                return docs, None

            if result.passed and result.score >= self._min_score:
                self._attach_metadata(docs, result)
                return docs, result

            # 未通过 → 重试
            if attempt < self._max_retry:
                revised = _build_retry_query(query, result)
                logger.info("验证未通过，第 %d 次重试: %s", attempt + 1, revised)
                try:
                    docs = retriever_fn(revised)
                except Exception as exc:
                    logger.warning("重试检索失败: %s", exc)
                    break
                if not docs:
                    break

        # 重试耗尽 → 放行最后一次结果
        if result is not None:
            self._attach_metadata(docs, result)
        return docs, result

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    @staticmethod
    def _attach_metadata(
        docs: list[Document],
        result: VerificationResult,
    ) -> None:
        """将验证结果注入每个文档的 metadata。"""
        meta = {
            "_verification_passed": result.passed,
            "_verification_score": result.score,
            "_verification_missing": result.missing_topics,
        }
        for doc in docs:
            doc.metadata.update(meta)


# ------------------------------------------------------------------
# 辅助
# ------------------------------------------------------------------


def _build_retry_query(original: str, result: VerificationResult) -> str:
    """根据验证结果构建重试查询。"""
    if result.missing_topics:
        topics = " ".join(result.missing_topics)
        return f"{original} {topics}"
    return f"{original} (补充检索)"
