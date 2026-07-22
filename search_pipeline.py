"""搜索管道：查询改写 + 检索 + 相关性判断 + 重试。

纯编排器，不依赖 Chroma / HybridRetriever / Registry / Router。
只依赖 Duck Typing 接口：retriever.search(query) -> list[Document]。
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod

from langchain_core.documents import Document

from config import settings
from prompts import (
    RELEVANCE_JUDGE_SYSTEM,
    RELEVANCE_JUDGE_USER,
    REWRITE_RETRY_PROMPT,
    REWRITE_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


# ============================================================
# 辅助函数
# ============================================================


def _parse_json_array(raw: str, fallback_count: int) -> list[int]:
    """容错解析 LLM 返回的 JSON 数组，失败时返回全 1（保守：视为弱相关）。"""
    try:
        parsed = json.loads(raw.strip())
        if isinstance(parsed, list) and all(isinstance(v, (int, float)) for v in parsed):
            return [int(v) for v in parsed]
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[([^\]]*)\]", raw)
    if match:
        try:
            values = [int(v.strip()) for v in match.group(1).split(",") if v.strip()]
            if values:
                return values
        except (ValueError, IndexError):
            pass

    logger.warning("无法解析相关性判断结果，降级为全部弱相关(1分): %s", raw[:200])
    return [1] * fallback_count


def _format_docs_for_judge(documents: list[Document]) -> str:
    """将文档列表格式化为判断提示词中的文本块。"""
    parts: list[str] = []
    for i, doc in enumerate(documents, start=1):
        content = doc.page_content[:800]
        parts.append(f"--- 文档 {i} ---\n{content}")
    return "\n\n".join(parts)


# ============================================================
# 抽象策略接口
# ============================================================


class RelevanceStrategy(ABC):
    """相关性判断的抽象接口。"""

    @abstractmethod
    def judge(self, query: str, documents: list[Document]) -> list[int]:
        """返回每个文档的相关性分数 (0-3)，顺序与输入一致。"""
        ...

    def has_relevant(self, scores: list[int], threshold: int) -> bool:
        """判断是否存在相关文档（任一文档分数 >= 阈值）。"""
        return any(s >= threshold for s in scores)


# ============================================================
# LLM 相关性判断
# ============================================================


class LLMRelevanceJudge(RelevanceStrategy):
    """使用 LLM 逐一判断文档与查询的语义相关性。

    LLM 实例由调用方注入，Judge 不负责创建。
    """

    def __init__(self, llm) -> None:
        """Args:
            llm: LangChain ChatModel 实例（由 llm_factory.create_llm 创建）。
        """
        self._llm = llm

    def judge(self, query: str, documents: list[Document]) -> list[int]:
        if not documents:
            return []

        docs_text = _format_docs_for_judge(documents)
        user_prompt = RELEVANCE_JUDGE_USER.format(query=query, documents_text=docs_text)

        try:
            response = self._llm.invoke([
                {"role": "system", "content": RELEVANCE_JUDGE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ])
            raw = response.content if hasattr(response, "content") else str(response)
            scores = _parse_json_array(raw, len(documents))
            if len(scores) < len(documents):
                scores += [1] * (len(documents) - len(scores))
            return scores[:len(documents)]
        except Exception as exc:
            logger.warning("LLM 相关性判断失败，降级为全部分相关(2分): %s", exc)
            return [2] * len(documents)


# ============================================================
# 查询改写器
# ============================================================


class QueryRewriter:
    """使用 LLM 改写用户查询，使其更适合检索。

    LLM 实例由调用方注入，Rewriter 不负责创建。
    """

    def __init__(self, llm) -> None:
        """Args:
            llm: LangChain ChatModel 实例（由 llm_factory.create_llm 创建）。
        """
        self._llm = llm

    def rewrite(self, original_query: str) -> str:
        """首次改写：优化检索关键词。"""
        try:
            response = self._llm.invoke([
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": original_query},
            ])
            result = response.content if hasattr(response, "content") else str(response)
            return result.strip() or original_query
        except Exception as exc:
            logger.warning("查询改写失败，使用原始查询: %s", exc)
            return original_query

    def rewrite_retry(self, original_query: str, last_rewrite: str) -> str:
        """重试改写：要求从不同角度重新表述。"""
        try:
            prompt = REWRITE_RETRY_PROMPT.format(
                original_query=original_query, last_rewrite=last_rewrite
            )
            response = self._llm.invoke([
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ])
            result = response.content if hasattr(response, "content") else str(response)
            return result.strip() or original_query
        except Exception as exc:
            logger.warning("重试改写失败，使用原始查询: %s", exc)
            return original_query


# ============================================================
# 搜索管道编排器
# ============================================================


class SearchPipeline:
    """编排完整的搜索增强流程。

    不持有 Retriever（Stateless），每次调用由外部传入。
    不依赖 Chroma / HybridRetriever / Registry / Router。

    流程（各阶段可独立关闭）：
        Rewrite → Retrieve → Judge → Retry → 返回 Documents
    """

    def __init__(
        self,
        rewriter: QueryRewriter | None = None,
        judge: RelevanceStrategy | None = None,
        *,
        rewrite_enabled: bool = settings.rewrite_enabled,
        judge_enabled: bool = settings.relevance_judge_enabled,
        max_retries: int = settings.max_retries,
        relevance_threshold: int = settings.relevance_threshold,
    ) -> None:
        """初始化搜索管道。

        Args:
            rewriter: 查询改写器（可选，改写关闭时不需要）。
            judge: 相关性判断策略（可选，判断关闭时不需要）。
            rewrite_enabled: 是否启用查询改写。
            judge_enabled: 是否启用相关性判断。
            max_retries: 最大重试次数。
            relevance_threshold: 相关性分数阈值 (0-3)。
        """
        self._rewriter = rewriter
        self._judge = judge
        self._rewrite_enabled = rewrite_enabled
        self._judge_enabled = judge_enabled
        self._max_retries = max_retries
        self._threshold = relevance_threshold

    # ------------------------------------------------------------------
    # 检索入口
    # ------------------------------------------------------------------

    def retrieve(self, query: str, retriever) -> list[Document]:
        """执行搜索增强流程。

        Args:
            query: 用户原始查询。
            retriever: 任意实现了 search(query) -> list[Document] 的对象。

        Returns:
            经过可选改写、检索、判断、重试后的文档列表。
        """
        # 快速路径：所有增强功能关闭 → 直接检索
        if not self._rewrite_enabled and not self._judge_enabled:
            return retriever.search(query)

        current_query = query
        last_docs: list[Document] = []

        for attempt in range(self._max_retries + 1):
            # ---- 1. 查询改写 ----
            if self._rewrite_enabled and self._rewriter is not None:
                try:
                    if attempt == 0:
                        current_query = self._rewriter.rewrite(query)
                    else:
                        current_query = self._rewriter.rewrite_retry(query, current_query)
                except Exception:
                    current_query = query  # 改写失败 → 回退原始查询

            logger.info("检索第 %d 次，查询: %s", attempt + 1, current_query)

            # ---- 2. 执行检索 ----
            try:
                documents = retriever.search(current_query)
            except Exception:
                # 改写后的查询可能产生异常 → 用原始查询重试
                documents = retriever.search(query)

            if not documents:
                logger.info("第 %d 次检索无结果", attempt + 1)
                continue

            # ---- 3. 相关性判断 ----
            if not self._judge_enabled or self._judge is None:
                return documents

            try:
                scores = self._judge.judge(query, documents)
            except Exception:
                logger.warning("相关性判断异常，返回全部文档")
                return documents

            logger.info("第 %d 次相关分数: %s", attempt + 1, scores)

            if self._judge.has_relevant(scores, self._threshold):
                relevant = [d for d, s in zip(documents, scores) if s >= self._threshold]
                logger.info(
                    "第 %d 次检索成功，%d/%d 个文档相关",
                    attempt + 1, len(relevant), len(documents),
                )
                return relevant

            last_docs = documents

        # ---- 降级：返回最后一次检索结果 ----
        logger.warning(
            "已达最大重试次数(%d)，降级返回最后 %d 个文档",
            self._max_retries, len(last_docs),
        )
        return last_docs if last_docs else retriever.search(query)
