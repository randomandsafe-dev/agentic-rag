"""
检索增强管道：查询改写 + 相关性判断 + 重试机制。

采用策略模式设计，相关性判断策略可插拔替换，与现有代码完全解耦。
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings
from prompts import (
    HYBRID_JUDGE_SYSTEM,
    RELEVANCE_JUDGE_SYSTEM,
    RELEVANCE_JUDGE_USER,
    REWRITE_RETRY_PROMPT,
    REWRITE_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


# ============================================================
# 辅助函数
# ============================================================

def _create_model(model_name: str | None = None) -> ChatOpenAI:
    """创建 LLM 实例；可指定不同模型用于改写/判断。"""
    return ChatOpenAI(
        model=model_name or settings.rewrite_model or settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=0,
    )


def _parse_json_array(raw: str, fallback_count: int) -> list[int]:
    """容错解析 LLM 返回的 JSON 数组，失败时返回全 1（保守：视为弱相关）。"""
    # 尝试直接解析
    try:
        parsed = json.loads(raw.strip())
        if isinstance(parsed, list) and all(isinstance(v, (int, float)) for v in parsed):
            return [int(v) for v in parsed]
    except json.JSONDecodeError:
        pass

    # 尝试提取 [...] 内容
    match = re.search(r"\[([^\]]*)\]", raw)
    if match:
        try:
            values = [int(v.strip()) for v in match.group(1).split(",") if v.strip()]
            if values:
                return values
        except (ValueError, IndexError):
            pass

    # 降级：返回全 1（保守策略，不丢弃任何文档）
    logger.warning("无法解析相关性判断结果，降级为全部弱相关(1分): %s", raw[:200])
    return [1] * fallback_count


def _format_docs_for_judge(documents: list[Document]) -> str:
    """将文档列表格式化为判断提示词中的文本块。"""
    parts: list[str] = []
    for i, doc in enumerate(documents, start=1):
        content = doc.page_content[:800]  # 截断过长文档
        parts.append(f"--- 文档 {i} ---\n{content}")
    return "\n\n".join(parts)


# ============================================================
# 抽象策略接口
# ============================================================

class RelevanceStrategy(ABC):
    """相关性判断的抽象接口。实现新策略只需继承此类。"""

    @abstractmethod
    def judge(self, query: str, documents: list[Document]) -> list[int]:
        """返回每个文档的相关性分数 (0-3)，顺序与输入一致。"""
        ...

    def has_relevant(self, scores: list[int], threshold: int) -> bool:
        """判断是否存在相关文档（任一文档分数 >= 阈值）。"""
        return any(s >= threshold for s in scores)


# ============================================================
# 策略 A：LLM 语义判断（默认，最准确）
# ============================================================

class LLMRelevanceJudge(RelevanceStrategy):
    """使用 LLM 逐一判断文档与查询的语义相关性。"""

    def __init__(self, model: ChatOpenAI | None = None) -> None:
        self._model = model

    @property
    def model(self) -> ChatOpenAI:
        if self._model is None:
            self._model = _create_model()
        return self._model

    def judge(self, query: str, documents: list[Document]) -> list[int]:
        if not documents:
            return []

        docs_text = _format_docs_for_judge(documents)
        user_prompt = RELEVANCE_JUDGE_USER.format(query=query, documents_text=docs_text)

        try:
            response = self.model.invoke([
                SystemMessage(content=RELEVANCE_JUDGE_SYSTEM),
                HumanMessage(content=user_prompt),
            ])
            raw = response.content if isinstance(response.content, str) else str(response.content)
            scores = _parse_json_array(raw, len(documents))
            # 补/截到正确长度
            if len(scores) < len(documents):
                scores += [1] * (len(documents) - len(scores))
            return scores[:len(documents)]
        except Exception as exc:
            logger.warning("LLM 相关性判断失败，降级为全部分相关(2分): %s", exc)
            return [2] * len(documents)


# ============================================================
# 策略 B：向量相似度分数（零额外成本）
# ============================================================

class VectorScoreJudge(RelevanceStrategy):
    """
    使用 Chroma 自带的 L2/IP 距离归一化为 0-3 分。
    对传入的文档列表做精确评分（通过内容匹配回搜索结果）。
    适合文档类型单一、向量质量高的场景。
    """

    def __init__(self, vector_store: Chroma) -> None:
        self._store = vector_store

    def judge(self, query: str, documents: list[Document]) -> list[int]:
        """通过 Chroma similarity_search_with_score 获取分数，匹配回输入文档。"""
        if not documents:
            return []

        try:
            results = self._store.similarity_search_with_score(
                query, k=max(len(documents), 4)
            )
        except Exception as exc:
            logger.warning("向量分数检索失败，降级为全部分相关(2分): %s", exc)
            return [2] * len(documents)

        if not results:
            return [0] * len(documents)

        # 用内容前 120 字符做键，将搜索结果分数映射回输入文档
        def _key(doc: Document) -> str:
            return doc.page_content[:120].strip()

        score_map: dict[str, float] = {}
        for doc, score in results:
            score_map[_key(doc)] = score

        # 归一化参数
        all_scores = [s for _, s in results]
        min_d, max_d = min(all_scores), max(all_scores)

        def _normalize(distance: float) -> int:
            """L2 距离 → 0-3 分（越小越相关）。"""
            if max_d - min_d < 1e-8:
                return 2
            normalized = 1.0 - (distance - min_d) / (max_d - min_d)  # 0~1
            return round(normalized * 3)  # 0~3

        scores: list[int] = []
        for doc in documents:
            k = _key(doc)
            if k in score_map:
                scores.append(_normalize(score_map[k]))
            else:
                scores.append(1)  # 未匹配 → 保守给弱相关

        return scores


# ============================================================
# 策略 C：混合判断（向量初筛 + LLM 终判）
# ============================================================

class HybridJudge(RelevanceStrategy):
    """
    先用向量分数筛掉明显无关的文档，剩余文档送 LLM 终判。
    平衡成本与准确性。
    """

    def __init__(
        self,
        vector_store: Chroma,
        vector_cutoff: float = 0.3,
        model: ChatOpenAI | None = None,
    ) -> None:
        self._store = vector_store
        self._vector_cutoff = vector_cutoff  # 归一化分数 < 此值的文档直接判 0
        self._model = model
        self._llm_judge = LLMRelevanceJudge(model)

    def judge(self, query: str, documents: list[Document]) -> list[int]:
        if not documents:
            return []

        # 第一轮：向量分数初筛
        vector_judge = VectorScoreJudge(self._store)
        vector_scores = vector_judge.judge(query, documents)

        # 找出需要 LLM 终判的文档（向量分数在中间地带）
        llm_indices: list[int] = []
        final_scores = [0] * len(documents)

        for i, score in enumerate(vector_scores):
            if score >= 2:  # 向量高分 → 直接采信
                final_scores[i] = score
            elif score >= self._vector_cutoff:  # 中间地带 → 需 LLM 确认
                llm_indices.append(i)
            else:  # 向量极低分 → 直接判 0
                final_scores[i] = 0

        # 第二轮：LLM 对中间地带的文档做精准判断
        if llm_indices:
            llm_docs = [documents[i] for i in llm_indices]
            llm_scores = self._llm_judge.judge(query, llm_docs)
            for idx, score in zip(llm_indices, llm_scores):
                final_scores[idx] = score

        return final_scores


# ============================================================
# 策略工厂
# ============================================================

def create_relevance_judge(
    vector_store: Chroma, model: ChatOpenAI | None = None
) -> RelevanceStrategy:
    """根据配置创建相关性判断策略，一行配置即可切换。"""
    strategy = settings.relevance_strategy
    if strategy == "vector":
        return VectorScoreJudge(vector_store)
    elif strategy == "hybrid":
        return HybridJudge(vector_store, model=model)
    else:  # "llm" 或非法值均默认 LLM
        return LLMRelevanceJudge(model)


# ============================================================
# 查询改写器
# ============================================================

class QueryRewriter:
    """使用 LLM 改写用户查询，使其更适合向量检索。"""

    def __init__(self, model: ChatOpenAI | None = None) -> None:
        self._model = model

    @property
    def model(self) -> ChatOpenAI:
        if self._model is None:
            self._model = _create_model()
        return self._model

    def rewrite(self, original_query: str) -> str:
        """首次改写：直接优化检索关键词。"""
        if not settings.rewrite_enabled:
            return original_query
        try:
            response = self.model.invoke([
                SystemMessage(content=REWRITE_SYSTEM_PROMPT),
                HumanMessage(content=original_query),
            ])
            result = response.content if isinstance(response.content, str) else str(response.content)
            return result.strip() or original_query
        except Exception as exc:
            logger.warning("查询改写失败，使用原始查询: %s", exc)
            return original_query

    def rewrite_retry(self, original_query: str, last_rewrite: str) -> str:
        """重试改写：要求从不同角度重新表述。"""
        if not settings.rewrite_enabled:
            return original_query
        try:
            prompt = REWRITE_RETRY_PROMPT.format(
                original_query=original_query, last_rewrite=last_rewrite
            )
            response = self.model.invoke([
                SystemMessage(content=REWRITE_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            result = response.content if isinstance(response.content, str) else str(response.content)
            return result.strip() or original_query
        except Exception as exc:
            logger.warning("重试改写失败，使用原始查询: %s", exc)
            return original_query


# ============================================================
# 检索管道编排器
# ============================================================

class RetrievalPipeline:
    """
    编排完整的检索增强流程：

    1. 改写查询 → 2. 向量检索 → 3. 相关性判断
       └─ 不相关且未达上限 → 换个角度改写，重试
       └─ 超限 → 降级返回最后一次检索结果
    """

    def __init__(
        self,
        vector_store: Chroma,
        rewriter: QueryRewriter | None = None,
        judge: RelevanceStrategy | None = None,
    ) -> None:
        self._store = vector_store
        self._retriever = vector_store.as_retriever(
            search_kwargs={"k": settings.top_k}
        )
        self._rewriter = rewriter or QueryRewriter()
        self._judge = judge or create_relevance_judge(vector_store)

    def retrieve(self, query: str) -> list[Document]:
        """
        执行检索，含改写、判断、重试。
        返回最终相关的文档列表。
        """
        max_retries = settings.max_retries
        threshold = settings.relevance_threshold
        rewrite_enabled = settings.rewrite_enabled
        judge_enabled = settings.relevance_judge_enabled

        current_rewrite = query
        all_attempts_docs: list[Document] = []

        for attempt in range(max_retries + 1):
            # ---- 1. 改写查询 ----
            if attempt == 0:
                current_rewrite = self._rewriter.rewrite(query) if rewrite_enabled else query
            else:
                current_rewrite = self._rewriter.rewrite_retry(query, current_rewrite)

            logger.info("检索第 %d 次，改写查询: %s", attempt + 1, current_rewrite)

            # ---- 2. 执行检索 ----
            documents = self._retriever.invoke(current_rewrite)
            if not documents:
                logger.info("第 %d 次检索无结果", attempt + 1)
                continue

            # ---- 3. 相关性判断 ----
            if not judge_enabled:
                return documents  # 不判断 → 直接返回

            scores = self._judge.judge(query, documents)
            logger.info("第 %d 次相关分数: %s", attempt + 1, scores)

            if self._judge.has_relevant(scores, threshold):
                # 返回分数 >= 阈值的文档
                relevant = [d for d, s in zip(documents, scores) if s >= threshold]
                logger.info("第 %d 次检索成功，%d/%d 个文档相关",
                            attempt + 1, len(relevant), len(documents))
                return relevant

            # 保存本轮结果用于最终降级
            all_attempts_docs = documents

        # ---- 降级：返回最后一次检索结果 ----
        logger.warning("已达最大重试次数(%d)，降级返回最后%d个文档",
                       max_retries, len(all_attempts_docs))
        return all_attempts_docs


# ============================================================
# 管道缓存（与现有 get_retriever/get_embeddings 模式一致）
# ============================================================

@lru_cache(maxsize=1)
def get_retrieval_pipeline() -> RetrievalPipeline:
    """创建并缓存检索管道单例。"""
    settings.validate()
    if not settings.persist_dir.exists():
        raise RuntimeError("尚未创建知识库。请先运行：python ingest.py")

    # 复用 embedding function（从 rag_agent 导入以避免重复创建）
    from rag_agent import get_embeddings

    store = Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.persist_dir),
        embedding_function=get_embeddings(),
    )
    return RetrievalPipeline(vector_store=store)
