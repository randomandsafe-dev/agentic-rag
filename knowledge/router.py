"""KnowledgeRouter —— 根据用户 query 将检索请求路由到最相关的知识库。

Phase 2：只负责 query → domain_id 映射，不直接访问 Chroma / Retriever。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from prompts import ROUTER_PROMPT_TEMPLATE

from knowledge.domain import KnowledgeDomain


# ------------------------------------------------------------------
# RoutingDecision
# ------------------------------------------------------------------


@dataclass
class RoutingDecision:
    """Router 的路由结果。

    Attributes:
        domain_id: 主目标知识库 ID（向后兼容）。
        domain_ids: 所有候选 domain ID（Phase 5 并发检索）。
        confidence: 路由置信度 0.0 ~ 1.0。
        strategy: 使用的路由策略标识（keyword / llm / single / fallback）。
    """

    domain_id: str = ""
    domain_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    strategy: str = "fallback"


# ------------------------------------------------------------------
# RouterStrategy (ABC)
# ------------------------------------------------------------------


class RouterStrategy(ABC):
    """路由策略抽象基类。

    每种策略接收 query 与 domain 元数据，返回 RoutingDecision 或 None。
    None 表示策略无法做出判断，由 KnowledgeRouter 统一 fallback。

    策略实现不得访问 Chroma / Retriever / Registry。
    """

    @abstractmethod
    def route(self, query: str, domains: list[KnowledgeDomain]) -> RoutingDecision | None:
        """根据 query 选择最相关的 domain。

        Args:
            query: 用户原始查询。
            domains: 全部可用 domain 的元数据列表。

        Returns:
            RoutingDecision 如果匹配成功，否则 None。
        """
        ...


# ------------------------------------------------------------------
# KeywordRouter
# ------------------------------------------------------------------


class KeywordRouter(RouterStrategy):
    """基于关键词匹配的零 LLM 路由策略。

    使用 retrieval.tokenize() 保持与 BM25 分词一致，对 domains 的
    keywords / name / description 分别加权匹配。

    匹配权重：
        keywords 命中   → +3 分/词
        name 命中       → +2 分/词
        description 命中 → +1 分/词

    得分最高者胜出；无命中时返回 None，由 KnowledgeRouter 兜底。
    """

    def route(self, query: str, domains: list[KnowledgeDomain]) -> RoutingDecision | None:
        """对每个 domain 打分；最高分 > 0 时返回 RoutingDecision，否则 None。"""
        from retrieval import tokenize

        query_tokens_lower: set[str] = set()
        for t in tokenize(query):
            stripped = t.lower().strip()
            if stripped and not stripped.isspace():
                query_tokens_lower.add(stripped)

        best_domain: KnowledgeDomain | None = None
        best_score = 0

        for domain in domains:
            score = 0.0
            kw_text = " ".join(domain.keywords).lower()
            name_text = domain.name.lower()
            desc_text = domain.description.lower()

            for token in query_tokens_lower:
                if token in kw_text:
                    score += 3
                elif token in name_text:
                    score += 2
                elif token in desc_text:
                    score += 1

            if score > best_score:
                best_score = score
                best_domain = domain

        if best_domain is None or best_score == 0:
            return None

        max_possible = len(query_tokens_lower) * 3
        confidence = min(best_score / max_possible, 1.0) if max_possible > 0 else 0.0
        return RoutingDecision(
            domain_id=best_domain.id,
            confidence=round(confidence, 2),
            strategy="keyword",
        )


# ------------------------------------------------------------------
# LLMRouter
# ------------------------------------------------------------------


class LLMRouter(RouterStrategy):
    """基于 LLM 的智能路由策略。

    LLM 实例由调用方（KnowledgeService）创建并通过构造函数注入，
    Router 不负责模型生命周期管理。
    """

    def __init__(self, llm) -> None:
        """初始化 LLM Router。

        Args:
            llm: LangChain ChatModel 实例，已配置好 model / api_key / base_url / temperature。
        """
        self._llm = llm

    def route(self, query: str, domains: list[KnowledgeDomain]) -> RoutingDecision | None:
        """使用 LLM 从 domains 中选择最相关的知识库；失败时返回 None。"""
        domain_lines: list[str] = []
        for d in domains:
            default_marker = " [default]" if d.default else ""
            domain_lines.append(
                f"- id: {d.id}, name: {d.name}, description: {d.description}{default_marker}"
            )

        prompt = ROUTER_PROMPT_TEMPLATE.format(
            domain_list="\n".join(domain_lines),
            query=query,
        )

        try:
            response = self._llm.invoke(prompt)
            raw_id = response.content if hasattr(response, "content") else str(response)
            candidate = raw_id.strip().strip('"').strip("'")

            valid_ids = {d.id for d in domains}
            if candidate in valid_ids:
                return RoutingDecision(
                    domain_id=candidate,
                    confidence=0.85,
                    strategy="llm",
                )
        except Exception:
            pass

        return None


# ------------------------------------------------------------------
# KnowledgeRouter
# ------------------------------------------------------------------


class KnowledgeRouter:
    """路由分发器 —— 将 query 映射到 domain_id。

    职责边界：
        - 单 KB：直接返回，零开销跳过策略。
        - 多 KB：委托给 RouterStrategy，结果为空时统一 fallback 至 default。
        - 兜底：所有未命中、异常、空结果由此层统一处理。

    不访问 Registry、Chroma、Retriever、settings。
    """

    def __init__(self, strategy: RouterStrategy) -> None:
        """初始化路由分发器。

        Args:
            strategy: 路由策略实例（KeywordRouter 或 LLMRouter）。
        """
        self._strategy = strategy

    def route(self, query: str, domains: list[KnowledgeDomain]) -> RoutingDecision:
        """根据 query 选择目标 domain。

        Args:
            query: 用户原始查询。
            domains: 全部可用 domain 元数据（由调用方从 Registry 获取）。

        Returns:
            RoutingDecision，保证 domain_id 非空。
        """
        if not domains:
            return RoutingDecision(
                domain_id="",
                domain_ids=[],
                confidence=0.0,
                strategy="fallback",
            )

        # 单 KB 快速路径
        if len(domains) == 1:
            return RoutingDecision(
                domain_id=domains[0].id,
                domain_ids=[domains[0].id],
                confidence=1.0,
                strategy="single",
            )

        # 多 KB → 策略分发
        decision = self._strategy.route(query, domains)

        # 策略返回 None 或空 domain_id → 统一 fallback
        if decision is None or not decision.domain_id:
            default = _find_default(domains)
            return RoutingDecision(
                domain_id=default.id,
                domain_ids=[default.id],
                confidence=0.0,
                strategy="fallback",
            )

        # 保证 domain_ids 至少包含 domain_id
        if not decision.domain_ids:
            decision.domain_ids = [decision.domain_id]
        return decision


# ------------------------------------------------------------------
# 内部辅助
# ------------------------------------------------------------------


def _find_default(domains: list[KnowledgeDomain]) -> KnowledgeDomain:
    """从 domain 列表中查找 default domain；无标记时返回第一个。"""
    for d in domains:
        if d.default:
            return d
    return domains[0]
