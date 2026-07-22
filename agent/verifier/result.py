"""VerificationResult — 检索验证的结构化输出。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VerificationResult:
    """检索验证的完整结果。

    Attributes:
        passed: 检索结果是否通过质量检查。
        score: 0.0 ~ 1.0 的综合质量分数。
        reason: 人类可读的验证说明。
        missing_topics: 检索遗漏的关键主题（供 Agent 决定是否重试）。
    """

    passed: bool = False
    score: float = 0.0
    reason: str = ""
    missing_topics: list[str] = field(default_factory=list)
