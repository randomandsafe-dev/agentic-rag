"""RetrievalVerifier — 检索结果质量验证。

对检索到的文档进行质量评估，判断是否充分支撑回答。
LLM 实例由调用方注入（依赖注入），Verifier 不负责创建。
"""

from __future__ import annotations

import json
import logging

from langchain_core.documents import Document

from agent.verifier.result import VerificationResult

logger = logging.getLogger(__name__)

# TODO: 迁移至 prompts.py 统一管理
_VERIFIER_SYSTEM_PROMPT = """你是一个检索质量验证器。你的任务是检查检索到的文档是否足以支撑对用户问题的回答。

评估标准：
1. 覆盖率：检索到的文档是否覆盖了问题的核心主题？
2. 相关性：文档内容是否与问题直接相关？
3. 充分性：文档信息是否足够做出完整回答？

请以 JSON 格式输出（不要输出其他内容）：
{
  "passed": true/false,
  "score": 0.0-1.0 的浮点数,
  "reason": "中文说明，简洁",
  "missing_topics": ["遗漏的主题1", "遗漏的主题2"]
}

评分指南：
- 0.8-1.0: 文档充分覆盖问题，可以直接回答
- 0.5-0.7: 文档部分相关，可能需要补充检索
- 0.0-0.4: 文档与问题无关或严重不足"""


class RetrievalVerifier:
    """检索结果质量验证器。

    仅负责评估检索质量，不参与权限判断、路由选择、检索执行。
    LLM 实例由调用方注入。
    """

    def __init__(self, llm) -> None:
        """Args:
            llm: LangChain ChatModel 实例（由 llm_factory.create_llm 创建）。
        """
        self._llm = llm

    def verify(
        self,
        question: str,
        retrieved_docs: list[Document],
        draft_answer: str | None = None,
    ) -> VerificationResult:
        """验证检索到的文档是否足以支撑回答。

        Args:
            question: 用户原始问题。
            retrieved_docs: 检索到的文档列表。
            draft_answer: 可选的草稿回答（Agent 预生成的回答）。

        Returns:
            VerificationResult，包含通过状态、分数和说明。
        """
        if not retrieved_docs:
            return VerificationResult(
                passed=False,
                score=0.0,
                reason="未检索到任何文档。",
                missing_topics=[question],
            )

        # 构建文档摘要
        doc_summaries: list[str] = []
        for i, doc in enumerate(retrieved_docs, start=1):
            source = doc.metadata.get("source", "未知")
            content = doc.page_content[:500]
            doc_summaries.append(f"--- 文档 {i} (来源: {source}) ---\n{content}")

        user_prompt = f"""用户问题：{question}

检索到的文档：
{"".join(doc_summaries)}

{'草稿回答：' + draft_answer if draft_answer else ''}

请评估检索质量，按 JSON 格式输出。"""

        try:
            response = self._llm.invoke([
                {"role": "system", "content": _VERIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ])
            raw = response.content if hasattr(response, "content") else str(response)
            result = self._parse_response(raw)
            return result
        except Exception as exc:
            logger.warning("检索验证失败，默认通过: %s", exc)
            return VerificationResult(
                passed=True,
                score=0.5,
                reason=f"验证过程异常（{exc}），默认放行。",
            )

    @staticmethod
    def _parse_response(raw: str) -> VerificationResult:
        """解析 LLM 返回的 JSON，容错处理。"""
        # 去除 markdown 代码块标记
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:])
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # 尝试提取 {...}
            import re
            match = re.search(r"\{[^{}]*\}", cleaned)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    return VerificationResult(
                        passed=True, score=0.5,
                        reason="无法解析验证结果，默认放行。",
                    )
            else:
                return VerificationResult(
                    passed=True, score=0.5,
                    reason="无法解析验证结果，默认放行。",
                )

        return VerificationResult(
            passed=bool(data.get("passed", True)),
            score=float(data.get("score", 0.5)),
            reason=str(data.get("reason", "")),
            missing_topics=list(data.get("missing_topics", [])),
        )
