"""答案验证——基于来源文档检查生成的回答是否可靠。"""

from __future__ import annotations

from langchain_core.documents import Document

from config import settings
from llm_factory import create_llm

VERIFY_SYSTEM_PROMPT = """你是一个严格的答案验证器。你的任务是检查 AI 助手给出的回答是否完全基于提供的参考资料。

按以下步骤逐条检查：
1. 逐句比对：将回答中的每个事实性陈述与参考资料对照
2. 标记幻觉：找出回答中存在但资料中无法找到支撑的内容
3. 评估覆盖：判断资料中的重要信息是否在回答中被遗漏
4. 综合评分：给出整体可靠性分数

请以以下 JSON 格式输出（不要输出其他内容）：
{
  "reliability": 1-5 的整数,
  "factual_accurate": true/false,
  "has_hallucination": true/false,
  "missing_info": true/false,
  "details": "详细说明，中文"
}"""


def verify_answer(
    question: str,
    answer: str,
    documents: list[Document],
) -> dict | None:
    """验证生成的回答是否忠实于检索到的来源文档。

    返回 None 表示跳过验证（未启用或没有文档）。
    返回 dict 包含 reliability / factual_accurate / has_hallucination / missing_info / details。
    """
    if not settings.verify_enabled:
        return None

    if not documents:
        return {
            "reliability": 0,
            "factual_accurate": False,
            "has_hallucination": False,
            "missing_info": False,
            "details": "没有检索到任何文档，无法进行验证。",
        }

    # 拼接参考资料
    doc_text_parts: list[str] = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "未知")
        doc_text_parts.append(f"--- 资料片段 {i}（来源: {source}）---\n{doc.page_content}")
    doc_text = "\n\n".join(doc_text_parts)

    user_prompt = f"""问题：{question}

AI 助手的回答：
{answer}

参考资料：
{doc_text}"""

    try:
        model = create_llm(temperature=0)
        response = model.invoke(
            [
                {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]
        )
        # 解析 JSON
        import json

        content = str(response.content)
        # 去除可能的 markdown 代码块标记
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content.rsplit("\n", 1)[0]
        return json.loads(content)
    except Exception as exc:
        return {
            "reliability": 0,
            "factual_accurate": False,
            "has_hallucination": False,
            "missing_info": False,
            "details": f"验证过程出错：{exc}",
        }


def format_verification(result: dict | None) -> str:
    """将验证结果格式化为可读文本。"""
    if result is None:
        return ""

    reliability = result.get("reliability", 0)
    bar = "█" * reliability + "░" * (5 - reliability)

    lines = [
        "\n📋 **答案验证报告**",
        f"可靠性：[{bar}] {reliability}/5",
        f"事实准确：{'✅ 是' if result.get('factual_accurate') else '❌ 否'}",
        f"存在幻觉：{'⚠️ 是' if result.get('has_hallucination') else '✅ 否'}",
        f"信息遗漏：{'⚠️ 是' if result.get('missing_info') else '✅ 否'}",
    ]
    if result.get("details"):
        lines.append(f"\n📝 详细说明：{result['details']}")

    return "\n".join(lines)
