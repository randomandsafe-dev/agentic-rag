"""Agent Retrieval Tools —— Agent 可调用的知识库工具集。

所有工具仅通过 KnowledgeService 公开 API 访问底层能力，
不直接接触 Router / Registry / Chroma / Retriever。

注意：对 rag_agent 的导入在函数体内惰性执行，以打破循环依赖：
  rag_agent → agent.tools.knowledge_tools → (lazy) rag_agent
"""

from __future__ import annotations

from langchain.tools import tool


@tool
def list_knowledge_bases(dummy: str = "") -> str:
    """列出当前用户可访问的所有知识库。Agent 应在用户询问有哪些知识库可用、
    或需要了解资料范围时调用此工具。

    Args:
        dummy: LangChain tool 要求至少一个参数，此参数无实际作用。
    """
    from rag_agent import _current_user
    from knowledge.service import get_knowledge_service

    try:
        domains = get_knowledge_service().list_domains(user=_current_user)
        if not domains:
            return "当前没有可用的知识库。"
        lines = [f"- {d['name']}：{d['description']}" for d in domains]
        return "可用知识库：\n" + "\n".join(lines)
    except Exception as exc:
        return f"获取知识库列表失败：{exc}"


@tool
def verify_retrieval_result(verification_input: str) -> str:
    """验证检索结果是否包含有效来源。

    输入格式: '回答: <answer> | 来源: <sources>'
    Agent 应在给出最终回答前调用此工具进行自检。

    Args:
        verification_input: 包含回答和来源的字符串，用 ' | 来源: ' 分隔。
    """
    parts = verification_input.split(" | 来源: ", 1)
    answer = parts[0].replace("回答: ", "", 1) if parts else verification_input
    sources = parts[1] if len(parts) > 1 else ""

    if not sources or sources.strip() in ("", "未检索到相关内容。"):
        return (
            "验证失败：未找到来源文档。"
            "请调用 search_knowledge_base 重新检索后再回答。"
        )

    if "未知来源" in sources and len(sources) < 100:
        return "验证警告：来源信息不完整，建议补充检索。"

    return "验证通过：回答有来源支撑。"
