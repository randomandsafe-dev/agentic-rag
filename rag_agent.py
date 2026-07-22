"""以检索工具为核心的 LangChain Agentic RAG。"""

from __future__ import annotations

from functools import lru_cache

from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain.tools import tool

from config import settings
from embeddings import get_embeddings  # re-export，保持向后兼容
from knowledge.access import UserContext
from llm_factory import create_llm

# Import is deferred to break circular dependency:
#   rag_agent -> knowledge.service (get_knowledge_service)
#   knowledge.service -> knowledge.registry
#   knowledge.registry -> embeddings (terminal, no back-reference)
# The import happens lazily inside search_knowledge_base().

# ------------------------------------------------------------------
# UserContext 透传
# ------------------------------------------------------------------

_current_user: UserContext | None = None


def set_agent_user(user: UserContext | None) -> None:
    """设置当前请求的 UserContext，供 search_knowledge_base 工具透明使用。

    UI / CLI 层在调用 Agent 之前设置此值。
    """
    global _current_user
    _current_user = user


def format_documents(documents: list[Document]) -> str:
    if not documents:
        return "未检索到相关内容。"
    sections = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "未知来源")
        page = document.metadata.get("page")
        page_label = f"，第 {page + 1} 页" if isinstance(page, int) else ""
        sections.append(f"[来源 {index}: {source}{page_label}]\n{document.page_content}")
    return "\n\n".join(sections)


@tool
def search_knowledge_base(query: str) -> str:
    """在本地知识库中检索与问题相关的材料。回答知识库相关问题前必须调用此工具。

    内部自动根据查询内容路由到最相关的知识库，执行 BM25+向量混合检索，
    经 Cross-Encoder 重排序后返回 top_k 结果。"""
    try:
        from knowledge.service import get_knowledge_service

        return format_documents(get_knowledge_service().search(query, user=_current_user))
    except Exception as exc:
        return (
            "知识库检索暂时不可用："
            f"{exc}。请检查 .env 的 EMBEDDING_PROVIDER 设置，并重新运行 python ingest.py。"
        )


SYSTEM_PROMPT = """你是一个严谨的中文知识库助手。
对于任何可能需要本地资料支撑的问题，先调用 search_knowledge_base；必要时可以用不同关键词多次检索。
只依据工具返回的资料作答，不要编造。若资料不足，请明确说明。
如果用户询问有哪些知识库可用，调用 list_knowledge_bases。
在给出最终回答前，如条件允许，调用 verify_retrieval_result 自检。
回答末尾以"来源：..."列出实际使用的文件路径；引用时使用工具结果中的来源路径。
普通寒暄无需调用工具。"""


def build_agent(checkpointer=None):
    """创建 LangChain Agent；它会自行决定何时、多次调用检索工具。

    Args:
        checkpointer: 可选的 LangGraph checkpointer（如 SqliteSaver）。
                      传入后 Agent 对话状态将自动持久化。
    """
    from agent.tools.knowledge_tools import list_knowledge_bases, verify_retrieval_result

    settings.validate()
    model = create_llm(temperature=0)
    return create_agent(
        model=model,
        tools=[search_knowledge_base, list_knowledge_bases, verify_retrieval_result],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
