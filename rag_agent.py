"""以检索工具为核心的 LangChain Agentic RAG。"""

from __future__ import annotations

from functools import lru_cache

from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import tool

from config import settings
from retrieval import HybridRetriever


@lru_cache(maxsize=1)
def get_embeddings():
    """获取嵌入模型；默认本地模型，避免聊天接口不支持 embeddings 的问题。"""
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.api_key,
            base_url=settings.base_url,
        )
    try:
        from langchain_community.embeddings import FastEmbedEmbeddings
    except ImportError as exc:
        raise RuntimeError("缺少本地嵌入依赖，请运行：pip install -r requirements.txt") from exc
    return FastEmbedEmbeddings(
        model_name=settings.local_embedding_model,
    )


@lru_cache(maxsize=1)
def get_vector_store():
    """打开已持久化的 Chroma 向量库。"""
    settings.validate()
    if not settings.persist_dir.exists():
        raise RuntimeError("尚未创建知识库。请先运行：python ingest.py")
    return Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.persist_dir),
        embedding_function=get_embeddings(),
    )


@lru_cache(maxsize=1)
def get_hybrid_retriever():
    """获取混合检索器（BM25 + 向量 + Reranker）。"""
    return HybridRetriever(get_vector_store())


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
    内部自动进行查询改写、相关性判断、最多两次重试（可通过 .env 配置）。"""
    try:
        return format_documents(get_hybrid_retriever().search(query))
    except Exception as exc:
        return (
            "知识库检索暂时不可用："
            f"{exc}。请检查 .env 的 EMBEDDING_PROVIDER 设置，并重新运行 python ingest.py。"
        )


SYSTEM_PROMPT = """你是一个严谨的中文知识库助手。
对于任何可能需要本地资料支撑的问题，先调用 search_knowledge_base；必要时可以用不同关键词多次检索。
只依据工具返回的资料作答，不要编造。若资料不足，请明确说明。
回答末尾以“来源：...”列出实际使用的文件路径；引用时使用工具结果中的来源路径。
普通寒暄无需调用工具。"""


def build_agent(checkpointer=None):
    """创建 LangChain Agent；它会自行决定何时、多次调用检索工具。

    Args:
        checkpointer: 可选的 LangGraph checkpointer（如 SqliteSaver）。
                      传入后 Agent 对话状态将自动持久化。
    """
    settings.validate()
    model = ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=0,
    )
    return create_agent(
        model=model,
        tools=[search_knowledge_base],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
