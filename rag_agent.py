"""以检索工具为核心的 LangChain Agentic RAG。"""

from __future__ import annotations

from functools import lru_cache

from langchain.agents import create_agent
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import tool

from config import settings
from retrieval import HybridRetriever
from web_search import format_web_documents, search_web_documents


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
    return "【本地知识库结果】\n" + "\n\n".join(sections)


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


@tool
def search_web(query: str) -> str:
    """搜索互联网以获取实时、公开的信息。适用于新闻、价格、时效性事实或本地知识库没有覆盖的问题。"""
    if not settings.tavily_api_key:
        return "联网搜索未启用：请在 .env 配置 TAVILY_API_KEY 后重启应用。"
    try:
        return format_web_documents(search_web_documents(query))
    except Exception as exc:
        return f"【联网搜索结果】\n联网搜索暂时不可用：{exc}"


def stream_grounded_answer(question: str, source_type: str, context: str):
    """仅使用一类给定来源生成回答，避免本地与联网证据混合。"""
    settings.validate()
    model = ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=0,
    )
    prompt = (
        f"你正在生成『{source_type}』的独立回答。\n"
        "只能使用下方提供的资料，不能引用其他来源、不能补充未出现的事实。"
        "若资料不足，请直接说明。请用中文简洁回答。\n\n"
        f"用户问题：{question}\n\n"
        f"可用资料：\n{context}"
    )
    return model.stream(
        [
            SystemMessage(content="你是严谨的资料问答助手。"),
            HumanMessage(content=prompt),
        ]
    )


SYSTEM_PROMPT = """你是一个严谨的中文知识库助手。
对于本地资料相关的问题，优先调用 search_knowledge_base；必要时可使用不同关键词多次检索。
对于新闻、价格、时效性事实、或本地资料不足的问题，调用 search_web 获取联网信息。
只依据工具返回的资料作答，不要编造。若资料不足，请明确说明。
必须严格区分来源：使用本地检索后，在回答末尾列出“本地来源：文件路径”；
使用联网搜索后，在回答末尾列出“联网来源：网页标题 - URL”。如果两种都用到，分别列出，绝不混合标注。
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
    tools = [search_knowledge_base]
    if settings.tavily_api_key:
        tools.append(search_web)
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
