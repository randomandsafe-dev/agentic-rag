"""Streamlit Web UI for the Agentic RAG application."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from config import settings
from ingest import ingest_documents
from rag_agent import build_agent, get_hybrid_retriever, get_vector_store


st.set_page_config(page_title="Agentic RAG", page_icon="📚", layout="wide")


def content_to_text(content: object) -> str:
    """Convert LangChain message content blocks into displayable text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)


def source_lines(answer: str) -> list[str]:
    """Extract the source line requested by the agent's system prompt."""
    return [line for line in answer.splitlines() if line.strip().startswith("来源：")]


@st.cache_resource
def load_agent():
    """Keep one Agent instance across Streamlit reruns."""
    return build_agent()


def reset_agent_caches() -> None:
    """Ensure a newly rebuilt index is used immediately."""
    get_vector_store.cache_clear()
    get_hybrid_retriever.cache_clear()
    build_agent.cache_clear()
    load_agent.clear()


def render_message(role: str, content: str) -> None:
    with st.chat_message(role):
        st.markdown(content)
        if role == "assistant":
            for source in source_lines(content):
                st.caption(source)


st.title("📚 Agentic RAG 知识库问答")
st.caption("DeepSeek 负责推理与工具调用，本地向量模型负责检索。")

with st.sidebar:
    st.header("知识库管理")
    st.caption("支持 Markdown、TXT、PDF。上传后点击“写入并重建索引”。")
    uploaded_files = st.file_uploader(
        "添加资料", type=["md", "txt", "pdf"], accept_multiple_files=True
    )
    rebuild = st.button("写入并重建索引", type="primary", use_container_width=True)
    st.divider()
    st.caption(f"单次检索最多返回 {settings.top_k} 个文本块")
    st.caption(f"嵌入模式：{settings.embedding_provider}")
    if st.button("清空本次对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if rebuild:
    try:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        uploaded_count = 0
        for uploaded_file in uploaded_files or []:
            # Drop any directory component supplied by the browser.
            target = settings.data_dir / Path(uploaded_file.name).name
            target.write_bytes(uploaded_file.getvalue())
            uploaded_count += 1

        with st.spinner("正在生成向量索引；首次使用本地模型时会下载模型文件..."):
            document_count, chunk_count = ingest_documents()
            reset_agent_caches()
        st.success(
            f"入库完成：新增/覆盖 {uploaded_count} 个文件；"
            f"当前索引包含 {document_count} 个文档、{chunk_count} 个文本块。"
        )
    except Exception as exc:
        st.error(f"入库失败：{exc}")

if "messages" not in st.session_state:
    st.session_state.messages: list[dict[str, str]] = []

for message in st.session_state.messages:
    render_message(message["role"], message["content"])

if question := st.chat_input("输入与知识库相关的问题..."):
    st.session_state.messages.append({"role": "user", "content": question})
    render_message("user", question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        status = st.status("Agent 正在分析问题...", expanded=False)
        answer = ""
        try:
            history = [
                HumanMessage(content=item["content"])
                if item["role"] == "user"
                else AIMessage(content=item["content"])
                for item in st.session_state.messages
            ]
            # messages mode yields model tokens as they arrive, including after tool calls.
            for chunk, metadata in load_agent().stream(
                {"messages": history}, stream_mode="messages"
            ):
                node = metadata.get("langgraph_node")
                if node == "tools":
                    status.update(label="正在检索知识库...", state="running")
                    continue
                text = content_to_text(chunk.content)
                if text:
                    answer += text
                    placeholder.markdown(answer + "▌")
            placeholder.markdown(answer)
            status.update(label="回答完成", state="complete", expanded=False)
        except Exception as exc:
            answer = f"抱歉，处理请求时发生错误：{exc}"
            placeholder.error(answer)
            status.update(label="处理失败", state="error", expanded=True)

        for source in source_lines(answer):
            st.caption(source)
    st.session_state.messages.append({"role": "assistant", "content": answer})
