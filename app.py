"""Streamlit Web UI for the Agentic RAG application —— 支持多会话管理。"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from config import settings
from ingest import ingest_documents
from memory import MemoryManager
from rag_agent import build_agent, get_hybrid_retriever, get_vector_store


st.set_page_config(page_title="Agentic RAG", page_icon="📚", layout="wide")


# ============================================================
# 辅助函数
# ============================================================


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


# ============================================================
# 资源缓存
# ============================================================


@st.cache_resource
def get_memory_manager() -> MemoryManager:
    """全局 MemoryManager 单例（跨 Streamlit 会话）。"""
    return MemoryManager(
        str(settings.memory_db_path),
        session_window=settings.session_window,
    )


@st.cache_resource
def load_agent(_checkpointer=None):
    """Keep one Agent instance across Streamlit reruns."""
    return build_agent(checkpointer=_checkpointer)


def reset_agent_caches() -> None:
    """Ensure a newly rebuilt index is used immediately."""
    get_vector_store.cache_clear()
    get_hybrid_retriever.cache_clear()
    load_agent.clear()
    get_memory_manager.clear()


# ============================================================
# 会话管理
# ============================================================


def _init_session_state(mm: MemoryManager) -> None:
    """初始化 session_state 中的会话相关字段。"""
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    if "messages" not in st.session_state:
        st.session_state.messages: list[dict[str, str]] = []


def _load_session_messages(mm: MemoryManager, session_id: str) -> None:
    """从持久化存储加载指定会话的消息到会话状态。"""
    msgs = mm.load_full_history(session_id)
    st.session_state.messages = [
        {"role": m.role, "content": m.content} for m in msgs
    ]


def _ensure_active_session(mm: MemoryManager) -> str | None:
    """确保存在一个活跃的会话。没有则返回 None。"""
    sessions = mm.sessions.list()
    if not sessions:
        return None
    sid = st.session_state.current_session_id
    if sid and any(s.id == sid for s in sessions):
        return sid
    # 默认选中第一个
    sid = sessions[0].id
    st.session_state.current_session_id = sid
    _load_session_messages(mm, sid)
    return sid


# ============================================================
# 对话框渲染
# ============================================================


def render_message(role: str, content: str) -> None:
    with st.chat_message(role):
        st.markdown(content)
        if role == "assistant":
            for source in source_lines(content):
                st.caption(source)


def render_sidebar() -> None:
    """渲染侧栏：会话管理 + 知识库管理。"""
    mm = get_memory_manager()
    _init_session_state(mm)

    # ---- 会话管理 ----
    st.sidebar.header("📁 会话")
    _render_session_panel(mm)
    st.sidebar.divider()

    # ---- 知识库管理 ----
    st.sidebar.header("📚 知识库管理")
    st.sidebar.caption("支持 Markdown、TXT、PDF。上传后点击「写入并重建索引」。")
    _render_kb_panel(mm)

    # ---- 信息 ----
    st.sidebar.divider()
    st.sidebar.caption(f"单次检索最多返回 {settings.top_k} 个文本块")
    st.sidebar.caption(f"嵌入模式：{settings.embedding_provider}")


def _render_session_panel(mm: MemoryManager) -> None:
    """渲染侧栏中的会话管理区域。"""
    sessions = mm.sessions.list()

    # --- 新建会话 ---
    new_name = st.sidebar.text_input("新建会话", placeholder="输入名称后回车", key="new_session_name")
    if new_name:
        session = mm.sessions.create(new_name)
        st.session_state.current_session_id = session.id
        _load_session_messages(mm, session.id)
        st.rerun()

    if not sessions:
        st.sidebar.caption("暂无会话，请先新建。")
        return

    # --- 会话列表 ---
    session_options = {
        s.id: f"{s.name} ({s.message_count}轮)"
        for s in sessions
    }
    current_id = st.session_state.current_session_id
    if current_id not in session_options:
        current_id = sessions[0].id

    # 找到当前选中项的 label 作为 default
    current_label = session_options.get(current_id, list(session_options.values())[0])

    selected_label = st.sidebar.selectbox(
        "切换会话",
        options=list(session_options.values()),
        index=list(session_options.keys()).index(current_id) if current_id in session_options else 0,
        key="session_selector",
    )

    # 反向查找选中的 session_id
    selected_id = None
    for sid, label in session_options.items():
        if label == selected_label:
            selected_id = sid
            break

    if selected_id and selected_id != st.session_state.current_session_id:
        st.session_state.current_session_id = selected_id
        _load_session_messages(mm, selected_id)
        st.rerun()

    # --- 当前会话操作 ---
    col1, col2 = st.sidebar.columns(2)
    cur_session = mm.sessions.get(st.session_state.current_session_id)
    if cur_session:
        with col2:
            if st.button("🗑 删除", use_container_width=True, key="delete_session_btn"):
                mm.sessions.delete(cur_session.id)
                st.session_state.current_session_id = None
                st.session_state.messages = []
                st.rerun()
    with col1:
        if st.button("🧹 清空对话", use_container_width=True, key="clear_session_btn"):
            mm.messages.delete_all(st.session_state.current_session_id)
            mm.sessions.update(st.session_state.current_session_id, updated_at="")
            st.session_state.messages = []
            st.rerun()


def _render_kb_panel(mm: MemoryManager) -> None:
    """渲染侧栏中的知识库管理区域。"""
    uploaded_files = st.sidebar.file_uploader(
        "添加资料", type=["md", "txt", "pdf"], accept_multiple_files=True,
        key="kb_uploader",
    )
    rebuild = st.sidebar.button("写入并重建索引", type="primary", use_container_width=True)

    if rebuild:
        try:
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            uploaded_count = 0
            for uploaded_file in uploaded_files or []:
                target = settings.data_dir / Path(uploaded_file.name).name
                target.write_bytes(uploaded_file.getvalue())
                uploaded_count += 1

            with st.spinner("正在生成向量索引；首次使用本地模型时会下载模型文件..."):
                document_count, chunk_count = ingest_documents()
                reset_agent_caches()
            st.sidebar.success(
                f"入库完成：新增/覆盖 {uploaded_count} 个文件；"
                f"当前索引包含 {document_count} 个文档、{chunk_count} 个文本块。"
            )
        except Exception as exc:
            st.sidebar.error(f"入库失败：{exc}")


# ============================================================
# 主界面
# ============================================================


def main() -> None:
    st.title("📚 Agentic RAG 知识库问答")
    st.caption("DeepSeek 负责推理与工具调用，本地向量模型负责检索。")

    mm = get_memory_manager()
    _init_session_state(mm)

    render_sidebar()

    # 确保有活跃会话
    sid = _ensure_active_session(mm)
    if sid is None:
        st.info("请在侧栏新建一个会话开始对话。")
        return

    # 获取/创建 Agent
    try:
        checkpointer = mm.checkpointer.get()
    except Exception:
        checkpointer = None
    agent = load_agent(_checkpointer=checkpointer)

    # 渲染历史消息
    for message in st.session_state.messages:
        render_message(message["role"], message["content"])

    # 聊天输入
    if question := st.chat_input("输入与知识库相关的问题..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": question})
        mm.messages.add(sid, "user", question)
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
                config = mm.checkpointer.get_config(sid)
                for chunk, metadata in agent.stream(
                    {"messages": history}, config=config, stream_mode="messages"
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

        # 持久化助手消息
        st.session_state.messages.append({"role": "assistant", "content": answer})
        mm.messages.add(sid, "assistant", answer)


if __name__ == "__main__":
    main()
