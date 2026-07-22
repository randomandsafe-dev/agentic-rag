"""Streamlit Web UI for the Agentic RAG application —— 支持多会话管理。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from config import settings
from ingest import ingest_documents
from knowledge.access import UserContext
from memory import MemoryManager
from knowledge.service import get_knowledge_service
from rag_agent import build_agent, set_agent_user


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
def load_agent():
    """Build an Agent without LangGraph's thread-bound SQLite checkpointer."""
    return build_agent()


def reset_agent_caches() -> None:
    """Ensure a newly rebuilt index is used immediately."""
    get_knowledge_service().invalidate()
    load_agent.clear()
    get_memory_manager.clear()


def delete_all_conversations() -> None:
    """删除全部会话、消息及 LangGraph checkpoint，且不依赖缓存对象。"""
    connection = sqlite3.connect(str(settings.memory_db_path))
    try:
        existing_tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        # 固定的表名列表；按依赖顺序先清理子表。
        for table_name in (
            "checkpoint_writes",
            "checkpoint_blobs",
            "checkpoints",
            "messages",
            "facts",
            "sessions",
        ):
            if table_name in existing_tables:
                connection.execute(f"DELETE FROM {table_name}")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


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

    # ---- 用户身份 ----
    st.sidebar.divider()
    st.sidebar.header("👤 用户身份")
    role = st.sidebar.selectbox(
        "角色",
        options=["viewer", "admin", "developer"],
        index=0,
        key="user_role_selector",
    )
    user_id = st.sidebar.text_input("用户 ID", value=role, key="user_id_input")
    set_agent_user(UserContext(user_id=user_id, role=role))

    # ---- 信息 ----
    st.sidebar.divider()
    st.sidebar.caption(f"单次检索最多返回 {settings.top_k} 个文本块")
    st.sidebar.caption(f"嵌入模式：{settings.embedding_provider}")


def _render_session_panel(mm: MemoryManager) -> None:
    """渲染侧栏中的会话管理区域。"""
    sessions = mm.sessions.list()

    # --- 新建会话 ---
    if st.session_state.pop("clear_new_session_name", False):
        st.session_state.new_session_name = ""
    new_name = st.sidebar.text_input(
        "会话名称", placeholder="例如：产品需求讨论", key="new_session_name"
    )
    if st.sidebar.button("＋ 新建对话", use_container_width=True, key="create_session_btn"):
        name = new_name.strip() or "新对话"
        session = mm.sessions.create(name)
        st.session_state.current_session_id = session.id
        _load_session_messages(mm, session.id)
        # 下一次渲染时再清空，避免修改已创建的 Streamlit widget 状态。
        st.session_state.clear_new_session_name = True
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
            try:
                mm.messages.delete_all(cur_session.id)
                # 触发 updated_at 自动更新，避免把时间写成空字符串。
                mm.sessions.update(cur_session.id, status=cur_session.status)
                st.session_state.messages = []
                st.sidebar.success("当前会话已清空。")
            except Exception as exc:
                st.sidebar.error(f"清空会话失败：{exc}")

    # --- 危险操作：删除全部会话 ---
    with st.sidebar.expander("危险操作"):
        st.caption("此操作会永久删除全部会话和聊天记录，知识库资料不会受影响。")
        confirmed = st.checkbox("我确认删除全部对话", key="confirm_delete_all_sessions")
        if st.button(
            "删除全部对话",
            type="secondary",
            use_container_width=True,
            disabled=not confirmed,
            key="delete_all_sessions_btn",
        ):
            try:
                delete_all_conversations()
                get_memory_manager.clear()
                st.session_state.current_session_id = None
                st.session_state.messages = []
                st.sidebar.success("全部对话已删除。")
            except Exception as exc:
                st.sidebar.error(f"删除全部对话失败：{exc}")


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

    # Web UI 使用消息表保存历史。LangGraph 的同步 SQLite checkpointer 会在
    # 工具调用线程间共享连接，因此这里不使用它。
    agent = load_agent()

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
                # 手动构建最近 N 条历史；MemoryManager 的每次数据库操作都会
                # 打开当前线程连接，因此与 Streamlit 的线程模型兼容。
                window_msgs = mm.load_history(sid)
                agent_input = {
                    "messages": [
                        HumanMessage(content=m.content)
                        if m.role == "user"
                        else AIMessage(content=m.content)
                        for m in window_msgs
                    ]
                }
                for chunk, metadata in agent.stream(
                    agent_input, stream_mode="messages"
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
