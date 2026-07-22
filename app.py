"""Streamlit Web UI for the Agentic RAG application —— 支持多会话管理。"""

from __future__ import annotations

import gc
import sqlite3
from pathlib import Path

import streamlit as st
from config import settings
from ingest import ingest_documents
from knowledge.access import UserContext
from knowledge.service import get_knowledge_service
from memory import MemoryManager
from rag_agent import (
    _current_user,
    format_documents,
    get_hybrid_retriever,
    get_vector_store,
    search_web,
    set_agent_user,
    stream_grounded_answer,
)
from search_router import route_question
from verify import verify_answer
from web_search import format_web_documents, search_web_documents


st.set_page_config(page_title="Agentic RAG", page_icon="📚", layout="wide")

ROUTE_LABELS = {
    "auto": "自动路由（推荐）",
    "local_only": "仅本地知识库",
    "web_only": "仅联网搜索",
    "both": "本地 + 联网",
}


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
    prefixes = ("来源：", "本地来源：", "联网来源：")
    return [line for line in answer.splitlines() if line.strip().startswith(prefixes)]


def append_retrieval_sources(
    answer: str, tool_results: list[tuple[str, str]]
) -> str:
    """Append source sections based on tool outputs, not model formatting choices."""
    local_sources: list[str] = []
    web_sources: list[str] = []
    for tool_name, content in tool_results:
        lines = [line for line in content.splitlines() if line.strip()]
        if tool_name == "search_knowledge_base":
            local_sources.extend(line for line in lines if line.startswith("[来源 "))
        elif tool_name == "search_web":
            web_sources.extend(line for line in lines if line.startswith("[联网来源 "))

    sections = []
    if local_sources:
        sections.append("**本地知识库结果**\n\n" + "\n".join(dict.fromkeys(local_sources)))
    if web_sources:
        sections.append("**联网搜索结果**\n\n" + "\n".join(dict.fromkeys(web_sources)))
    return answer if not sections else answer.rstrip() + "\n\n---\n\n" + "\n\n".join(sections)


def render_verification_report(result: dict | None) -> None:
    """Render a compact, source-specific factuality report."""
    if result is None:
        return
    score = result.get("reliability", 0)
    factual = result.get("factual_accurate", False)
    hallucination = result.get("has_hallucination", False)
    with st.expander(f"答案验证报告：可信度 {score}/5", expanded=False):
        st.markdown(
            f"- 事实与资料一致：{'是' if factual else '否'}\n"
            f"- 检测到潜在幻觉：{'是' if hallucination else '否'}\n"
            f"- 信息可能遗漏：{'是' if result.get('missing_info', False) else '否'}"
        )
        if result.get("details"):
            st.caption(result["details"])


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


def reset_agent_caches() -> None:
    """Release cached retrievers so a rebuilt index can replace its files."""
    get_vector_store.cache_clear()
    get_hybrid_retriever.cache_clear()
    get_knowledge_service().invalidate()
    get_memory_manager.clear()
    # Chroma can keep memory-mapped index files open until the retriever objects
    # are collected.  This matters on Windows, where an open .bin file cannot be
    # removed while rebuilding the persistent directory.
    gc.collect()


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

    # ---- KB 状态 ----
    st.sidebar.divider()
    st.sidebar.header("📋 知识库状态")
    _render_kb_status()

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
    if settings.tavily_api_key:
        st.sidebar.success("联网搜索：已启用")
    else:
        st.sidebar.info("联网搜索：未启用（配置 TAVILY_API_KEY 后可用）")
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
                # This must happen before ingest_documents(), which replaces the
                # whole Chroma directory.  Doing it afterwards is too late on
                # Windows because the old cached store still holds data_level*.bin.
                reset_agent_caches()
                document_count, chunk_count = ingest_documents()
                reset_agent_caches()
            st.sidebar.success(
                f"入库完成：新增/覆盖 {uploaded_count} 个文件；"
                f"当前索引包含 {document_count} 个文档、{chunk_count} 个文本块。"
            )
        except Exception as exc:
            st.sidebar.error(f"入库失败：{exc}")


def _render_kb_status() -> None:
    """渲染侧栏中的 KB 状态列表。"""
    try:
        from knowledge.management import KnowledgeManager
        mgr = KnowledgeManager()
        kbs = mgr.list_knowledge_bases()
    except Exception as exc:
        st.sidebar.caption(f"无法加载 KB 列表：{exc}")
        return

    if not kbs:
        st.sidebar.caption("暂无已配置的知识库。")
        return

    for kb in kbs:
        status_icon = "🟢" if kb["enabled"] else "🔴"
        default_mark = " ⭐" if kb["default"] else ""
        st.sidebar.caption(
            f"{status_icon} **{kb['name']}**{default_mark}\n\n"
            f"`{kb['id']}` — {kb.get('description', '')}"
        )


# ============================================================
# 主界面
# ============================================================


def main() -> None:
    st.title("📚 Agentic RAG 知识库问答")
    st.caption("DeepSeek 负责推理与工具调用，本地向量模型负责检索。")

    route_keys = list(ROUTE_LABELS)
    default_route = (
        settings.web_route_mode
        if settings.web_route_mode in ROUTE_LABELS
        else "auto"
    )
    route_col, route_help_col = st.columns([1, 2])
    with route_col:
        st.selectbox(
            "搜索路由",
            options=route_keys,
            index=route_keys.index(default_route),
            format_func=lambda key: ROUTE_LABELS[key],
            key="web_route_mode",
        )
    with route_help_col:
        st.caption("自动模式默认优先保护内部资料；仅在问题需要实时公开信息时联网。")

    mm = get_memory_manager()
    _init_session_state(mm)

    render_sidebar()

    # 确保有活跃会话
    sid = _ensure_active_session(mm)
    if sid is None:
        st.info("请在侧栏新建一个会话开始对话。")
        return

    # 渲染历史消息
    for message in st.session_state.messages:
        render_message(message["role"], message["content"])

    # 聊天输入
    if question := st.chat_input("输入与知识库相关的问题..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": question})
        mm.messages.add(sid, "user", question)
        render_message("user", question)

        decision = route_question(
            question,
            requested_mode=st.session_state.web_route_mode,
            web_available=bool(settings.tavily_api_key),
        )
        st.caption(f"本次路由：**{decision.label}**。{decision.reason}")
        source_contexts: list[tuple[str, str, str, list]] = []

        if decision.use_local:
            with st.status("正在检索本地知识库...", expanded=False) as status:
                try:
                    # 通过 KnowledgeService 检索，保留多 KB 路由和权限过滤
                    local_documents = get_knowledge_service().search(
                        question, user=_current_user
                    )
                    local_context = format_documents(local_documents)
                    status.update(label="本地资料检索完成", state="complete")
                except Exception as exc:
                    local_documents = []
                    local_context = f"【本地知识库结果】\n本地检索失败：{exc}"
                    status.update(label="本地资料检索失败", state="error")
            source_contexts.append(
                ("本地知识库回答", "本地知识库", local_context, local_documents)
            )

        if decision.use_web:
            with st.status("正在联网搜索...", expanded=False) as status:
                try:
                    web_documents = search_web_documents(question)
                    web_context = format_web_documents(web_documents)
                    status.update(label="联网搜索、可信度评分与重排序完成", state="complete")
                except Exception as exc:
                    web_documents = []
                    web_context = f"【联网搜索结果】\n联网搜索失败：{exc}"
                    status.update(label="联网搜索失败", state="error")
            source_contexts.append(("联网搜索回答", "联网搜索", web_context, web_documents))

        answers: list[tuple[str, str, str]] = []
        for title, source_type, context, documents in source_contexts:
            with st.chat_message("assistant"):
                st.markdown(f"### {title}")
                placeholder = st.empty()
                answer = ""
                try:
                    for chunk in stream_grounded_answer(question, source_type, context):
                        text = content_to_text(chunk.content)
                        if text:
                            answer += text
                            placeholder.markdown(answer + "▌")
                    placeholder.markdown(answer)
                except Exception as exc:
                    answer = f"抱歉，生成{title}时发生错误：{exc}"
                    placeholder.error(answer)
                render_verification_report(verify_answer(question, answer, documents))
                st.divider()
                st.markdown(context)
            answers.append((title, answer, context))

        combined_answer = "\n\n".join(
            f"## {title}\n\n{answer}\n\n{context}" for title, answer, context in answers
        )
        st.session_state.messages.append({"role": "assistant", "content": combined_answer})
        mm.messages.add(sid, "assistant", combined_answer)


if __name__ == "__main__":
    main()
