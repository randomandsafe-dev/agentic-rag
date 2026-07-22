"""命令行聊天入口 —— 支持多会话管理。"""

from __future__ import annotations

import argparse

from langchain_core.messages import AIMessage, HumanMessage

from config import settings
from knowledge.access import UserContext
from memory import MemoryManager
from rag_agent import build_agent, set_agent_user
from verify import verify_answer, format_verification
from web_search import search_web_documents


def message_text(message: AIMessage) -> str:
    """兼容不同模型返回的文本内容格式。"""
    if isinstance(message.content, str):
        return message.content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in message.content
    )


def _select_or_create_session(mm: MemoryManager) -> str:
    """交互式选择或创建会话。返回 session_id。"""
    print("\n现有会话：")
    sessions = mm.sessions.list()

    if sessions:
        for i, s in enumerate(sessions, start=1):
            print(f"  [{i}] {s.name}  ({s.message_count}轮)  {s.updated_at[:16]}")
        print(f"  [n] 新建会话")
        while True:
            choice = input("\n选择会话: ").strip()
            if choice.lower() == "n":
                break
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(sessions):
                    session = sessions[idx]
                    print(f"已切换到「{session.name}」。")
                    return session.id
            except ValueError:
                pass
            print("无效选择，请重试。")

    # 新建会话
    name = input("新会话名称: ").strip()
    if not name:
        name = "默认会话"
    session = mm.sessions.create(name)
    print(f"✓ 已创建会话「{session.name}」。")
    return session.id


def main(user: UserContext | None = None) -> None:
    set_agent_user(user)

    mm = MemoryManager(
        str(settings.memory_db_path),
        session_window=settings.session_window,
    )

    # --- 会话选择 ---
    session_id = _select_or_create_session(mm)

    # --- 构建 Agent（带 checkpointer） ---
    try:
        checkpointer = mm.checkpointer.get()
    except Exception:
        checkpointer = None
    agent = build_agent(checkpointer=checkpointer)

    print("\n命令: /new <名称> | /list | /switch <编号> | /delete <编号>")
    print("输入 exit 或 quit 退出。")

    while True:
        raw = input("\n你：").strip()

        # ---- 退出 ----
        if raw.lower() in {"exit", "quit"}:
            break

        # ---- 命令处理 ----
        if raw.startswith("/"):
            new_id = _handle_command(mm, raw)
            if new_id is not None:
                session_id = new_id
            continue

        if not raw:
            continue

        # ---- 正常对话 ----
        question = raw

        # 持久化用户消息
        mm.messages.add(session_id, "user", question)

        # 调用 Agent
        config = mm.checkpointer.get_config(session_id)
        if checkpointer is not None:
            # LangGraph 从 checkpointer 自动恢复历史，只传本轮新消息避免重复
            result = agent.invoke(
                {"messages": [HumanMessage(content=question)]}, config=config
            )
        else:
            # 降级：无 checkpointer 时手动构建窗口内历史
            history = _build_history(mm, session_id)
            result = agent.invoke({"messages": history})
        answer = result["messages"][-1]
        answer_text = message_text(answer)

        print(f"\n助手：{answer_text}")

        # 持久化助手消息
        mm.messages.add(session_id, "assistant", answer_text)

        # --- 答案验证 ---
        if settings.verify_enabled:
            try:
                from knowledge.service import get_knowledge_service

                verify_docs = list(get_knowledge_service().search(question))

                # 检测 Agent 本轮是否使用了联网搜索
                _web_used = any(
                    getattr(msg, "name", None) == "search_web"
                    for msg in result.get("messages", [])
                )
                if _web_used and settings.tavily_api_key:
                    try:
                        web_docs = search_web_documents(question)
                        verify_docs.extend(web_docs)
                    except Exception:
                        pass  # 联网文档获取失败不影响验证

                verification = verify_answer(question, answer_text, verify_docs)
                if verification:
                    print(format_verification(verification))
            except Exception:
                pass  # 验证失败不应中断对话


def _build_history(mm: MemoryManager, session_id: str) -> list[HumanMessage | AIMessage]:
    """从持久化存储加载窗口内的消息，构建 LangChain 历史。"""
    messages = mm.load_history(session_id)
    history: list[HumanMessage | AIMessage] = []
    for msg in messages:
        if msg.role == "user":
            history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            history.append(AIMessage(content=msg.content))
    return history


def _handle_command(mm: MemoryManager, raw: str) -> str | None:
    """处理 / 命令。

    Returns:
        如果命令导致当前会话切换，返回新 session_id；否则返回 None。
    """
    parts = raw.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "/list":
        print("\n会话列表：")
        sessions = mm.sessions.list()
        if not sessions:
            print("  (无会话)")
            return None
        for i, s in enumerate(sessions, start=1):
            print(f"  [{i}] {s.name}  ({s.message_count}轮)  {s.updated_at[:16]}")
        return None

    elif cmd == "/new":
        name = arg if arg else input("会话名称: ").strip()
        if not name:
            name = "默认会话"
        session = mm.sessions.create(name)
        print(f"✓ 已创建并切换到「{session.name}」。")
        return session.id

    elif cmd == "/switch":
        if not arg:
            print("用法: /switch <编号>")
            return None
        sessions = mm.sessions.list()
        try:
            idx = int(arg) - 1
            if 0 <= idx < len(sessions):
                session = sessions[idx]
                print(f"已切换到「{session.name}」。")
                return session.id
            else:
                print("无效的会话编号。")
        except ValueError:
            print("请输入有效的数字编号。")
        return None

    elif cmd == "/delete":
        if not arg:
            print("用法: /delete <编号>")
            return None
        sessions = mm.sessions.list()
        try:
            idx = int(arg) - 1
            if 0 <= idx < len(sessions):
                session = sessions[idx]
                confirm = input(f"确认删除「{session.name}」({session.message_count}轮)? [y/N]: ").strip().lower()
                if confirm == "y":
                    mm.sessions.delete(session.id)
                    print("✓ 已删除。")
                else:
                    print("已取消。")
            else:
                print("无效的会话编号。")
        except ValueError:
            print("请输入有效的数字编号。")
        return None

    else:
        print(f"未知命令: {cmd}。可用: /new /list /switch /delete")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic RAG 命令行聊天")
    parser.add_argument("--user", type=str, default=None, help="用户 ID")
    parser.add_argument("--role", type=str, default="viewer", help="用户角色 (admin/developer/viewer)")
    args = parser.parse_args()

    user = UserContext(user_id=args.user, role=args.role) if args.user else None
    main(user=user)
