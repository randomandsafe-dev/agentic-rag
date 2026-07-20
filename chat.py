"""命令行聊天入口 —— 支持多会话管理。"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from config import ROOT_DIR, settings
from memory import MemoryManager
from rag_agent import build_agent
from verify import verify_answer, format_verification


def message_text(message: AIMessage) -> str:
    """兼容不同模型返回的文本内容格式。"""
    if isinstance(message.content, str):
        return message.content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block)
        for block in message.content
    )


def _print_sessions(mm: MemoryManager) -> None:
    """打印会话列表。"""
    sessions = mm.sessions.list()
    if not sessions:
        print("   (无现有会话)")
        return
    for i, s in enumerate(sessions, start=1):
        marker = ""
        print(f"  [{i}] {s.name}  ({s.message_count}轮)  {s.updated_at[:16]}{marker}")


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


def main() -> None:
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
            _handle_command(mm, raw)
            # 如果切换了会话，更新 session_id
            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower()
            if cmd in {"/switch", "/new"}:
                sessions = mm.sessions.list()
                if sessions:
                    session_id = sessions[0].id  # 最新创建的或切换到的
            continue

        if not raw:
            continue

        # ---- 正常对话 ----
        question = raw

        # 持久化用户消息
        mm.messages.add(session_id, "user", question)

        # 加载窗口内历史
        history = _build_history(mm, session_id)

        # 调用 Agent
        config = mm.checkpointer.get_config(session_id)
        result = agent.invoke({"messages": history}, config=config)
        answer = result["messages"][-1]
        answer_text = message_text(answer)

        print(f"\n助手：{answer_text}")

        # 持久化助手消息
        mm.messages.add(session_id, "assistant", answer_text)

        # --- 答案验证 ---
        if settings.verify_enabled:
            try:
                from rag_agent import get_hybrid_retriever
                retriever = get_hybrid_retriever()
                verify_docs = retriever.search(question)
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


def _handle_command(mm: MemoryManager, raw: str) -> None:
    """处理 / 命令。"""
    parts = raw.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd == "/list":
        print("\n会话列表：")
        sessions = mm.sessions.list()
        if not sessions:
            print("  (无会话)")
            return
        for i, s in enumerate(sessions, start=1):
            print(f"  [{i}] {s.name}  ({s.message_count}轮)  {s.updated_at[:16]}")

    elif cmd == "/new":
        name = arg if arg else input("会话名称: ").strip()
        if not name:
            name = "默认会话"
        session = mm.sessions.create(name)
        print(f"✓ 已创建并切换到「{session.name}」。")

    elif cmd == "/switch":
        if not arg:
            print("用法: /switch <编号>")
            return
        sessions = mm.sessions.list()
        try:
            idx = int(arg) - 1
            if 0 <= idx < len(sessions):
                session = sessions[idx]
                mm.sessions.update(session.id)
                print(f"已切换到「{session.name}」。")
            else:
                print("无效的会话编号。")
        except ValueError:
            print("请输入有效的数字编号。")

    elif cmd == "/delete":
        if not arg:
            print("用法: /delete <编号>")
            return
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

    else:
        print(f"未知命令: {cmd}。可用: /new /list /switch /delete")


if __name__ == "__main__":
    main()
