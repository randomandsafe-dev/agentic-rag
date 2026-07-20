"""MemoryManager：Memory Layer 的统一门面。

应用层和 Agent 层只通过此类访问记忆，不直接依赖 backend 或 checkpointer。

用法::

    mm = MemoryManager("conversations.db", session_window=20)
    session = mm.sessions.create("新会话")
    mm.messages.add(session.id, "user", "你好")
    agent = build_agent(checkpointer=mm.checkpointer.get())
"""

from __future__ import annotations

from memory.backend import SqliteBackend
from memory.checkpointer import CheckpointerAdapter
from memory.models import Message, Session


class _SessionStore:
    """Session CRUD 子门面。"""

    def __init__(self, backend: SqliteBackend) -> None:
        self._backend = backend

    def create(self, name: str) -> Session:
        return self._backend.create_session(name)

    def list(self) -> list[Session]:
        return self._backend.list_sessions()

    def get(self, session_id: str) -> Session | None:
        return self._backend.get_session(session_id)

    def update(self, session_id: str, **kwargs: object) -> None:
        self._backend.update_session(session_id, **kwargs)

    def delete(self, session_id: str) -> None:
        self._backend.delete_session(session_id)


class _MessageStore:
    """Message CRUD 子门面。"""

    def __init__(self, backend: SqliteBackend) -> None:
        self._backend = backend

    def add(self, session_id: str, role: str, content: str) -> Message:
        return self._backend.add_message(session_id, role, content)

    def list(
        self, session_id: str, limit: int | None = None, offset: int = 0
    ) -> list[Message]:
        return self._backend.list_messages(session_id, limit=limit, offset=offset)

    def count(self, session_id: str) -> int:
        return self._backend.count_messages(session_id)

    def delete_all(self, session_id: str) -> None:
        self._backend.delete_messages(session_id)


class MemoryManager:
    """记忆层统一门面。

    ── 属性 ──
    .sessions      → _SessionStore    会话管理
    .messages      → _MessageStore    消息管理
    .checkpointer  → CheckpointerAdapter  LangGraph checkpointer
    .session_window → int             上下文窗口大小
    """

    def __init__(self, db_path: str, session_window: int = 20) -> None:
        self._backend = SqliteBackend(db_path)
        self.sessions = _SessionStore(self._backend)
        self.messages = _MessageStore(self._backend)
        self.checkpointer = CheckpointerAdapter(self._backend)
        self.session_window = session_window

    def load_history(
        self, session_id: str
    ) -> list[Message]:
        """加载会话的窗口内消息（最近 N 条，按时间正序）。"""
        total = self.messages.count(session_id)
        if total <= self.session_window:
            return self.messages.list(session_id)
        offset = total - self.session_window
        return self.messages.list(session_id, limit=self.session_window, offset=offset)

    def load_full_history(
        self, session_id: str
    ) -> list[Message]:
        """加载会话的完整消息历史。"""
        return self.messages.list(session_id)
