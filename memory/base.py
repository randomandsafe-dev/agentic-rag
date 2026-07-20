"""MemoryBackend 抽象接口。

所有持久化后端必须实现此接口。
V1 仅提供 SqliteBackend，未来可替换为 PostgresBackend 等。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from memory.models import Message, Session


class MemoryBackend(ABC):
    """内存后端的抽象基类。

    定义会话、消息、事实的完整 CRUD 接口。
    子类实现时需处理所有异常并转换为 MemoryError。
    """

    # ---- Session CRUD ----

    @abstractmethod
    def create_session(self, name: str) -> Session:
        """创建新会话并返回带 ID 的 Session 对象。"""
        ...

    @abstractmethod
    def list_sessions(self) -> list[Session]:
        """按更新时间降序列出所有会话。"""
        ...

    @abstractmethod
    def get_session(self, session_id: str) -> Session | None:
        """获取单个会话，不存在返回 None。"""
        ...

    @abstractmethod
    def update_session(self, session_id: str, **kwargs: object) -> None:
        """更新会话字段（name, status 等）。"""
        ...

    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        """删除会话及其关联的所有消息。"""
        ...

    # ---- Message CRUD ----

    @abstractmethod
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> Message:
        """向指定会话添加一条消息。"""
        ...

    @abstractmethod
    def list_messages(
        self,
        session_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Message]:
        """按时间倒序列出会话消息。limit 为 None 表示不限。"""
        ...

    @abstractmethod
    def count_messages(self, session_id: str) -> int:
        """返回会话的消息总数。"""
        ...

    @abstractmethod
    def delete_messages(self, session_id: str) -> None:
        """删除指定会话的所有消息。"""
        ...
