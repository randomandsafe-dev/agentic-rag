"""SqliteBackend：MemoryBackend 的 SQLite 实现。

管理 sessions、messages、facts 三张表的 CRUD。
同时对外暴露同一个 SQLite 连接，供 CheckpointerAdapter 复用。
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from memory.base import MemoryBackend
from memory.models import Message, Session


def _now() -> str:
    """返回 ISO 8601 格式的当前 UTC 时间字符串。"""
    return datetime.now(timezone.utc).isoformat()


class SqliteBackend(MemoryBackend):
    """SQLite 持久化后端。

    ── sessions ──
    实际写入：id, name, status, created_at, updated_at
    预留字段（仅建表）：user_id, kb_ids, project, agent_config, metadata

    ── messages ──
    实际写入：session_id, role, content, created_at
    预留字段（仅建表）：agent_id, tool_calls, token_count, metadata

    ── facts ──
    仅建表，V1 无业务代码。
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._init_tables()

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """获取同一个数据库文件的连接（供 CheckpointerAdapter 复用）。"""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    # ------------------------------------------------------------------
    # 建表
    # ------------------------------------------------------------------

    def _init_tables(self) -> None:
        """创建所有表（幂等：IF NOT EXISTS）。"""
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id            TEXT PRIMARY KEY,
                    name          TEXT NOT NULL,
                    user_id       TEXT,
                    kb_ids        TEXT,
                    project       TEXT,
                    status        TEXT DEFAULT 'active',
                    agent_config  TEXT,
                    metadata      TEXT,
                    created_at    TEXT NOT NULL,
                    updated_at    TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id    TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role          TEXT NOT NULL,
                    agent_id      TEXT,
                    content       TEXT NOT NULL,
                    tool_calls    TEXT,
                    token_count   INTEGER,
                    metadata      TEXT,
                    created_at    TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id, created_at);

                CREATE TABLE IF NOT EXISTS facts (
                    id            TEXT PRIMARY KEY,
                    session_id    TEXT REFERENCES sessions(id),
                    key           TEXT NOT NULL,
                    value         TEXT NOT NULL,
                    category      TEXT DEFAULT 'knowledge',
                    metadata      TEXT,
                    created_at    TEXT NOT NULL
                );
            """)
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    def create_session(self, name: str) -> Session:
        session = Session(
            id=str(uuid.uuid4()),
            name=name,
            status="active",
            created_at=_now(),
            updated_at=_now(),
        )
        conn = self._get_conn()
        try:
            conn.execute(
                "INSERT INTO sessions (id, name, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session.id, session.name, session.status,
                 session.created_at, session.updated_at),
            )
            conn.commit()
        finally:
            conn.close()
        return session

    def list_sessions(self) -> list[Session]:
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT s.id, s.name, s.status, s.created_at, s.updated_at, "
                "       s.user_id, s.kb_ids, s.project, s.agent_config, s.metadata, "
                "       COUNT(m.id) AS msg_count "
                "FROM sessions s "
                "LEFT JOIN messages m ON m.session_id = s.id "
                "GROUP BY s.id "
                "ORDER BY s.updated_at DESC"
            ).fetchall()
        finally:
            conn.close()

        results: list[Session] = []
        for row in rows:
            session = Session(
                id=row[0],
                name=row[1],
                status=row[2],
                created_at=row[3],
                updated_at=row[4],
                user_id=row[5],
                kb_ids=None,
                project=row[7],
                agent_config=None,
                metadata=None,
                message_count=row[10],
            )
            results.append(session)
        return results

    def get_session(self, session_id: str) -> Session | None:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT s.id, s.name, s.status, s.created_at, s.updated_at, "
                "       s.user_id, s.kb_ids, s.project, s.agent_config, s.metadata, "
                "       COUNT(m.id) AS msg_count "
                "FROM sessions s "
                "LEFT JOIN messages m ON m.session_id = s.id "
                "WHERE s.id = ? "
                "GROUP BY s.id",
                (session_id,),
            ).fetchone()
        finally:
            conn.close()

        if row is None:
            return None
        return Session(
            id=row[0],
            name=row[1],
            status=row[2],
            created_at=row[3],
            updated_at=row[4],
            user_id=row[5],
            kb_ids=None,
            project=row[7],
            agent_config=None,
            metadata=None,
            message_count=row[10],
        )

    def update_session(self, session_id: str, **kwargs: object) -> None:
        allowed = {"name", "status", "updated_at"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        updates.setdefault("updated_at", _now())
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [session_id]

        conn = self._get_conn()
        try:
            conn.execute(
                f"UPDATE sessions SET {set_clause} WHERE id = ?", values
            )
            conn.commit()
        finally:
            conn.close()

    def delete_session(self, session_id: str) -> None:
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Message CRUD
    # ------------------------------------------------------------------

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> Message:
        now = _now()
        conn = self._get_conn()
        try:
            cursor = conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, role, content, now),
            )
            # 同时刷新会话的 updated_at
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            conn.commit()
            msg_id = cursor.lastrowid
        finally:
            conn.close()

        return Message(
            id=msg_id,
            session_id=session_id,
            role=role,
            content=content,
            created_at=now,
        )

    def list_messages(
        self,
        session_id: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Message]:
        conn = self._get_conn()
        try:
            if limit is not None:
                rows = conn.execute(
                    "SELECT id, session_id, role, content, created_at, "
                    "       agent_id, tool_calls, token_count, metadata "
                    "FROM messages "
                    "WHERE session_id = ? "
                    "ORDER BY id ASC "
                    "LIMIT ? OFFSET ?",
                    (session_id, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, session_id, role, content, created_at, "
                    "       agent_id, tool_calls, token_count, metadata "
                    "FROM messages "
                    "WHERE session_id = ? "
                    "ORDER BY id ASC",
                    (session_id,),
                ).fetchall()
        finally:
            conn.close()

        results: list[Message] = []
        for row in rows:
            results.append(Message(
                id=row[0],
                session_id=row[1],
                role=row[2],
                content=row[3],
                created_at=row[4],
                agent_id=row[5],
                tool_calls=None,
                token_count=row[7],
                metadata=None,
            ))
        return results

    def count_messages(self, session_id: str) -> int:
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        finally:
            conn.close()
        return row[0] if row else 0

    def delete_messages(self, session_id: str) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (session_id,),
            )
            conn.commit()
        finally:
            conn.close()
