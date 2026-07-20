"""Memory Layer 数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Session:
    """会话元数据。

    同时作为 LangGraph 的 thread_id，一个会话 = 一个 LangGraph thread。
    """

    id: str
    name: str
    status: str = "active"  # active | archived
    created_at: str = ""
    updated_at: str = ""
    message_count: int = 0  # 动态计算，非存储字段

    # ---- 预留字段（V1 建表但不写入业务逻辑） ----
    user_id: str | None = None
    kb_ids: list[str] | None = None
    project: str | None = None
    agent_config: dict | None = None
    metadata: dict | None = None


@dataclass
class Message:
    """单条对话消息。"""

    session_id: str
    role: str  # user | assistant | system | tool
    content: str
    id: int | None = None
    created_at: str = ""

    # ---- 预留字段 ----
    agent_id: str | None = None
    tool_calls: dict | None = None
    token_count: int | None = None
    metadata: dict | None = None


@dataclass
class Fact:
    """长期记忆事实（V1 仅定义，不实现业务逻辑）。"""

    id: str
    session_id: str
    key: str
    value: str
    category: str = "knowledge"  # preference | knowledge | custom
    metadata: dict | None = None
    created_at: str = ""
