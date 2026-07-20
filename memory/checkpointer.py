"""CheckpointerAdapter：将 LangGraph SqliteSaver 接入 MemoryBackend。

此适配器复用 SqliteBackend 的 SQLite 连接，
确保 LangGraph checkpoint 表与业务表在同一 DB 文件中。
"""

from __future__ import annotations

from memory.backend import SqliteBackend


class CheckpointerAdapter:
    """包装 LangGraph 的 SqliteSaver，与业务表共享同一个 SQLite 连接。

    用法::

        adapter = CheckpointerAdapter(backend)
        checkpointer = adapter.get()
        config = adapter.get_config(session_id)
        agent.invoke({"messages": [...]}, config=config)
    """

    def __init__(self, backend: SqliteBackend) -> None:
        self._backend = backend

    def get(self):
        """返回当前线程专属的 SqliteSaver 实例（供 create_agent 使用）。

        Streamlit 每次交互都可能在不同线程执行脚本。SQLite 连接不能跨线程
        复用，因此不能缓存带有连接的 SqliteSaver；每次调用都创建一个新的连接。
        """
        from langgraph.checkpoint.sqlite import SqliteSaver

        return SqliteSaver(self._backend._get_conn())

    def get_config(self, session_id: str) -> dict:
        """返回 LangGraph 的 config dict。

        thread_id = session_id，保证业务会话与 LangGraph 线程一一对应。
        """
        return {"configurable": {"thread_id": session_id}}
