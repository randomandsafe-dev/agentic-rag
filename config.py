"""项目配置：所有可变配置都来自环境变量。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")


def _bool_env(key: str, default: bool = True) -> bool:
    """解析布尔型环境变量。"""
    val = os.getenv(key, "").strip().lower()
    if not val:
        return default
    return val in ("true", "1", "yes", "on")


@dataclass(frozen=True)
class Settings:
    """Agentic RAG 的运行时设置。"""

    data_dir: Path = ROOT_DIR / "data"
    persist_dir: Path = ROOT_DIR / "chroma_db"
    collection_name: str = "knowledge_base"
    model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "local").lower()
    local_embedding_model: str = os.getenv(
        "LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
    )
    openai_embedding_model: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )
    api_key: str | None = os.getenv("OPENAI_API_KEY")
    base_url: str | None = os.getenv("OPENAI_BASE_URL")
    top_k: int = int(os.getenv("RAG_TOP_K", "4"))
    # --- 混合检索 ---
    hybrid_enabled: bool = os.getenv("HYBRID_ENABLED", "true").lower() == "true"
    # --- Reranker ---
    reranker_enabled: bool = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "10"))
    # --- 答案验证 ---
    verify_enabled: bool = os.getenv("VERIFY_ENABLED", "true").lower() == "true"

    # ---- 会话记忆 ----
    memory_db_path: Path = ROOT_DIR / os.getenv("MEMORY_DB_PATH", "conversations.db")
    session_window: int = int(os.getenv("SESSION_WINDOW", "20"))

    # ---- 检索增强配置 (新增) ----
    rewrite_enabled: bool = field(
        default_factory=lambda: _bool_env("REWRITE_ENABLED", True)
    )
    relevance_judge_enabled: bool = field(
        default_factory=lambda: _bool_env("RELEVANCE_JUDGE_ENABLED", True)
    )
    relevance_strategy: str = os.getenv("RELEVANCE_STRATEGY", "llm").lower()
    max_retries: int = int(os.getenv("MAX_RETRIES", "2"))
    relevance_threshold: int = int(os.getenv("RELEVANCE_THRESHOLD", "2"))
    rewrite_model: str | None = os.getenv("REWRITE_MODEL") or None

    def validate(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "未找到 OPENAI_API_KEY。请复制 .env.example 为 .env 并填写密钥。"
            )
        if self.embedding_provider not in {"local", "openai"}:
            raise RuntimeError(
                "EMBEDDING_PROVIDER 只能是 local 或 openai。"
            )
        if self.relevance_strategy not in {"llm", "vector", "hybrid"}:
            raise RuntimeError(
                "RELEVANCE_STRATEGY 只能是 llm、vector 或 hybrid。"
            )


settings = Settings()
