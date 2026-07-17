"""项目配置：所有可变配置都来自环境变量。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")


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

    def validate(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "未找到 OPENAI_API_KEY。请复制 .env.example 为 .env 并填写密钥。"
            )
        if self.embedding_provider not in {"local", "openai"}:
            raise RuntimeError(
                "EMBEDDING_PROVIDER 只能是 local 或 openai。"
            )


settings = Settings()
