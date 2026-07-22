"""Embedding 模型工厂 —— 独立于 Agent 层，供 Knowledge 层和 Ingest 使用。

从 rag_agent.py 抽取，消除 knowledge.registry → rag_agent 的反向依赖。
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from config import settings


@lru_cache(maxsize=1)
def get_embeddings():
    """获取嵌入模型；默认本地模型，避免聊天接口不支持 embeddings 的问题。"""
    if settings.embedding_provider == "openai":
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.api_key,
            base_url=settings.base_url,
        )
    try:
        from langchain_community.embeddings import FastEmbedEmbeddings
    except ImportError as exc:
        raise RuntimeError("缺少本地嵌入依赖，请运行：pip install -r requirements.txt") from exc
    return FastEmbedEmbeddings(
        model_name=settings.local_embedding_model,
    )
