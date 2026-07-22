"""统一的 LLM 创建入口。

所有模块（Agent、Router、Pipeline、Verifier）通过此函数获取 LLM 实例，
禁止直接手写 ChatOpenAI(...)。
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from config import settings


@lru_cache(maxsize=4)
def create_llm(*, temperature: float = 0, model: str | None = None) -> ChatOpenAI:
    """创建 ChatOpenAI 实例。

    Args:
        temperature: 温度参数。Router/Judge/Verifier 用 0（确定性），Agent 可用更高值。
        model: 可选模型覆盖，None 时使用 settings.model。

    Returns:
        配置好的 ChatOpenAI 实例，已设置 model / api_key / base_url / temperature。
    """
    return ChatOpenAI(
        model=model or settings.model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=temperature,
    )
