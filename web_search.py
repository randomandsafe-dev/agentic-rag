"""Governed web-search pipeline: score, rerank, and format external evidence."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from urllib.parse import urlparse

from langchain_core.documents import Document

from config import settings


TRUSTED_SUFFIXES = (".gov", ".edu", ".ac.")
TRUSTED_DOMAINS = {
    "who.int", "un.org", "openai.com", "docs.python.org", "wikipedia.org",
}


def domain_trust_score(url: str) -> float:
    """Return a transparent, lightweight domain-trust prior between 0 and 1."""
    domain = urlparse(url).netloc.lower().removeprefix("www.")
    if domain in TRUSTED_DOMAINS or domain.endswith(TRUSTED_SUFFIXES):
        return 1.0
    if domain.endswith((".org", ".com", ".net")):
        return 0.6
    return 0.4


def recency_score(published_date: str | None) -> float:
    """Score a result's publication date when the search provider supplies one."""
    if not published_date:
        return 0.5
    try:
        value = published_date.replace("Z", "+00:00")
        published = datetime.fromisoformat(value)
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        age_days = max(0, (datetime.now(timezone.utc) - published).days)
        if age_days <= 7:
            return 1.0
        if age_days <= 30:
            return 0.8
        if age_days <= 365:
            return 0.6
    except ValueError:
        pass
    return 0.4


@lru_cache(maxsize=1)
def get_web_reranker():
    """Load the same optional Cross-Encoder used by local retrieval."""
    if not settings.reranker_enabled:
        return None
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(settings.reranker_model, max_length=512)
    except Exception:
        return None


def rerank_web_documents(query: str, documents: list[Document]) -> list[Document]:
    """Rerank external evidence by semantic relevance; retain provider order on fallback."""
    reranker = get_web_reranker()
    if reranker is None or len(documents) < 2:
        return documents[: settings.top_k]
    try:
        pairs = [(query, doc.page_content[:2000]) for doc in documents]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(documents, scores), key=lambda item: item[1], reverse=True)
        result = []
        for document, score in ranked[: settings.top_k]:
            document.metadata["rerank_score"] = float(score)
            result.append(document)
        return result
    except Exception:
        return documents[: settings.top_k]


def search_web_documents(query: str) -> list[Document]:
    """Search Tavily, apply trust/recency ordering, then semantic reranking."""
    if not settings.tavily_api_key:
        raise RuntimeError("未配置 TAVILY_API_KEY，无法执行联网搜索。")
    from tavily import TavilyClient

    response = TavilyClient(api_key=settings.tavily_api_key).search(
        query=query,
        max_results=max(settings.web_search_max_results, settings.top_k),
        search_depth="advanced",
    )
    documents: list[Document] = []
    for result in response.get("results", []):
        url = result.get("url", "")
        published_date = result.get("published_date")
        documents.append(
            Document(
                page_content=result.get("content", ""),
                metadata={
                    "source": url,
                    "title": result.get("title", "未命名网页"),
                    "published_date": published_date,
                    "provider_score": float(result.get("score", 0.0)),
                    "domain_trust": domain_trust_score(url),
                    "recency_score": recency_score(published_date),
                    "source_type": "web",
                },
            )
        )
    # Before the semantic reranker is available, use source quality as a deterministic prior.
    documents.sort(
        key=lambda doc: (
            doc.metadata["provider_score"] * 0.6
            + doc.metadata["domain_trust"] * 0.25
            + doc.metadata["recency_score"] * 0.15
        ),
        reverse=True,
    )
    return rerank_web_documents(query, documents)


def format_web_documents(documents: list[Document]) -> str:
    if not documents:
        return "【联网搜索结果】\n未找到可用网页。"
    sections = []
    for index, document in enumerate(documents, start=1):
        title = document.metadata.get("title", "未命名网页")
        url = document.metadata.get("source", "")
        date = document.metadata.get("published_date") or "发布时间未知"
        trust = document.metadata.get("domain_trust", 0.0)
        sections.append(
            f"[联网来源 {index}: {title}]({url})\n"
            f"发布时间：{date}；域名可信度：{trust:.1f}\n"
            f"{document.page_content}"
        )
    return "【联网搜索结果】\n" + "\n\n".join(sections)
