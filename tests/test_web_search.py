"""web_search 模块单元测试 —— 域名可信度 / 时效性评分 / 格式化 / 重排序。

全部使用纯函数测试，不依赖 Tavily API / Cross-Encoder / 外部服务。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from web_search import (
    domain_trust_score,
    format_web_documents,
    recency_score,
    rerank_web_documents,
)


# ============================================================
# domain_trust_score
# ============================================================


def test_gov_suffix_scores_1():
    assert domain_trust_score("https://www.example.gov/page") == 1.0


def test_edu_suffix_scores_1():
    assert domain_trust_score("https://cs.stanford.edu/paper") == 1.0


def test_ac_suffix_scores_1():
    """`.ac.` 后缀精确匹配域名末尾 —— 注意：仅匹配 `foo.ac.` 形式，
    不会匹配 `.ac.jp` / `.ac.uk` 等常见学术二级域名（这是 web_search.py 的一个已知局限）。
    """
    assert domain_trust_score("https://example.ac./research") == 1.0


def test_trusted_domain_scores_1():
    for domain in ("who.int", "un.org", "openai.com", "docs.python.org", "wikipedia.org"):
        assert domain_trust_score(f"https://{domain}/path") == 1.0


def test_org_suffix_scores_0_6():
    assert domain_trust_score("https://example.org/page") == 0.6


def test_com_suffix_scores_0_6():
    assert domain_trust_score("https://example.com/page") == 0.6


def test_net_suffix_scores_0_6():
    assert domain_trust_score("https://example.net/page") == 0.6


def test_unknown_tld_scores_0_4():
    assert domain_trust_score("https://example.xyz/page") == 0.4


def test_no_subdomain_www_removed():
    assert domain_trust_score("https://www.example.org/page") == 0.6


def test_empty_url_handled():
    result = domain_trust_score("")
    assert 0.0 <= result <= 1.0


# ============================================================
# recency_score
# ============================================================


def test_published_none_returns_0_5():
    assert recency_score(None) == 0.5


def test_invalid_date_returns_0_4():
    """无法解析的日期字符串触发 ValueError → 0.4；空字符串走 `not published_date` → 0.5。"""
    assert recency_score("not-a-date") == 0.4


def test_within_7_days_scores_1():
    from datetime import datetime, timedelta, timezone
    recent = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    assert recency_score(recent) == 1.0


def test_within_30_days_scores_0_8():
    from datetime import datetime, timedelta, timezone
    mid = (datetime.now(timezone.utc) - timedelta(days=15)).isoformat()
    assert recency_score(mid) == 0.8


def test_within_365_days_scores_0_6():
    from datetime import datetime, timedelta, timezone
    old = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
    assert recency_score(old) == 0.6


def test_older_than_365_days_scores_0_4():
    from datetime import datetime, timedelta, timezone
    ancient = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
    assert recency_score(ancient) == 0.4


def test_z_suffix_handled():
    from datetime import datetime, timedelta, timezone
    recent = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert recency_score(recent) == 1.0


def test_timezone_offset_handled():
    from datetime import datetime, timedelta, timezone
    recent = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    assert recency_score(recent) == 1.0


def test_boundary_7_days():
    from datetime import datetime, timedelta, timezone
    exactly_7 = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    assert recency_score(exactly_7) == 1.0


def test_boundary_30_days():
    from datetime import datetime, timedelta, timezone
    exactly_30 = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    assert recency_score(exactly_30) == 0.8


# ============================================================
# format_web_documents
# ============================================================


def test_format_empty_returns_empty_message():
    result = format_web_documents([])
    assert "未找到可用网页" in result


def test_format_single_document():
    doc = Document(
        page_content="test content",
        metadata={
            "source": "https://example.com",
            "title": "测试标题",
            "published_date": "2026-07-20",
            "domain_trust": 0.8,
        },
    )
    result = format_web_documents([doc])
    assert "联网来源 1" in result
    assert "测试标题" in result
    assert "https://example.com" in result
    assert "test content" in result
    assert "2026-07-20" in result


def test_format_multiple_documents_uses_sequential_indices():
    docs = [
        Document(page_content=f"content {i}", metadata={
            "source": f"https://example{i}.com",
            "title": f"Title {i}",
        })
        for i in range(3)
    ]
    result = format_web_documents(docs)
    assert "联网来源 1" in result
    assert "联网来源 2" in result
    assert "联网来源 3" in result


def test_format_missing_metadata_defaults():
    doc = Document(page_content="content", metadata={"source": "https://x.com"})
    result = format_web_documents([doc])
    assert "未命名网页" in result
    assert "发布时间未知" in result


# ============================================================
# rerank_web_documents
# ============================================================


def make_docs(n: int) -> list[Document]:
    return [
        Document(
            page_content=f"document {i}",
            metadata={"source": f"https://example{i}.com", "title": f"Title {i}"},
        )
        for i in range(n)
    ]


def test_rerank_no_reranker_returns_top_k_unsorted():
    """Without reranker, just truncate to top_k."""
    docs = make_docs(10)
    with patch("web_search.get_web_reranker", return_value=None):
        result = rerank_web_documents("query", docs)
    assert len(result) <= 4  # default top_k


def test_rerank_single_doc_returns_unchanged():
    docs = make_docs(1)
    with patch("web_search.get_web_reranker", return_value=None):
        result = rerank_web_documents("query", docs)
    assert len(result) == 1
    assert result[0].page_content == "document 0"


def test_rerank_with_cross_encoder_sorts_by_score():
    docs = make_docs(5)
    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = [0.3, 0.9, 0.1, 0.7, 0.5]
    with patch("web_search.get_web_reranker", return_value=mock_reranker):
        result = rerank_web_documents("query", docs)
    # Sorted descending by score: 0.9, 0.7, 0.5, 0.3
    assert result[0].page_content == "document 1"  # score 0.9
    assert result[1].page_content == "document 3"  # score 0.7
    assert "rerank_score" in result[0].metadata


def test_rerank_cross_encoder_error_falls_back_to_truncation():
    docs = make_docs(8)
    mock_reranker = MagicMock()
    mock_reranker.predict.side_effect = RuntimeError("model crash")
    with patch("web_search.get_web_reranker", return_value=mock_reranker):
        result = rerank_web_documents("query", docs)
    assert len(result) <= 4  # falls back to top_k truncation


def test_rerank_respects_top_k():
    """Result count should not exceed settings.top_k."""
    docs = make_docs(20)
    mock_reranker = MagicMock()
    mock_reranker.predict.return_value = list(range(20, 0, -1))
    with patch("web_search.get_web_reranker", return_value=mock_reranker):
        result = rerank_web_documents("query", docs)
    assert len(result) <= 4


def test_rerank_fewer_docs_than_top_k_returns_all():
    docs = make_docs(2)
    with patch("web_search.get_web_reranker", return_value=None):
        result = rerank_web_documents("query", docs)
    assert len(result) == 2
