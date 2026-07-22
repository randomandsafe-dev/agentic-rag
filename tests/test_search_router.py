"""search_router 模块单元测试 —— 路由决策 / 敏感词检测 / 模式切换。

全部纯函数测试，不依赖外部服务。
"""

from __future__ import annotations

import pytest

from search_router import (
    LOCAL_TERMS,
    SENSITIVE_TERMS,
    WEB_TERMS,
    RouteDecision,
    SearchRoute,
    route_question,
)


# ============================================================
# SearchRoute enum
# ============================================================


def test_search_route_values():
    assert SearchRoute.LOCAL_ONLY == "local_only"
    assert SearchRoute.WEB_ONLY == "web_only"
    assert SearchRoute.BOTH == "both"


# ============================================================
# RouteDecision dataclass
# ============================================================


def test_local_only_decision():
    d = RouteDecision(SearchRoute.LOCAL_ONLY, "reason")
    assert d.use_local is True
    assert d.use_web is False
    assert d.label == "仅本地知识库"


def test_web_only_decision():
    d = RouteDecision(SearchRoute.WEB_ONLY, "reason")
    assert d.use_local is False
    assert d.use_web is True
    assert d.label == "仅联网搜索"


def test_both_decision():
    d = RouteDecision(SearchRoute.BOTH, "reason")
    assert d.use_local is True
    assert d.use_web is True
    assert d.label == "本地知识库 + 联网搜索"


def test_decision_is_frozen():
    with pytest.raises(Exception):
        RouteDecision(SearchRoute.LOCAL_ONLY, "").route = SearchRoute.BOTH  # type: ignore[misc]


# ============================================================
# route_question — default (auto) mode
# ============================================================


def test_auto_defaults_to_local():
    """Auto mode without specific intent → local only."""
    decision = route_question("什么是Python", requested_mode="auto", web_available=True)
    assert decision.route == SearchRoute.LOCAL_ONLY


def test_auto_web_intent_triggers_web_only():
    """Time-sensitive terms → web search."""
    for term in ("今天天气", "最新新闻", "现在的股价"):
        decision = route_question(term, requested_mode="auto", web_available=True)
        assert decision.route == SearchRoute.WEB_ONLY, f"Failed for: {term}"


def test_auto_local_plus_web_intent_triggers_both():
    """Internal doc + real-time terms → both."""
    decision = route_question(
        "我们公司项目的最新政策是什么",
        requested_mode="auto",
        web_available=True,
    )
    assert decision.route == SearchRoute.BOTH


def test_auto_web_unavailable_forces_local():
    """Even with time-sensitive terms, no web → local."""
    decision = route_question("今天最新新闻", requested_mode="auto", web_available=False)
    assert decision.route == SearchRoute.LOCAL_ONLY


# ============================================================
# route_question — sensitive term detection
# ============================================================


def test_sensitive_term_forces_local():
    """Privacy-sensitive queries must never leave local."""
    sensitive_queries = [
        "我的密码是什么",
        "API key 怎么配置",
        "忘记了自己的身份证号码",
        "客户名单在哪里",
        "合同编号 2024-001",
        "这是内部机密",
        "this is confidential",
        "secret project details",
    ]
    for q in sensitive_queries:
        decision = route_question(q, requested_mode="auto", web_available=True)
        assert decision.route == SearchRoute.LOCAL_ONLY, (
            f"Sensitive query should be local: {q}"
        )


def test_sensitive_term_overrides_manual_mode():
    """Even in manual 'both' mode, sensitive terms force local."""
    decision = route_question(
        "我的密码是多少",
        requested_mode="both",
        web_available=True,
    )
    assert decision.route == SearchRoute.LOCAL_ONLY


def test_sensitive_term_overrides_web_only():
    decision = route_question(
        "客户的手机号在哪里",
        requested_mode="web_only",
        web_available=True,
    )
    assert decision.route == SearchRoute.LOCAL_ONLY


def test_sensitive_detection_case_insensitive():
    decision = route_question("問我 Token 是什麼", requested_mode="auto", web_available=True)
    assert decision.route == SearchRoute.LOCAL_ONLY


# ============================================================
# route_question — manual modes
# ============================================================


def test_manual_local_only():
    decision = route_question("latest news today", requested_mode="local_only", web_available=True)
    assert decision.route == SearchRoute.LOCAL_ONLY


def test_manual_web_only():
    decision = route_question("内部项目文档", requested_mode="web_only", web_available=True)
    assert decision.route == SearchRoute.WEB_ONLY


def test_manual_both():
    decision = route_question("hello", requested_mode="both", web_available=True)
    assert decision.route == SearchRoute.BOTH


def test_manual_web_only_but_web_unavailable():
    """Requested web_only but no Tavily key → fall back to local."""
    decision = route_question("some query", requested_mode="web_only", web_available=False)
    # web_unavailable check runs AFTER sensitive check, BEFORE manual routing
    # So manual mode is still honoured even without web...
    # Actually the code checks web_available BEFORE manual routing, so web_only + no web → local
    assert decision.route == SearchRoute.LOCAL_ONLY


# ============================================================
# term lists — regression guards
# ============================================================


def test_sensitive_terms_not_empty():
    assert len(SENSITIVE_TERMS) > 5


def test_local_terms_not_empty():
    assert len(LOCAL_TERMS) > 5


def test_web_terms_not_empty():
    assert len(WEB_TERMS) > 5


def test_all_terms_are_non_empty_strings():
    for term in SENSITIVE_TERMS + LOCAL_TERMS + WEB_TERMS:
        assert isinstance(term, str)
        assert len(term.strip()) > 0
