"""Enterprise-oriented routing policy for local RAG and web search."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SearchRoute(str, Enum):
    LOCAL_ONLY = "local_only"
    WEB_ONLY = "web_only"
    BOTH = "both"


@dataclass(frozen=True)
class RouteDecision:
    route: SearchRoute
    reason: str

    @property
    def use_local(self) -> bool:
        return self.route in {SearchRoute.LOCAL_ONLY, SearchRoute.BOTH}

    @property
    def use_web(self) -> bool:
        return self.route in {SearchRoute.WEB_ONLY, SearchRoute.BOTH}

    @property
    def label(self) -> str:
        return {
            SearchRoute.LOCAL_ONLY: "仅本地知识库",
            SearchRoute.WEB_ONLY: "仅联网搜索",
            SearchRoute.BOTH: "本地知识库 + 联网搜索",
        }[self.route]


# A conservative policy: do not send potentially sensitive queries to third-party search.
SENSITIVE_TERMS = (
    "密码", "密钥", "api key", "token", "身份证", "手机号", "银行账号",
    "客户名单", "合同编号", "内部机密", "confidential", "secret",
)
LOCAL_TERMS = (
    "知识库", "内部", "公司", "项目", "文档", "上传", "文件", "制度",
    "流程", "会议纪要", "客户", "员工", "私有",
)
WEB_TERMS = (
    "今天", "今日", "最新", "目前", "实时", "新闻", "股价", "汇率", "天气",
    "价格", "政策", "行情", "刚刚", "现在", "本周", "本月", "latest", "news",
    "current", "price", "weather",
)


def route_question(
    question: str,
    requested_mode: str = "auto",
    web_available: bool = False,
) -> RouteDecision:
    """Choose a source route while enforcing privacy and availability guardrails."""
    normalized = question.casefold()
    has_sensitive = any(term.casefold() in normalized for term in SENSITIVE_TERMS)
    if has_sensitive:
        return RouteDecision(
            SearchRoute.LOCAL_ONLY,
            "检测到可能包含敏感信息，已禁止向第三方联网搜索服务发送查询。",
        )

    if not web_available:
        return RouteDecision(
            SearchRoute.LOCAL_ONLY,
            "未配置联网搜索服务，已使用本地知识库。",
        )

    manual_routes = {
        "local_only": SearchRoute.LOCAL_ONLY,
        "web_only": SearchRoute.WEB_ONLY,
        "both": SearchRoute.BOTH,
    }
    if requested_mode in manual_routes:
        return RouteDecision(manual_routes[requested_mode], "根据界面中的手动路由选择。")

    has_local_intent = any(term.casefold() in normalized for term in LOCAL_TERMS)
    has_web_intent = any(term.casefold() in normalized for term in WEB_TERMS)
    if has_local_intent and has_web_intent:
        return RouteDecision(SearchRoute.BOTH, "同时检测到内部资料与实时公开信息需求。")
    if has_web_intent:
        return RouteDecision(SearchRoute.WEB_ONLY, "检测到实时或公开信息需求。")
    return RouteDecision(SearchRoute.LOCAL_ONLY, "默认优先使用本地知识库，避免不必要的外部数据发送。")
