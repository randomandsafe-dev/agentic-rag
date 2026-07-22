"""Knowledge Layer 包 —— 知识库注册、检索、路由。"""

from knowledge.domain import KnowledgeDomain
from knowledge.registry import KnowledgeBaseRegistry
from knowledge.router import RoutingDecision, RouterStrategy
from knowledge.service import KnowledgeService, get_knowledge_service

__all__ = [
    "KnowledgeDomain",
    "KnowledgeBaseRegistry",
    "KnowledgeService",
    "get_knowledge_service",
    "RoutingDecision",
    "RouterStrategy",
]
