"""Permission Layer — 知识库级访问控制。

Phase 3：纯 Python + YAML 策略，零外部依赖。
不涉及 Router / Registry / Retriever / Agent。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from knowledge.domain import KnowledgeDomain

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# UserContext
# ------------------------------------------------------------------


@dataclass
class UserContext:
    """最小用户身份标识。

    Phase 3 基于 role 做访问控制，不引入认证机制。
    """

    user_id: str
    role: str = "viewer"
    metadata: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------
# AccessPolicy
# ------------------------------------------------------------------


class AccessPolicy:
    """加载 policy.yaml 并提供用户权限查询。

    规则优先级：
    1. allowed_kbs 包含 "*" → 全部允许
    2. domain_id 在 allowed_kbs 中 → 允许
    3. 回退到 default 规则
    4. 无匹配 → 拒绝
    """

    def __init__(self, policy_path: str | Path | None = None) -> None:
        """Args:
            policy_path: policy.yaml 路径；None 时使用项目根目录下的 config/policy.yaml。
        """
        self._users: dict[str, dict[str, Any]] = {}
        self._default: dict[str, Any] = {"role": "viewer", "allowed_kbs": []}
        self._loaded = False

        if policy_path is None:
            policy_path = Path(__file__).resolve().parent.parent / "config" / "policy.yaml"
        self._policy_path = Path(policy_path)

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """惰性加载 policy.yaml；文件不存在时使用默认策略。"""
        if self._loaded:
            return

        if not self._policy_path.exists():
            self._loaded = True
            self._default = {"role": "viewer", "allowed_kbs": ["*"]}
            logger.warning("policy.yaml 未找到 (%s)，使用默认策略：全部允许。", self._policy_path)
            return

        try:
            with open(self._policy_path, "r", encoding="utf-8") as fh:
                config = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise RuntimeError(
                f"policy.yaml 格式错误 ({self._policy_path}): {exc}"
            ) from exc

        self._users = config.get("users", {})
        self._default = config.get("default", {"role": "viewer", "allowed_kbs": []})
        self._loaded = True

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def is_allowed(self, user: UserContext, domain_id: str) -> bool:
        """检查 user 是否有权访问 domain_id。"""
        self._ensure_loaded()

        policy = self._users.get(user.user_id)
        if policy is None:
            # 用户不在策略中 → 使用默认规则
            allowed = self._default.get("allowed_kbs", [])
            return "*" in allowed or domain_id in allowed

        allowed_kbs: list[str] = policy.get("allowed_kbs", [])
        if "*" in allowed_kbs:
            return True
        return domain_id in allowed_kbs


# ------------------------------------------------------------------
# AccessGuard
# ------------------------------------------------------------------


class AccessGuard:
    """权限过滤器 —— 根据 UserContext 过滤可访问的 domain 列表。

    纯过滤器，不访问 Chroma / Retriever / Router / Registry。
    """

    def __init__(self, policy: AccessPolicy | None = None) -> None:
        """Args:
            policy: AccessPolicy 实例；None 时使用默认路径加载。
        """
        self._policy = policy or AccessPolicy()

    def filter_domains(
        self,
        user: UserContext | None,
        domains: list[KnowledgeDomain],
    ) -> list[KnowledgeDomain]:
        """返回 user 有权访问的 domain 子集。

        Args:
            user: 用户身份；None 表示无过滤（返回全部 domains）。
            domains: 全部可用 domain 列表。

        Returns:
            过滤后的 domain 列表；保证非空。
        """
        # 无用户 → 向后兼容：全部通过
        if user is None:
            return list(domains)

        allowed = [d for d in domains if self._policy.is_allowed(user, d.id)]

        # 安全网：至少保留 default domain
        if not allowed:
            for d in domains:
                if d.default:
                    return [d]
            # 无标记 default → 返回第一个
            if domains:
                return [domains[0]]

        return allowed
