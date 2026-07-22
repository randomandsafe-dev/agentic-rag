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

    匹配优先级: user_id → role → default。
    规则: deny 优先于 allow（denied_kbs > allowed_kbs）。
    """

    def __init__(self, policy_path: str | Path | None = None) -> None:
        """Args:
            policy_path: policy.yaml 路径；None 时使用项目根目录下的 config/policy.yaml。
        """
        self._users: dict[str, dict[str, Any]] = {}
        self._roles: dict[str, dict[str, Any]] = {}
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
        self._roles = config.get("roles", {})
        self._default = config.get("default", {"role": "viewer", "allowed_kbs": []})
        self._validate_schema()
        self._loaded = True

    def _validate_schema(self) -> None:
        """轻量 schema 校验：检查字段类型，异常时输出 warning 并使用安全 fallback。"""
        for section_name, section in [
            ("users", self._users),
            ("roles", self._roles),
        ]:
            if not isinstance(section, dict):
                logger.warning(
                    "policy.yaml: '%s' 应为 dict，当前类型 %s，已重置为空。",
                    section_name, type(section).__name__,
                )
                if section_name == "users":
                    self._users = {}
                else:
                    self._roles = {}
                continue

            for key, policy in section.items():
                self._validate_policy_entry(f"{section_name}.{key}", policy)

        self._validate_policy_entry("default", self._default)

    def _validate_policy_entry(self, path: str, policy: Any) -> None:
        """校验单个策略条目的字段类型。"""
        if not isinstance(policy, dict):
            logger.warning(
                "policy.yaml: '%s' 应为 dict，已跳过。", path,
            )
            return

        for field in ("allowed_kbs", "denied_kbs"):
            value = policy.get(field)
            if value is not None and not isinstance(value, list):
                logger.warning(
                    "policy.yaml: '%s.%s' 应为 list，当前类型 %s，已重置为 []。",
                    path, field, type(value).__name__,
                )
                policy[field] = []

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get_effective_policy(self, user: UserContext) -> dict[str, Any]:
        """字段级合并获取用户最终权限策略。

        继承链: default → role override → user override。
        未声明字段继续继承，已声明字段覆盖。

        Returns:
            {"allowed_kbs": [...], "denied_kbs": [...], "allowed_levels": [...]}
        """
        self._ensure_loaded()

        # 1. base = default
        effective = {
            "allowed_kbs": list(self._default.get("allowed_kbs", [])),
            "denied_kbs": list(self._default.get("denied_kbs", [])),
            "allowed_levels": list(
                self._default.get("allowed_levels", ["public"])
            ),
        }

        # 2. role 覆盖
        role_policy = self._roles.get(user.role)
        if role_policy is not None:
            _field_override(effective, role_policy)

        # 3. user 覆盖
        user_policy = self._users.get(user.user_id)
        if user_policy is not None:
            _field_override(effective, user_policy)

        return effective

    def is_allowed(self, user: UserContext, domain_id: str) -> bool:
        """检查 user 是否有权访问 domain_id。"""
        return self._check_policy(self.get_effective_policy(user), domain_id)

    def is_level_allowed(self, user: UserContext, level: str) -> bool:
        """检查 user 是否有权访问指定 access_level 的 KB。"""
        effective = self.get_effective_policy(user)
        allowed_levels = effective.get("allowed_levels", ["public"])
        return "*" in allowed_levels or level in allowed_levels

    @staticmethod
    def _check_policy(policy: dict[str, Any], domain_id: str) -> bool:
        """检查单个策略是否允许访问 domain_id。

        规则: deny 优先于 allow。
        """
        denied_kbs: list[str] = policy.get("denied_kbs", [])
        if "*" in denied_kbs or domain_id in denied_kbs:
            return False

        allowed_kbs: list[str] = policy.get("allowed_kbs", [])
        if "*" in allowed_kbs:
            return True
        return domain_id in allowed_kbs


def _field_override(base: dict[str, Any], override: dict[str, Any]) -> None:
    """override 中显式声明的字段覆盖 base 对应字段。"""
    for field in ("allowed_kbs", "denied_kbs", "allowed_levels"):
        if field in override:
            base[field] = list(override[field])


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

        effective = self._policy.get_effective_policy(user)

        allowed = []
        for d in domains:
            # 1. domain_id 权限检查 (deny > allow)
            if not AccessPolicy._check_policy(effective, d.id):
                continue
            # 2. access_level 权限检查
            allowed_levels = effective.get("allowed_levels", ["public"])
            if "*" not in allowed_levels and d.access_level not in allowed_levels:
                continue
            allowed.append(d)

        # 安全网：至少保留 default domain
        if not allowed:
            for d in domains:
                if d.default:
                    return [d]
            # 无标记 default → 返回第一个
            if domains:
                return [domains[0]]

        return allowed
