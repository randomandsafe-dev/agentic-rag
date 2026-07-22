# Phase 3 — Permission Layer Design

> **Status**: Draft — awaiting review before implementation.
> **Target**: Multi-user, multi-KB access control without OAuth/JWT/database.
> **Principle**: Pure Python + YAML policy. Zero new dependencies.

---

## 1. Goal

Enable knowledge-base-level access control for multi-user scenarios.

**Scope (Phase 3)**:
- Per-user KB access control via YAML policy
- `UserContext` passed through the search call chain
- `AccessGuard` filters domains before routing

**Out of scope (Phase 4+)**:
- Authentication (OAuth, JWT, sessions)
- Row-level document permissions
- Dynamic policy updates at runtime
- UI for policy management

---

## 2. Current Architecture & Integration Point

### 2.1 Current Call Chain (Phase 2.5)

```
User
  ↓
Agent (search_knowledge_base)
  ↓
KnowledgeService.search(query)
  ├─ Registry.list_domains()           → [KnowledgeDomain]
  ├─ Router.route(query, domains)      → RoutingDecision
  ├─ Registry.get_retriever(domain_id) → HybridRetriever
  └─ Pipeline.retrieve(query, retriever) → list[Document]
```

### 2.2 Target Call Chain (Phase 3)

```
User
  ↓
Agent (search_knowledge_base)          ← unchanged
  ↓
KnowledgeService.search(query, user=None)
  ├─ Registry.list_domains()           → all domains
  ├─ AccessGuard.filter(domains, user) → allowed domains   ← NEW
  ├─ Router.route(query, allowed)      → RoutingDecision
  ├─ Registry.get_retriever(domain_id) → HybridRetriever
  └─ Pipeline.retrieve(query, retriever) → list[Document]
```

### 2.3 Why Between Service and Router

| Alternative | Problem |
|---|---|
| Guard before Service | Agent must know about UserContext — breaks encapsulation |
| Guard after Router | Router may select unauthorized domain → wasted LLM call + need re-route |
| **Guard between list_domains and route** | ✅ Router only sees authorized domains; Guard is a pure filter; Service orchestrates |

---

## 3. Data Model Design

### 3.1 UserContext

```python
# knowledge/access.py

from dataclasses import dataclass, field

@dataclass
class UserContext:
    """Minimal user identity for access control.

    Phase 3 uses role-based access. Future phases may add
    groups, tenant_id, or token-based authentication.
    """

    user_id: str
    role: str = "viewer"
    metadata: dict[str, str] = field(default_factory=dict)
```

**Design decisions**:
- `user_id`: unique identifier. Phase 3 is a string label (no auth).
- `role`: maps to `roles.*` in policy.yaml.
- `metadata`: extension point for future fields (tenant, department) without breaking the dataclass.
- `role` defaults to `"viewer"` — safe default for unknown users.

### 3.2 KnowledgeDomain Extension

No changes to `knowledge/domain.py` in Phase 3. Access policy is external (policy.yaml), not embedded in domain metadata. This keeps the domain model focused on storage/retrieval concerns.

If future phases require per-domain policy metadata, a single optional field can be added:

```python
# Future (NOT Phase 3):
access_policy: str | None = None  # policy name override
```

---

## 4. AccessPolicy Design

### 4.1 policy.yaml

```yaml
# 知识库访问策略 (Phase 3)
# 位置: 项目根目录，与 knowledge_bases.yaml 同级

users:
  # ---- 管理员：全部 KB ----
  admin:
    role: admin
    allowed_kbs:
      - "*"

  # ---- 开发者：仅技术文档 ----
  developer:
    role: developer
    allowed_kbs:
      - tech_docs
      - api_docs

  # ---- 普通用户：仅公开 KB ----
  viewer:
    role: viewer
    allowed_kbs:
      - default

# ---- 角色级别默认规则 ----
roles:
  admin:
    allowed_levels:
      - public
      - internal
      - restricted

  developer:
    allowed_levels:
      - public
      - internal

  viewer:
    allowed_levels:
      - public
```

### 4.2 Policy Loading

```python
class AccessPolicy:
    """Load and evaluate access rules from policy.yaml.

    Rules (evaluated in order):
    1. If user has "*" in allowed_kbs → access all
    2. If domain_id in user's allowed_kbs → access granted
    3. If domain has no explicit policy → fallback_allow (configurable)
    4. Otherwise → denied
    """

    def __init__(self, policy_path: str | Path | None = None) -> None:
        ...

    def is_allowed(self, user: UserContext, domain_id: str) -> bool:
        ...
```

### 4.3 Fallback Behavior

| Scenario | Behavior | Rationale |
|---|---|---|
| `policy.yaml` not found | All domains allowed + log warning | Backward compatible. Existing single-user deployments have no policy file. |
| Domain not listed in any policy | Allowed (default open) | Phase 3 is KB-level isolation, not zero-trust. Phase 4 may add `default_deny` mode. |
| User not listed in policy | Fallback to role-based rules | New users get role-default access. |
| `user=None` | Bypass all checks | Backward compatible — no auth = full access. |

---

## 5. AccessGuard Responsibility

### 5.1 Single Responsibility

`AccessGuard` has exactly one job:

```
list[KnowledgeDomain] + UserContext → list[KnowledgeDomain]
```

It is a **pure filter**. It does not:
- Route queries
- Retrieve documents
- Call LLMs
- Access Chroma
- Modify domains
- Log or audit (that's KnowledgeService's job)

### 5.2 Interface

```python
class AccessGuard:
    """Filter knowledge base domains by user access policy."""

    def __init__(self, policy: AccessPolicy | None = None) -> None:
        """Args:
            policy: AccessPolicy instance. If None, loads from default path.
        """
        ...

    def filter(
        self,
        domains: list[KnowledgeDomain],
        user: UserContext | None,
    ) -> list[KnowledgeDomain]:
        """Return the subset of domains the user is authorized to access.

        Args:
            domains: Full list from Registry.
            user: User identity; None means no filtering (full access).

        Returns:
            Filtered domain list. Guaranteed non-empty — at least
            the default domain is always included.
        """
        ...
```

### 5.3 Safety Guarantee

`filter()` must never return an empty list. If all domains are denied, it returns the default domain (conservative fallback — better to show public KB than crash).

---

## 6. API Design

### 6.1 KnowledgeService Changes

```python
class KnowledgeService:
    def search(
        self,
        query: str,
        user: UserContext | None = None,  # ← Phase 3 addition
    ) -> list[Document]:
        domains = self._registry.list_domains()

        if user is not None:                              # ← Phase 3
            domains = self._access_guard.filter(domains, user)

        decision = self._router.route(query, domains)
        retriever = self._registry.get_retriever(decision.domain_id)
        return self._pipeline.retrieve(query, retriever)

    def list_domains(
        self,
        user: UserContext | None = None,  # ← Phase 3 addition
    ) -> list[dict[str, object]]:
        domains = self._registry.list_domains()

        if user is not None:                              # ← Phase 3
            domains = self._access_guard.filter(domains, user)

        return [{"id": d.id, "name": d.name, ...} for d in domains]
```

### 6.2 Agent Layer (Unchanged)

```python
# rag_agent.py — NO changes in Commit 2
@tool
def search_knowledge_base(query: str) -> str:
    """..."""
    return format_documents(get_knowledge_service().search(query))
```

The `user` parameter is `None` by default. Agent calls remain identical. UserContext injection happens at a higher level (Web UI / CLI), not inside the Agent tool.

---

## 7. Integration Plan

### Commit 1: `feat: UserContext + AccessPolicy + AccessGuard`

| Item | Detail |
|---|---|
| **New files** | `knowledge/access.py` — UserContext, AccessPolicy, AccessGuard |
| | `policy.yaml` — example policy with admin/developer/viewer |
| **Modified** | None |
| **Tests** | Unit tests: filter with roles, "*" wildcard, missing user, missing policy file |
| **Risk** | Zero — new module, not wired to anything |

### Commit 2: `feat: KnowledgeService 接入 AccessGuard`

| Item | Detail |
|---|---|
| **Modified** | `knowledge/service.py` — `search()` and `list_domains()` add `user` parameter |
| | `knowledge/service.py` — `__init__` creates AccessGuard |
| **API** | `search(query, user=None)` — fully backward compatible |
| **Tests** | Integration: `user=None` identical behavior, `user=admin` all KBs, `user=viewer` filtered |
| **Risk** | Low — `user` defaults to `None`, existing callers unchanged |

### Commit 3: `feat: Agent/UI 透传 UserContext`

| Item | Detail |
|---|---|
| **Modified** | `rag_agent.py` — optional `user` parameter on tool (internal) |
| | `app.py` — sidebar role selector, passes UserContext |
| | `chat.py` — CLI option `--user` / `--role` |
| **API** | CLI/Web UI gain user selection; Agent tool signature unchanged externally |
| **Risk** | Low — all new parameters have defaults |

### Commit 4: `test: AccessGuard integration tests`

| Item | Detail |
|---|---|
| **New files** | `tests/test_access.py` |
| **Coverage** | admin full access, viewer filtered, wildcard "*", user not in policy, no policy file fallback, empty result safety net |
| **Risk** | Zero |

---

## 8. Compatibility Requirements

| Requirement | How |
|---|---|
| `search_knowledge_base(query)` still works | `user=None` default → no filtering |
| CLI runs without login | `policy.yaml` absent → all allowed + warning |
| Web UI unchanged | No user selector until Commit 3 |
| No new dependencies | Pure Python + PyYAML (already a dependency) |
| No config changes | `policy.yaml` is independent of `config.py` |
| No performance regression | `filter()` is O(n) on domain count (typically <10) |

---

## 9. Error Handling

| Scenario | Behavior |
|---|---|
| `policy.yaml` not found | Log warning. All domains allowed. |
| `policy.yaml` invalid YAML | RuntimeError on import. Fail fast. |
| User not in `users:` | Fall back to `roles.<user.role>` if defined; else viewer defaults. |
| Domain not in any policy | Allowed (default open). |
| All domains denied | Return at minimum the `default` domain. |

---

## 10. Future Extension Points (NOT Phase 3)

- **Authentication**: Replace `UserContext(user_id="...")` with token validation. AccessGuard interface unchanged.
- **Per-document ACL**: Add `document.metadata["access_level"]` and filter in SearchPipeline.
- **Dynamic policy**: Watch `policy.yaml` for changes; reload without restart.
- **Audit logging**: KnowledgeService logs `user_id` + `domain_id` per search.
- **RBAC with groups**: `UserContext.groups: list[str]` → AccessPolicy resolves group permissions.
