# Phase 3 Permission Layer — Release v0.3.0

> **Branch**: `feature/multi-kb-search-pipeline`
> **Date**: 2026-07-22
> **Status**: Release Candidate

---

## 1. Release Summary

Phase 2.5 + Phase 3 合并发布，标志着项目从单用户单知识库原型演进为支持多知识库路由和 KB 级权限控制的稳定架构。

**项目定位**：本地知识库问答平台，支持多知识库、可插拔搜索增强管道、角色级权限隔离。

### 包含里程碑

| Phase | 内容 |
|---|---|
| Phase 1 | Knowledge Layer 基础设施（domain / registry / service） |
| Phase 1.5 | Retriever 配置解耦（persist_dir 从 domain 显式传入） |
| Phase 2 | KnowledgeRouter 多 KB 自动路由（Keyword + LLM） |
| Phase 2.5 | SearchPipeline 统一搜索编排 + LLMFactory + 循环依赖消除 |
| Phase 3 | Permission Layer（UserContext + AccessPolicy + AccessGuard） |

---

## 2. Architecture Snapshot

### 2.1 调用链

```
UI (app.py) / CLI (chat.py)
  │ set_agent_user(UserContext(...))
  ▼
Agent (search_knowledge_base)
  │ get_knowledge_service().search(query, user=_current_user)
  ▼
KnowledgeService.search(query, user)
  ├─ AccessGuard.filter_domains(user, domains)     ← Phase 3
  ├─ KnowledgeRouter.route(query, allowed_domains)  ← Phase 2
  ├─ KnowledgeBaseRegistry.get_retriever(id)        ← Phase 1
  └─ SearchPipeline.retrieve(query, retriever)      ← Phase 2.5
       ├─ [if enabled] QueryRewriter.rewrite()
       ├─ retriever.search()                        ← BM25 + Vector + RRF + Reranker
       ├─ [if enabled] LLMRelevanceJudge.judge()
       └─ [if not relevant] retry
```

### 2.2 模块职责

```
┌─────────────────────────────────────────────────────────┐
│  L4  Presentation    app.py, chat.py, ingest.py          │
├─────────────────────────────────────────────────────────┤
│  L3  Application     rag_agent.py, verify.py             │
├─────────────────────────────────────────────────────────┤
│  L2  Orchestration   knowledge/service.py                │
├─────────────────────────────────────────────────────────┤
│  L1  Domain          knowledge/router.py                 │
│                      knowledge/access.py                 │
│                      search_pipeline.py                  │
│                      knowledge/domain.py                 │
├─────────────────────────────────────────────────────────┤
│  L0  Infrastructure  knowledge/registry.py               │
│                      retrieval.py, embeddings.py         │
│                      llm_factory.py, config.py           │
│                      prompts.py, memory/                 │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Feature Matrix

| Feature | Status | Phase |
|---|---|---|
| Multi-turn Chat | ✅ Stable | 1.0 |
| Agent-based Retrieval | ✅ Stable | 1.0 |
| Hybrid Retrieval (BM25 + Vector + RRF) | ✅ Stable | 1.0 |
| Cross-Encoder Reranker | ✅ Stable | 1.0 |
| Query Rewrite | ✅ Configurable | 2.5 |
| Relevance Judge | ✅ Configurable | 2.5 |
| Retry Mechanism | ✅ Configurable | 2.5 |
| Session Memory (SQLite) | ✅ Stable | 1.0 |
| Multi Knowledge Base | ✅ Stable | 2.0 |
| KB Auto-Routing (Keyword) | ✅ Stable | 2.0 |
| KB Auto-Routing (LLM) | ✅ Stable | 2.0 |
| Permission Control (Role-based) | ✅ Stable | 3.0 |
| Web UI (Streamlit) | ✅ Stable | 1.0 |
| Answer Verification | ✅ CLI only | 1.0 |
| Policy Hot Reload | ❌ Not planned | — |
| Multi-Agent | ❌ Phase 5 | — |

---

## 4. Permission Layer Design

### 4.1 UserContext

```python
@dataclass
class UserContext:
    user_id: str
    role: str = "viewer"
    metadata: dict[str, Any] = field(default_factory=dict)
```

### 4.2 AccessPolicy

- 从 `config/policy.yaml` 加载（惰性）
- 支持 `"*"` 通配符（全部 KB）
- 未在 `users:` 中列出的用户使用 `default` 规则
- 文件不存在 → 全部允许（向后兼容）
- 格式错误 → RuntimeError

### 4.3 AccessGuard

```python
class AccessGuard:
    def filter_domains(
        self,
        user: UserContext | None,
        domains: list[KnowledgeDomain],
    ) -> list[KnowledgeDomain]
```

- `user=None` → 返回全部 domains（向后兼容）
- `user` 存在 → 按 `allowed_kbs` 过滤
- 安全网：全部被拒绝时返回 default domain

### 4.4 policy.yaml

```yaml
users:
  admin:
    role: admin
    allowed_kbs:
      - "*"                 # 全部 KB

  developer:
    role: developer
    allowed_kbs:
      - tech_docs
      - api_docs

default:
  role: viewer
  allowed_kbs:
    - default               # 默认仅访问公开 KB
```

---

## 5. API Stability

### 保持兼容

| API | 签名 |
|---|---|
| `KnowledgeService.search(query)` | `(str) → list[Document]` |
| `KnowledgeService.list_domains()` | `() → list[dict]` |
| `KnowledgeService.invalidate()` | `() → None` |
| `search_knowledge_base(query)` | LangChain tool |
| `build_agent(checkpointer)` | `(Checkpointer?) → Agent` |
| `get_knowledge_service()` | `() → KnowledgeService` |

### 新增（向后兼容）

| API | 签名 |
|---|---|
| `KnowledgeService.search(query, user=None)` | `(str, UserContext?) → list[Document]` |
| `KnowledgeService.list_domains(user=None)` | `(UserContext?) → list[dict]` |
| `create_llm(*, temperature, model)` | `() → ChatOpenAI` |
| `UserContext(user_id, role, metadata)` | dataclass |
| `cli: --user / --role` | optional arguments |

### 删除

| API | 替代 |
|---|---|
| `get_default_retriever()` | `get_retriever(domain_id)` |

---

## 6. Testing Status

```
57 passed in 1.4s

tests/test_access.py                 14  ✅  AccessPolicy unit tests
tests/test_service_access.py          8  ✅  KnowledgeService + Guard
tests/test_agent_access.py            5  ✅  Agent user pass-through
tests/test_permission_integration.py  9  ✅  E2E permission flows
tests/test_search_pipeline.py        21  ✅  SearchPipeline

All tests offline — zero external dependencies.
```

---

## 7. Known Limitations

| Limitation | Impact | Target |
|---|---|---|
| 真实 Chroma Multi-KB E2E 测试未覆盖 | 低 — Mock 覆盖完整 | Phase 4 |
| Web UI ingest 不支持 domain 选择 | 中 — 仅 CLI 支持 `--domain` | Phase 4 |
| `policy.yaml` 不支持热加载 | 低 — 需重启生效 | Phase 4+ |
| 并发用户隔离依赖模块级全局变量 | 中 — 当前单用户模型足够 | Phase 5 |
| 答案验证仅 CLI 可用（Web UI 缺失） | 低 | Phase 4 |
| `relevance_strategy` 配置仅支持 `llm` | 低 — 其他策略已删除 | Phase 4+ |

---

## 8. Next Phase Roadmap

### Phase 4 — Multi-Agent & Web UI Enhancement

| Priority | Task |
|---|---|
| P0 | Web UI ingest 支持 domain 选择（TD-6） |
| P0 | 真实 Chroma Multi-KB E2E 测试 |
| P1 | KB 管理 UI（注册/查看/删除 domain） |
| P1 | Web UI 接入 answer verification |
| P2 | 多 Agent 基础架构（Planner / Retriever / Verifier） |

### Phase 5 — Production Hardening

| Priority | Task |
|---|---|
| P0 | 并发用户隔离（替换模块级全局变量） |
| P1 | `policy.yaml` 热加载 |
| P1 | 审计日志 |
| P2 | 性能优化（Pipeline 结果缓存） |
| P2 | 配置治理（字段分组、废弃清理） |
