# Phase 4 — Multi-KB & Agentic RAG Enhancement Design

> **Status**: Draft — design review pending.
> **Depends on**: v0.3.0 (Phase 3 Permission Layer, merged to main).
> **Target**: Multi-KB runtime, Agentic retrieval enhancement, real E2E environment.

---

## 1. Current Architecture Review

### 1.1 Frozen Interfaces (Phase 3)

These interfaces are **stable and must not be modified** in Phase 4:

```python
# knowledge/service.py — stable
KnowledgeService.search(query: str, user: UserContext | None = None) -> list[Document]
KnowledgeService.list_domains(user: UserContext | None = None) -> list[dict]

# knowledge/access.py — stable
AccessGuard.filter_domains(user: UserContext | None, domains: list[KnowledgeDomain]) -> list[KnowledgeDomain]
AccessPolicy.is_allowed(user: UserContext, domain_id: str) -> bool

# knowledge/router.py — stable
KnowledgeRouter.route(query: str, domains: list[KnowledgeDomain]) -> RoutingDecision
RouterStrategy.route(query: str, domains: list[KnowledgeDomain]) -> RoutingDecision | None

# knowledge/registry.py — stable
KnowledgeBaseRegistry.get_retriever(domain_id: str) -> HybridRetriever
KnowledgeBaseRegistry.list_domains() -> list[KnowledgeDomain]

# search_pipeline.py — stable
SearchPipeline.retrieve(query: str, retriever) -> list[Document]
```

### 1.2 Extension Points (Phase 4)

Where new behavior can be added without modifying existing code:

| Layer | Extension Point | Mechanism |
|---|---|---|
| Agent | Add new tools alongside `search_knowledge_base` | `create_agent(tools=[...])` |
| Routing | Add new `RouterStrategy` implementations | `RouterStrategy(ABC)` |
| Registry | Add domain management methods | New methods on `KnowledgeBaseRegistry` |
| Pipeline | Add `RelevanceStrategy` implementations | `RelevanceStrategy(ABC)` |
| Config | Add `knowledge_bases.yaml` domains | YAML parsing in `Registry._load()` |

---

## 2. Phase 4 Goals

### A. Multi Knowledge Base — Runtime & Management

**Current** (Phase 3):
- Multiple KBs defined in `knowledge_bases.yaml`
- Each maps to a data dir + Chroma persist dir
- CLI `--domain` support for ingest
- Web UI ingest always targets default domain

**Target** (Phase 4):
- Web UI domain selector for ingest (fix TD-6)
- Domain metadata display (doc count, chunk count, last ingest time)
- KB health check (index exists, doc count)
- Ingest history tracking per domain
- Multi-KB E2E with real Chroma collections

**Design**:

```
knowledge_bases.yaml
  ↓
KnowledgeBaseRegistry (Phase 1)
  ├─ list_domains()        → existing
  ├─ get_retriever(id)     → existing
  ├─ get_domain_stats(id)  → NEW: doc count, chunk count, last ingest
  └─ health_check(id)       → NEW: index exists, Chroma reachable

ingest.py
  ├─ ingest_documents(domain_id)  → existing
  └─ ingest_metadata              → NEW: save ingest timestamp + doc count
```

**Domain metadata storage**: `.meta.json` file per domain persist_dir.
```json
{
  "last_ingest": "2026-07-22T10:00:00",
  "document_count": 15,
  "chunk_count": 320
}
```

### B. Agentic Retrieval Enhancement

**Current** (Phase 3):
- Single tool: `search_knowledge_base(query)`
- Agent decides when to call it
- No multi-step retrieval orchestration
- Answer verification only in CLI

**Target** (Phase 4):

```
Agent (with multiple tools)
  ├─ search_knowledge_base(query)     → existing, search authorized KBs
  ├─ list_available_kbs()             → NEW: tell user what KBs are available
  ├─ compare_viewpoints(query, kbs)   → NEW: search multiple KBs, compare results
  └─ verify_answer(question, answer)  → NEW: self-verify before responding
```

**New System Prompt**:
```
你是一个严谨的中文知识库助手。
可用知识库: {kb_summary}  ← dynamically injected

对于需要本地资料的问题:
1. 先用 search_knowledge_base 检索
2. 必要时用不同关键词多次检索
3. 对比不同来源，综合回答

如果用户询问有哪些知识库可用，使用 list_available_kbs。
回答末尾列出实际使用的来源文件。
```

**Tool design — `list_available_kbs`**:
```python
@tool
def list_available_kbs(dummy: str = "") -> str:
    """列出当前用户可访问的知识库列表。Agent 应在用户询问可用资料范围时调用。"""
    domains = get_knowledge_service().list_domains(user=_current_user)
    return "\n".join(f"- {d['name']}: {d['description']}" for d in domains)
```

**Tool design — `verify_answer`** (moved to Agent tool):
```python
@tool
def verify_answer_tool(question_and_answer: str) -> str:
    """验证上一轮回答是否忠实于检索到的来源文档。格式: '问题: ... 回答: ...'"""
    # reuse verify.py logic, but as a tool the Agent can self-invoke
```

**Architectural note**: Tools only call `KnowledgeService` public API. They do not access Router/Registry/Chroma directly. This preserves the frozen interface constraint.

### C. Real E2E Environment

**Current** (Phase 3): All 56 tests use Mock. Zero real Chroma tests.

**Target** (Phase 4): Real Chroma integration tests.

```
tests/
  conftest.py                    ← NEW: pytest fixtures for test KBs
  test_e2e_multi_kb.py           ← NEW: real Chroma + real Router + real Pipeline
  test_e2e_permission.py         ← NEW: real Chroma + AccessGuard + Router
```

**Test KB setup** (conftest.py):
```python
@pytest.fixture
def multi_kb_env(tmp_path):
    """Create 3 real Chroma KBs with distinct content for E2E testing."""
    # 1. Create 3 data dirs with distinct .md files
    # 2. Run ingest_documents(domain_id) for each
    # 3. Return KnowledgeService with real Router + Registry
```

**E2E test scenarios**:
- Query "Python deployment" → routes to tech_docs (not hr_docs)
- Admin user → can search all KBs
- Restricted user → only searches allowed KB
- KB with no index → raises clear error, doesn't crash

---

## 3. Architecture Constraints

### 3.1 Must NOT Modify

| Module | Protected Reason |
|---|---|
| `AccessGuard.filter_domains()` | Phase 3 stable. Permission logic frozen. |
| `KnowledgeRouter.route()` | Phase 2 stable. Routing logic frozen. |
| `KnowledgeBaseRegistry.get_retriever()` | Phase 1 stable. Domain→Retriever mapping frozen. |
| `SearchPipeline.retrieve()` | Phase 2.5 stable. Pipeline orchestration frozen. |
| `HybridRetriever.search()` | Phase 1 stable. Retrieval logic frozen. |

### 3.2 Allowed Extensions

| Module | What Can Be Added |
|---|---|
| `rag_agent.py` | New tools (`list_available_kbs`, self-verify) |
| `rag_agent.py` | Enhanced `SYSTEM_PROMPT` with dynamic KB summary |
| `knowledge/registry.py` | New methods: `get_domain_stats()`, `health_check()` |
| `ingest.py` | Metadata persistence (`.meta.json`) |
| `app.py` | Domain selector for ingest, KB status display |
| `chat.py` | New CLI flags, KB info commands |
| `config.py` | New fields (if needed for Phase 4 features) |
| `tests/` | Real E2E tests with Chroma fixtures |

---

## 4. Commit Plan

### Commit 1: `feat: Multi-KB infrastructure — domain metadata & health check`

| Item | Detail |
|---|---|
| **Files** | `knowledge/registry.py` (+ `get_domain_stats`, `health_check`) |
| | `ingest.py` (write `.meta.json` after ingest) |
| | `config.py` (optional: `kb_meta_enabled` flag) |
| **Tests** | `tests/test_registry_stats.py` |
| **Risk** | Low — new methods, additive only |
| **Depends on** | Nothing |

### Commit 2: `feat: Web UI domain-aware ingest + KB status`

| Item | Detail |
|---|---|
| **Files** | `app.py` (domain selector in KB panel, ingest passes domain_id) |
| | `app.py` (KB status display: doc count, last ingest) |
| **Tests** | Manual verification + Streamlit smoke test |
| **Risk** | Low — UI changes only |
| **Depends on** | Commit 1 (for `get_domain_stats`) |

### Commit 3: `feat: Agentic retrieval — new tools + enhanced system prompt`

| Item | Detail |
|---|---|
| **Files** | `rag_agent.py` (new tools: `list_available_kbs`) |
| | `rag_agent.py` (dynamic `SYSTEM_PROMPT` with KB summary) |
| | `verify.py` → Agent tool refactor (optional) |
| **Tests** | `tests/test_agent_tools.py` |
| **Risk** | Medium — Agent behavior change, needs prompt tuning |
| **Depends on** | Commit 1 |

### Commit 4: `test: Real Chroma E2E — multi-KB + permission integration`

| Item | Detail |
|---|---|
| **Files** | `tests/conftest.py` (Chroma fixtures) |
| | `tests/test_e2e_multi_kb.py` |
| | `tests/test_e2e_permission.py` |
| **Tests** | Real Chroma collections, real Router, real Pipeline |
| **Risk** | Low — tests only, no production code |
| **Depends on** | All previous commits |

---

## 5. Migration Strategy

### 5.1 API Compatibility Guarantees

| v0.3.0 API | v0.4.0 Behavior | Compatible? |
|---|---|---|
| `KnowledgeService.search(query)` | Identical | ✅ |
| `KnowledgeService.search(query, user=None)` | Identical | ✅ |
| `search_knowledge_base(query)` | Identical (tool unchanged) | ✅ |
| `build_agent(checkpointer)` | May add `tools` / `system_prompt` params with defaults | ✅ |
| `KnowledgeBaseRegistry.list_domains()` | Identical | ✅ |
| `AccessGuard.filter_domains()` | Identical | ✅ |
| `ingest_documents(domain_id)` | Now writes `.meta.json` alongside Chroma index | ✅ (additive) |

### 5.2 Single-KB Continuity

- Default `knowledge_bases.yaml` with single domain → behavior identical to v0.3.0
- No `config/policy.yaml` → all users have full access
- All new Agent tools are additive; existing tool unchanged
- `SYSTEM_PROMPT` changes are additive (more guidance, same core behavior)

### 5.3 Rollback

- `.meta.json` is independent of Chroma index. Deleting it has no effect on retrieval.
- New Agent tools can be disabled by not adding them to `build_agent()`.
- E2E tests use `tmp_path` fixtures; no permanent data created.

---

## 6. Risk Analysis

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Agent prompt changes cause worse answers | Medium | Medium | Keep old prompt as fallback; A/B test prompts |
| Real Chroma E2E tests slow CI | Medium | Low | Use `tmp_path` + small test docs; mark as `slow` |
| `.meta.json` drift from actual Chroma state | Low | Low | Rebuild on next `ingest_documents()` |
| Web UI domain selector UX confusion | Low | Low | Default to "default" domain; show clear labels |
| New tools make Agent decision loop longer | Low | Medium | `list_available_kbs` is zero-LLM; `search_knowledge_base` unchanged |

---

## 7. Success Criteria

| Criterion | How to Verify |
|---|---|
| Multi-KB E2E works | `pytest tests/test_e2e_multi_kb.py` passes |
| Permission + routing works with real Chroma | `pytest tests/test_e2e_permission.py` passes |
| Web UI ingest supports domain | Manual: upload file → select domain → verify correct Chroma collection |
| Agent tools don't break existing flow | All 56 existing tests continue to pass |
| Single-KB deployment unchanged | `python chat.py` with single-domain YAML works identically |
| v0.3.0 APIs unchanged | All frozen interfaces have identical signatures |
