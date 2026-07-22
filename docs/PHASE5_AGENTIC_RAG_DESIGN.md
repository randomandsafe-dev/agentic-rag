# Phase 5 — Agentic RAG System Design

> **Status**: Draft — design review pending.
> **Depends on**: v0.4.0 (Phase 4 Multi-KB Agentic RAG, merged to main).
> **Target**: Production-ready Agentic Retrieval with real E2E, verification agent, concurrent multi-KB search.

---

## 1. Current Architecture Review

### 1.1 Frozen Interfaces (v0.4.0)

These interfaces are **stable across v0.3.0 → v0.4.0 and must not change** in Phase 5:

```python
# knowledge/service.py
KnowledgeService.search(query: str, user: UserContext | None = None) -> list[Document]
KnowledgeService.list_domains(user: UserContext | None = None) -> list[dict]

# knowledge/access.py
AccessGuard.filter_domains(user: UserContext | None, domains: list[KnowledgeDomain]) -> list[KnowledgeDomain]

# knowledge/router.py
KnowledgeRouter.route(query: str, domains: list[KnowledgeDomain]) -> RoutingDecision

# knowledge/registry.py
KnowledgeBaseRegistry.get_retriever(domain_id: str) -> HybridRetriever
KnowledgeBaseRegistry.list_domains() -> list[KnowledgeDomain]

# search_pipeline.py
SearchPipeline.retrieve(query: str, retriever) -> list[Document]
```

### 1.2 Current Agent Flow (v0.4.0)

```
User Query
  ↓
Agent (build_agent)
  ├─ search_knowledge_base(query)         → single KB
  ├─ list_knowledge_bases()               → KB names
  └─ verify_retrieval_result(input)       → rule-based check
  ↓
KnowledgeService → AccessGuard → Router → Registry → Pipeline
  ↓
Answer (with sources)
```

### 1.3 Limitations to Address

| Limitation | v0.4.0 | v0.5.0 Target |
|---|---|---|
| E2E tests | All Mock (98 tests) | Real Chroma + real Router + real Pipeline |
| Retrieval quality check | Rule-based tool | Dedicated Verifier Agent with LLM |
| Multi-KB search | Single domain per query | Concurrent search across multiple domains |
| KB management | YAML-only | Web UI management + stats display |

---

## 2. Phase 5 Goals

### A. Real Chroma E2E

Replace Mock-based integration tests with real Chroma collections in CI-safe fixtures.

```
tests/conftest.py
  ├─ Fixture: tmp_chroma_env → creates N real Chroma collections
  ├─ Fixture: multi_kb_service → KnowledgeService with real Registry + Router
  └─ Fixture: seeded_docs → distinct .md files per KB

tests/test_e2e_chroma.py
  ├─ Real ingest → real Chroma → real HybridRetriever
  ├─ Multi-KB routing with real collections
  ├─ Permission filter with real Chroma
  └─ SearchPipeline with real retriever
```

**Constraints**: Use `tmp_path` fixtures. Tests must complete in <30s. No permanent data.

### B. Verifier Agent

Upgrade from the current rule-based `verify_retrieval_result` tool to a dedicated verification step.

```
Current (v0.4.0):                     Target (v0.5.0):
  Agent → answer                        Agent → draft answer
           ↑                                     ↓
  verify_retrieval_result (rule)        Verifier Agent (LLM)
                                           ├─ Check source coverage
                                           ├─ Check factual consistency
                                           ├─ Suggest re-retrieval if needed
                                           └─ Return: {pass, retry_needed, missing_topics}
```

**Design**:

```python
# agent/verifier.py
class RetrievalVerifier:
    """Dedicated verifier — checks if retrieved docs support the answer."""

    def __init__(self, llm):
        self._llm = llm  # DI, from create_llm()

    def verify(
        self,
        question: str,
        draft_answer: str,
        retrieved_docs: list[Document],
    ) -> VerificationResult:
        """Returns pass/fail + missing topics for re-retrieval."""
```

**Architecture note**: Verifier is NOT an Agent tool. It's a pipeline step called by
KnowledgeService before returning results to the Agent. This keeps the Agent
simple and the verification logic centralized.

### C. Multi-KB Concurrent Retrieval

Router currently returns a single `domain_id`. For Phase 5, enable concurrent
search across multiple domains when routing confidence is moderate.

```
Current:                              Target:
  Router → domain_id                    Router → [domain_id_1, domain_id_2]
  Registry → retriever                  Registry → [retriever_1, retriever_2]
  Pipeline → search(one)                Pipeline → concurrent search(all)
                                           ↓
                                        Merge results (dedup + re-rank)
```

**Design**:

```python
# knowledge/service.py — enhanced search()
def search(self, query: str, user=None) -> list[Document]:
    domains = self._registry.list_domains()
    if user:
        domains = self._access_guard.filter_domains(user, domains)

    decision = self._router.route(query, domains)

    if decision.concurrent and len(decision.domain_ids) > 1:
        # Concurrent multi-KB search
        retrievers = [self._registry.get_retriever(did) for did in decision.domain_ids]
        results = []
        for r in retrievers:
            results.extend(self._pipeline.retrieve(query, r))
        return self._merge_results(results)  # dedup + re-rank
    else:
        # Single-KB (existing behavior)
        retriever = self._registry.get_retriever(decision.domain_id)
        return self._pipeline.retrieve(query, retriever)
```

**RoutingDecision extension**:

```python
@dataclass
class RoutingDecision:
    domain_ids: list[str]       # was domain_id: str in v0.3.0 → list for Phase 5
    primary_id: str             # always populated (backward compat alias)
    confidence: float
    strategy: str
    concurrent: bool = False    # NEW: whether to search all domain_ids in parallel
```

### D. KB Management

Basic Web UI and API for managing knowledge bases at runtime.

```
Web UI sidebar                          CLI
  ├─ KB list (name, status, doc count)    ├─ python manage.py kb list
  ├─ Enable / Disable toggle              ├─ python manage.py kb enable <id>
  ├─ Ingest per KB (--domain selector)    └─ python manage.py kb disable <id>
  └─ Create new KB (YAML editor)
```

---

## 3. Architecture Design

### 3.1 Target Call Chain

```
User Query
  ↓
Agent (build_agent)
  ├─ search_knowledge_base(query)           → single KB search
  ├─ search_all_knowledge_bases(query)      → concurrent multi-KB (NEW)
  ├─ list_knowledge_bases()                 → existing
  └─ verify_retrieval_result(input)         → existing (upgraded)
  ↓
KnowledgeService.search(query, user)
  ├─ AccessGuard.filter_domains()           — unchanged
  ├─ Router.route() → RoutingDecision       — extended (concurrent flag)
  ├─ Registry.get_retriever() / get_retrievers()  — NEW multi-get
  ├─ SearchPipeline.retrieve() × N          — concurrent calls
  └─ Verifier.verify()                      — NEW post-retrieval check
  ↓
Final Answer (with verified sources)
```

### 3.2 Component Responsibility

| Component | Phase 5 Role | Change |
|---|---|---|
| **Agent** | Decision: which tool to call, when to retry | Enhanced SYSTEM_PROMPT only |
| **Verifier** | Post-retrieval: check answer quality, flag missing topics | NEW module `agent/verifier.py` |
| **KnowledgeService** | Orchestrate Guard → Router → Registry → Pipeline → Verifier | Add verifier step; concurrent path |
| **Router** | Return multiple domain_ids with concurrency flag | Extend RoutingDecision |
| **Registry** | Batch get_retrievers for concurrent search | NEW method |
| **SearchPipeline** | Unchanged. Receives retriever, returns docs. | None |

### 3.3 Why NOT a Full Planner/Executor/Verifier Loop

A full multi-agent loop (Planner → Executor → Verifier → loop) would require:
- Inter-agent message protocol
- State machine for multi-turn retrieval
- Significant LangGraph custom graph building

This is beyond Phase 5 scope. Instead, we achieve the same effect through:
- Enhanced Agent SYSTEM_PROMPT (planner behavior)
- `search_all_knowledge_bases` tool (executor with concurrency)
- `Verifier` as post-retrieval check in KnowledgeService (verifier)
- Agent's own retry logic via `search_knowledge_base` re-calls

---

## 4. Commit Plan

### Commit 1: `test: real Chroma E2E test infrastructure`

| Item | Detail |
|---|---|
| **Files** | `tests/conftest.py` (Chroma fixtures), `tests/test_e2e_chroma.py` |
| **Content** | `tmp_chroma_env` fixture: creates N Chroma collections with seed docs |
| | Tests: real ingest → Chroma → HybridRetriever → Pipeline |
| | Multi-KB routing with real collections |
| | Permission filter with real Chroma (admin/developer/unknown) |
| **Tests** | ~8 E2E tests, all using tmp_path |
| **Risk** | Medium — Chroma must be installed, tests slower (5-15s) |
| **Depends on** | Nothing |

### Commit 2: `feat: retrieval verifier agent`

| Item | Detail |
|---|---|
| **Files** | `agent/verifier.py` (NEW), `knowledge/service.py` |
| **Content** | `RetrievalVerifier`: LLM-based semantic check |
| | `VerificationResult`: pass/fail + missing_topics + retry_suggested |
| | `KnowledgeService.search()` optionally calls verifier after retrieval |
| | Replace rule-based `verify_retrieval_result` tool with verifier integration |
| **Tests** | `tests/test_verifier.py` (Mock LLM, all scenarios) |
| **Risk** | Low — verifier is optional, disabled by default |
| **Depends on** | Nothing |

### Commit 3: `feat: concurrent multi-KB retrieval`

| Item | Detail |
|---|---|
| **Files** | `knowledge/router.py`, `knowledge/registry.py`, `knowledge/service.py` |
| **Content** | `RoutingDecision.concurrent` flag |
| | `RoutingDecision.domain_ids: list[str]` (extend from single str) |
| | `Registry.get_retrievers(domain_ids)` batch method |
| | `search_all_knowledge_bases` Agent tool |
| | `KnowledgeService.search()` concurrent path with merge |
| **Tests** | `tests/test_concurrent_retrieval.py` |
| **Risk** | Medium — extends RoutingDecision (backward compat via `primary_id`) |
| **Depends on** | Commit 1 (real Chroma tests validate concurrency) |

### Commit 4: `feat: KB management API and Web UI`

| Item | Detail |
|---|---|
| **Files** | `app.py`, `chat.py`, `knowledge/registry.py` |
| **Content** | Web UI: KB list with status/stats, enable/disable toggle |
| | Web UI: domain selector for ingest (fix TD-6) |
| | CLI: `python chat.py --kb-list` / `--kb-manage` |
| | KB metadata display (doc count from `.meta.json`) |
| **Tests** | Manual smoke test |
| **Risk** | Low — UI-only changes, no backend logic change |
| **Depends on** | Commit 1 |

---

## 5. Migration & Compatibility

### 5.1 Backward Compatibility

| v0.4.0 | v0.5.0 | Status |
|---|---|---|
| `RoutingDecision.domain_id: str` | `domain_ids: list[str]` + `primary_id: str` | ✅ Compatible via property |
| `search(query, user)` | Identical signature | ✅ |
| `search_knowledge_base(query)` | Identical tool | ✅ |
| `build_agent(checkpointer)` | Identical | ✅ |
| `user=None` bypass | Identical | ✅ |
| E2E tests | Real Chroma tests added alongside Mock tests | ✅ Additive |

### 5.2 Single-KB Continuity

- Single domain in `knowledge_bases.yaml` → behavior identical to v0.4.0
- Verifier disabled by default → no extra LLM calls
- Concurrent path only triggers when Router returns multiple domains
- `merge_results()` is a no-op for single result list

### 5.3 Rollback

- Verifier: disabled via config flag
- Concurrent search: only when Router strategy returns multiple IDs
- Real Chroma E2E: marked `@pytest.mark.slow`, skippable in CI

---

## 6. Risk Analysis

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Real Chroma tests slow CI | High | Low | Mark as slow; run on schedule, not per-PR |
| RoutingDecision extension breaks callers | Low | Medium | `primary_id` property preserves backward compat |
| Concurrent search doubles LLM calls | Medium | Medium | Only when confidence is moderate; configurable threshold |
| Verifier LLM hallucination | Low | Medium | Use temperature=0; verifier only flags issues, doesn't modify answer |
| KB management YAML corruption | Low | High | Atomic write; backup before save |

---

## 7. Success Criteria

| Criterion | How to Verify |
|---|---|
| Real Chroma E2E passes | `pytest tests/test_e2e_chroma.py -m slow` |
| Verifier catches missing sources | Test: deliberately incomplete retrieval → verifier flags |
| Concurrent search finds more relevant docs | Test: query matches multiple KBs → merged results > single |
| KB management works | Manual: enable/disable KB → list_domains() reflects change |
| v0.4.0 APIs unchanged | All 98 existing tests continue to pass |
| Backward compatible | Single KB config → identical behavior to v0.4.0 |
