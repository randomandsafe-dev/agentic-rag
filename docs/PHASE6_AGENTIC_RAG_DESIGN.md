# Phase 6 — Production Agentic RAG Design

> **Status**: Draft — design review pending.
> **Depends on**: v0.5.0 (Phase 5 Advanced Agentic RAG, merged to main).
> **Target**: Closed-loop retrieval with verification, self-correction, and production hardening.

---

## 1. Current Architecture Review

### 1.1 Frozen Interfaces (v0.5.0)

These must **not** be modified in Phase 6:

| Layer | Interface | Frozen Since |
|---|---|---|
| **Permission** | `AccessGuard.filter_domains(user, domains)` | Phase 3 |
| **Routing** | `KnowledgeRouter.route(query, domains)` | Phase 2 |
| **Registry** | `KnowledgeBaseRegistry.get_retriever(id)` | Phase 1 |
| **Pipeline** | `SearchPipeline.retrieve(query, retriever)` | Phase 2.5 |
| **Concurrent** | `ConcurrentRetriever.search(query, retrievers)` | Phase 5 |
| **Verifier** | `RetrievalVerifier.verify(question, docs, draft)` | Phase 5 |
| **Service** | `KnowledgeService.search(query, user=None)` | Phase 3 |
| **Agent Tool** | `search_knowledge_base(query)` | Phase 1 |

### 1.2 Current Flow

```
Agent (build_agent) — 3 tools
  │
  ▼
KnowledgeService.search(query, user)
  ├─ AccessGuard.filter_domains()
  ├─ Router.route() → RoutingDecision
  ├─ [single KB] Pipeline.retrieve(query, retriever)
  └─ [multi KB]  ConcurrentRetriever.search(query, retrievers)
  ▼
Documents → Agent formats answer

RetrievalVerifier (EXISTS but NOT WIRED)
ConcurrentRetriever (EXISTS but bypasses Pipeline for rerank/judge)
```

### 1.3 Gaps to Address

| Gap | Current | Target |
|---|---|---|
| Verifier not wired | Exists in `agent/verifier/`, never called in production flow | Config-gated, called post-retrieval |
| Concurrent results not reranked | Raw merged docs, no Pipeline processing | Pass concurrent results through Pipeline |
| No self-correction | Agent retries manually based on tool output | Structured retry triggered by Verifier feedback |
| No production telemetry | `logging.warning()` only | Structured logs, latency metrics |

---

## 2. Phase 6 Goals

### Goal 1: Verifier Integration

Wire `RetrievalVerifier` into the retrieval flow with a configuration switch.

**Design**:

```yaml
# config/verification.yaml (NEW)
verification:
  enabled: false          # Phase 6 default: off (opt-in)
  min_score: 0.5          # score below this → trigger retry
  max_retries: 1          # max verification retries
```

**Flow**:

```
KnowledgeService.search(query, user)
  ├─ ... retrieve docs ...
  ├─ [if verification.enabled]
  │    result = verifier.verify(query, docs)
  │    if not result.passed and retries < max:
  │        rewrite query using result.missing_topics
  │        re-retrieve
  │        re-verify
  └─ return docs
```

**Integration point**: Inside `KnowledgeService.search()`, after retrieval, before return. Controlled by a single config flag. When disabled, behavior is identical to v0.5.0.

### Goal 2: Agent Self-Correction Loop

Give the Agent structured feedback to decide whether to re-search.

**Design**:

```
Agent calls search_knowledge_base(query)
  ↓
KnowledgeService returns docs + verification_result (in document metadata)
  ↓
Agent reads _verification metadata:
  - passed=true  → Agent proceeds to answer
  - passed=false → Agent sees missing_topics, decides to re-search
  ↓
Agent optionally calls search_knowledge_base(new_query) with revised query
```

**Key principle**: The Agent remains in control. We don't build a separate planner loop — we enrich the Agent's context so it can make better decisions within the existing LangChain `create_agent` framework.

**Implementation**: Add `_verification` metadata to returned documents:

```python
# In KnowledgeService.search(), after verification:
for doc in docs:
    doc.metadata["_verification"] = {
        "passed": result.passed,
        "score": result.score,
        "missing_topics": result.missing_topics,
    }
```

**SYSTEM_PROMPT enhancement**:

```
如果检索结果的来源文档中包含验证信息显示检索不足，
请用不同的关键词重新调用 search_knowledge_base。
```

### Goal 3: Concurrent Search Pipeline Integration

Currently, concurrent results bypass `SearchPipeline` (no reranker, no judge). Wire them through.

**Design**:

```python
# knowledge/service.py — concurrent path enhanced
if len(decision.domain_ids) > 1:
    retrievers = {did: self._registry.get_retriever(did) for did in decision.domain_ids}
    raw_docs = self._concurrent.search(query, retrievers)
    # NEW: pass merged docs through pipeline for final processing
    # Create a pass-through retriever that returns raw_docs
    return self._pipeline.retrieve(query, _PassthroughRetriever(raw_docs))
```

**PassthroughRetriever**: A minimal adapter that implements `search(query) -> list[Document]`, returning pre-retrieved documents. This lets `SearchPipeline` apply relevance judge + reranker on the merged result set.

### Goal 4: Production Readiness

Structured logging, latency tracking, and configuration management.

**Logging**:

```python
# knowledge/service.py
logger.info("search_start", extra={"query": query[:100], "user": user_id})
logger.info("search_end", extra={"latency_ms": elapsed, "doc_count": len(docs)})
```

**Metrics** (lightweight, no external dependency):

```python
# knowledge/metrics.py (NEW)
class SearchMetrics:
    """In-memory metrics collector. Phase 6: console output. Phase 7: Prometheus."""
    def record_search(self, latency_ms, doc_count, verifier_score):
        ...
    def summary(self) -> dict:
        ...
```

**Config management**:

- All new features gated behind config flags (default: off)
- `config/verification.yaml` for verifier settings
- No new env vars without `config.py` declaration

---

## 3. Architecture Design

### 3.1 Target Call Chain (Phase 6)

```
Agent (build_agent) — 3 tools + verification-aware SYSTEM_PROMPT
  ↓
KnowledgeService.search(query, user)
  ├─ 1. AccessGuard.filter_domains()
  ├─ 2. Router.route()
  ├─ 3. Retrieve
  │     ├─ [single KB] Pipeline.retrieve(query, retriever)
  │     └─ [multi KB]  ConcurrentRetriever.search(query, retrievers)
  │                      └─ Pipeline.retrieve(query, PassthroughRetriever)
  ├─ 4. Verify (if enabled)
  │     ├─ Verifier.verify(query, docs)
  │     ├─ Attach _verification metadata
  │     └─ [if not passed] retry from step 3 with revised query
  └─ 5. Return docs
  ↓
Agent reads _verification metadata → decides to re-search or answer
```

### 3.2 Component Map

```
Phase 6 NEW:
  knowledge/metrics.py       — SearchMetrics collector
  config/verification.yaml   — Verifier configuration

Phase 6 MODIFIED:
  knowledge/service.py       — + verifier integration + metrics + concurrent pipeline
  agent/verifier/verifier.py — minor: prompt refinements
  rag_agent.py               — SYSTEM_PROMPT enhancement for verification awareness

Phase 6 UNCHANGED:
  knowledge/access.py        — AccessGuard, AccessPolicy
  knowledge/router.py        — KnowledgeRouter, RouterStrategy
  knowledge/registry.py      — KnowledgeBaseRegistry
  knowledge/concurrent.py    — ConcurrentRetriever
  search_pipeline.py         — SearchPipeline
  retrieval.py               — HybridRetriever
```

---

## 4. Commit Plan

### Commit 1: `feat: wire verifier into knowledge service`

| Item | Detail |
|---|---|
| **Files** | `knowledge/service.py`, `config/verification.yaml` (NEW) |
| **Content** | `verification.enabled` config flag; post-retrieval verifier call; `_verification` metadata on docs; verification-triggered retry |
| **Tests** | `tests/test_verifier_integration.py`: verifier on/off, retry on low score, metadata attachment |
| **Risk** | Low — disabled by default, identical behavior when off |
| **API impact** | Zero — `search(query, user)` signature unchanged |

### Commit 2: `feat: concurrent search pipeline integration`

| Item | Detail |
|---|---|
| **Files** | `knowledge/service.py`, `search_pipeline.py` (PassthroughRetriever) |
| **Content** | Concurrent results go through Pipeline for reranker/judge; PassthroughRetriever adapter |
| **Tests** | `tests/test_concurrent_pipeline.py`: merged docs reranked, judge applied |
| **Risk** | Medium — changes concurrent result quality (improvement, but behavior change) |
| **API impact** | Zero — internal only |

### Commit 3: `feat: agent self-correction with verification feedback`

| Item | Detail |
|---|---|
| **Files** | `rag_agent.py` (SYSTEM_PROMPT only) |
| **Content** | Enhanced SYSTEM_PROMPT: guide Agent to read `_verification` metadata and re-search if needed |
| **Tests** | `tests/test_agent_self_correction.py`: Agent re-searches when verification fails; Agent proceeds when passed |
| **Risk** | Low — prompt change only, behavior improvement |
| **API impact** | Zero |

### Commit 4: `feat: production metrics and logging`

| Item | Detail |
|---|---|
| **Files** | `knowledge/metrics.py` (NEW), `knowledge/service.py` |
| **Content** | `SearchMetrics`: latency, doc count, verifier score tracking; structured logging at key pipeline stages |
| **Tests** | `tests/test_metrics.py`: metrics collection, summary output |
| **Risk** | Low — additive only |
| **API impact** | Zero |

### Commit 5: `chore: Phase 6 config consolidation`

| Item | Detail |
|---|---|
| **Files** | `config.py`, `config/verification.yaml`, `.env.example` |
| **Content** | Consolidate scattered config; document all Phase 6 flags; update .env.example |
| **Tests** | Config validation tests |
| **Risk** | Low — config changes only |
| **API impact** | Zero |

---

## 5. Architecture Constraints

### MUST NOT

| Constraint | Reason |
|---|---|
| Large-scale Agent rewrite | LangChain `create_agent` is stable; self-correction via prompt + metadata is sufficient |
| Replace LangChain | Deep integration; migration cost >> benefit |
| Delete Permission Layer | `AccessGuard` is a foundational capability |
| Break `search(query, user=None)` | Core backward compatibility contract |
| Bypass AccessGuard | Concurrent path must still go through Guard → Router → Registry |
| Introduce new Agent loop framework | Keep within existing `create_agent` + tools pattern |
| Remove existing tools | `search_knowledge_base`, `list_knowledge_bases`, `verify_retrieval_result` all preserved |

### MUST

| Requirement | Implementation |
|---|---|
| All new features config-gated | Default: off. Opt-in via YAML or env. |
| Backward compatible single-KB | `verification.enabled=false` → identical to v0.5.0 |
| Zero new dependencies | stdlib `logging`, `time` for metrics |
| All existing tests pass | 129 tests must remain green |

---

## 6. Risk Analysis

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Verifier LLM call adds latency | High (when enabled) | Medium | Disabled by default; configurable timeout |
| Verifier false-negatives cause unnecessary re-search | Medium | Medium | `max_retries=1` cap; agent can override |
| Concurrent + Pipeline integration changes result quality | Low | Low | Pipeline on merged docs is strictly better than raw merge |
| SYSTEM_PROMPT change degrades Agent behavior | Medium | Medium | A/B testable; old prompt preserved as fallback |
| Metrics overhead | Low | Low | In-memory counters only; no external service dependency |

---

## 7. Success Criteria

| Criterion | How to Verify |
|---|---|
| Verifier wired and config-gated | `verification.enabled=false` → identical behavior; `true` → verifier called |
| Self-correction works | Agent re-searches when verifier reports missing_topics |
| Concurrent results reranked | Multi-KB search → Pipeline reranker applied → higher quality top results |
| Metrics trackable | `SearchMetrics.summary()` returns valid latency/doc counts |
| Zero API breakage | All 129 existing tests pass |
| Single-KB unchanged | `search(query)` without verification → identical to v0.5.0 |
| Permission layer preserved | AccessGuard still filters before any retrieval |
