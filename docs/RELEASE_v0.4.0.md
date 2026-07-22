# Phase 4 Multi-KB Agentic RAG — Release v0.4.0

> **Branch**: `main`
> **Date**: 2026-07-22
> **Previous**: [v0.3.0-phase3-permission]
> **Status**: Release Candidate

---

## 1. Release Summary

v0.4.0 在 v0.3.0 Permission Layer 基础上，新增 Multi-KB Runtime Management 和 Agentic Retrieval Tools，使项目从单一检索工具演进为多工具协作的智能检索 Agent。

### 里程碑

| Version | Phase | Content |
|---|---|---|
| v0.1.0 | 1.0 | Baseline: Web UI, hybrid retrieval, session memory |
| v0.2.5 | 2.5 | SearchPipeline + LLMFactory + circular dependency resolution |
| v0.3.0 | 3.0 | Permission Layer: UserContext + AccessPolicy + AccessGuard |
| **v0.4.0** | **4.0** | **Multi-KB runtime + Agentic retrieval tools** |

### v0.4.0 Changes

| Commit | Content |
|---|---|
| `14d6db6` | Multi-KB registry infrastructure (enabled field, kb_loader) |
| `25ecffe` | Runtime management (KnowledgeDomain.metadata, kb_loader extraction) |
| `a770e9f` | Agentic tools (list_knowledge_bases, verify_retrieval_result) |
| `ce3a906` | E2E integration tests (12 scenarios) |

---

## 2. Architecture Snapshot

### 2.1 Full Call Chain

```
Web UI (app.py) / CLI (chat.py)
  │ set_agent_user(UserContext)
  ▼
Agent (build_agent)
  ├─ search_knowledge_base(query)          — Phase 1
  ├─ list_knowledge_bases()                — Phase 4 NEW
  └─ verify_retrieval_result(input)        — Phase 4 NEW
       │
       ▼
KnowledgeService.search(query, user)
  ├─ AccessGuard.filter_domains(user)       — Phase 3
  ├─ KnowledgeRouter.route(query, domains)  — Phase 2
  ├─ KnowledgeBaseRegistry.get_retriever()  — Phase 1 + 4 (kb_loader)
  └─ SearchPipeline.retrieve(query, retriever) — Phase 2.5
       ├─ [if] QueryRewriter.rewrite()
       ├─ retriever.search() — BM25+Vector+RRF+Reranker
       ├─ [if] LLMRelevanceJudge.judge()
       └─ [if] Retry
```

### 2.2 Module Map

```
L4  Presentation     app.py, chat.py, ingest.py
L3  Application      rag_agent.py, agent/tools/, verify.py
L2  Orchestration    knowledge/service.py
L1  Domain           knowledge/router.py, knowledge/access.py,
                     search_pipeline.py, knowledge/domain.py,
                     knowledge/kb_loader.py
L0  Infrastructure   knowledge/registry.py, retrieval.py,
                     embeddings.py, llm_factory.py, prompts.py,
                     config.py, memory/
```

---

## 3. Multi-KB Design

### 3.1 Configuration (`config/knowledge_bases.yaml`)

```yaml
knowledge_bases:
  - id: default
    name: "默认知识库"
    collection: knowledge_base
    default: true
    enabled: true

  - id: tech_docs
    name: "技术文档"
    collection: kb_tech_docs
    enabled: false

  - id: product_docs
    name: "产品文档"
    collection: kb_product_docs
    enabled: false
```

### 3.2 KnowledgeDomain

```python
@dataclass
class KnowledgeDomain:
    id: str
    name: str
    description: str
    data_dir: Path
    persist_dir: Path
    collection_name: str
    default: bool = False
    enabled: bool = True
    keywords: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
```

### 3.3 kb_loader

```
kb_loader.load_domains(config_path) -> dict[str, KnowledgeDomain]

Fallback: config/knowledge_bases.yaml -> knowledge_bases.yaml -> settings
Supports: knowledge_bases (new) + domains (old) YAML keys
```

### 3.4 Registry

Registry 仅负责 `domain -> Retriever` 映射。Domain 加载委托给 `kb_loader`。

```
KnowledgeBaseRegistry.__init__()
  self._domains = load_domains(config_path)
  self._retrievers = {}  (lazy)

list_domains()   -> enabled domains only
list_all_domains() -> all domains (including disabled)
```

---

## 4. Agent Tools

### 4.1 search_knowledge_base (Phase 1, stable)

```python
@tool
def search_knowledge_base(query: str) -> str
```

检索知识库，自动路由到最相关的 KB。内部经过 BM25+Vector+RRF+Reranker。

### 4.2 list_knowledge_bases (Phase 4 NEW)

```python
@tool
def list_knowledge_bases(dummy: str = "") -> str
```

列出当前用户可访问的知识库。Agent 在用户询问资料范围时调用。

### 4.3 verify_retrieval_result (Phase 4 NEW)

```python
@tool
def verify_retrieval_result(verification_input: str) -> str
```

验证检索结果是否包含有效来源。Agent 在给出最终回答前自检。

### 4.4 Tool Access Pattern

```
All tools → KnowledgeService public API only
          → No direct Router/Registry/Chroma access
          → UserContext passed via rag_agent._current_user
```

---

## 5. API Stability

### Preserved from v0.3.0

| API | Status |
|---|---|
| `KnowledgeService.search(query)` | ✅ Identical |
| `KnowledgeService.search(query, user=None)` | ✅ Identical |
| `KnowledgeService.list_domains()` | ✅ Identical |
| `KnowledgeService.list_domains(user=None)` | ✅ Identical |
| `search_knowledge_base(query)` | ✅ Identical |
| `build_agent(checkpointer)` | ✅ Returns agent (now includes 2 new tools) |
| `set_agent_user(UserContext)` | ✅ Identical |
| `AccessGuard.filter_domains()` | ✅ Identical |
| `KnowledgeRouter.route()` | ✅ Identical |
| `SearchPipeline.retrieve()` | ✅ Identical |
| `ingest_documents(domain_id=None)` | ✅ Identical |

### New in v0.4.0

| API | Module |
|---|---|
| `KnowledgeDomain.enabled` | `knowledge/domain.py` |
| `KnowledgeDomain.metadata` | `knowledge/domain.py` |
| `KnowledgeBaseRegistry.list_all_domains()` | `knowledge/registry.py` |
| `load_domains(config_path)` | `knowledge/kb_loader.py` |
| `list_knowledge_bases` (Agent tool) | `agent/tools/knowledge_tools.py` |
| `verify_retrieval_result` (Agent tool) | `agent/tools/knowledge_tools.py` |
| `config/knowledge_bases.yaml` (new format) | `config/` |

---

## 6. Testing Status

```
98 passed in 1.5s

test_access.py                 14  ✅  Permission unit tests
test_agent_access.py            5  ✅  Agent pass-through
test_agent_tools.py             9  ✅  New agent tools
test_kb_loader.py              10  ✅  Config loading
test_multi_kb_registry.py      12  ✅  Registry + enabled filter
test_permission_integration.py   9  ✅  Permission E2E
test_phase4_e2e.py             12  ✅  Phase 4 full chain
test_search_pipeline.py        21  ✅  SearchPipeline
test_service_access.py          8  ✅  Service + Guard

All tests offline — zero external dependencies (no Chroma, no LLM API).
```

---

## 7. Known Limitations

| # | Limitation | Impact | Phase 5 |
|---|---|---|---|
| 1 | Real Chroma Multi-KB E2E not covered | Medium — Mock covers full chain correctly | P0 |
| 2 | Router returns single domain (no concurrent multi-KB search) | Medium — queries hit one KB at a time | P1 |
| 3 | `verify_retrieval_result` is rule-based, not LLM | Low — catches missing sources; semantic check deferred | P1 |
| 4 | KB management UI missing (Web UI domain selector for ingest) | Low — CLI `--domain` works; Web UI limited to default | P1 |
| 5 | `list_knowledge_bases` has unused `dummy` param | Low — LangChain tool decorator requirement | P2 |
| 6 | `relevance_strategy` config only supports `llm` | Low — other strategies removed in Phase 2.5 | P2 |
| 7 | KB metadata not displayed in Web UI | Low — doc count/chunk count stored but not shown | P2 |

---

## 8. Phase 5 Roadmap

### P0 — Must Have

| Task | Description |
|---|---|
| Real Chroma E2E | `tests/conftest.py` with Chroma fixtures; multi-KB + permission real tests |
| QA pass | Full manual test: multi-KB ingest + routing + permission + agent tools |

### P1 — Should Have

| Task | Description |
|---|---|
| Concurrent multi-KB retrieval | Router returns list of domains; Pipeline searches multiple KBs in parallel |
| LLM-based verification | Upgrade `verify_retrieval_result` to use LLM for semantic fact-checking |
| KB management Web UI | Sidebar: domain selector for ingest, enable/disable toggle, KB stats |
| KB metadata in Web UI | Display doc count, chunk count, last ingest time per domain |

### P2 — Nice to Have

| Task | Description |
|---|---|
| KB health check API | `GET /health` returning index status per domain |
| Policy hot reload | Watch `policy.yaml` changes without restart |
| Tool param cleanup | Remove `dummy` param when LangChain supports zero-arg tools |
| Config governance | Group settings by module, remove dead fields |
