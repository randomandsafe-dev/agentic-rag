# Phase 6 — Production Agentic RAG Release Notes

> **Branch**: `feature/multi-kb-search-pipeline`
> **Status**: Release Candidate

---

## Summary

Phase 6 将 Agentic RAG 从基础检索工具升级为具备自纠正闭环、指标可观测性的生产级系统。

### Key Features

| Feature | Description | Default |
|---|---|---|
| **Verifier Integration** | 检索后 LLM 质量验证，判断文档是否充分支撑回答 | Off |
| **Self-Correction Loop** | 验证失败 → 查询修正 → 重新检索 → 再验证的闭环 | Off |
| **Concurrent Pipeline** | 多 KB 并发结果通过 SearchPipeline 统一后处理 (reranker) | Auto |
| **Metrics Layer** | 请求级指标收集：延迟、文档数、验证分数、重试次数 | Off |
| **Runtime Config** | 统一 `runtime.yaml` 配置入口，typed config objects | — |

### Architecture

```
Agent (3 tools + verification-aware prompt)
  ↓
KnowledgeService.search(query, user)
  ├─ AccessGuard.filter_domains()        — Phase 3
  ├─ Router.route()                      — Phase 2
  ├─ _retrieve_docs()
  │    ├─ [single] Pipeline.retrieve()   — Phase 2.5
  │    └─ [multi]  ConcurrentPipeline    — Phase 6
  ├─ SelfCorrectionController.run()      — Phase 6
  │    └─ RetrievalVerifier.verify()     — Phase 5
  └─ MetricsCollector.record()           — Phase 6
```

### Test Coverage

```
176 passed, 6 skipped in 1.8s

  9 test files:
    test_access.py                 14  ✅
    test_agent_access.py            5  ✅
    test_agent_tools.py             9  ✅
    test_concurrent_pipeline.py     7  ✅  Phase 6 C2
    test_concurrent_search.py      11  ✅
    test_kb_loader.py              10  ✅
    test_kb_management.py          10  ✅
    test_metrics.py                 9  ✅  Phase 6 C4
    test_multi_kb_registry.py      12  ✅
    test_permission_integration.py   9  ✅
    test_phase4_e2e.py             12  ✅
    test_real_chroma_e2e.py         6  ⏭
    test_runtime_config.py          8  ✅  Phase 6 C5
    test_search_pipeline.py        21  ✅
    test_self_correction.py        11  ✅  Phase 6 C3
    test_service_access.py          8  ✅
    test_verification_integration.py 10  ✅  Phase 6 C1
    test_verifier.py               11  ✅
```

### Known Limitations

| # | Limitation | Target |
|---|---|---|
| 1 | Real Chroma E2E requires `fastembed` install | CI setup |
| 2 | Self-correction depends on LLM quality for query rewriting | LLM model selection |
| 3 | Metrics collector is console-only (no external export) | Phase 7 Prometheus |
| 4 | Concurrent Pipeline PassthroughRetriever skips BM25 (already searched) | Acceptable tradeoff |
| 5 | `verification.enabled` and `self_correction.enabled` both default off | Opt-in by design |

### Upgrade from v0.5.0

- All APIs remain backward compatible
- `config/runtime.yaml` is additive — absent file → all features off
- `config/verification.yaml` and `config/metrics.yaml` still readable as fallback
- Single KB behavior unchanged when all features disabled

### Commit History (Phase 6)

```
c37cf0c feat: add agentic rag metrics layer
426346c feat: add agentic self correction loop
f24e707 feat: integrate concurrent retrieval with pipeline
b38cb71 feat: integrate retrieval verifier into agent flow
334fec1 docs: add Phase 6 production agentic RAG design
```
