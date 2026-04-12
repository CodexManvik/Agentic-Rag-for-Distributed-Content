---
goal: Implement Short-Circuit Routing for Simple Fact-Lookup Queries
version: 1.0
date_created: 2026-04-07
last_updated: 2026-04-07
owner: Development Team
status: 'Planned'
tags: [feature, optimization, supervisor, routing, performance]
---

# Introduction

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

This plan implements Short-Circuit Routing within the Supervisor Agent architecture. When the QueryAnalyzer detects a simple fact-lookup query (e.g., "What is X?", "Define Y", "When did Z happen?"), the system bypasses the TaskPlanner entirely and routes directly to the RAG Agent. This optimization reduces latency for straightforward queries by eliminating unnecessary planning overhead while preserving full planning capabilities for complex, multi-step queries.

**Parent Plan**: `refactor-agentic-system-production-1.md` (Phase 3: Supervisor Agent Architecture)

---

## 1. Requirements & Constraints

### Functional Requirements

- **REQ-001**: QueryAnalyzer must classify query complexity into discrete levels: `SIMPLE_LOOKUP`, `MODERATE`, `COMPLEX`, `MULTI_STEP`
- **REQ-002**: Queries classified as `SIMPLE_LOOKUP` must bypass TaskPlanner and route directly to RAG Agent
- **REQ-003**: Short-circuit routing must be configurable (enable/disable via settings)
- **REQ-004**: System must log when short-circuit routing is activated for observability
- **REQ-005**: Short-circuit decisions must be included in trace events for debugging
- **REQ-006**: Fallback to full planning if RAG Agent returns low-confidence result after short-circuit

### Performance Requirements

- **PER-001**: Short-circuit path must reduce average latency by at least 40% for simple queries
- **PER-002**: Query classification must complete within 100ms (single LLM call or heuristic)
- **PER-003**: No additional LLM calls for short-circuit path compared to direct RAG execution

### Design Constraints

- **CON-001**: Must integrate with existing QueryAnalyzer (TASK-025 in parent plan)
- **CON-002**: Must not break planning mode when enabled for complex queries
- **CON-003**: Must maintain backward compatibility with existing API contracts
- **CON-004**: Classification logic must work without external API calls (local inference only)

### Design Guidelines

- **GUD-001**: Use hybrid classification: fast heuristics first, LLM fallback for ambiguous cases
- **GUD-002**: Complexity thresholds must be tunable without code changes
- **GUD-003**: Short-circuit path must produce identical output format as planned execution path

---

## 2. Implementation Steps

### Phase 1: Query Complexity Classification

- **GOAL-001**: Extend QueryAnalyzer to detect simple fact-lookup queries with high accuracy

| Task | Description | Completed | Date |
| -------- | ----------- | --------- | ---- |
| TASK-001 | Define `QueryComplexity` enum in `backend/app/supervisor/schemas.py` with values: `SIMPLE_LOOKUP = "simple_lookup"`, `MODERATE = "moderate"`, `COMPLEX = "complex"`, `MULTI_STEP = "multi_step"`. Add field `complexity: QueryComplexity` to existing `QueryAnalysis` model | | |
| TASK-002 | Create `backend/app/supervisor/complexity_classifier.py` with class `ComplexityClassifier`. Implement method `classify(query: str) -> QueryComplexity` using hybrid approach (heuristics + optional LLM) | | |
| TASK-003 | Implement heuristic classifier in `ComplexityClassifier._heuristic_classify(query: str) -> QueryComplexity | None`. Rules: (1) Single question word + single entity = `SIMPLE_LOOKUP`, (2) Contains "and", "then", "after", "compare" = `COMPLEX`, (3) Word count > 30 = `MODERATE`, (4) Contains numbered steps = `MULTI_STEP`. Return `None` if ambiguous | | |
| TASK-004 | Implement LLM-based classifier in `ComplexityClassifier._llm_classify(query: str) -> QueryComplexity`. Use structured output with `UnifiedLLM.with_structured_output()`. Only called when heuristic returns `None` | | |
| TASK-005 | Create classification prompt template `backend/app/supervisor/prompts/complexity_classification.txt` with few-shot examples for each complexity level. Format: system instruction + 8 examples (2 per category) + query placeholder | | |
| TASK-006 | Add complexity classification patterns config `backend/app/supervisor/complexity_patterns.py`: Dataclass `ComplexityPatterns` with fields: `simple_patterns: list[str]` (regex), `complex_indicators: list[str]`, `multi_step_indicators: list[str]`, `ambiguity_threshold: float` | | |
| TASK-007 | Update `QueryAnalyzer.analyze()` method in `backend/app/supervisor/query_analyzer.py`: Integrate `ComplexityClassifier`, populate `QueryAnalysis.complexity` field before returning | | |

### Phase 2: Short-Circuit Router Implementation

- **GOAL-002**: Implement routing logic that bypasses TaskPlanner for simple queries

| Task | Description | Completed | Date |
| -------- | ----------- | --------- | ---- |
| TASK-008 | Create `backend/app/supervisor/short_circuit.py` with class `ShortCircuitRouter`. Constructor accepts `config: SupervisorConfig`, `rag_agent_name: str = "rag"` | | |
| TASK-009 | Implement `ShortCircuitRouter.should_short_circuit(analysis: QueryAnalysis) -> bool`. Returns `True` if: (1) `analysis.complexity == SIMPLE_LOOKUP`, (2) `config.enable_short_circuit == True`, (3) `analysis.required_capabilities` contains only `["retrieval", "synthesis"]` or subset | | |
| TASK-010 | Implement `ShortCircuitRouter.create_direct_route(analysis: QueryAnalysis) -> RoutingDecision`. Returns `RoutingDecision` with single agent (RAG), no dependencies, `is_short_circuited=True` flag | | |
| TASK-011 | Add `is_short_circuited: bool = False` field to `RoutingDecision` model in `backend/app/supervisor/schemas.py` | | |
| TASK-012 | Add `short_circuit_confidence_threshold: float = 0.7` field to `SupervisorConfig` in `backend/app/supervisor/config.py`. If RAG returns confidence below threshold after short-circuit, trigger fallback | | |
| TASK-013 | Update `AgentRouter.route()` method in `backend/app/supervisor/agent_router.py`: Check `ShortCircuitRouter.should_short_circuit()` first, return direct route if True, else proceed with normal routing logic | | |

### Phase 3: Supervisor Integration

- **GOAL-003**: Integrate short-circuit routing into SupervisorAgent with fallback handling

| Task | Description | Completed | Date |
| -------- | ----------- | --------- | ---- |
| TASK-014 | Update `SupervisorAgent.process()` method in `backend/app/supervisor/supervisor_agent.py`: After `QueryAnalyzer.analyze()`, check `routing_decision.is_short_circuited`. If True, skip `TaskPlanner.create_plan()` and execute RAG agent directly | | |
| TASK-015 | Implement fallback logic in `SupervisorAgent._handle_short_circuit_result(result: AgentResult) -> SupervisorResponse | None`. If `result.confidence < config.short_circuit_confidence_threshold`, return `None` to trigger full planning. Else return `SupervisorResponse` | | |
| TASK-016 | Add trace event emission in `SupervisorAgent.process()`: When short-circuit activates, emit trace event `{"node": "Supervisor", "status": "short_circuit", "detail": "Simple lookup detected, bypassing planner", "complexity": "<complexity_value>"}` | | |
| TASK-017 | Add fallback trace event: When short-circuit fallback triggers, emit `{"node": "Supervisor", "status": "short_circuit_fallback", "detail": "Low confidence result, escalating to full planning", "confidence": "<value>"}` | | |
| TASK-018 | Update `backend/app/supervisor/config.py`: Add fields `enable_short_circuit: bool = True`, `short_circuit_log_enabled: bool = True` to `SupervisorConfig` | | |

### Phase 4: Configuration and API

- **GOAL-004**: Expose short-circuit settings via configuration and API

| Task | Description | Completed | Date |
| -------- | ----------- | --------- | ---- |
| TASK-019 | Update `backend/app/config.py`: Add settings `enable_short_circuit_routing: bool = True`, `short_circuit_confidence_threshold: float = 0.7`, `short_circuit_use_heuristics_only: bool = False` | | |
| TASK-020 | Update `/settings` GET endpoint in `backend/app/main.py`: Include `enable_short_circuit_routing`, `short_circuit_confidence_threshold` in response | | |
| TASK-021 | Update `/settings` POST endpoint in `backend/app/main.py`: Allow updating `enable_short_circuit_routing`, `short_circuit_confidence_threshold` (add to `allowed` set) | | |
| TASK-022 | Add `short_circuited: bool` field to `ChatResponse` schema in `backend/app/api/schemas.py`. Indicates whether response was generated via short-circuit path | | |
| TASK-023 | Update `/chat` endpoint response building in `backend/app/main.py`: Populate `short_circuited` field from workflow result | | |

### Phase 5: Testing and Validation

- **GOAL-005**: Comprehensive testing of short-circuit routing behavior

| Task | Description | Completed | Date |
| -------- | ----------- | --------- | ---- |
| TASK-024 | Create `backend/tests/unit/test_complexity_classifier.py`: Test heuristic classification for 20 sample queries (5 per complexity level). Assert correct classification for unambiguous cases | | |
| TASK-025 | Create `backend/tests/unit/test_short_circuit_router.py`: Test `should_short_circuit()` with various `QueryAnalysis` inputs. Test `create_direct_route()` output structure | | |
| TASK-026 | Create `backend/tests/integration/test_short_circuit_flow.py`: End-to-end test with simple query "What is machine learning?" Assert: (1) `short_circuited=True` in response, (2) No planner trace events, (3) RAG agent executed | | |
| TASK-027 | Create `backend/tests/integration/test_short_circuit_fallback.py`: Test fallback scenario with mock low-confidence RAG result. Assert full planning triggered after fallback | | |
| TASK-028 | Create `backend/tests/benchmarks/bench_short_circuit_latency.py`: Measure latency for 100 simple queries with short-circuit enabled vs disabled. Assert >= 40% latency reduction | | |

---

## 3. Alternatives

- **ALT-001**: **LLM-only classification** - Use LLM for all complexity classification. Rejected because it adds 200-500ms latency per query, negating short-circuit benefits. Hybrid approach uses fast heuristics for clear cases.

- **ALT-002**: **Keyword-based routing only** - Use pure regex/keyword matching without LLM fallback. Rejected because edge cases (sarcasm, complex phrasing) would be misrouted. Hybrid provides accuracy while maintaining speed for obvious cases.

- **ALT-003**: **Confidence-based routing without complexity** - Route based solely on QueryAnalyzer confidence score. Rejected because confidence measures intent clarity, not query complexity. A confidently understood complex query still needs planning.

- **ALT-004**: **No fallback mechanism** - Always trust short-circuit decision without fallback. Rejected because misclassified queries would return poor results. Confidence-based fallback ensures quality while maintaining optimization benefits.

---

## 4. Dependencies

### Internal Dependencies

- **DEP-001**: `backend/app/supervisor/query_analyzer.py` (TASK-025 in parent plan) - Must be implemented first
- **DEP-002**: `backend/app/supervisor/schemas.py` (TASK-031 in parent plan) - Must exist with base models
- **DEP-003**: `backend/app/supervisor/config.py` (TASK-033 in parent plan) - Must exist for configuration
- **DEP-004**: `backend/app/agents/builtin/rag_agent.py` (TASK-046 in parent plan) - RAG agent must be implemented
- **DEP-005**: `backend/app/supervisor/supervisor_agent.py` (TASK-028 in parent plan) - Base supervisor must exist

### External Dependencies

- **DEP-006**: No new external dependencies required. Uses existing `pydantic` for models and `loguru` for logging.

---

## 5. Files

### New Files

- **FILE-001**: `backend/app/supervisor/complexity_classifier.py` - Query complexity classification logic
- **FILE-002**: `backend/app/supervisor/complexity_patterns.py` - Configurable classification patterns
- **FILE-003**: `backend/app/supervisor/short_circuit.py` - Short-circuit routing logic
- **FILE-004**: `backend/app/supervisor/prompts/complexity_classification.txt` - LLM prompt for classification
- **FILE-005**: `backend/tests/unit/test_complexity_classifier.py` - Unit tests for classifier
- **FILE-006**: `backend/tests/unit/test_short_circuit_router.py` - Unit tests for router
- **FILE-007**: `backend/tests/integration/test_short_circuit_flow.py` - Integration tests
- **FILE-008**: `backend/tests/integration/test_short_circuit_fallback.py` - Fallback tests
- **FILE-009**: `backend/tests/benchmarks/bench_short_circuit_latency.py` - Performance benchmarks

### Modified Files

- **FILE-010**: `backend/app/supervisor/schemas.py` - Add `QueryComplexity` enum, update `QueryAnalysis`, `RoutingDecision`
- **FILE-011**: `backend/app/supervisor/query_analyzer.py` - Integrate complexity classification
- **FILE-012**: `backend/app/supervisor/agent_router.py` - Add short-circuit check
- **FILE-013**: `backend/app/supervisor/supervisor_agent.py` - Integrate short-circuit flow and fallback
- **FILE-014**: `backend/app/supervisor/config.py` - Add short-circuit configuration fields
- **FILE-015**: `backend/app/config.py` - Add global short-circuit settings
- **FILE-016**: `backend/app/api/schemas.py` - Add `short_circuited` to `ChatResponse`
- **FILE-017**: `backend/app/main.py` - Update settings endpoints and chat response

---

## 6. Testing

### Unit Tests

- **TEST-001**: `test_heuristic_simple_lookup` - Input: "What is Python?" → Assert: `SIMPLE_LOOKUP`
- **TEST-002**: `test_heuristic_complex` - Input: "Compare Python and Java, then explain which is better for ML" → Assert: `COMPLEX`
- **TEST-003**: `test_heuristic_multi_step` - Input: "1. Find X, 2. Calculate Y, 3. Summarize" → Assert: `MULTI_STEP`
- **TEST-004**: `test_heuristic_ambiguous_returns_none` - Input: "Tell me about the history" → Assert: `None` (triggers LLM)
- **TEST-005**: `test_should_short_circuit_true` - Input: `QueryAnalysis(complexity=SIMPLE_LOOKUP, ...)` → Assert: `True`
- **TEST-006**: `test_should_short_circuit_false_complex` - Input: `QueryAnalysis(complexity=COMPLEX, ...)` → Assert: `False`
- **TEST-007**: `test_should_short_circuit_false_disabled` - Config: `enable_short_circuit=False` → Assert: `False`
- **TEST-008**: `test_create_direct_route_structure` - Assert: Single agent, no deps, `is_short_circuited=True`

### Integration Tests

- **TEST-009**: `test_end_to_end_short_circuit` - Query: "Define RAG" → Assert: Response has `short_circuited=True`, trace shows "short_circuit" event
- **TEST-010**: `test_end_to_end_no_short_circuit` - Query: "Compare RAG and fine-tuning, then recommend for my use case" → Assert: `short_circuited=False`, planner trace events present
- **TEST-011**: `test_fallback_on_low_confidence` - Mock RAG returns confidence=0.3 → Assert: Full planning triggered, "short_circuit_fallback" trace event
- **TEST-012**: `test_settings_toggle` - POST `/settings` with `enable_short_circuit_routing=False` → Assert: Subsequent queries never short-circuit

### Performance Tests

- **TEST-013**: `bench_simple_query_latency` - Measure 100 simple queries, assert mean latency < 1.5s with short-circuit
- **TEST-014**: `bench_latency_comparison` - Compare short-circuit vs full-plan for same simple queries, assert >= 40% improvement

---

## 7. Risks & Assumptions

### Risks

- **RISK-001**: Heuristic patterns may misclassify edge cases (e.g., "What is the comparison between X and Y?" looks simple but is comparative). Mitigation: Tune patterns based on production query logs, implement LLM fallback for ambiguous cases.

- **RISK-002**: Short-circuit may return lower quality answers for borderline queries. Mitigation: Confidence-based fallback ensures quality; log all short-circuit decisions for monitoring.

- **RISK-003**: Users may disable short-circuit globally, losing performance benefits. Mitigation: Default to enabled; document performance impact clearly.

- **RISK-004**: LLM classification fallback negates latency savings if triggered frequently. Mitigation: Tune heuristics to handle 80%+ of queries without LLM; monitor fallback rate.

### Assumptions

- **ASSUMPTION-001**: RAG Agent (TASK-046) returns a `confidence` score in its output that can be used for fallback decisions
- **ASSUMPTION-002**: Simple fact-lookup queries constitute at least 40% of total query volume (justifies optimization effort)
- **ASSUMPTION-003**: Heuristic classification can achieve 90%+ accuracy on unambiguous cases
- **ASSUMPTION-004**: Parent plan Phase 3 (Supervisor Architecture) is implemented before this feature
- **ASSUMPTION-005**: LLM inference latency for classification is ~200-500ms (acceptable for fallback path only)

---

## 8. Related Specifications / Further Reading

- [Parent Plan: refactor-agentic-system-production-1.md](./refactor-agentic-system-production-1.md) - Phase 3: Supervisor Agent Architecture
- [Query Classification Patterns Research](https://arxiv.org/abs/2305.14627) - Intent classification techniques
- [LangGraph Conditional Routing](https://python.langchain.com/docs/langgraph/how-tos/branching) - Routing patterns in LangGraph
