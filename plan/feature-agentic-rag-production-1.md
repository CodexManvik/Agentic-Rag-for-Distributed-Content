---
goal: Production-Ready Agentic RAG System with Dynamic Agent Management
version: 1.0
date_created: 2026-04-05
last_updated: 2026-04-05
owner: Development Team
status: "Planned"
tags: ["feature", "architecture", "streaming", "agents", "production"]
---

# Agentic RAG System - Production Implementation Plan

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

Comprehensive 3-phase implementation plan to transform the existing RAG system into a production-ready local-first agentic platform with dynamic agent management, real-time streaming, and extensible tool integration.

---

## 1. Framework Strategy

### Purpose
Define a clear, flexible path for orchestration layer evolution without premature refactoring.

### Current Choice: LangGraph
- **Why**: Excellent state management, fine-grained control, proven stability
- **Timeline**: Primary orchestration framework for Phases 1-2 (Weeks 1-6)
- **Scope**: Handles navigation workflow, agent execution, state threading

### Future Evaluation Options (Phase 3+)
These are **NOT planned for current implementation**. They are evaluation paths IF system requirements change:

#### Option A: LlamaIndex Integration
- **When to consider**: If RAG-specific workflows grow complex (multi-index querying, dynamic routing)
- **Role**: Query engine layer, NOT orchestration
- **Coexistence**: Can augment LangGraph (not replace)
- **Decision gate**: Post-Phase 2, if retrieval patterns require specialized handling

#### Option B: AutoGen for Multi-Agent Coordination
- **When to consider**: If Phase 3 multi-agent needs exceed LangGraph coordination capabilities
- **Role**: Multi-agent orchestration, NOT individual agent execution
- **Coexistence strategy**: EITHER replace LangGraph orchestration OR use as wrapper
- **Decision rule**: No overlapping frameworks solving the same problem

#### Option C: DSPy for Prompt Optimization
- **When to consider**: Only after system stabilizes (post-Phase 2) and evaluation data exists
- **Role**: Optimization layer for planning/synthesis prompts
- **Coexistence**: Works alongside any orchestration framework
- **MVP status**: NOT included; requires offline evaluation dataset first

#### Option D: CrewAI for User-Facing Multi-Agent
- **When to consider**: End of Phase 3, if users need visual multi-agent workflow builder
- **Role**: High-level agent coordination abstraction
- **Coexistence**: REPLACES LangGraph orchestration IF adopted
- **Decision rule**: Incompatible with simultaneous LangGraph orchestration at same level

### Evaluation Criteria
Any framework change must satisfy:
1. **Stability**: ≥6 months in production externally
2. **Requirement fit**: Solves stated problem better than current approach
3. **Integration cost**: < 2 weeks to adopt without breaking Phases 1-2
4. **Exit cost**: Can revert to LangGraph in <1 week if needed

### Architectural Principle
> **"Use LangGraph now, stay flexible later."**

No framework lock-in. All agent definitions are framework-agnostic (stored as JSON configs), enabling future swaps.

---

## 2. Framework Extensions (Optional Path)

These are **NOT part of MVP**. They represent **future evaluation gates** only.

### DSPy: Prompt Optimization (Phase 3+, Experimental)

**Current state**: Hand-crafted prompts in agent configs  
**Future option**: Optimize via DSPy optimizer  
**Timeline**: After ≥500 executed queries with eval annotations  
**Cost**: Requires offline dataset creation (2-3 weeks)  
**Decision**: Pursue only if prompt quality plateaus  

**Minimal schema addition** (for future use):
```json
{
  "optimization": {
    "type": "dspy",
    "enabled": false,
    "metrics": ["answer_quality", "citation_accuracy"],
    "optimizer": "BootstrapFewShot",
    "min_samples": 500
  }
}
```

### CrewAI: Commercial Multi-Agent UI (Phase 4+, If Needed)

**Current state**: Registry-based agent discovery (backend only)  
**Future option**: Visual workflow builder for users  
**Timeline**: Only if users request agent composition UI  
**Cost**: 3-4 weeks integration + rewrite of orchestrator  
**Trade-off**: Either use CrewAI OR hand-built orchestration, not both  

**Preconditions**:
- [ ] Phase 3 multi-agent system proven stable in production
- [ ] User feedback demands visual agent composition
- [ ] Performance acceptable without CrewAI abstractions

---

## 3. Requirements & Constraints

### Functional Requirements

- **REQ-001**: Desktop UI must connect to backend via HTTP/WebSocket (no IPC layer for API)
- **REQ-002**: All API responses must stream in real-time (SSE or WebSocket)
- **REQ-003**: Reasoning traces must render in a collapsible UI component (open by default)
- **REQ-004**: System must support local Ollama models only (no cloud services)
- **REQ-005**: Users must be able to select models from UI (with auto-download for missing models)
- **REQ-006**: Agents must be defined via JSON configs (no hardcoded pipeline nodes)
- **REQ-007**: Agent registry must support filesystem-based discovery (initial phase)
- **REQ-008**: AI must be able to generate agent configs but cannot auto-execute them
- **REQ-009**: Source visualization must display URL, PDF name, or document reference
- **REQ-010**: Tool execution disabled by default (opt-in security model)

### Technical Constraints

- **CON-001**: Single-user local system only (no cloud scaling required)
- **CON-002**: No multi-tenancy support needed
- **CON-003**: LangGraph is primary orchestration framework initially, with flexibility to evaluate alternatives post-Phase 2 per Framework Strategy guidelines
- **CON-004**: All data must be stored locally (no cloud dependencies)
- **CON-005**: External tools must be explicitly whitelisted before execution
- **CON-006**: Agent schemas must be strict (validation before loading)
- **CON-007**: No uncontrolled agent generation loops allowed

### Non-Functional Requirements

- **PERF-001**: API response time target: < 1s initial response, streaming chunks < 100ms
- **AVAIL-001**: System must gracefully handle Ollama unavailable state
- **SEC-001**: All tool access must be logged
- **OBS-001**: All agent executions must generate complete trace logs
- **COMPAT-001**: Must run on Windows, macOS, Linux

### Guidelines

- **GUD-001**: Prefer working system over perfect architecture (iterative improvement)
- **GUD-002**: All generated artifacts must include validation before execution
- **GUD-003**: Keep local-first security posture throughout design
- **GUD-004**: Document all agent manifest formats for extensibility

---

## 4. Implementation Overview

### PHASE 1: Backend-Frontend Connectivity & Streaming
**Duration**: Weeks 1-3
**Priority**: CRITICAL - Blocks all other work
**Deliverables**:
- ✅ Working desktop-to-backend HTTP connection
- ✅ Real-time SSE streaming for chat responses
- ✅ Updated UI with trace/source visualization
- ✅ Model selection with auto-download
- ✅ Error handling for Ollama unavailability

### PHASE 2: Agent Registry & Dynamic Execution
**Duration**: Weeks 4-6
**Priority**: HIGH - Enables extensibility
**Deliverables**:
- ✅ Agent manifest schema and registry
- ✅ Filesystem-based agent discovery
- ✅ Dynamic workflow execution from configs
- ✅ LLM-based agent generation
- ✅ Agent validation and versioning

### PHASE 3: Multi-Agent Coordination System (OPTIONAL)
**Duration**: Weeks 7-9 (if pursuing)
**Priority**: LOW - Advanced/Experimental
**Status**: Feature is OPTIONAL; system fully functional without it
**Deliverables** (if implemented):
- ✅ Sub-agent spawning and coordination
- ✅ Parallel and sequential agent execution
- ✅ Inter-agent communication
- ✅ Tool registry and whitelist system
- ✅ Distributed request tracing

**Decision gate before starting Phase 3**:
- [ ] Phase 1 & 2 stable in production (≥2 weeks)
- [ ] Defined use case requiring multi-agent workflows
- [ ] Sufficient team capacity for experimental feature
- [ ] Clear rollback plan if removed later

**Note**: System must function completely without Phase 3. It is an optional enhancement, not a core requirement.

---

## 5. Phase 1: Detailed Implementation

### Phase 1 Architecture

**Streaming Architecture**:
```
Desktop App (SSE Client)
    ↓
HTTP Connection (localhost:8000)
    ↓
Backend FastAPI Server
    ↓
/chat/stream Endpoint (EventSource)
    ↓
LangGraph Workflow
    ↓
Event Emitters (trace, chunk, complete)
    ↓
SSE Response Stream
    ↓
Desktop App Renders in Real-Time
```

### Task Breakdown

#### Phase 1 Execution: 5 Sequential Steps (NOT Parallel)

Execute in strict order. Do not start next step until previous validates successfully.

**Step 1: Backend Connectivity (Day 1)**

| Task | File | LOC | Validates By |
|------|------|-----|--|
| P1-S1-1 | Fix CORS in FastAPI | backend/app/main.py | 30 lines | `curl -X OPTIONS http://localhost:8000 -H "Origin: http://localhost:5173"` returns 200 |
| P1-S1-2 | Add basic GET endpoint | backend/app/main.py | 20 lines | `curl http://localhost:8000/health` returns healthy |

**Step 2: SSE Streaming with Dummy Data (Day 2)**

| Task | File | LOC | Validates By |
|------|------|-----|--|
| P1-S2-1 | Create GET /chat/stream endpoint (dummy response) | backend/app/main.py | 50 lines | `curl -N http://localhost:8000/chat/stream?query=test` returns SSE events |
| P1-S2-2 | Implement EventSource client in React | desktop-app/src/renderer/context/ChatContext.tsx | 80 lines | Browser console shows `trace`, `chunk`, `complete` events received |

**Step 3: Workflow Integration (Day 3)**

| Task | File | LOC | Validates By |
|------|------|-----|--|
| P1-S3-1 | Replace dummy data with `workflow.astream()` | backend/app/main.py | 60 lines | Real trace events from workflow execution received |
| P1-S3-2 | Verify streaming chunks during execution | backend/app/graph/workflow.py | 20 lines | Answer accumulates in real-time in frontend |

**Step 4: UI Components (Day 4)**

| Task | File | LOC | Validates By |
|------|------|-----|--|
| P1-S4-1 | Create TraceViewer component | desktop-app/src/renderer/components/TraceViewer.tsx | 180 lines | Trace dropdown displays and is collapsible |
| P1-S4-2 | Create SourcePanel component | desktop-app/src/renderer/components/SourcePanel.tsx | 140 lines | Citations render with sources and snippets |
| P1-S4-3 | Update ChatWindow for streaming | desktop-app/src/renderer/components/ChatWindow.tsx | 100 lines | Message accumulates on screen in real-time |

**Step 5: Model Management (Day 5)**

| Task | File | LOC | Validates By |
|------|------|-----|--|
| P1-S5-1 | Create model manager service | backend/app/services/model_manager.py | 250 lines | `curl http://localhost:8000/api/models` lists Ollama models |
| P1-S5-2 | Add model selection UI | desktop-app/src/renderer/components/ModelSelector.tsx | 120 lines | Dropdown shows available models, non-downloaded grayed out |
| P1-S5-3 | Add error handling | desktop-app/src/renderer/hooks/useChat_API.ts | 80 lines | Shows "Ollama not running" message when unavailable |

**Total Phase 1: 5 days with daily checkpoints (not concurrent tasks)**

### Key Implementation Details

#### P1-S2-1: GET SSE Streaming Endpoint with Real astream()

```python
@app.get("/chat/stream")
async def stream_chat(request: ChatRequest):
    async def event_generator():
        try:
            initial_state = NavigatorState(...)
            
            # Hook into workflow for events
            for event in workflow.stream(initial_state):
                if event.get("type") == "trace":
                    yield f"event: <type>\ndata: {json.dumps(event)}\n\n"
                elif event.get("type") == "chunk":
                    yield f"event: <type>\ndata: {json.dumps(event)}\n\n"
            
            # Final response
            yield f"event: <type>\ndata: {json.dumps({'type': 'complete', 'data': final_response})}\n\n"
        except Exception as e:
            yield f"event: <type>\ndata: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

#### P1-B-1: SSE Client in React

```typescript
const sendMessage = async (query: string) => {
    const eventSource = new EventSource(
        `${BACKEND_URL}/chat/stream?query=${encodeURIComponent(query)}`
    );
    
    eventSource.addEventListener('trace', (event) => {
        const data = JSON.parse(event.data);
        setResponse(data);
        eventSource.close();
    });
    
    eventSource.addEventListener('chunk', (event) => {
        const data = JSON.parse(event.data);
        setMessage(prev => prev + data);
    });
    
    eventSource.addEventListener('complete', (event) => {
        const response = JSON.parse(event.data);
        setResponse(response);
        eventSource.close();
    });
};
```

---

## 4. Phase 2: Detailed Implementation

### Agent System Architecture

```
Agent Manifest (JSON)
    ↓
Agent Registry (loads from disk)
    ↓
Schema Validator (validates before loading)
    ↓
Agent Executor (executes based on type)
    ↓
Outputs → State
```

### Agent Manifest Example

```json
{
  "name": "planning_agent",
  "version": "1.0.0",
  "type": "planner",
  "description": "Generates sub-queries from user query",
  "inputs": {
    "query": {
      "type": "string",
      "required": true
    }
  },
  "outputs": {
    "sub_queries": {
      "type": "array",
      "items": {"type": "string"}
    }
  },
  "execution": {
    "type": "python",
    "entrypoint": "app.agents.builtin.agents.planning_agent",
    "timeout_seconds": 10
  }
}
```

### Dynamic Workflow

```python
# OLD (hardcoded):
graph.add_node("planning", planning_agent_node)

# NEW (dynamic from registry):
for agent_spec in pipeline_definition:
    agent = registry.get_agent(agent_spec['name'])
    graph.add_node(agent.name, executor.create_node(agent))
```

---

## 7. Phase 3: Detailed Implementation

### Multi-Agent Execution Model

**Sequential**: agent1 → agent2 → agent3 (output of N becomes input of N+1)
**Parallel**: [agent1, agent2, agent3] all run simultaneously, results merged

### Sub-Agent Pattern

```python
async def autonomous_planner(state):
    # Spawn sub-agents for parallel analysis
    results = await orchestrator.execute_parallel([
        {"agent": "document_analyzer", "inputs": state},
        {"agent": "query_expander", "inputs": state},
        {"agent": "relevance_scorer", "inputs": state},
    ], parent_context=current_context)
    
    # Aggregate results
    return merge_results(results)
```

---

## 8. Testing Strategy

### Phase 1 Test Cases

```bash
# Test CORS
curl -X OPTIONS http://localhost:8000/chat \
  -H "Origin: http://localhost:5173"

# Test streaming
curl http://localhost:8000/chat/stream \
  -d '{"query":"test"}' \
  -H 'Content-Type: application/json'

# Test model listing
curl http://localhost:8000/api/models
```

### Phase 2 Test Cases

```python
# Test agent loading
agent = registry.get_agent("planning_agent", "1.0.0")
assert agent is not None

# Test agent execution
result = await executor.execute("planning_agent", {"query": "test"})
assert "sub_queries" in result
```

### Phase 3 Test Cases

```python
# Test sub-agent spawning
result = await orchestrator.spawn_agent(
    "document_analyzer",
    {"doc": "..."},
    parent_context
)
assert result["status"] == "success"
```

### Phase 2 Framework Tests

| TEST-P2-008 | Test workflow execution type (composed agents) | Unit | IMPORTANT |
| TEST-P2-009 | Test framework-agnostic agent executor | Unit | IMPORTANT |

---

## 9. Risks, Assumptions & Mitigations

### Critical Risks

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Streaming connection drops | Loss of response data | Implement client-side buffering + reconnect |
| Agent config bugs crash system | Data loss, downtime | Strict validation + timeout enforcement |
| Memory bloat from traces | Memory exhaustion | Trace cleanup, configurable retention |
| Ollama unavailable | Complete system failure | Graceful error messages, health checks || Framework lock-in via LangGraph tight coupling | Expensive refactoring if swap needed | Keep agent definitions as JSON configs; support multiple execution backends; design executor abstraction |
### Key Assumptions

- Ollama running locally before app starts
- Single-user, single-session usage
- Agent configs < 1MB
- LLM can generate valid JSON
- Stable local network
- User has write access to agent directory

---

## 10. Success Criteria

### Phase 1 Done When:
- ✅ Desktop app connects to backend and sends/receives messages
- ✅ Responses stream in real-time without gaps
- ✅ UI displays trace and sources correctly formatted
- ✅ Model selection works with auto-download
- ✅ Error states handled gracefully (Ollama down, etc.)
- ✅ No console errors in Electron dev tools
- ✅ E2E test suite passes

### Phase 2 Done When:
- ✅ Agents load from JSON files without errors
- ✅ New agents can be registered and discovered
- ✅ Workflow executes using registry instead of imports
- ✅ LLM can generate valid agent configs
- ✅ Invalid agents rejected before execution
- ✅ Version conflicts handled correctly

### Phase 3 Done When:
- ✅ Sub-agents spawn and execute successfully
- ✅ Parallel execution returns correct aggregated results
- ✅ Tools can be called within agents
- ✅ Tool whitelist enforced
- ✅ Distributed tracing works across agents
- ✅ No circular execution or infinite loops