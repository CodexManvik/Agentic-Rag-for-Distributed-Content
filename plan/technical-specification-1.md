---
title: Technical Specification - Agentic RAG Production System
version: 1.0
date: 2026-04-05
---

# Technical Specification Document

## Part A: API Schemas & Contracts

### 1. Streaming Chat Endpoint

**Endpoint**: `GET /chat/stream?query={query_string}`
**Response-Type**: `text/event-stream`
**Query Parameters**:
- `query` (required): User query string (URL-encoded, 3-2000 chars)

**Example Request**:
```
GET /chat/stream?query=what%20is%20RAG
```

**CRITICAL**: EventSource only supports GET requests. POST will not work.

**Response Events**:

Event: `trace`
```json
{
  "type": "trace",
  "data": {
    "node": "string",
    "status": "started|running|completed|failed",
    "detail": "string",
    "ts": "ISO8601",
    "duration_ms": "number (optional)"
  }
}
```

Event: `chunk`
```json
{
  "type": "chunk",
  "data": "string (text fragment)"
}
```

Event: `complete`
```json
{
  "type": "complete",
  "data": {
    "answer": "string",
    "citations": [
      {
        "index": "number",
        "source": "string",
        "url": "string (optional)",
        "snippet": "string",
        "source_type": "string (pdf|url|document)",
        "section": "string (optional)",
        "page_number": "number (optional)"
      }
    ],
    "sub_queries": ["string"],
    "confidence": "number (0-1)",
    "abstained": "boolean",
    "abstain_reason": "string (null if not abstained)",
    "trace": [
      {
        "node": "string",
        "status": "string",
        "detail": "string",
        "ts": "ISO8601",
        "duration_ms": "number"
      }
    ],
    "retrieval_quality": {
      "max_score": "number",
      "avg_score": "number",
      "source_diversity": "number",
      "chunk_count": "number",
      "adequate": "boolean",
      "reason": "string"
    },
    "stage_timings": {
      "[stage_name]": "number (milliseconds)"
    }
  }
}
```

Event: `error`
```json
{
  "type": "error",
  "data": {
    "error": "string",
    "code": "string (OLLAMA_DOWN|MODEL_NOT_FOUND|NETWORK_ERROR|etc)"
  }
}
```

---

### 2. Model Management Endpoints

**Endpoint**: `GET /api/models`
**Response**:
```json
{
  "chat_models": [
    {
      "name": "qwen:4b",
      "digest": "sha256:...",
      "size": 2500000,
      "modified_at": "ISO8601",
      "downloaded": true
    }
  ],
  "embedding_models": [
    {
      "name": "nomic-embed-text:latest",
      "digest": "sha256:...",
      "size": 500000,
      "modified_at": "ISO8601",
      "downloaded": true
    }
  ],
  "current_chat_model": "qwen:4b",
  "current_embedding_model": "nomic-embed-text:latest"
}
```

**Endpoint**: `POST /api/models/select`
**Request**:
```json
{
  "model_name": "mistral:7b"
}
```
**Response**:
```json
{
  "status": "success|downloading|error",
  "current_model": "mistral:7b"
}
```

**Endpoint**: `POST /api/models/pull`
**Request**:
```json
{
  "model_name": "llama2:13b"
}
```
**Response**: Event Stream
```json
{
  "type": "progress",
  "data": {
    "status": "downloading",
    "completed": 500000,
    "total": 5000000,
    "percent": 10
  }
}
{
  "type": "complete",
  "data": {
    "model": "llama2:13b",
    "size": 5000000
  }
}
```

---

### 3. Agent Management Endpoints

**Endpoint**: `GET /api/agents`
**Query Params**: `?type=planner&tag=query-expansion`
**Response**:
```json
{
  "agents": [
    {
      "id": "planning_agent@1.0.0",
      "name": "planning_agent",
      "version": "1.0.0",
      "type": "planner",
      "description": "...",
      "tags": ["query", "expansion"],
      "inputs": {...},
      "outputs": {...}
    }
  ],
  "total": 15,
  "loaded_from": "filesystem"
}
```

**Endpoint**: `GET /api/agents/{name}/{version}`
**Response**: Full agent manifest

**Endpoint**: `POST /api/agents/generate`
**Request**:
```json
{
  "task_description": "Create an agent that analyzes documents for sentiment",
  "auto_save": false
}
```
**Response**:
```json
{
  "agent_id": "sentiment_analyzer@1.0.0",
  "manifest": {...},
  "valid": true,
  "errors": [],
  "requires_review": true,
  "pending_review_id": "uuid-for-approval"
}
```

**Endpoint**: `POST /api/agents/review`
**Request**:
```json
{
  "pending_review_id": "uuid",
  "approved": true,
  "notes": "Looks good to deploy"
}
```

**Endpoint**: `POST /api/agents/{name}/{version}/execute`
**Request**:
```json
{
  "inputs": {
    "query": "test query",
    "context": {...}
  }
}
```
**Response**:
```json
{
  "execution_id": "uuid",
  "status": "success|error",
  "result": {...},
  "trace": [...]
}
```

---

### 4. Tool Management Endpoints

**Endpoint**: `GET /api/tools`
**Response**:
```json
{
  "all_tools": [...],
  "whitelisted_tools": ["web_search", "pdf_extract"],
  "available_for_use": ["web_search", "pdf_extract"]
}
```

**Endpoint**: `POST /api/tools/whitelist`
**Request**:
```json
{
  "tool_name": "web_search",
  "approved": true
}
```

**Endpoint**: `POST /api/tools/{name}/call`
**Request**:
```json
{
  "args": {...},
  "require_approval": false
}
```

---

## API Request Method Reference

| Endpoint | Method | Reason |
|----------|--------|--------|
| `/chat/stream` | **GET** (not POST) | EventSource only supports GET requests |
| `/api/models` | GET | Standard REST data retrieval |
| `/api/models/pull/{name}` | POST | Mutation, initiates download |
| `/api/agents/{name}/{version}/execute` | POST | Mutation, initiates execution |

**Critical**: Do NOT use POST for EventSource. Always use GET with query parameters.

### Future: Model Context Protocol (MCP) Integration

**Current implementation** (Phase 1-2):
- Filesystem-based tool registry
- Manual whitelist management
- Static tool definitions

**Future evolution path** (Phase 4+):
- Tools as MCP servers
- Dynamic tool discovery
- Runtime whitelist negotiation

**Why not now**:
- MCP spec still evolving
- Adds HTTP/networking complexity to local system
- Current registry approach sufficient for MVP

**Migration path** (when/if adopted):
1. Keep registry API surface unchanged
2. Implement MCP backend adapter
3. Tools become MCP servers (optional per tool)
4. No breaking changes to agent configs

**Decision gate**: Pursue only if tool ecosystem grows >20 tools or requires external integrations.

---

## Part B: Directory Structure & File Organization

### Backend Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                          # FastAPI app, all endpoints
│   ├── config.py                        # Settings (existing)
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── schemas.py                   # Pydantic models (updated)
│   │
│   ├── agents/                          # NEW: Agent system
│   │   ├── __init__.py
│   │   ├── schema.json                  # Agent manifest JSON schema
│   │   ├── registry.py                  # Agent registry
│   │   ├── loader.py                    # Dynamic agent loading
│   │   ├── executor.py                  # Agent execution engine
│   │   ├── validator.py                 # Schema validation
│   │   ├── orchestrator.py              # Multi-agent coordination
│   │   ├── messaging.py                 # Inter-agent communication
│   │   ├── context.py                   # Execution context
│   │   │
│   │   └── builtin/                     # Built-in agents
│   │       ├── __init__.py
│   │       ├── agents.py                # Agent implementations
│   │       ├── normalize_query.json
│   │       ├── planning_agent.json
│   │       ├── retrieval_agent.json
│   │       ├── adequacy_check.json
│   │       ├── reformulation_agent.json
│   │       ├── synthesis_agent.json
│   │       └── citation_validation.json
│   │
│   ├── tools/                           # NEW: Tool system
│   │   ├── __init__.py
│   │   ├── schema.json                  # Tool manifest schema
│   │   ├── registry.py                  # Tool registry
│   │   ├── whitelist.py                 # Tool access control
│   │   └── builtin/                     # Built-in tools (e.g., web_search)
│   │
│   ├── trace/                           # NEW: Tracing system
│   │   ├── __init__.py
│   │   ├── tracer.py                    # Distributed tracing
│   │   └── aggregator.py                # Trace aggregation
│   │
│   ├── graph/                           # LangGraph (updated)
│   │   ├── __init__.py
│   │   ├── state.py                     # State types (existing)
│   │   ├── nodes.py                     # Node implementations (keep for backwards compat)
│   │   └── workflow.py                  # Workflow builder (refactored)
│   │
│   └── services/                        # Services (updated)
│       ├── __init__.py
│       ├── llm.py                       # LLM client (existing)
│       ├── vector_store.py              # Vector store (existing)
│       ├── ingestion.py                 # Ingestion (existing)
│       ├── guardrails.py                # Guardrails (existing)
│       ├── policy.py                    # Policy (existing)
│       ├── compliance.py                # Compliance (existing)
│       ├── chunking.py                  # Chunking (existing)
│       └── model_manager.py             # NEW: Model management
│
├── tests/                               # Updated tests
│   ├── test_agent_registry.py           # NEW
│   ├── test_agent_execution.py          # NEW
│   ├── test_streaming.py                # NEW
│   └── ...existing tests...
│
├── Dockerfile
├── requirements.txt                     # Updated with new deps
└── run_ingestion.py
```

### Frontend Structure

```
desktop-app/
├── src/
│   ├── renderer/
│   │   ├── App.tsx                      # Main app (existing)
│   │   ├── index.css                    # Styles
│   │   │
│   │   ├── components/
│   │   │   ├── ChatWindow.tsx           # UPDATED: Streaming support
│   │   │   ├── Message.tsx              # Component for messages
│   │   │   ├── FileUpload.tsx
│   │   │   ├── ModelSelector.tsx        # NEW: Model selection UI
│   │   │   ├── TraceViewer.tsx          # NEW: Trace visualization
│   │   │   ├── SourcePanel.tsx          # NEW: Source display
│   │   │   ├── ModelDownloadProgress.tsx # NEW: Download progress
│   │   │   └── Sidebar.tsx              # Sidebar
│   │   │
│   │   ├── context/
│   │   │   └── ChatContext.tsx          # UPDATED: SSE client, streaming
│   │   │
│   │   └── hooks/
│   │       ├── useChat_API.ts           # UPDATED: API calls
│   │       ├── useElectronFile.ts       # File handling
│   │       └── useSSE.ts                # NEW: SSE streaming hook
│   │
│   ├── main/
│   │   └── main.ts                      # UPDATED: IPC handlers
│   │
│   └── preload/
│       └── preload.ts                   # Electron preload
│
├── .env.local                           # NEW: Backend URL config
├── package.json
├── vite.config.ts
├── tsconfig.json
└── index.html
```

---

## Part C: Configuration Files

### .env.local (Desktop App)

```bash
# Backend connection
VITE_BACKEND_URL=http://localhost:8000

# Optional: Override paths
VITE_AGENTS_DIR=~/.agentic_rag/agents
VITE_TOOLS_DIR=~/.agentic_rag/tools
```

### config.py (Backend - Updated)

```python
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Existing settings...
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "qwen:4b"
    
    # NEW: Agent system
    agents_registry_path: Path = Path.home() / ".agentic_rag" / "agents"
    agent_max_depth: int = 5
    agent_max_siblings: int = 10
    agent_execution_timeout: float = 300
    
    # NEW: Tool system
    tools_registry_path: Path = Path.home() / ".agentic_rag" / "tools"
    tools_whitelisted: list[str] = []  # Empty by default (opt-in)
    
    # NEW: Streaming
    streaming_chunk_size: int = 1000  # chars per SSE event
    streaming_timeout: float = 600
    
    # NEW: Tracing
    trace_retention_hours: int = 24
    trace_max_entries: int = 1000
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

---

## Part D: Key Implementation Patterns

### Pattern 1: Adding Streaming to Workflow

Current (blocking):
```python
final_state = workflow.invoke(initial_state)
return ChatResponse(...)
```

New (streaming):
```python
# In workflow.py
async for event in workflow.astream(initial_state):
    if event.get("trace"):
        yield StreamEvent(type="trace", data=event["trace"])
    if event.get("chunk"):
        yield StreamEvent(type="chunk", data=event["chunk"])
```

### Pattern 2: Agent Discovery

```python
# Initialize registry
registry = AgentRegistry(Path(settings.agents_registry_path))

# List agents
all_agents = registry.list_agents()
planners = registry.list_agents(type_filter="planner")

# Get specific agent
agent = registry.get_agent("planning_agent", version="1.0.0")

# Execute agent
result = await executor.execute("planning_agent", inputs)
```

### Pattern 3: Dynamic Workflow Building

```python
# Old: Static graph
graph.add_node("planning", planning_agent_node)
graph.add_edge("normalize", "planning")

# New: Registry-based
pipeline = [
    {"agent": "normalize_query", "version": "1.0.0"},
    {"agent": "planning_agent", "version": "1.0.0"},
    {"agent": "retrieval_agent", "version": "1.0.0"},
]

for spec in pipeline:
    agent = registry.get_agent(spec["agent"], spec["version"])
    node_func = executor.create_node(agent)
    graph.add_node(agent.name, node_func)
```

### Pattern 4: Tool Whitelisting

```python
# Define available tools
AVAILABLE_TOOLS = {
    "web_search": {
        "description": "Search the web",
        "requires_api_key": False,
        "whitelisted": False,  # Not approved by default
    },
    "pdf_extract": {
        "description": "Extract text from PDF",
        "requires_api_key": False,
        "whitelisted": True,  # Approved for use
    }
}

# Check before using
if not is_tool_whitelisted(tool_name):
    raise SecurityError(f"Tool {tool_name} not whitelisted")

# Log usage
log_tool_usage(agent_name, tool_name, timestamp)
```

### Pattern 3B: Supporting Workflow Agents

```python
async def _execute_workflow(self, agent: AgentManifest, inputs: dict) -> dict:
    """Execute a composed workflow (sequence of agents)."""
    workflow_spec = agent.execution.get('workflow', [])
    execution_mode = agent.execution.get('execution_mode', 'sequence')
    
    if execution_mode == 'sequence':
        return await self._execute_sequence(workflow_spec, inputs)
    elif execution_mode == 'parallel':
        return await self._execute_parallel(workflow_spec, inputs)
    else:
        raise ValueError(f"Unknown workflow mode: {execution_mode}")
```

This allows agents to compose other agents without heavyweight orchestration framework.

---

## Part E: Development Workflow

### Setting Up for Phase 1

1. **Create agent system directories**:
   ```bash
   mkdir -p backend/app/agents/builtin
   mkdir -p backend/app/tools/builtin
   mkdir -p backend/app/trace
   ```

2. **Install new dependencies**:
   ```bash
   pip install jsonschema==4.20.0
   ```

3. **Update requirements.txt**:
   ```
   jsonschema==4.20.0
   ```

4. **Create agent schema**:
   Copy `agent_schema.json` to `backend/app/agents/schema.json`

5. **Create registry module**:
   Implement `backend/app/agents/registry.py`

6. **Update FastAPI main.py**:
   - Add CORS middleware
   - Add SSE `/chat/stream` endpoint
   - Add `/api/models` endpoints

7. **Update Desktop App**:
   - Add `.env.local` with backend URL
   - Update `ChatContext.tsx` with SSE client
   - Add `TraceViewer.tsx`, `SourcePanel.tsx`
   - Update `ChatWindow.tsx` to render streaming

8. **Test**:
   ```bash
   # Backend streaming
   curl http://localhost:8000/chat/stream -d '{"query":"test"}' -H 'Content-Type: application/json'
   
   # Frontend connects
   npm run dev --prefix desktop-app
   ```

### Setting Up for Phase 2

1. **Refactor existing nodes as agents**:
   - Move node implementations to `backend/app/agents/builtin/agents.py`
   - Create JSON configs for each node

2. **Create agent executor**:
   - Implement dynamic function loading
   - Support different execution types (python, llm_chain, tool_call)

3. **Update workflow.py**:
   - Remove hardcoded node imports
   - Build graph from registry

4. **Add agent generation**:
   - Implement `agent_generator.py`
   - Create `/api/agents/generate` endpoint

### Setting Up for Phase 3

1. **Implement orchestrator**:
   - Support sub-agent spawning
   - Support parallel/sequential execution

2. **Add messaging layer**:
   - Inter-agent communication
   - Message queuing

3. **Build tool system**:
   - Tool registry
   - Whitelist manager
   - Tool execution layer

4. **Add tracing**:
   - Distributed tracing across agents
   - Trace aggregation

---

## Part F: Testing & Validation

### Unit Tests (Phase 1)

```python
# tests/test_streaming.py
def test_sse_endpoint_returns_stream():
    response = client.post("/chat/stream", json={"query": "test"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"

def test_model_list_endpoint():
    response = client.get("/api/models")
    assert response.status_code == 200
    assert "chat_models" in response.json()
```

### Integration Tests (Phase 2)

```python
# tests/test_agent_registry.py
def test_agent_registry_loads_agents():
    registry = AgentRegistry(Path("backend/app/agents/builtin"))
    agents = registry.list_agents()
    assert len(agents) > 0
    assert "planning_agent" in [a.name for a in agents]

def test_agent_execution():
    agent = registry.get_agent("planning_agent")
    result = executor.execute("planning_agent", {"query": "test"})
    assert "sub_queries" in result
```

### E2E Tests (Phase 1+)

```bash
# Using Playwright
npx playwright test desktop-app/tests/e2e/

# Test: Desktop app connects to backend
# Test: Messages stream in real-time
# Test: Trace displays correctly
# Test: Sources render with links
```

---

## Part G: Deployment & Release

### Release Checklist (End of Each Phase)

**Phase 1 Release (Week 3)**:
- ✅ All Phase 1 tests passing
- ✅ No console errors in dev tools
- ✅ Streaming works reliably over 5-minute session
- ✅ All R requirements met
- ✅ Documentation updated
- ✅ Release notes prepared

**Phase 2 Release (Week 6)**:
- ✅ All Phase 2 tests passing
- ✅ Existing RAG pipeline still works
- ✅ New agents can be created and executed
- ✅ No breaking changes to Phase 1 features

**Phase 3 Release (Week 9)**:
- ✅ Multi-agent coordination tested
- ✅ Tool whitelist working
- ✅ No security issues
- ✅ Release to production-ready status

---

**End of Technical Specification**