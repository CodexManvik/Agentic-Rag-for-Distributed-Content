---
goal: Refactor to Production-Ready Agentic System with User-Created Agents and Unified Inference
version: 1.0
date_created: 2026-04-07
last_updated: 2026-04-07
owner: Development Team
status: 'Planned'
tags: [refactor, architecture, agentic, llm, lancedb, supervisor, production]
---

# Introduction

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

This plan refactors the existing Agentic RAG system into a production-ready, state-of-the-art agentic platform. The system will enable users to create custom agents via AI-assisted natural language or YAML/JSON configuration. A sophisticated global Supervisor Agent will orchestrate all agent interactions with dynamic routing and optional planning capabilities. The backend will use a unified inference layer supporting llama.cpp, Ollama, and local models with HuggingFace integration. ChromaDB will be replaced with LanceDB for improved local performance. Target hardware is consumer GPUs with 4-12GB VRAM.

---

## 1. Requirements & Constraints

### Functional Requirements

- **REQ-001**: Users can create agents via AI-assisted natural language descriptions
- **REQ-002**: Users can create agents via YAML/JSON configuration files
- **REQ-003**: Agents can invoke sub-agents (hierarchical agent composition)
- **REQ-004**: Single global Supervisor Agent handles all agent orchestration and routing
- **REQ-005**: Supervisor performs dynamic agent selection based on query analysis
- **REQ-006**: Optional "Planning Mode" for multi-step task decomposition when enabled
- **REQ-007**: Unified inference layer supporting llama.cpp, Ollama, and local GGUF models
- **REQ-008**: HuggingFace model download integration with quantization selection (Q4, Q5, Q8, GGUF)
- **REQ-009**: Model registry with metadata (context length, VRAM requirements, recommended use)
- **REQ-010**: Different models configurable per agent type
- **REQ-011**: LanceDB as vector store with multiple knowledge bases per user
- **REQ-012**: Session-scoped KV cache for context persistence within conversations
- **REQ-013**: File access tool (read-only, user-uploaded files only)
- **REQ-014**: Optional web search tool (toggleable by user)
- **REQ-015**: Single adaptive preset based on detected hardware capabilities
- **REQ-016**: Agent types: RAG, Code, Data Processing, Web Search, File Analysis

### Security Requirements

- **SEC-001**: No direct filesystem write/delete access for agents
- **SEC-002**: File access restricted to user-uploaded files only
- **SEC-003**: Web search tool disabled by default, opt-in only
- **SEC-004**: Agent execution sandboxed within defined tool boundaries
- **SEC-005**: No external API calls without explicit user configuration

### Performance Constraints

- **CON-001**: Target hardware: Single GPU with 4-12GB VRAM
- **CON-002**: No batched inference (single request processing)
- **CON-003**: Session-scoped KV cache only (no distributed caching)
- **CON-004**: Speculative decoding deferred to future implementation
- **CON-005**: Memory footprint must fit within consumer hardware limits

### Design Guidelines

- **GUD-001**: Minimal UI changes; adapt existing React/Vite frontend to new backend
- **GUD-002**: Settings panel remains minimal and user-friendly
- **GUD-003**: No system resource monitoring in UI (VRAM, GPU usage)
- **GUD-004**: Single codebase supporting all inference backends
- **GUD-005**: Modular architecture allowing future backend additions

### Architecture Patterns

- **PAT-001**: Strategy pattern for inference backend abstraction
- **PAT-002**: Factory pattern for agent instantiation
- **PAT-003**: Registry pattern for model and agent management
- **PAT-004**: Observer pattern for session state updates
- **PAT-005**: Command pattern for tool execution

---

## 2. Implementation Steps

### Phase 1: Core Infrastructure Refactoring

- **GOAL-001**: Establish unified inference layer and replace vector store

| Task | Description | Completed | Date |
| -------- | ----------- | --------- | ---- |
| TASK-001 | Create `backend/app/inference/` directory with `__init__.py`, `base.py` (abstract base class `InferenceBackend` with methods: `load_model()`, `generate()`, `embed()`, `unload_model()`, `get_model_info()`) | | |
| TASK-002 | Implement `backend/app/inference/llama_cpp_backend.py` using llama-cpp-python library. Class `LlamaCppBackend(InferenceBackend)` with GGUF model loading, generation with streaming support, and KV cache management | | |
| TASK-003 | Implement `backend/app/inference/ollama_backend.py` wrapping existing Ollama integration. Class `OllamaBackend(InferenceBackend)` maintaining backward compatibility with current `langchain-ollama` usage | | |
| TASK-004 | Implement `backend/app/inference/local_backend.py` for direct local model file loading. Class `LocalModelBackend(InferenceBackend)` supporting user-specified model paths | | |
| TASK-005 | Create `backend/app/inference/backend_factory.py` with `InferenceBackendFactory` class implementing factory pattern. Method `create_backend(backend_type: str, config: dict) -> InferenceBackend` | | |
| TASK-006 | Create `backend/app/inference/unified_llm.py` with `UnifiedLLM` class that wraps `InferenceBackend` and provides LangChain-compatible interface (`invoke()`, `stream()`, `with_structured_output()`) | | |
| TASK-007 | Replace ChromaDB with LanceDB: Create `backend/app/services/lancedb_store.py` with class `LanceDBVectorStore` implementing methods: `create_table()`, `add_documents()`, `search()`, `delete()`, `list_tables()` | | |
| TASK-008 | Implement multi-knowledge-base support in `lancedb_store.py`: Method `create_knowledge_base(user_id: str, kb_name: str)`, `switch_knowledge_base()`, `list_knowledge_bases()` | | |
| TASK-009 | Create `backend/app/inference/model_registry.py` with `ModelRegistry` class. Fields: `model_id`, `name`, `path`, `backend_type`, `context_length`, `vram_requirement_gb`, `quantization`, `recommended_use`, `is_downloaded` | | |
| TASK-010 | Implement HuggingFace model download in `backend/app/inference/hf_downloader.py`: Class `HuggingFaceDownloader` with methods `search_models()`, `download_model()`, `list_quantizations()`, `get_download_progress()` | | |
| TASK-011 | Create hardware detection module `backend/app/system/hardware.py`: Function `detect_hardware() -> HardwareInfo` returning GPU name, VRAM, CPU cores, RAM. Use `GPUtil` or `pynvml` for GPU detection | | |
| TASK-012 | Implement adaptive preset in `backend/app/system/adaptive_config.py`: Class `AdaptiveConfig` that generates optimal settings based on `HardwareInfo` (context length, batch size, model recommendations) | | |
| TASK-013 | Update `backend/app/config.py`: Remove `runtime_profile` presets (low_latency, balanced, high_quality). Add fields: `inference_backend`, `default_model_path`, `hf_cache_dir`, `lancedb_path`, `enable_planning_mode`, `enable_web_search` | | |
| TASK-014 | Create migration script `backend/scripts/migrate_chromadb_to_lancedb.py` to transfer existing ChromaDB data to LanceDB format | | |
| TASK-015 | Update `requirements.txt`: Add `llama-cpp-python>=0.2.50`, `lancedb>=0.5.0`, `huggingface-hub>=0.20.0`, `GPUtil>=1.4.0`. Remove `chromadb`, `llama-index-vector-stores-chroma` | | |

### Phase 2: Session and KV Cache Management

- **GOAL-002**: Implement session persistence and simple KV cache for context continuity

| Task | Description | Completed | Date |
| -------- | ----------- | --------- | ---- |
| TASK-016 | Create `backend/app/session/` directory with `__init__.py` | | |
| TASK-017 | Implement `backend/app/session/session_manager.py`: Class `SessionManager` with methods `create_session()`, `get_session()`, `update_session()`, `delete_session()`, `list_sessions()`. Store session data in SQLite or JSON files | | |
| TASK-018 | Create `backend/app/session/session_state.py`: Dataclass `SessionState` with fields: `session_id`, `user_id`, `created_at`, `last_active`, `conversation_history: list[Message]`, `active_agents: list[str]`, `knowledge_base_id`, `model_config` | | |
| TASK-019 | Implement `backend/app/session/kv_cache.py`: Class `SessionKVCache` wrapping llama.cpp KV cache. Methods: `save_state()`, `load_state()`, `clear()`, `get_context_tokens()`. Cache scoped to session lifetime | | |
| TASK-020 | Create `backend/app/session/conversation.py`: Class `ConversationHistory` with methods `add_message()`, `get_messages()`, `truncate_to_context_length()`, `serialize()`, `deserialize()` | | |
| TASK-021 | Update `backend/app/api/schemas.py`: Add `SessionCreateRequest`, `SessionResponse`, `ConversationMessage` schemas | | |
| TASK-022 | Add session endpoints to `backend/app/main.py`: `POST /sessions`, `GET /sessions/{id}`, `DELETE /sessions/{id}`, `GET /sessions` | | |
| TASK-023 | Modify `/chat` and `/chat/stream` endpoints to accept optional `session_id` parameter and use session context | | |

### Phase 3: Supervisor Agent Architecture

- **GOAL-003**: Build state-of-the-art Supervisor Agent with dynamic routing and planning

| Task | Description | Completed | Date |
| -------- | ----------- | --------- | ---- |
| TASK-024 | Create `backend/app/supervisor/` directory with `__init__.py` | | |
| TASK-025 | Implement `backend/app/supervisor/query_analyzer.py`: Class `QueryAnalyzer` with method `analyze(query: str) -> QueryAnalysis` returning: `intent`, `complexity`, `required_capabilities: list[str]`, `suggested_agents: list[str]`, `requires_planning: bool` | | |
| TASK-026 | Implement `backend/app/supervisor/agent_router.py`: Class `AgentRouter` with method `route(analysis: QueryAnalysis, available_agents: list[AgentManifest]) -> RoutingDecision` containing selected agents and execution order | | |
| TASK-027 | Create `backend/app/supervisor/planner.py`: Class `TaskPlanner` with method `create_plan(query: str, agents: list[str]) -> ExecutionPlan`. Returns list of `PlanStep` with agent, input_transform, output_transform, dependencies | | |
| TASK-028 | Implement `backend/app/supervisor/supervisor_agent.py`: Class `SupervisorAgent` integrating `QueryAnalyzer`, `AgentRouter`, `TaskPlanner`. Main method `process(query: str, session: SessionState) -> SupervisorResponse` | | |
| TASK-029 | Create supervisor prompt templates in `backend/app/supervisor/prompts/`: `query_analysis.txt`, `routing_decision.txt`, `plan_generation.txt`, `plan_execution.txt` | | |
| TASK-030 | Implement `backend/app/supervisor/execution_engine.py`: Class `ExecutionEngine` with method `execute_plan(plan: ExecutionPlan, state: AgentState) -> ExecutionResult`. Handles sequential and parallel agent execution | | |
| TASK-031 | Create `backend/app/supervisor/schemas.py`: Pydantic models `QueryAnalysis`, `RoutingDecision`, `PlanStep`, `ExecutionPlan`, `SupervisorResponse` | | |
| TASK-032 | Update `backend/app/graph/workflow.py`: Replace current supervisor_node with new `SupervisorAgent`. Integrate with LangGraph StateGraph | | |
| TASK-033 | Create supervisor configuration in `backend/app/supervisor/config.py`: `SupervisorConfig` with fields: `planning_enabled`, `max_plan_steps`, `routing_model`, `analysis_model`, `fallback_agent` | | |

### Phase 4: Agent Creation System

- **GOAL-004**: Enable users to create custom agents via AI-assisted or YAML/JSON configuration

| Task | Description | Completed | Date |
| -------- | ----------- | --------- | ---- |
| TASK-034 | Create `backend/app/agent_builder/` directory with `__init__.py` | | |
| TASK-035 | Define enhanced agent manifest schema `backend/app/agent_builder/manifest_schema.json`: Add fields `capabilities: list[str]`, `required_tools: list[str]`, `sub_agents: list[str]`, `model_requirements`, `prompts` | | |
| TASK-036 | Implement `backend/app/agent_builder/ai_builder.py`: Class `AIAgentBuilder` with method `build_from_description(description: str) -> AgentManifest`. Uses LLM to generate agent configuration | | |
| TASK-037 | Create AI builder prompts `backend/app/agent_builder/prompts/`: `agent_generation.txt`, `capability_extraction.txt`, `tool_selection.txt` | | |
| TASK-038 | Implement `backend/app/agent_builder/yaml_builder.py`: Class `YAMLAgentBuilder` with method `build_from_yaml(yaml_content: str) -> AgentManifest`. Validates against schema | | |
| TASK-039 | Implement `backend/app/agent_builder/json_builder.py`: Class `JSONAgentBuilder` with method `build_from_json(json_content: str) -> AgentManifest`. Validates against schema | | |
| TASK-040 | Create `backend/app/agent_builder/validator.py`: Class `AgentManifestValidator` with method `validate(manifest: dict) -> ValidationResult`. Checks schema compliance, tool availability, sub-agent existence | | |
| TASK-041 | Implement `backend/app/agent_builder/agent_factory.py`: Class `AgentFactory` with method `create_agent(manifest: AgentManifest) -> BaseAgent`. Instantiates agent with tools, prompts, model config | | |
| TASK-042 | Update `backend/app/agents/registry.py`: Add methods `register_user_agent()`, `update_user_agent()`, `delete_user_agent()`, `list_user_agents()`. Separate system agents from user-created agents | | |
| TASK-043 | Create API endpoints in `backend/app/main.py`: `POST /agents/create` (AI-assisted), `POST /agents/upload` (YAML/JSON), `GET /agents`, `GET /agents/{id}`, `PUT /agents/{id}`, `DELETE /agents/{id}` | | |
| TASK-044 | Add agent creation schemas to `backend/app/api/schemas.py`: `AgentCreateRequest`, `AgentUploadRequest`, `AgentResponse`, `AgentListResponse` | | |

### Phase 5: Built-in Agent Types

- **GOAL-005**: Implement core agent types with specialized capabilities

| Task | Description | Completed | Date |
| -------- | ----------- | --------- | ---- |
| TASK-045 | Create `backend/app/agents/builtin/` directory with `__init__.py` | | |
| TASK-046 | Implement `backend/app/agents/builtin/rag_agent.py`: Class `RAGAgent` specialized for retrieval-augmented generation. Methods: `retrieve()`, `synthesize()`, `cite()` | | |
| TASK-047 | Implement `backend/app/agents/builtin/code_agent.py`: Class `CodeAgent` for code analysis and generation. Methods: `analyze_code()`, `generate_code()`, `explain_code()` | | |
| TASK-048 | Implement `backend/app/agents/builtin/data_agent.py`: Class `DataProcessingAgent` for structured data operations. Methods: `parse_data()`, `transform()`, `summarize_data()` | | |
| TASK-049 | Implement `backend/app/agents/builtin/search_agent.py`: Class `WebSearchAgent` for web search (optional, toggleable). Methods: `search()`, `extract_content()`, `summarize_results()` | | |
| TASK-050 | Implement `backend/app/agents/builtin/file_agent.py`: Class `FileAnalysisAgent` for user-uploaded file analysis. Methods: `read_file()`, `extract_text()`, `analyze_content()` | | |
| TASK-051 | Create agent manifests in `backend/app/agents/manifests/builtin/`: `rag.agent.json`, `code.agent.json`, `data.agent.json`, `search.agent.json`, `file.agent.json` | | |
| TASK-052 | Implement sub-agent calling in `backend/app/agents/base.py`: Method `call_sub_agent(agent_name: str, inputs: dict) -> dict` enabling hierarchical agent composition | | |

### Phase 6: Tool System Refactoring

- **GOAL-006**: Implement secure tool system with file access and optional web search

| Task | Description | Completed | Date |
| -------- | ----------- | --------- | ---- |
| TASK-053 | Create `backend/app/tools/file_access/` directory with `__init__.py` | | |
| TASK-054 | Implement `backend/app/tools/file_access/file_manager.py`: Class `UserFileManager` with methods `upload_file()`, `list_files()`, `read_file()`, `get_file_metadata()`. Files stored in user-specific directories | | |
| TASK-055 | Implement `backend/app/tools/file_access/file_tool.py`: Class `FileAccessTool` wrapping `UserFileManager`. Methods: `read()`, `list()`, `search_content()`. No write/delete operations | | |
| TASK-056 | Create `backend/app/tools/web_search/` directory with `__init__.py` | | |
| TASK-057 | Implement `backend/app/tools/web_search/search_tool.py`: Class `WebSearchTool` with method `search(query: str, num_results: int) -> list[SearchResult]`. Uses DuckDuckGo or SearXNG (configurable) | | |
| TASK-058 | Implement `backend/app/tools/web_search/content_extractor.py`: Class `ContentExtractor` with method `extract(url: str) -> ExtractedContent`. Extracts main content from URLs | | |
| TASK-059 | Update `backend/app/tools/registry.py`: Add tool categorization (system, user-enabled). Method `get_enabled_tools(user_settings: dict) -> list[Tool]` | | |
| TASK-060 | Update `backend/app/tools/whitelist.py`: Add user-level tool permissions. Method `is_tool_enabled(tool_name: str, user_id: str) -> bool` | | |
| TASK-061 | Create tool manifests: `backend/app/tools/manifests/file_access.tool.json`, `backend/app/tools/manifests/web_search.tool.json` | | |
| TASK-062 | Add file upload endpoint `POST /files/upload` and file listing `GET /files` in `backend/app/main.py` | | |

### Phase 7: Model Management API

- **GOAL-007**: Implement model download, configuration, and per-agent model assignment

| Task | Description | Completed | Date |
| -------- | ----------- | --------- | ---- |
| TASK-063 | Create `backend/app/models/` directory with `__init__.py` | | |
| TASK-064 | Implement `backend/app/models/model_manager.py`: Class `ModelManager` with methods `list_local_models()`, `download_model()`, `delete_model()`, `get_model_info()`, `set_default_model()` | | |
| TASK-065 | Implement `backend/app/models/quantization.py`: Enum `QuantizationType` (Q4_0, Q4_K_M, Q5_0, Q5_K_M, Q8_0, F16). Function `get_vram_requirement(quant: QuantizationType, params: int) -> float` | | |
| TASK-066 | Create `backend/app/models/model_config.py`: Dataclass `ModelConfig` with fields: `model_id`, `context_length`, `temperature`, `top_p`, `top_k`, `repeat_penalty`, `max_tokens` | | |
| TASK-067 | Implement `backend/app/models/agent_model_mapping.py`: Class `AgentModelMapping` with methods `set_model_for_agent()`, `get_model_for_agent()`, `get_default_model()`. Stored in SQLite or JSON | | |
| TASK-068 | Add model API endpoints in `backend/app/main.py`: `GET /models/local`, `GET /models/available` (HuggingFace), `POST /models/download`, `DELETE /models/{id}`, `POST /models/agent-mapping` | | |
| TASK-069 | Update settings endpoint to include model configuration: `GET /settings/models`, `POST /settings/models` | | |

### Phase 8: Frontend Adaptation

- **GOAL-008**: Update React frontend to support new backend capabilities with minimal UI changes

| Task | Description | Completed | Date |
| -------- | ----------- | --------- | ---- |
| TASK-070 | Update `frontend/src/api/` to add new endpoints: `sessions.ts`, `agents.ts`, `models.ts`, `files.ts` | | |
| TASK-071 | Create session management hook `frontend/src/hooks/useSession.ts`: Manages session lifecycle, conversation history | | |
| TASK-072 | Update settings panel `frontend/src/components/Settings/`: Remove preset selector, add model selector, planning mode toggle, web search toggle | | |
| TASK-073 | Create agent management component `frontend/src/components/Agents/AgentList.tsx`: Lists available agents (system + user-created) | | |
| TASK-074 | Create agent creation modal `frontend/src/components/Agents/CreateAgent.tsx`: Tab interface for AI-assisted (text input) and manual (YAML/JSON upload) | | |
| TASK-075 | Update chat interface `frontend/src/components/Chat/`: Add session indicator, agent activity display, planning visualization (when enabled) | | |
| TASK-076 | Create file upload component `frontend/src/components/Files/FileUpload.tsx`: Drag-drop file upload, file list display | | |
| TASK-077 | Update model selector `frontend/src/components/Settings/ModelSelector.tsx`: Show local models, download progress, per-agent assignment | | |
| TASK-078 | Create knowledge base selector `frontend/src/components/KnowledgeBase/KBSelector.tsx`: List, create, switch knowledge bases | | |
| TASK-079 | Update `frontend/src/types/`: Add TypeScript interfaces for new API responses | | |

### Phase 9: Integration and Testing

- **GOAL-009**: Integrate all components and ensure system stability

| Task | Description | Completed | Date |
| -------- | ----------- | --------- | ---- |
| TASK-080 | Create integration tests `backend/tests/integration/test_unified_inference.py`: Test all inference backends with same prompts | | |
| TASK-081 | Create integration tests `backend/tests/integration/test_supervisor.py`: Test query analysis, routing, planning | | |
| TASK-082 | Create integration tests `backend/tests/integration/test_agent_creation.py`: Test AI-assisted and manual agent creation | | |
| TASK-083 | Create integration tests `backend/tests/integration/test_session_management.py`: Test session lifecycle, KV cache | | |
| TASK-084 | Create integration tests `backend/tests/integration/test_lancedb.py`: Test vector operations, multi-KB support | | |
| TASK-085 | Update existing tests to work with new architecture | | |
| TASK-086 | Create end-to-end test `backend/tests/e2e/test_full_workflow.py`: Complete user journey from query to response | | |
| TASK-087 | Performance benchmarks `backend/tests/benchmarks/`: Inference latency, memory usage, response quality | | |
| TASK-088 | Update `docker-compose.yml` to remove Ollama dependency (make optional), add volume mounts for models | | |

### Phase 10: Documentation and Cleanup

- **GOAL-010**: Document new architecture and clean up deprecated code

| Task | Description | Completed | Date |
| -------- | ----------- | --------- | ---- |
| TASK-089 | Update `README.md` with new architecture overview, setup instructions, hardware requirements | | |
| TASK-090 | Create `docs/architecture.md`: System architecture diagram, component interactions, data flow | | |
| TASK-091 | Create `docs/agent-creation.md`: Guide for creating agents via AI-assisted and manual methods | | |
| TASK-092 | Create `docs/model-management.md`: Guide for downloading, configuring, and assigning models | | |
| TASK-093 | Create `docs/api-reference.md`: Complete API documentation for all new endpoints | | |
| TASK-094 | Remove deprecated files: `backend/app/services/llm.py` (replaced by unified inference), ChromaDB references | | |
| TASK-095 | Remove preset-related code from `backend/app/config.py` and related files | | |
| TASK-096 | Final code review and cleanup of unused imports, dead code | | |

---

## 3. Alternatives

- **ALT-001**: **vLLM instead of llama-cpp-python** - Rejected because vLLM requires significant VRAM (16GB+) and is optimized for server deployments, not consumer hardware with 4-12GB VRAM
- **ALT-002**: **Keep ChromaDB** - Rejected because LanceDB offers better embedded performance, native Lance format efficiency, and simpler multi-table (multi-KB) support without external server
- **ALT-003**: **Hierarchical multi-supervisor architecture** - Deferred to future implementation; single global supervisor sufficient for current scope
- **ALT-004**: **ExLlamaV2 as inference backend** - Considered but llama-cpp-python has broader model compatibility and better cross-platform support; can be added later as additional backend
- **ALT-005**: **Milvus Lite instead of LanceDB** - Rejected because LanceDB has lower memory overhead and native Python integration without gRPC complexity
- **ALT-006**: **Speculative decoding for faster inference** - Deferred to future implementation per user request
- **ALT-007**: **Distributed KV cache** - Rejected per user request; session-scoped cache only

---

## 4. Dependencies

### Python Dependencies (New)

- **DEP-001**: `llama-cpp-python>=0.2.50` - Core inference engine with CUDA/Metal support
- **DEP-002**: `lancedb>=0.5.0` - Vector store replacement for ChromaDB
- **DEP-003**: `huggingface-hub>=0.20.0` - Model download and repository access
- **DEP-004**: `GPUtil>=1.4.0` - GPU detection and monitoring
- **DEP-005**: `pynvml>=11.5.0` - NVIDIA GPU information (alternative to GPUtil)
- **DEP-006**: `duckduckgo-search>=4.0.0` - Web search functionality (optional tool)
- **DEP-007**: `trafilatura>=1.6.0` - Web content extraction

### Python Dependencies (Existing - Keep)

- **DEP-008**: `fastapi>=0.115.6` - API framework
- **DEP-009**: `langgraph>=0.2.62` - Agent orchestration graphs
- **DEP-010**: `langchain>=0.3.13` - LLM abstraction layer
- **DEP-011**: `pydantic>=2.10.3` - Data validation
- **DEP-012**: `loguru>=0.7.2` - Logging

### Python Dependencies (Remove)

- **DEP-013**: Remove `chromadb` - Replaced by LanceDB
- **DEP-014**: Remove `llama-index-vector-stores-chroma` - No longer needed
- **DEP-015**: Remove `langchain-ollama` after migration complete (optional, keep for backward compatibility)

### System Dependencies

- **DEP-016**: CUDA Toolkit 11.8+ or 12.x (for NVIDIA GPU acceleration)
- **DEP-017**: CMake (for llama-cpp-python compilation)

---

## 5. Files

### New Files to Create

- **FILE-001**: `backend/app/inference/__init__.py` - Inference module initialization
- **FILE-002**: `backend/app/inference/base.py` - Abstract InferenceBackend class
- **FILE-003**: `backend/app/inference/llama_cpp_backend.py` - llama.cpp integration
- **FILE-004**: `backend/app/inference/ollama_backend.py` - Ollama wrapper
- **FILE-005**: `backend/app/inference/local_backend.py` - Local model loading
- **FILE-006**: `backend/app/inference/backend_factory.py` - Backend factory
- **FILE-007**: `backend/app/inference/unified_llm.py` - Unified LLM interface
- **FILE-008**: `backend/app/inference/model_registry.py` - Model registry
- **FILE-009**: `backend/app/inference/hf_downloader.py` - HuggingFace downloader
- **FILE-010**: `backend/app/services/lancedb_store.py` - LanceDB vector store
- **FILE-011**: `backend/app/session/session_manager.py` - Session management
- **FILE-012**: `backend/app/session/session_state.py` - Session state model
- **FILE-013**: `backend/app/session/kv_cache.py` - KV cache wrapper
- **FILE-014**: `backend/app/session/conversation.py` - Conversation history
- **FILE-015**: `backend/app/supervisor/supervisor_agent.py` - Main supervisor
- **FILE-016**: `backend/app/supervisor/query_analyzer.py` - Query analysis
- **FILE-017**: `backend/app/supervisor/agent_router.py` - Agent routing
- **FILE-018**: `backend/app/supervisor/planner.py` - Task planning
- **FILE-019**: `backend/app/supervisor/execution_engine.py` - Plan execution
- **FILE-020**: `backend/app/agent_builder/ai_builder.py` - AI-assisted builder
- **FILE-021**: `backend/app/agent_builder/yaml_builder.py` - YAML config builder
- **FILE-022**: `backend/app/agent_builder/json_builder.py` - JSON config builder
- **FILE-023**: `backend/app/agent_builder/agent_factory.py` - Agent factory
- **FILE-024**: `backend/app/system/hardware.py` - Hardware detection
- **FILE-025**: `backend/app/system/adaptive_config.py` - Adaptive preset
- **FILE-026**: `backend/app/tools/file_access/file_tool.py` - File access tool
- **FILE-027**: `backend/app/tools/web_search/search_tool.py` - Web search tool
- **FILE-028**: `backend/app/models/model_manager.py` - Model management

### Files to Modify

- **FILE-029**: `backend/app/config.py` - Remove presets, add new settings
- **FILE-030**: `backend/app/main.py` - Add new API endpoints
- **FILE-031**: `backend/app/api/schemas.py` - Add new request/response schemas
- **FILE-032**: `backend/app/graph/workflow.py` - Integrate new supervisor
- **FILE-033**: `backend/app/agents/registry.py` - Add user agent methods
- **FILE-034**: `backend/app/tools/registry.py` - Add tool categorization
- **FILE-035**: `requirements.txt` - Update dependencies
- **FILE-036**: `frontend/src/components/Settings/` - Update settings panel

### Files to Remove

- **FILE-037**: `backend/app/services/vector_store.py` - Replaced by lancedb_store.py
- **FILE-038**: References to ChromaDB in all files

---

## 6. Testing

- **TEST-001**: Unit test `test_inference_backends.py` - Test each inference backend independently with mock models
- **TEST-002**: Unit test `test_model_registry.py` - Test model registration, lookup, metadata operations
- **TEST-003**: Unit test `test_lancedb_store.py` - Test vector operations, multi-KB, search accuracy
- **TEST-004**: Unit test `test_session_manager.py` - Test session CRUD, conversation history, KV cache
- **TEST-005**: Unit test `test_supervisor_agent.py` - Test query analysis, routing decisions, planning
- **TEST-006**: Unit test `test_agent_builder.py` - Test AI-assisted and manual agent creation
- **TEST-007**: Unit test `test_file_tool.py` - Test file upload, read, list operations
- **TEST-008**: Unit test `test_web_search.py` - Test search, content extraction (with mocks)
- **TEST-009**: Integration test `test_full_inference_pipeline.py` - End-to-end inference with real models
- **TEST-010**: Integration test `test_agent_orchestration.py` - Multi-agent workflows
- **TEST-011**: Integration test `test_session_persistence.py` - Session save/restore with KV cache
- **TEST-012**: Performance test `bench_inference_latency.py` - Measure inference speed across backends
- **TEST-013**: Performance test `bench_vector_search.py` - Measure LanceDB search latency
- **TEST-014**: E2E test `test_user_journey.py` - Complete flow: session → query → agent → response

---

## 7. Risks & Assumptions

### Risks

- **RISK-001**: llama-cpp-python compilation issues on Windows may require pre-built wheels or manual CUDA setup
- **RISK-002**: Memory pressure on 4GB VRAM GPUs may require aggressive quantization (Q4) and small context windows
- **RISK-003**: LanceDB migration may cause data loss if ChromaDB export is incomplete; mitigate with backup before migration
- **RISK-004**: AI-assisted agent creation may generate invalid configurations; mitigate with strict validation
- **RISK-005**: Web search tool rate limiting may affect user experience; mitigate with caching and fallbacks
- **RISK-006**: KV cache memory consumption may exceed available RAM for long sessions; mitigate with automatic truncation

### Assumptions

- **ASSUMPTION-001**: Users have Python 3.10+ installed
- **ASSUMPTION-002**: Users have NVIDIA GPU with CUDA support OR Apple Silicon with Metal support OR capable CPU
- **ASSUMPTION-003**: Users have stable internet connection for HuggingFace model downloads
- **ASSUMPTION-004**: Existing agent manifests in `backend/app/agents/manifests/` are compatible with new schema after minor updates
- **ASSUMPTION-005**: Frontend React/Vite setup remains unchanged; only component updates required
- **ASSUMPTION-006**: LangGraph StateGraph API remains stable and backward compatible
- **ASSUMPTION-007**: User-uploaded files are reasonably sized (< 100MB each)

---

## 8. Related Specifications / Further Reading

- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [llama-cpp-python GitHub](https://github.com/abetlen/llama-cpp-python)
- [LanceDB Documentation](https://lancedb.github.io/lancedb/)
- [HuggingFace Hub Documentation](https://huggingface.co/docs/huggingface_hub)
- [GGUF Model Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [Existing Project Plan: feature-agentic-rag-production-1.md](./feature-agentic-rag-production-1.md)
- [Existing Technical Specification: technical-specification-1.md](./technical-specification-1.md)
