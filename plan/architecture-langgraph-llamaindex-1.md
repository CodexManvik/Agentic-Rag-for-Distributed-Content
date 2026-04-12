---
goal: Migrate custom RAG orchestrator to a hybrid LangGraph + LlamaIndex architecture
version: 1.0
date_created: 2026-04-06
owner: Principal AI Software Engineer
status: 'Planned'
tags: [architecture, refactor, backend, rag, langgraph, llamaindex]
---

# Introduction

![Status: Planned](https://img.shields.io/badge/status-Planned-blue)

This implementation plan outlines the migration of the existing custom, manual while-loop state-machine RAG orchestrator to a robust, hybrid architecture leveraging LangGraph and LlamaIndex. The current architecture suffers from brittleness, primarily due to manual regex/JSON parsing of Ollama model outputs, which often break when reasoning models inject `<think>` tags. The new architecture will adopt the Supervisor Agent Pattern via LangGraph for robust state management and routing. LlamaIndex will handle specialized data ingestion, chunking, embeddings, and vector search. To eliminate parse errors, LangChain's `with_structured_output` will enforce strict Pydantic schema compliance for local Ollama LLM outputs.

## 1. Requirements & Constraints

- **REQ-001**: Adopt the Supervisor Agent Pattern using LangGraph for state management and routing.
- **REQ-002**: Use LangChain's `with_structured_output` with Pydantic models to strictly enforce valid JSON outputs and bypass `<think>` tag parsing issues.
- **REQ-003**: Replace custom vector store operations with LlamaIndex `VectorStoreIndex` and `QueryEngine`.
- **CON-001**: **Separation of Concerns**: LlamaIndex strictly handles data ingestion, embeddings, and vector search. LangGraph strictly handles state, agent routing, and execution loops. Mix-ins of agentic capabilities are prohibited.
- **CON-002**: **No LangChain Chains**: Avoid heavy abstractions like `ConversationalRetrievalChain`. Use `langchain_core` only for messages and Pydantic bindings.
- **CON-003**: **Local Models Only**: All LLM and embedding calls must route strictly through local Ollama endpoints (`llama3.2:3b`, `nomic-embed-text`, etc.).

## 2. Implementation Steps

### Implementation Phase 1: Environment & State Foundation

- GOAL-001: Update system dependencies and establish the new LangGraph state schema to replace the custom `NavigatorState`.

| Task     | Description                                                                                                                                           | Completed | Date |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ---- |
| TASK-001 | Update `requirements.txt` to include `langgraph`, `langchain-ollama`, `llama-index-core`, `llama-index-llms-ollama`, `llama-index-embeddings-ollama`, and `llama-index-vector-stores-chroma`. |           |      |
| TASK-002 | Modify `backend/app/graph/state.py` to define a LangGraph `TypedDict` named `AgentState`. Must include keys: `messages` (using `Annotated` reducer), `query`, `retrieved_chunks`, `final_response`, `citations`, and `retrieval_quality`. |           |      |

### Implementation Phase 2: Schemas & Retrieval Service

- GOAL-002: Define strict output schemas and implement the isolated LlamaIndex retrieval service.

| Task     | Description                                                                                                                                                                 | Completed | Date |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ---- |
| TASK-003 | Create `backend/app/agents/schemas.py`. Define Pydantic `BaseModel` classes: `SupervisorDecision` (fields: `reasoning`, `next_agent`) and `SynthesisOutput` (fields: `answer`, `citations`, `confidence`). |           |      |
| TASK-004 | Replace `backend/app/services/vector_store.py` with `backend/app/services/retrieval.py`. Initialize a LlamaIndex `ChromaVectorStore` pointing to existing `./chroma_data`. |           |      |
| TASK-005 | Configure global LlamaIndex `Settings` in `retrieval.py` to use Ollama for embeddings (`nomic-embed-text`) and LLM.                                                        |           |      |
| TASK-006 | Implement `get_query_engine()` in `retrieval.py` returning a configured LlamaIndex query engine instance (incorporating `SimilarityPostprocessor` or `SentenceWindowNodeParser` if needed). |           |      |

### Implementation Phase 3: LangGraph Worker Nodes

- GOAL-003: Build the isolated worker nodes that perform specific graph actions securely.

| Task     | Description                                                                                                                                                                                            | Completed | Date |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------- | ---- |
| TASK-007 | Update `backend/app/graph/nodes.py` to implement `retrieval_node`. Extends `AgentState`, extracts query, invokes LlamaIndex `query_engine.query()`, formats results to chunk schema, returns `{"retrieved_chunks": results}`. |           |      |
| TASK-008 | Update `backend/app/graph/nodes.py` to implement `synthesis_node`. Formats prompt with query and chunks, invokes `ChatOllama.with_structured_output(SynthesisOutput)`, returns `{"final_response": ..., "citations": ...}`. |           |      |

### Implementation Phase 4: Supervisor Node & Graph Compilation

- GOAL-004: Implement the central supervisor routing logic and compile the LangGraph workflow.

| Task     | Description                                                                                                                                                                  | Completed | Date |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ---- |
| TASK-009 | Create `supervisor_node(state: AgentState)` in `backend/app/graph/nodes.py`. Format state analysis prompt, invoke `ChatOllama.with_structured_output(SupervisorDecision)`, return `{"next_step": decision.next_agent}`. |           |      |
| TASK-010 | Refactor `backend/app/graph/workflow.py`. Delete custom `run_workflow` while-loop. Instantiate `StateGraph(AgentState)`.                                                   |           |      |
| TASK-011 | In `workflow.py`, add graph nodes: `Supervisor`, `RetrievalAgent`, `SynthesisAgent`. Add necessary edges and conditional routing logic from Supervisor based on `next_step`. |           |      |
| TASK-012 | Compile the StateGraph into `graph = builder.compile()` in `workflow.py`.                                                                                                    |           |      |

### Implementation Phase 5: API Integration

- GOAL-005: Connect the compiled graph back to the FastAPI endpoints.

| Task     | Description                                                                                                                                                     | Completed | Date |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ---- |
| TASK-013 | Update `backend/app/main.py` endpoints (`/chat` and `/chat/stream`) to invoke the new compiled LangGraph workflow.                                              |           |      |
| TASK-014 | Implement response mapping in `backend/app/main.py` ensuring the final `AgentState` payload translates accurately to the existing FastAPI `ChatResponse` model. |           |      |

## 3. Alternatives

- **ALT-001**: Continue using custom regex parsing with heavy defensive coding against `<think>` tags. (Rejected due to high maintenance overhead and fragility with reasoning models like DeepSeek/Gemma).
- **ALT-002**: Adopt a full LangChain `ConversationalRetrievalChain`. (Rejected due to lack of transparency, difficulty in debugging, and violation of the constraint against heavy abstractions).
- **ALT-003**: Use LlamaIndex for both retrieval and orchestration. (Rejected because LangGraph offers superior low-level control for the Supervisor Agent Pattern, cyclical loops, and state inspection).

## 4. Dependencies

- **DEP-001**: Python packages: `langgraph`, `langchain-ollama`, `llama-index-core`, `llama-index-llms-ollama`, `llama-index-embeddings-ollama`, `llama-index-vector-stores-chroma`, `pydantic`.
- **DEP-002**: Local running Ollama instance with models: `llama3.2:3b`, `gemma4:e4b`, `nomic-embed-text`.
- **DEP-003**: Existing `./chroma_data` vector database.

## 5. Files

- **FILE-001**: `requirements.txt`
- **FILE-002**: `backend/app/graph/state.py`
- **FILE-003**: `backend/app/agents/schemas.py`
- **FILE-004**: `backend/app/services/retrieval.py`
- **FILE-005**: `backend/app/graph/nodes.py`
- **FILE-006**: `backend/app/graph/workflow.py`
- **FILE-007**: `backend/app/main.py`

## 6. Testing

- **TEST-001**: Verify that `<think>` tags injected by models do not break the API endpoints and `SynthesisOutput` parsing.
- **TEST-002**: Validate that `retrieved_chunks` from the LlamaIndex engine are correctly structured in the `AgentState`.
- **TEST-003**: Trace the Supervisor node routing correctly to `RetrievalAgent`, then `SynthesisAgent`, then `FINISH` based on the graph state.

## 7. Risks & Assumptions

- **RISK-001**: `ChatOllama`'s structured output capabilities might occasionally fail if the underlying local model (e.g., 3B parameter model) struggles to adhere strictly to the Pydantic schema geometry.
- **ASSUMPTION-001**: The existing `./chroma_data` database structure is perfectly compatible with `llama-index-vector-stores-chroma` without requiring a full re-ingestion of the document repository.

## 8. Related Specifications / Further Reading

- [LangGraph Documentation (Context7: /langchain-ai/langgraph)](https://langchain-ai.github.io/langgraph/)
- [LlamaIndex Documentation (Context7: /run-llama/llama_index)](https://docs.llamaindex.ai/)