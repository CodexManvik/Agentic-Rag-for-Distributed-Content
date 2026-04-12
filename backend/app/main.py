from typing import Any, cast
import asyncio
import hashlib
import json
import time
import shutil, tempfile
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse, Response
from loguru import logger
from prometheus_client import Counter, Histogram, generate_latest

from app.config import settings
from app.api.schemas import (
    ChatRequest, ChatResponse, Citation, RetrievalQuality, TraceEvent,
    CreateSessionRequest, CreateSessionResponse, UpdateSessionRequest, 
    SessionStateSchema, SessionListResponse, SessionStatsResponse, ErrorResponse,
    ConversationMessageSchema, ModelConfigSchema,
)
from app.graph.workflow import run_workflow, workflow
from app.graph.state import NavigatorState
from app.services.llm import check_ollama_readiness
from app.services.lancedb_ingestion import ingest_pdf, ingest_text_file
from app.session.session_manager import SessionManager
from app.session.session_state import ConversationMessage, MessageRole, SessionState, ModelConfig

# Prometheus metrics
rag_queries_total = Counter(
    'rag_queries_total',
    'Total number of RAG queries processed',
    ['endpoint', 'status']
)
rag_query_latency = Histogram(
    'rag_query_latency_seconds',
    'RAG query latency in seconds',
    ['endpoint']
)
retrieval_quality_score = Histogram(
    'retrieval_quality_score',
    'Retrieval quality score (0-1)',
    ['endpoint'],
    buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
)

_health_state: dict[str, str] = {"status": "starting", "reason": "initializing"}

# Global session manager instance
session_manager: SessionManager | None = None
MAX_SESSION_LIST_LIMIT = 100


def _to_session_state_schema(session: SessionState) -> SessionStateSchema:
    """Convert internal SessionState dataclass to API schema."""
    history = [
        ConversationMessageSchema(
            role=message.role,
            content=message.content,
            timestamp=message.timestamp,
            metadata=message.metadata or {},
        )
        for message in session.conversation_history
    ]

    model_config_schema: ModelConfigSchema | None = None
    if session.model_configuration:
        model_conf = session.model_configuration
        if isinstance(model_conf, dict):
            model_config_schema = ModelConfigSchema(**model_conf)
        elif isinstance(model_conf, ModelConfig):
            model_config_schema = ModelConfigSchema(
                backend=model_conf.backend,
                model_name=model_conf.model_name,
                temperature=model_conf.temperature,
                max_tokens=model_conf.max_tokens,
                context_length=model_conf.context_length,
                additional_params=model_conf.additional_params,
            )

    return SessionStateSchema(
        session_id=session.session_id,
        user_id=session.user_id,
        created_at=session.created_at,
        last_active=session.last_active,
        conversation_history=history,
        active_agents=session.active_agents,
        knowledge_base_id=session.knowledge_base_id,
        model_configuration=model_config_schema,
        session_metadata=session.session_metadata,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager for startup and shutdown events."""
    logger.info(f"🚀 Smart Knowledge Navigator API starting (v0.1.0, profile={settings.runtime_profile})")
    
    # Initialize session manager
    global session_manager
    session_db_path = Path("data") / "sessions.db"
    session_db_path.parent.mkdir(exist_ok=True)
    session_manager = SessionManager(db_path=str(session_db_path))
    logger.info("✓ Session manager initialized")
    
    # Startup: run checks before server accepts requests
    # Wrap blocking I/O in asyncio.to_thread to avoid blocking the event loop
    logger.info("⏳ Checking Ollama readiness...")
    ready, reason = await asyncio.to_thread(check_ollama_readiness)
    if not ready:
        logger.warning(f"⚠️ Ollama check failed: {reason}")
        _health_state["status"] = "degraded"
        _health_state["reason"] = reason
        if settings.fail_fast_on_startup:
            logger.error(f"❌ Startup failed: {reason}")
            raise RuntimeError(f"Startup readiness failed: {reason}")
    else:
        logger.info("✓ Ollama ready")
        _health_state["status"] = "ok"
        _health_state["reason"] = "ready"
    
    yield
    # Shutdown: cleanup code here if needed
    if session_manager:
        session_manager.cleanup_old_sessions()
        logger.info("✓ Session manager cleaned up")
    logger.info("🛑 Smart Knowledge Navigator API shutting down")


app = FastAPI(title="Smart Knowledge Navigator API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,  # Configured in .env or defaults to localhost:3000, localhost:5173
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods including OPTIONS
    allow_headers=["*"],  # Allow all headers
)


@app.get("/health")
def health() -> dict[str, str]:
    logger.debug(f"health check: {_health_state['status']}")
    return _health_state


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus metrics endpoint for monitoring and observability."""
    logger.debug("metrics endpoint accessed")
    return Response(generate_latest(), media_type="text/plain")


@app.get("/models")
async def get_available_models() -> dict[str, list[str]]:
    """Get list of available models from Ollama."""
    logger.info("📨 /models endpoint called to fetch available models")
    try:
        # Use asyncio.to_thread to avoid blocking the event loop with requests.get
        import requests
        response = await asyncio.to_thread(
            requests.get,
            f"{settings.ollama_base_url}/api/tags",
            timeout=5
        )
        if response.ok:
            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]
            logger.info(f"✓ /models returned {len(models)} available models")
            return {"models": models}
        else:
            logger.warning(f"⚠️ Ollama /api/tags returned {response.status_code}")
            # Fallback to configured model
            return {"models": [settings.ollama_chat_model]}
    except Exception as e:
        logger.error(f"❌ Failed to fetch models from Ollama: {e}")
        # Return the configured default model as fallback
        return {"models": [settings.ollama_chat_model]}


@app.get("/api/models")
async def get_available_models_api() -> dict[str, list[str]]:
    """Phase 1 endpoint alias for model listing used by frontend."""
    return await get_available_models()

@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)) -> dict:
    """Ingest an uploaded file (PDF, TXT, or MD) into the vector store."""
    logger.info(f"📥 /ingest endpoint called with file: {file.filename}")
    
    if not file.filename:
        logger.error("❌ No filename provided")
        raise HTTPException(status_code=400, detail="No filename provided")
 
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Accepted: .pdf, .txt, .md",
        )
 
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
 
    try:
        if suffix == ".pdf":
            stats = ingest_pdf(tmp_path)
        else:
            stats = ingest_text_file(tmp_path)
 
        if stats["errors"]:
            logger.warning(f"Ingestion warnings for {file.filename}: {stats['errors']}")
 
        return {
            "filename": file.filename,
            "chunks_added": stats["chunks_added"],
            "skipped_duplicates": stats["skipped_duplicates"],
            "errors": stats["errors"],
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)
 
@app.get("/settings")
def get_settings() -> dict:
    """Return the current set of tunable runtime settings."""
    return {
        "model_temperature": settings.model_temperature,
        "ollama_chat_model": settings.ollama_chat_model,
        "context_chunk_limit": settings.context_chunk_limit,
        "context_chunk_char_limit": settings.context_chunk_char_limit,
        "planner_max_subqueries": settings.planner_max_subqueries,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "runtime_profile": settings.runtime_profile,
        "retrieval_top_k": settings.effective_retrieval_top_k,
        "max_retrieval_retries": settings.max_retrieval_retries,
        "max_validation_retries": settings.max_validation_retries,
        "enable_short_circuit_routing": settings.enable_short_circuit_routing,
        "short_circuit_confidence_threshold": settings.short_circuit_confidence_threshold,
    }
 
 
@app.post("/settings")
def update_settings(payload: dict) -> dict:
    """
    Dynamically patch runtime settings.
    Only the keys listed in `allowed` can be changed.
    """
    allowed = {
        "model_temperature",
        "context_chunk_limit",
        "context_chunk_char_limit",
        "planner_max_subqueries",
        "runtime_profile",
        "max_retrieval_retries",
        "max_validation_retries",
        "enable_short_circuit_routing",
        "short_circuit_confidence_threshold",
    }
    updated: dict = {}
    for key, value in payload.items():
        if key not in allowed:
            continue
        if not hasattr(settings, key):
            continue
        try:
            current_type = type(getattr(settings, key))
            if current_type is bool and isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "1", "yes", "y", "on"}:
                    parsed_value = True
                elif normalized in {"false", "0", "no", "n", "off"}:
                    parsed_value = False
                else:
                    raise ValueError(f"Invalid boolean string: {value}")
                setattr(settings, key, parsed_value)
                updated[key] = parsed_value
            else:
                cast_value = current_type(value)
                setattr(settings, key, cast_value)
                updated[key] = cast_value
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid value for '{key}': {exc}")
    return {"updated": updated}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    start_time = time.time()
    # Log endpoint call without exposing user query (PII risk)
    query_hash = hash(payload.query) & 0x7FFFFFFF  # Deterministic hash for tracking
    session_info = f" session_id: {payload.session_id}" if payload.session_id else ""
    logger.info(f"📨 /chat endpoint called (query_hash={query_hash})" + (f" model: {payload.model}" if payload.model else "") + session_info)
    
    if _health_state["status"] != "ok":
        error_msg = f"Service unavailable: {_health_state['reason']}"
        logger.warning(f"⚠️ /chat rejected (service degraded): {error_msg}")
        rag_queries_total.labels(endpoint="/chat", status="degraded").inc()
        raise HTTPException(status_code=503, detail=error_msg)

    session_state = None
    if payload.session_id:
        if not session_manager:
            logger.warning(f"⚠️ Session requested but session manager not available")
        else:
            session_state = session_manager.get_session(payload.session_id)
            if not session_state:
                logger.warning(f"⚠️ Session {payload.session_id} not found")
                raise HTTPException(status_code=404, detail="Session not found")

    try:
        logger.info(f"🔄 Running workflow for query")
        result = run_workflow(payload.query, model=payload.model)
        latency = time.time() - start_time
        
        final_answer = result["final_response"]
        
        # Use getattr to safely check for hide_reasoning in case ChatRequest schema isn't fully updated yet
        if getattr(payload, "hide_reasoning", True):
            final_answer = re.sub(r'<think>.*?</think>\n*', '', final_answer, flags=re.DOTALL).strip()
        
        citations = [Citation(**c) for c in result["citations"]]
        trace = [TraceEvent(**t) for t in result["trace"]]
        retrieval_quality = RetrievalQuality(**result["retrieval_quality"])
        
        # Update session with conversation history
        if session_state and session_manager:
            # Add user message
            session_state.add_message(
                ConversationMessage(
                    role=MessageRole.USER,
                    content=payload.query,
                )
            )
            # Add assistant response
            session_state.add_message(
                ConversationMessage(
                    role=MessageRole.ASSISTANT,
                    content=final_answer,
                    metadata={
                        "citations": [c.model_dump() for c in citations],
                        "trace": [t.model_dump() for t in trace],
                        "stage_timings": result.get("stage_timings", {}),
                        "confidence": result["confidence"],
                        "abstained": result["abstained"],
                        "sub_queries": result["sub_queries"],
                        "short_circuited": bool(result.get("short_circuited", False)),
                    },
                )
            )
            # Save updated session
            session_manager.update_session(session_state)
            logger.info(f"✓ Updated session {payload.session_id} with new conversation")
        
        # Record metrics
        rag_queries_total.labels(endpoint="/chat", status="success").inc()
        rag_query_latency.labels(endpoint="/chat").observe(latency)
        # Use the correct histogram metric for retrieval quality
        retrieval_quality_score.labels(endpoint="/chat").observe(retrieval_quality.max_score)
        
        logger.info(
            f"✓ /chat completed in {latency:.2f}s, "
            f"citations={len(citations)}, confidence={result['confidence']:.2f}"
        )
        
        return ChatResponse(
            answer=final_answer,
            citations=citations,
            sub_queries=result["sub_queries"],
            confidence=result["confidence"],
            abstained=result["abstained"],
            abstain_reason=result["abstain_reason"],
            trace=trace,
            retrieval_quality=retrieval_quality,
            stage_timings=result.get("stage_timings", {}),
            short_circuited=bool(result.get("short_circuited", False)),
            session_id=payload.session_id  # Include session_id in response
        )
    except HTTPException:
        raise
    except Exception as exc:
        latency = time.time() - start_time
        logger.error(f"❌ /chat failed after {latency:.2f}s: {exc}", exc_info=True)
        rag_queries_total.labels(endpoint="/chat", status="error").inc()
        raise HTTPException(
            status_code=500,
            detail="Workflow failed due to internal processing error. Check server logs.",
        ) from exc


@app.get("/chat/stream")
async def chat_stream(
    query: str, 
    model: str | None = None,
    hide_reasoning: bool = True,
    session_id: str | None = None
) -> StreamingResponse:
    query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    session_info = f" session_id: {session_id}" if session_id else ""
    logger.info(f"📨 /chat/stream endpoint called (query_hash={query_hash})" + (f" model: {model}" if model else "") + session_info)
    
    # Validate session if provided
    session_state = None
    if session_id:
        if not session_manager:
            logger.warning(f"⚠️ Session requested but session manager not available")
        else:
            session_state = session_manager.get_session(session_id)
            if not session_state:
                logger.warning(f"⚠️ Session {session_id} not found")
                raise HTTPException(status_code=404, detail="Session not found")
    
    if _health_state["status"] != "ok":
        error_msg = f"Service unavailable: {_health_state['reason']}"
        logger.warning(f"⚠️ /chat/stream rejected (service degraded)")
        rag_queries_total.labels(endpoint="/chat/stream", status="degraded").inc()
        raise HTTPException(status_code=503, detail=error_msg)

    async def generate():
        start_time = time.time()
        result: NavigatorState | None = None
        error_message: str | None = None
        emitted_answer = ""
        emitted_trace_count = 0
        emitted_trace_event = False
        emitted_chunk_event = False

        try:
            # Build initial workflow state (same shape used by run_workflow)
            initial_state: NavigatorState = {
                "query": query,
                "original_query": query,
                "sub_queries": [],
                "retrieved_chunks": [],
                "final_response": "",
                "citations": [],
                "retrieval_quality": {
                    "max_score": 0.0,
                    "avg_score": 0.0,
                    "source_diversity": 0,
                    "chunk_count": 0,
                    "adequate": False,
                    "reason": "Not evaluated",
                },
                "retries_used": 0,
                "validation_retries_used": 0,
                "validation_errors": [],
                "abstained": False,
                "abstain_reason": None,
                "confidence": 0.0,
                "used_deterministic_fallback": False,
                "cited_indices": [],
                "synthesis_output": {
                    "answer": "",
                    "cited_indices": [],
                    "confidence": 0.0,
                    "abstain_reason": None,
                },
                "trace": [],
                "stage_timings": {},
                "stage_timestamps": {},
                "selected_model": model or settings.ollama_chat_model,
            }
            state_accumulator: dict[str, Any] = dict(initial_state)
            node_update_keys = {"Supervisor", "RetrievalAgent", "SynthesisAgent"}
            logger.info(f"🔄 Streaming workflow execution with astream (model: {initial_state['selected_model']})")
            async for current_state in workflow.astream(initial_state):
                if not isinstance(current_state, dict):
                    continue

                # LangGraph astream typically yields per-node deltas like:
                # {"NodeName": {...state update...}}. Merge them into a full state.
                if any(key in node_update_keys for key in current_state.keys()):
                    for node_name, node_delta in current_state.items():
                        if node_name in node_update_keys and isinstance(node_delta, dict):
                            state_accumulator.update(node_delta)
                else:
                    # Fallback for direct state/delta payloads.
                    state_accumulator.update(current_state)

                current = cast(NavigatorState, dict(state_accumulator))
                result = current

                # Emit newly added trace entries as trace events.
                trace_events = current.get("trace", [])
                while emitted_trace_count < len(trace_events):
                    trace_event = trace_events[emitted_trace_count]
                    emitted_trace_count += 1
                    emitted_trace_event = True
                    yield f"event: trace\ndata: {json.dumps(trace_event)}\n\n"

                # Emit answer delta as chunk events (progressive text).
                answer = str(current.get("final_response", ""))
                
                # Dynamically strip unclosed or closed thinking tags
                if hide_reasoning:
                    answer = re.sub(r'<think>.*?(?:</think>\n*|$)', '', answer, flags=re.DOTALL).lstrip()

                if len(answer) > len(emitted_answer):
                    delta = answer[len(emitted_answer):]
                    for i in range(0, len(delta), 24):
                        token = delta[i:i + 24]
                        emitted_answer += token
                        emitted_chunk_event = True
                        yield f"event: chunk\ndata: {json.dumps({'text': token})}\n\n"

            if result is None:
                logger.warning("astream produced no states; falling back to run_workflow for stream completion")
                result = await asyncio.to_thread(run_workflow, query, model)

            # Compatibility fallback: still provide trace/chunk signals if astream
            # yields sparse state updates and nothing was emitted above.
            if result and not emitted_trace_event:
                trace_events = result.get("trace", [])
                if trace_events:
                    for trace_event in trace_events:
                        yield f"event: trace\ndata: {json.dumps(trace_event)}\n\n"
                        emitted_trace_count += 1
                        emitted_trace_event = True
                        
            if result and not emitted_chunk_event:
                answer_fallback = str(result.get("final_response", ""))
                if hide_reasoning:
                    answer_fallback = re.sub(r'<think>.*?(?:</think>\n*|$)', '', answer_fallback, flags=re.DOTALL).lstrip()
                if answer_fallback:
                    yield f"event: chunk\ndata: {json.dumps({'text': answer_fallback})}\n\n"
                    emitted_answer = answer_fallback
                    emitted_chunk_event = True

            # Final consistency flush: emit any trace entries not yet streamed.
            if result:
                final_trace = result.get("trace", [])
                while emitted_trace_count < len(final_trace):
                    trace_event = final_trace[emitted_trace_count]
                    emitted_trace_count += 1
                    emitted_trace_event = True
                    yield f"event: trace\ndata: {json.dumps(trace_event)}\n\n"

                final_answer = str(result.get("final_response", ""))
                if hide_reasoning:
                    final_answer = re.sub(r'<think>.*?(?:</think>\n*|$)', '', final_answer, flags=re.DOTALL).lstrip()
                    result["final_response"] = final_answer # Ensure payload sends cleaned version
                    
                if len(final_answer) > len(emitted_answer):
                    tail = final_answer[len(emitted_answer):]
                    for i in range(0, len(tail), 24):
                        token = tail[i:i + 24]
                        emitted_answer += token
                        emitted_chunk_event = True
                        yield f"event: chunk\ndata: {json.dumps({'text': token})}\n\n"

            final_answer_value = str(result.get("final_response", "")) if result else emitted_answer
            answer_is_empty = len(final_answer_value.strip()) == 0
            if result and "abstained" in result:
                abstained_value = bool(result.get("abstained"))
            else:
                # If upstream state is sparse (e.g. streaming step payloads), infer from final answer.
                abstained_value = answer_is_empty

            final_payload = {
                "answer": final_answer_value,
                "citations": result.get("citations", []) if result else [],
                "sub_queries": result.get("sub_queries", []) if result else [],
                "confidence": result.get("confidence", 0.0) if result else 0.0,
                "abstained": abstained_value,
                "abstain_reason": result.get("abstain_reason") if result else None,
                "trace": result.get("trace", []) if result else [],
                "retrieval_quality": result.get("retrieval_quality", {}) if result else {},
                "stage_timings": result.get("stage_timings", {}) if result else {},
                "short_circuited": bool(result.get("short_circuited", False)) if result else False,
                "session_id": session_id  # Include session_id in response
            }
            
            # Update session with conversation history if session is active
            if session_state and session_manager and result:
                try:
                    final_answer = str(result.get("final_response", ""))
                    if hide_reasoning:
                        final_answer = re.sub(r'<think>.*?(?:</think>\n*|$)', '', final_answer, flags=re.DOTALL).lstrip()
                    
                    # Add user message and assistant response to conversation history
                    session_state.add_message(
                        ConversationMessage(
                            role=MessageRole.USER,
                            content=query,
                        )
                    )
                    session_state.add_message(
                        ConversationMessage(
                            role=MessageRole.ASSISTANT,
                            content=final_answer,
                            metadata={
                                "citations": final_payload.get("citations", []),
                                "trace": final_payload.get("trace", []),
                                "stage_timings": final_payload.get("stage_timings", {}),
                                "confidence": final_payload.get("confidence", 0.0),
                                "abstained": final_payload.get("abstained", False),
                                "sub_queries": final_payload.get("sub_queries", []),
                                "short_circuited": final_payload.get("short_circuited", False),
                            },
                        )
                    )
                    
                    # Save updated session
                    session_manager.update_session(session_state)
                    logger.info(f"✓ Updated session {session_id} with streaming conversation")
                except Exception as session_exc:
                    logger.error(f"❌ Failed to update session {session_id}: {session_exc}", exc_info=True)
            
            yield f"event: complete\ndata: {json.dumps(final_payload)}\n\n"
        except (GeneratorExit, asyncio.CancelledError):
            # Client disconnected or request was cancelled
            logger.info("ℹ️ /chat/stream client disconnected")
            # Don't re-raise; gracefully end the stream
        except Exception as exc:
            error_message = str(exc)
            logger.error(f"❌ Stream generation error: {exc}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': error_message})}\n\n"
        finally:
            latency = time.time() - start_time
            status = "error" if error_message else "success"
            rag_queries_total.labels(endpoint="/chat/stream", status=status).inc()
            rag_query_latency.labels(endpoint="/chat/stream").observe(latency)
            logger.info(f"✓ /chat/stream completed in {latency:.2f}s, status={status}")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/chat/debug")
def chat_debug(payload: ChatRequest) -> dict[str, Any]:
    logger.info(f"🐛 /chat/debug endpoint called (debug_trace_enabled={settings.debug_trace_enabled})")
    
    if not settings.debug_trace_enabled:
        logger.warning(f"⚠️ /chat/debug rejected (debug endpoint disabled)")
        raise HTTPException(status_code=404, detail="Debug endpoint is disabled")
    
    try:
        logger.info(f"🔄 Running workflow in debug mode")
        start_time = time.time()
        result = dict(run_workflow(payload.query))
        latency = time.time() - start_time
        logger.info(f"✓ /chat/debug completed in {latency:.2f}s")
        return result
    except Exception as exc:
        logger.error(f"❌ /chat/debug failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Workflow failed due to internal processing error. Check server logs.",
        ) from exc


# ==================== SESSION MANAGEMENT ENDPOINTS ====================

@app.post("/sessions", response_model=CreateSessionResponse)
def create_session(request: CreateSessionRequest) -> CreateSessionResponse:
    """Create a new chat session."""
    try:
        if not session_manager:
            raise HTTPException(status_code=500, detail="Session manager not initialized")
        
        session = session_manager.create_session(
            user_id=request.user_id,
            knowledge_base_id=request.knowledge_base_id,
            model_config=request.model_configuration.model_dump() if request.model_configuration else None
        )
        
        logger.info(f"✓ Created session {session.session_id} for user {request.user_id}")
        
        return CreateSessionResponse(
            session_id=session.session_id,
            created_at=session.created_at,
            message="Session created successfully"
        )
    except Exception as exc:
        logger.error(f"❌ Failed to create session: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")



@app.get("/sessions/stats", response_model=SessionStatsResponse)
def get_session_stats() -> SessionStatsResponse:
    """Get session statistics."""
    try:
        if not session_manager:
            raise HTTPException(status_code=500, detail="Session manager not initialized")
        
        stats = session_manager.get_stats()
        
        return SessionStatsResponse(
            total_sessions=stats["total_sessions"],
            active_sessions=stats["active_sessions"],
            messages_today=stats["messages_today"],
            avg_session_length=stats["avg_session_length"],
            top_knowledge_bases=stats["top_knowledge_bases"]
        )
    except Exception as exc:
        logger.error(f"❌ Failed to get session stats: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/sessions/{session_id}", response_model=SessionStateSchema)
def get_session(session_id: str) -> SessionStateSchema:
    """Get session state by ID."""
    try:
        if not session_manager:
            raise HTTPException(status_code=500, detail="Session manager not initialized")
        
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return _to_session_state_schema(session)
    except HTTPException:
        raise
    except Exception as exc:
        logger.opt(exception=True).error("❌ Failed to get session {}: {}", session_id, exc)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.put("/sessions/{session_id}", response_model=SessionStateSchema)
def update_session(session_id: str, request: UpdateSessionRequest) -> SessionStateSchema:
    """Update session metadata or configuration."""
    try:
        if not session_manager:
            raise HTTPException(status_code=500, detail="Session manager not initialized")
        
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Update fields if provided
        if request.knowledge_base_id is not None:
            session.knowledge_base_id = request.knowledge_base_id
        if request.model_configuration is not None:
            session.model_configuration = request.model_configuration.model_dump()
        if request.session_metadata is not None:
            session.session_metadata.update(request.session_metadata)
        
        # Save updated session
        session_manager.update_session(session)
        
        logger.info(f"✓ Updated session {session_id}")
        
        return _to_session_state_schema(session)
    except HTTPException:
        raise
    except Exception as exc:
        logger.opt(exception=True).error("❌ Failed to update session {}: {}", session_id, exc)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, str]:
    """Delete a session."""
    try:
        if not session_manager:
            raise HTTPException(status_code=500, detail="Session manager not initialized")
        
        success = session_manager.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")
        
        logger.info(f"✓ Deleted session {session_id}")
        
        return {"message": "Session deleted successfully"}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"❌ Failed to delete session {session_id}: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/sessions", response_model=SessionListResponse)
def list_sessions(user_id: str | None = None, limit: int = 20, offset: int = 0) -> SessionListResponse:
    """List sessions, optionally filtered by user ID."""
    try:
        if not session_manager:
            raise HTTPException(status_code=500, detail="Session manager not initialized")

        if limit <= 0 or limit > MAX_SESSION_LIST_LIMIT:
            raise HTTPException(
                status_code=400,
                detail=f"limit must be between 1 and {MAX_SESSION_LIST_LIMIT}",
            )
        if offset < 0:
            raise HTTPException(status_code=400, detail="offset must be >= 0")
        
        sessions = session_manager.list_sessions(
            user_id=user_id,
            limit=limit,
            offset=offset
        )
        
        session_list = [_to_session_state_schema(session) for session in sessions]
        
        total_count = session_manager.get_session_count(user_id=user_id)
        
        return SessionListResponse(
            sessions=session_list,
            total=total_count,
            limit=limit,
            offset=offset
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.opt(exception=True).error("❌ Failed to list sessions: {}", exc)
        raise HTTPException(status_code=500, detail="Internal server error")


