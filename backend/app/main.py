from typing import Any, cast
import asyncio
import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse, Response
from loguru import logger
from prometheus_client import Counter, Histogram, generate_latest

from app.config import settings
from app.api.schemas import ChatRequest, ChatResponse, Citation, RetrievalQuality, TraceEvent
from app.graph.workflow import run_workflow, workflow
from app.graph.state import NavigatorState
from app.services.llm import check_ollama_readiness
from app.services.vector_store import build_bm25_index

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager for startup and shutdown events."""
    logger.info(f"🚀 Smart Knowledge Navigator API starting (v0.1.0, profile={settings.runtime_profile})")
    
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
        try:
            logger.info("⏳ Building BM25 index...")
            await asyncio.to_thread(build_bm25_index)
            _health_state["status"] = "ok"
            _health_state["reason"] = "ready"
            logger.info("✓ BM25 index ready, service is healthy")
        except Exception as exc:
            logger.error(f"❌ BM25 cache warmup failed: {exc}")
            _health_state["status"] = "degraded"
            _health_state["reason"] = f"BM25 cache warmup failed: {exc}"
            if settings.fail_fast_on_startup:
                raise RuntimeError(_health_state["reason"]) from exc
    
    yield
    # Shutdown: cleanup code here if needed
    logger.info("🛑 Smart Knowledge Navigator API shutting down")


app = FastAPI(title="Smart Knowledge Navigator API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,  # Configured in .env or defaults to localhost:3000, localhost:5173
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Restrict to only necessary HTTP methods
    allow_headers=["Content-Type"],  # Only allow essential headers
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


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    start_time = time.time()
    # Log endpoint call without exposing user query (PII risk)
    query_hash = hash(payload.query) & 0x7FFFFFFF  # Deterministic hash for tracking
    logger.info(f"📨 /chat endpoint called (query_hash={query_hash})")
    
    if _health_state["status"] != "ok":
        error_msg = f"Service unavailable: {_health_state['reason']}"
        logger.warning(f"⚠️ /chat rejected (service degraded): {error_msg}")
        rag_queries_total.labels(endpoint="/chat", status="degraded").inc()
        raise HTTPException(status_code=503, detail=error_msg)

    try:
        logger.info(f"🔄 Running workflow for query")
        result = run_workflow(payload.query)
        latency = time.time() - start_time
        
        citations = [Citation(**c) for c in result["citations"]]
        trace = [TraceEvent(**t) for t in result["trace"]]
        retrieval_quality = RetrievalQuality(**result["retrieval_quality"])
        
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
            answer=result["final_response"],
            citations=citations,
            sub_queries=result["sub_queries"],
            confidence=result["confidence"],
            abstained=result["abstained"],
            abstain_reason=result["abstain_reason"],
            trace=trace,
            retrieval_quality=retrieval_quality,
            stage_timings=result.get("stage_timings", {}),
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
async def chat_stream(query: str) -> StreamingResponse:
    logger.info(f"📨 /chat/stream endpoint called with query: {query[:100]}...")
    
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
            }

            logger.info("🔄 Streaming workflow execution with astream")
            async for current_state in workflow.astream(initial_state):
                if not isinstance(current_state, dict):
                    continue

                current = cast(NavigatorState, current_state)
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
                if len(answer) > len(emitted_answer):
                    delta = answer[len(emitted_answer):]
                    for i in range(0, len(delta), 24):
                        token = delta[i:i + 24]
                        emitted_answer += token
                        emitted_chunk_event = True
                        yield f"event: chunk\ndata: {json.dumps({'text': token})}\n\n"

            if result is None:
                logger.warning("astream produced no states; falling back to run_workflow for stream completion")
                result = await asyncio.to_thread(run_workflow, query)

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
                if len(final_answer) > len(emitted_answer):
                    tail = final_answer[len(emitted_answer):]
                    for i in range(0, len(tail), 24):
                        token = tail[i:i + 24]
                        emitted_answer += token
                        emitted_chunk_event = True
                        yield f"event: chunk\ndata: {json.dumps({'text': token})}\n\n"

            final_payload = {
                "answer": str(result.get("final_response", "")) if result else emitted_answer,
                "citations": result.get("citations", []) if result else [],
                "sub_queries": result.get("sub_queries", []) if result else [],
                "confidence": result.get("confidence", 0.0) if result else 0.0,
                "abstained": result.get("abstained", True) if result else True,
                "abstain_reason": result.get("abstain_reason") if result else None,
                "trace": result.get("trace", []) if result else [],
                "retrieval_quality": result.get("retrieval_quality", {}) if result else {},
                "stage_timings": result.get("stage_timings", {}) if result else {},
            }
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
