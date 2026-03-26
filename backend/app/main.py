from fastapi import FastAPI, HTTPException

from app.config import settings
from app.api.schemas import ChatRequest, ChatResponse, Citation, RetrievalQuality, TraceEvent
from app.graph.workflow import run_workflow
from app.services.llm import check_ollama_readiness
from app.services.vector_store import refresh_bm25_cache

app = FastAPI(title="Smart Knowledge Navigator API", version="0.1.0")


_health_state: dict[str, str] = {"status": "starting", "reason": "initializing"}


@app.on_event("startup")
def startup_checks() -> None:
    ready, reason = check_ollama_readiness()
    if not ready:
        _health_state["status"] = "degraded"
        _health_state["reason"] = reason
        if settings.fail_fast_on_startup:
            raise RuntimeError(f"Startup readiness failed: {reason}")
        return

    try:
        refresh_bm25_cache()
    except Exception as exc:
        _health_state["status"] = "degraded"
        _health_state["reason"] = f"BM25 cache warmup failed: {exc}"
        if settings.fail_fast_on_startup:
            raise RuntimeError(_health_state["reason"]) from exc
        return

    _health_state["status"] = "ok"
    _health_state["reason"] = "ready"


@app.get("/health")
def health() -> dict[str, str]:
    return _health_state


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    if _health_state["status"] != "ok":
        raise HTTPException(status_code=503, detail=f"Service unavailable: {_health_state['reason']}")

    try:
        result = run_workflow(payload.query)
    except Exception as exc:
        print(f"Workflow failed for /chat request: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Workflow failed due to internal processing error. Check server logs.",
        ) from exc

    citations = [Citation(**c) for c in result["citations"]]
    trace = [TraceEvent(**t) for t in result["trace"]]
    retrieval_quality = RetrievalQuality(**result["retrieval_quality"])
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


@app.post("/chat/debug")
def chat_debug(payload: ChatRequest) -> dict:
    if not settings.debug_trace_enabled:
        raise HTTPException(status_code=404, detail="Debug endpoint is disabled")
    try:
        return run_workflow(payload.query)
    except Exception as exc:
        print(f"Workflow failed for /chat/debug request: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Workflow failed due to internal processing error. Check server logs.",
        ) from exc
