from fastapi import FastAPI, HTTPException

from app.config import settings
from app.api.schemas import ChatRequest, ChatResponse, Citation, RetrievalQuality, TraceEvent
from app.graph.workflow import run_workflow

app = FastAPI(title="Smart Knowledge Navigator API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        result = run_workflow(payload.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Workflow failed: {exc}") from exc

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
    )


@app.post("/chat/debug")
def chat_debug(payload: ChatRequest) -> dict:
    if not settings.debug_trace_enabled:
        raise HTTPException(status_code=404, detail="Debug endpoint is disabled")
    try:
        return run_workflow(payload.query)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Workflow failed: {exc}") from exc
