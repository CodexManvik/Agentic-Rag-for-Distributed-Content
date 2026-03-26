from fastapi import FastAPI

from app.api.schemas import ChatRequest, ChatResponse, Citation
from app.graph.workflow import run_workflow

app = FastAPI(title="Smart Knowledge Navigator API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    result = run_workflow(payload.query)
    citations = [Citation(**c) for c in result["citations"]]
    return ChatResponse(
        answer=result["final_response"],
        citations=citations,
        sub_queries=result["sub_queries"],
    )
