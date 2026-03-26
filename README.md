# Smart Knowledge Navigator

Agentic RAG prototype that ingests public Confluence/web pages and files, retrieves relevant evidence from ChromaDB, and returns cited answers through a LangGraph multi-agent workflow.

## Stack

- Orchestration: LangGraph
- Backend: FastAPI (Python)
- Frontend: Streamlit
- Vector Store: ChromaDB
- LLM: Azure OpenAI
- Infra: Docker Compose

## Architecture

StateGraph nodes:

1. Planning Agent: decomposes the user question into retrieval sub-queries.
2. Retrieval Agent: fetches relevant chunks from ChromaDB across ingested sources.
3. Synthesis Agent: produces final answer with strict footnote citations.

State tracks:

- original query
- planned sub-queries
- retrieved chunks
- final synthesized response
- citation metadata

## Setup

1. Copy environment template.
2. Start services.

```bash
cp .env.example .env
docker compose up --build
```

Backend: http://localhost:8000/docs  
Frontend: http://localhost:8501

## API

`POST /chat`

```json
{
  "query": "What does the onboarding policy require?"
}
```

Response contains:

- `answer`: synthesized text with `[n]` citations
- `citations`: citation metadata list
- `sub_queries`: planning output

## Ingestion

Use `backend/app/services/ingestion.py` to ingest public URLs and PDFs before querying.

- `ingest_web_page(url)`
- `ingest_pdf(file_path)`

## Notes

- Only ingest public or approved data.
- If Azure OpenAI variables are not set, workflow uses a deterministic fallback for planning and synthesis.
