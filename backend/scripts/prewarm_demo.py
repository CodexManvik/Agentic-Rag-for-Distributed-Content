import argparse
import sys
import time
from pathlib import Path

# Add backend directory to path so imports work from any location
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.graph.workflow import run_workflow
from app.services.llm import check_ollama_readiness, invoke_chat_with_timeout, get_embedding_model
from app.services.vector_store import refresh_bm25_cache


def _warmup_queries() -> list[str]:
    """Diverse warmup queries to exercise different retrieval patterns."""
    return [
        # Factual lookup
        "What is vector retrieval in RAG?",
        # Multi-hop reasoning
        "How do citation validation and synthesis interact in LangGraph workflows?",
        # Edge case - should abstain
        "When should the assistant abstain due to weak evidence?",
        # Longer context query
        "Explain the complete workflow from query ingestion through response synthesis with citations.",
        # Keyword-heavy (tests BM25)
        "embedding vector database similarity metrics",
    ]


def main(queries: int) -> None:
    start = time.perf_counter()
    print("[Prewarm] Checking Ollama readiness...")
    ready, reason = check_ollama_readiness()
    if not ready:
        raise RuntimeError(f"Readiness failed: {reason}")

    print("[Prewarm] Loading embedding model into VRAM...")
    try:
        # Prewarm embedding model explicitly
        embedding_model = get_embedding_model()
        embedding_model.embed_documents(["warmup"])
        print("[Prewarm] ✓ Embedding model loaded")
    except Exception as exc:
        print(f"[Prewarm] ⚠ Embedding model load warning: {exc}")

    print("[Prewarm] Refreshing BM25 retrieval cache...")
    refresh_bm25_cache()

    warmups = _warmup_queries()[: max(1, queries)]
    for idx, query in enumerate(warmups, start=1):
        try:
            result = run_workflow(query)
            if isinstance(result, dict):
                abstained = bool(result.get('abstained', False))
                stage_timings = result.get('stage_timings', {})
                print(
                    f"[Prewarm] {idx}/{len(warmups)} | "
                    f"query='{query[:40]}...' | "
                    f"abstained={abstained} | "
                    f"latency={sum(stage_timings.values()) if stage_timings else 0:.0f}ms"
                )
            else:
                print(f"[Prewarm] {idx}/{len(warmups)} | query='{query[:40]}...' | ✓ completed")
        except Exception as exc:
            print(f"[Prewarm] {idx}/{len(warmups)} | ⚠ workflow error: {exc}")

    print("[Prewarm] Loading language model into VRAM...")
    try:
        invoke_chat_with_timeout(
            "Reply with exactly: ready",
            purpose="demo_prewarm",
            timeout_seconds=20.0,
        )
        print("[Prewarm] ✓ Language model loaded")
    except Exception as exc:
        print(f"[Prewarm] ⚠ Language model load warning: {exc}")

    elapsed = (time.perf_counter() - start) * 1000
    print(f"[Prewarm] Complete in {elapsed:.1f}ms - embeddings and model ready in VRAM")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prewarm retrieval, BM25, and model runtime for demo runs")
    parser.add_argument("--queries", type=int, default=3, help="Number of warmup workflow queries to run")
    args = parser.parse_args()
    main(args.queries)
