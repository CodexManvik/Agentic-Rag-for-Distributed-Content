import argparse
import time

from app.graph.workflow import run_workflow
from app.services.llm import check_ollama_readiness, invoke_chat_with_timeout
from app.services.vector_store import refresh_bm25_cache


def _warmup_queries() -> list[str]:
    return [
        "How is vector retrieval used in RAG pipelines?",
        "How do citation validation and synthesis interact in LangGraph workflows?",
        "When should the assistant abstain due to weak evidence?",
    ]


def main(queries: int) -> None:
    start = time.perf_counter()
    ready, reason = check_ollama_readiness()
    if not ready:
        raise RuntimeError(f"Readiness failed: {reason}")

    refresh_bm25_cache()
    warmups = _warmup_queries()[: max(1, queries)]
    for idx, query in enumerate(warmups, start=1):
        result = run_workflow(query)
        print(
            f"Warmup {idx}/{len(warmups)} | "
            f"abstained={bool(result.get('abstained', False))} | "
            f"stages={result.get('stage_timings', {})}"
        )

    invoke_chat_with_timeout(
        "Reply with exactly: warm",
        purpose="demo_prewarm",
        timeout_seconds=12.0,
    )
    elapsed = (time.perf_counter() - start) * 1000
    print(f"Prewarm complete in {elapsed:.1f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prewarm retrieval, BM25, and model runtime for demo runs")
    parser.add_argument("--queries", type=int, default=3, help="Number of warmup workflow queries to run")
    args = parser.parse_args()
    main(args.queries)
