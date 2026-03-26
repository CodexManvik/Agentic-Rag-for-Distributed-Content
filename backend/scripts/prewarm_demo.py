import time

from app.services.llm import check_ollama_readiness, invoke_chat_with_timeout
from app.services.vector_store import refresh_bm25_cache


def main() -> None:
    start = time.perf_counter()
    ready, reason = check_ollama_readiness()
    if not ready:
        raise RuntimeError(f"Readiness failed: {reason}")

    refresh_bm25_cache()
    invoke_chat_with_timeout(
        "Reply with exactly: warm",
        purpose="demo_prewarm",
        timeout_seconds=8.0,
    )
    elapsed = (time.perf_counter() - start) * 1000
    print(f"Prewarm complete in {elapsed:.1f}ms")


if __name__ == "__main__":
    main()
