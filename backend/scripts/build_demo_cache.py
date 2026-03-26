import argparse
import json
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fallback cached answers for live demo backup")
    parser.add_argument("--backend-url", default="http://localhost:8000/chat", help="Backend chat endpoint")
    parser.add_argument("--output", default="backend/resources/demo_cached_answers.json", help="Output JSON cache file")
    args = parser.parse_args()

    demo_queries = [
        "What are the key concepts behind LangChain RAG pipelines?",
        "Compare orchestration patterns across LangChain and LangGraph docs and cite each claim.",
        "Describe internal salary bands for a non-public enterprise.",
    ]

    cached: dict[str, dict] = {}
    for query in demo_queries:
        try:
            response = requests.post(args.backend_url, json={"query": query}, timeout=90)
            response.raise_for_status()
            payload = response.json()
            cached[query] = payload if isinstance(payload, dict) else {"answer": str(payload)}
            print(f"Cached: {query}")
        except Exception as exc:
            cached[query] = {
                "answer": "I do not have sufficient information in the retrieved documents to answer this query.",
                "abstained": True,
                "abstain_reason": f"Cache fallback due to backend error: {exc}",
                "citations": [],
                "confidence": 0.0,
                "trace": [],
                "retrieval_quality": {
                    "max_score": 0.0,
                    "avg_score": 0.0,
                    "source_diversity": 0,
                    "chunk_count": 0,
                    "adequate": False,
                    "reason": "Cached fallback",
                },
                "stage_timings": {},
            }
            print(f"Fallback cached: {query}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cached, indent=2), encoding="utf-8")
    print(f"Saved cache: {output_path}")


if __name__ == "__main__":
    main()
