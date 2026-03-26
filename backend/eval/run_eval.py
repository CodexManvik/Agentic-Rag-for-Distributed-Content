import argparse
import json
import os
from pathlib import Path
import platform
import time
from typing import Any
import sys

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.graph.workflow import run_workflow
from app.config import settings


DATASET_PATH = Path(__file__).parent / "dataset.jsonl"
JSON_REPORT_PATH = Path(__file__).parent / "eval_report.json"
MD_REPORT_PATH = Path(__file__).parent / "eval_report.md"


def _load_dataset(limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if limit > 0:
        return rows[:limit]
    return rows


def _source_hit(expected: list[str], retrieved_sources: list[str]) -> tuple[bool, float]:
    if not expected:
        return False, 0.0
    expected_lower = [e.lower() for e in expected]
    for rank, src in enumerate(retrieved_sources, start=1):
        normalized = src.lower()
        if any(e in normalized for e in expected_lower):
            return True, 1.0 / rank
    return False, 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


def _run_profile_eval(
    dataset: list[dict[str, Any]],
    profile: str,
    show_progress: bool = True,
) -> dict[str, Any]:
    prev_profile = settings.runtime_profile
    settings.runtime_profile = profile
    try:
        total = len(dataset)
        hit_count = 0
        reciprocal_rank_sum = 0.0
        citation_tp = 0
        citation_fp = 0
        support_covered = 0
        support_total = 0
        abstain_tp = 0
        abstain_fp = 0
        abstain_fn = 0
        latencies_ms: list[float] = []
        validation_error_categories: dict[str, int] = {}

        rows: list[dict[str, Any]] = []

        total_items = len(dataset)
        for idx, item in enumerate(dataset, start=1):
            start = time.perf_counter()
            result = run_workflow(item["query"])
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed_ms)
            if show_progress:
                print(
                    f"[{profile}] {idx}/{total_items} "
                    f"{elapsed_ms:.1f}ms | abstained={bool(result.get('abstained', False))}"
                )
            retrieved = result.get("retrieved_chunks", [])
            citations = result.get("citations", [])
            abstained = bool(result.get("abstained", False))

            retrieved_sources = [str(chunk.get("source", "")) for chunk in retrieved]
            hit, rr = _source_hit(item.get("expected_sources", []), retrieved_sources)

            if hit:
                hit_count += 1
            reciprocal_rank_sum += rr

            expected_sources = [s.lower() for s in item.get("expected_sources", [])]
            for citation in citations:
                source = str(citation.get("source", "")).lower()
                if expected_sources and any(exp in source for exp in expected_sources):
                    citation_tp += 1
                elif expected_sources:
                    citation_fp += 1

            if expected_sources:
                support_total += len(expected_sources)
                for exp in expected_sources:
                    if any(exp in src.lower() for src in retrieved_sources):
                        support_covered += 1

            answerable = bool(item.get("answerable", True))
            if not answerable and abstained:
                abstain_tp += 1
            if answerable and abstained:
                abstain_fp += 1
            if (not answerable) and (not abstained):
                abstain_fn += 1

            for err in result.get("validation_errors", []):
                category = str(err).split(":", 1)[0].strip().lower()
                if category:
                    validation_error_categories[category] = validation_error_categories.get(category, 0) + 1

            rows.append(
                {
                    "query": item["query"],
                    "answerable": answerable,
                    "abstained": abstained,
                    "hit": hit,
                    "rr": rr,
                    "latency_ms": round(elapsed_ms, 2),
                    "retrieval_quality": result.get("retrieval_quality", {}),
                    "validation_errors": result.get("validation_errors", []),
                }
            )

        hit_at_k = hit_count / total if total else 0.0
        mrr = reciprocal_rank_sum / total if total else 0.0
        citation_precision = (
            citation_tp / (citation_tp + citation_fp) if (citation_tp + citation_fp) else 0.0
        )
        support_coverage = support_covered / support_total if support_total else 0.0
        abstain_precision = abstain_tp / (abstain_tp + abstain_fp) if (abstain_tp + abstain_fp) else 0.0
        abstain_recall = abstain_tp / (abstain_tp + abstain_fn) if (abstain_tp + abstain_fn) else 0.0
        p50_ms = _percentile(latencies_ms, 50)
        p95_ms = _percentile(latencies_ms, 95)

        return {
            "profile": profile,
            "dataset_size": total,
            "metrics": {
                "hit_at_k": hit_at_k,
                "mrr": mrr,
                "citation_precision": citation_precision,
                "support_coverage": support_coverage,
                "abstain_precision": abstain_precision,
                "abstain_recall": abstain_recall,
                "latency_p50_ms": p50_ms,
                "latency_p95_ms": p95_ms,
            },
            "validation_error_categories": validation_error_categories,
            "rows": rows,
        }
    finally:
        settings.runtime_profile = prev_profile


def run_eval(limit: int = 0, profiles: list[str] | None = None, show_progress: bool = True) -> dict[str, Any]:
    dataset = _load_dataset(limit=limit)
    selected_profiles = profiles or ["balanced", "low_latency"]

    profile_results: dict[str, Any] = {}
    for profile in selected_profiles:
        profile_results[profile] = _run_profile_eval(dataset, profile=profile, show_progress=show_progress)

    report = {
        "dataset_size": len(dataset),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "runtime_profile_compared": selected_profiles,
            "chat_model": settings.ollama_chat_model,
            "embedding_model": settings.ollama_embedding_model,
        },
        "profiles": profile_results,
    }

    with JSON_REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    primary_profile = selected_profiles[0]
    secondary_profile = selected_profiles[1] if len(selected_profiles) > 1 else selected_profiles[0]
    p = report["profiles"][primary_profile]["metrics"]
    s = report["profiles"][secondary_profile]["metrics"]

    md = [
        "# Eval Report",
        "",
        f"Dataset size: {len(dataset)}",
        "",
        "## Hardware + Runtime Profile",
        "",
        f"- Python: {report['runtime']['python']}",
        f"- Platform: {report['runtime']['platform']}",
        f"- Chat model: {report['runtime']['chat_model']}",
        f"- Embedding model: {report['runtime']['embedding_model']}",
        "",
        "## Profile Comparison",
        "",
        f"| Metric | {primary_profile} | {secondary_profile} |",
        "|---|---:|---:|",
        f"| Hit@k | {p['hit_at_k']:.3f} | {s['hit_at_k']:.3f} |",
        f"| MRR | {p['mrr']:.3f} | {s['mrr']:.3f} |",
        f"| Citation precision | {p['citation_precision']:.3f} | {s['citation_precision']:.3f} |",
        f"| Support coverage | {p['support_coverage']:.3f} | {s['support_coverage']:.3f} |",
        f"| Abstain precision | {p['abstain_precision']:.3f} | {s['abstain_precision']:.3f} |",
        f"| Abstain recall | {p['abstain_recall']:.3f} | {s['abstain_recall']:.3f} |",
        f"| Latency P50 (ms) | {p['latency_p50_ms']:.1f} | {s['latency_p50_ms']:.1f} |",
        f"| Latency P95 (ms) | {p['latency_p95_ms']:.1f} | {s['latency_p95_ms']:.1f} |",
    ]
    with MD_REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("\n".join(md))

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Agentic RAG evaluation and latency benchmarking")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional dataset row limit for quick validation runs (0 = full dataset)",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["balanced", "low_latency"],
        help="Runtime profiles to evaluate, e.g. balanced low_latency",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable per-query progress logs",
    )
    args = parser.parse_args()
    result = run_eval(limit=args.limit, profiles=args.profiles, show_progress=not args.no_progress)
    print(json.dumps(result["profiles"], indent=2))
