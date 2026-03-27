import argparse
import json
import os
from pathlib import Path
import platform
import time
from collections import defaultdict
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


def _load_dataset(dataset_path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(_normalize_row(json.loads(line), len(rows)))
    if limit > 0:
        return rows[:limit]
    return rows


def _normalize_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    query = str(row.get("query", "")).strip()
    should_abstain = bool(row.get("should_abstain", False))
    if "answerable" in row:
        answerable = bool(row.get("answerable", True))
        should_abstain = not answerable if not should_abstain else should_abstain
    else:
        answerable = not should_abstain

    must_cite = row.get("must_cite_sources", row.get("expected_sources", []))
    must_cite_sources = [str(s).lower() for s in must_cite if isinstance(s, str)]

    tags = row.get("tags", [])
    tags_clean = [str(t).lower() for t in tags if isinstance(t, str)]
    bucket = str(row.get("bucket", "")).strip().lower()
    if not bucket:
        if should_abstain:
            bucket = "unanswerable_out_of_scope"
        elif any("compare" in query.lower() for _ in [0]):
            bucket = "comparison_questions"
        elif any(word in query.lower() for word in ["how", "steps", "procedure"]):
            bucket = "procedure_how_to"
        else:
            bucket = "fact_lookup"

    difficulty = str(row.get("difficulty", "medium")).strip().lower()
    if difficulty not in {"easy", "medium", "hard"}:
        difficulty = "medium"

    requires_multi_hop = bool(row.get("requires_multi_hop", bucket in {"multi_hop_synthesis", "comparison_questions"}))

    return {
        "id": str(row.get("id", f"sample-{index+1:04d}")),
        "query": query,
        "expected_answer": str(row.get("expected_answer", "")),
        "must_cite_sources": must_cite_sources,
        "difficulty": difficulty,
        "requires_multi_hop": requires_multi_hop,
        "should_abstain": should_abstain,
        "reason_if_abstain": str(row.get("reason_if_abstain", "")),
        "tags": tags_clean,
        "bucket": bucket,
        "answerable": answerable,
    }


def _source_hit(expected: list[str], retrieved_sources: list[str]) -> tuple[bool, float]:
    if not expected:
        return True, 0.0
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
        bucket_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "hits": 0, "eligible": 0})
        difficulty_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "hits": 0})
        citation_type_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0})
        latencies_ms: list[float] = []
        validation_error_categories: dict[str, int] = {}
        adversarial_total = 0
        adversarial_correct_abstain = 0

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
            expected_sources = [s.lower() for s in item.get("must_cite_sources", [])]
            hit, rr = _source_hit(expected_sources, retrieved_sources)

            if hit:
                hit_count += 1
            reciprocal_rank_sum += rr

            for citation in citations:
                source = str(citation.get("source", "")).lower()
                source_type = str(citation.get("source_type", "unknown") or "unknown").lower()
                if expected_sources and any(exp in source for exp in expected_sources):
                    citation_tp += 1
                    citation_type_counts[source_type]["tp"] += 1
                elif expected_sources:
                    citation_fp += 1
                    citation_type_counts[source_type]["fp"] += 1

            if expected_sources:
                support_total += len(expected_sources)
                for exp in expected_sources:
                    if any(exp in src.lower() for src in retrieved_sources):
                        support_covered += 1

            answerable = not bool(item.get("should_abstain", False))
            if not answerable and abstained:
                abstain_tp += 1
            if answerable and abstained:
                abstain_fp += 1
            if (not answerable) and (not abstained):
                abstain_fn += 1

            bucket = str(item.get("bucket", "unknown"))
            difficulty = str(item.get("difficulty", "unknown"))
            bucket_counts[bucket]["total"] += 1
            difficulty_counts[difficulty]["total"] += 1
            include_for_bucket_hit = not (bool(item.get("should_abstain", False)) and abstained)
            if include_for_bucket_hit:
                bucket_counts[bucket]["eligible"] += 1
                if hit:
                    bucket_counts[bucket]["hits"] += 1
            if hit:
                difficulty_counts[difficulty]["hits"] += 1

            if "adversarial" in bucket.lower():
                adversarial_total += 1
                if bool(item.get("should_abstain", False)) and abstained:
                    adversarial_correct_abstain += 1

            for err in result.get("validation_errors", []):
                category = str(err).split(":", 1)[0].strip().lower()
                if category:
                    validation_error_categories[category] = validation_error_categories.get(category, 0) + 1

            rows.append(
                {
                    "query": item["query"],
                    "id": item["id"],
                    "answerable": answerable,
                    "abstained": abstained,
                    "bucket": bucket,
                    "difficulty": difficulty,
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
        adversarial_abstain_rate = (
            adversarial_correct_abstain / adversarial_total if adversarial_total else 0.0
        )

        per_bucket_hit_at_k: dict[str, float | None] = {}
        for bucket, values in bucket_counts.items():
            eligible_bucket = values["eligible"]
            per_bucket_hit_at_k[bucket] = (values["hits"] / eligible_bucket) if eligible_bucket else None

        per_difficulty_hit_at_k: dict[str, float] = {}
        for difficulty, values in difficulty_counts.items():
            total_difficulty = values["total"]
            per_difficulty_hit_at_k[difficulty] = (values["hits"] / total_difficulty) if total_difficulty else 0.0

        citation_precision_by_source_type: dict[str, float] = {}
        for source_type, values in citation_type_counts.items():
            denom = values["tp"] + values["fp"]
            citation_precision_by_source_type[source_type] = (values["tp"] / denom) if denom else 0.0

        abstain_total = sum(1 for item in dataset if bool(item.get("should_abstain", False)))
        abstain_subset = {
            "required_count": abstain_total,
            "precision": abstain_precision,
            "recall": abstain_recall,
            "tp": abstain_tp,
            "fp": abstain_fp,
            "fn": abstain_fn,
        }

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
                "adversarial_abstain_rate": adversarial_abstain_rate,
                "latency_p50_ms": p50_ms,
                "latency_p95_ms": p95_ms,
            },
            "per_bucket": {
                "hit_at_k": per_bucket_hit_at_k,
            },
            "per_difficulty": {
                "hit_at_k": per_difficulty_hit_at_k,
            },
            "citation_precision_by_source_type": citation_precision_by_source_type,
            "abstain_subset": abstain_subset,
            "adversarial_abstain": {
                "total": adversarial_total,
                "correct_abstain": adversarial_correct_abstain,
                "rate": adversarial_abstain_rate,
            },
            "validation_error_categories": validation_error_categories,
            "rows": rows,
        }
    finally:
        settings.runtime_profile = prev_profile


def run_eval(
    limit: int = 0,
    profiles: list[str] | None = None,
    show_progress: bool = True,
    dataset_path: str | None = None,
) -> dict[str, Any]:
    selected_dataset = Path(dataset_path) if dataset_path else DATASET_PATH
    dataset = _load_dataset(selected_dataset, limit=limit)
    selected_profiles = profiles or ["balanced", "low_latency"]

    profile_results: dict[str, Any] = {}
    for profile in selected_profiles:
        profile_results[profile] = _run_profile_eval(dataset, profile=profile, show_progress=show_progress)

    report = {
        "dataset_size": len(dataset),
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dataset_path": str(selected_dataset),
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
        f"Dataset path: {selected_dataset}",
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
        f"| Adversarial abstain rate | {p['adversarial_abstain_rate']:.3f} | {s['adversarial_abstain_rate']:.3f} |",
        f"| Latency P50 (ms) | {p['latency_p50_ms']:.1f} | {s['latency_p50_ms']:.1f} |",
        f"| Latency P95 (ms) | {p['latency_p95_ms']:.1f} | {s['latency_p95_ms']:.1f} |",
        "",
        "## Per-Bucket Hit@k",
    ]

    bucket_keys = sorted(
        set(report["profiles"][primary_profile].get("per_bucket", {}).get("hit_at_k", {}).keys())
        | set(report["profiles"][secondary_profile].get("per_bucket", {}).get("hit_at_k", {}).keys())
    )
    if bucket_keys:
        md.extend([
            "",
            f"| Bucket | {primary_profile} | {secondary_profile} |",
            "|---|---:|---:|",
        ])
        for key in bucket_keys:
            p_val = report["profiles"][primary_profile].get("per_bucket", {}).get("hit_at_k", {}).get(key, 0.0)
            s_val = report["profiles"][secondary_profile].get("per_bucket", {}).get("hit_at_k", {}).get(key, 0.0)
            p_display = f"{p_val:.3f}" if isinstance(p_val, (float, int)) else "n/a"
            s_display = f"{s_val:.3f}" if isinstance(s_val, (float, int)) else "n/a"
            md.append(f"| {key} | {p_display} | {s_display} |")

    md.extend([
        "",
        "## Abstain Subset",
        "",
        f"- {primary_profile}: {report['profiles'][primary_profile].get('abstain_subset', {})}",
        f"- {secondary_profile}: {report['profiles'][secondary_profile].get('abstain_subset', {})}",
        "",
        "## Citation Precision by Source Type",
        "",
        f"- {primary_profile}: {report['profiles'][primary_profile].get('citation_precision_by_source_type', {})}",
        f"- {secondary_profile}: {report['profiles'][secondary_profile].get('citation_precision_by_source_type', {})}",
    ])

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
    parser.add_argument(
        "--dataset",
        default=str(DATASET_PATH),
        help="Path to dataset JSONL (supports strengthened schema)",
    )
    args = parser.parse_args()
    result = run_eval(
        limit=args.limit,
        profiles=args.profiles,
        show_progress=not args.no_progress,
        dataset_path=args.dataset,
    )
    print(json.dumps(result["profiles"], indent=2))
