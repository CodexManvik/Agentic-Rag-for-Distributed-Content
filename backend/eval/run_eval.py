import json
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.graph.workflow import run_workflow


DATASET_PATH = Path(__file__).parent / "dataset.jsonl"
JSON_REPORT_PATH = Path(__file__).parent / "eval_report.json"
MD_REPORT_PATH = Path(__file__).parent / "eval_report.md"


def _load_dataset() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
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


def run_eval() -> dict[str, Any]:
    dataset = _load_dataset()

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

    rows: list[dict[str, Any]] = []

    for item in dataset:
        result = run_workflow(item["query"])
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

        rows.append(
            {
                "query": item["query"],
                "answerable": answerable,
                "abstained": abstained,
                "hit": hit,
                "rr": rr,
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

    report = {
        "dataset_size": total,
        "metrics": {
            "hit_at_k": hit_at_k,
            "mrr": mrr,
            "citation_precision": citation_precision,
            "support_coverage": support_coverage,
            "abstain_precision": abstain_precision,
            "abstain_recall": abstain_recall,
        },
        "rows": rows,
    }

    with JSON_REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md = [
        "# Eval Report",
        "",
        f"Dataset size: {total}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Hit@k | {hit_at_k:.3f} |",
        f"| MRR | {mrr:.3f} |",
        f"| Citation precision | {citation_precision:.3f} |",
        f"| Support coverage | {support_coverage:.3f} |",
        f"| Abstain precision | {abstain_precision:.3f} |",
        f"| Abstain recall | {abstain_recall:.3f} |",
    ]
    with MD_REPORT_PATH.open("w", encoding="utf-8") as f:
        f.write("\n".join(md))

    return report


if __name__ == "__main__":
    result = run_eval()
    print(json.dumps(result["metrics"], indent=2))
