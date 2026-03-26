import argparse
import json
import random
from pathlib import Path
from typing import Any


def _normalize_row(row: dict[str, Any], idx: int) -> dict[str, Any]:
    query = str(row.get("query", "")).strip()
    should_abstain = bool(row.get("should_abstain", False))
    if "answerable" in row:
        should_abstain = not bool(row.get("answerable", True)) if not should_abstain else should_abstain

    must_cite_sources = row.get("must_cite_sources", row.get("expected_sources", []))
    must_cite_sources = [str(s).lower() for s in must_cite_sources if isinstance(s, str)]

    bucket = str(row.get("bucket", "")).strip().lower()
    if not bucket:
        if should_abstain:
            bucket = "unanswerable_out_of_scope"
        elif "compare" in query.lower():
            bucket = "comparison_questions"
        elif "how" in query.lower() or "steps" in query.lower():
            bucket = "procedure_how_to"
        else:
            bucket = "fact_lookup"

    difficulty = str(row.get("difficulty", "medium")).strip().lower()
    if difficulty not in {"easy", "medium", "hard"}:
        difficulty = "medium"

    return {
        "id": str(row.get("id", f"row-{idx+1:04d}")),
        "query": query,
        "expected_answer": str(row.get("expected_answer", "")),
        "must_cite_sources": must_cite_sources,
        "difficulty": difficulty,
        "requires_multi_hop": bool(row.get("requires_multi_hop", bucket in {"multi_hop_synthesis", "comparison_questions"})),
        "should_abstain": should_abstain,
        "reason_if_abstain": str(row.get("reason_if_abstain", "missing_or_out_of_scope" if should_abstain else "")),
        "tags": [str(t).lower() for t in row.get("tags", []) if isinstance(t, str)],
        "bucket": bucket,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rows.append(_normalize_row(json.loads(line), idx))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dev/hidden splits with strengthened schema")
    parser.add_argument("--input", default="backend/eval/dataset.jsonl", help="Input dataset JSONL")
    parser.add_argument("--dev-output", default="backend/eval/dataset_dev.jsonl", help="Dev split output JSONL")
    parser.add_argument("--hidden-output", default="backend/eval/dataset_hidden.jsonl", help="Hidden split output JSONL")
    parser.add_argument("--dev-ratio", type=float, default=0.7, help="Dev split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    args = parser.parse_args()

    rows = _load_jsonl(Path(args.input))
    random.Random(args.seed).shuffle(rows)

    cut = int(len(rows) * args.dev_ratio)
    dev_rows = rows[:cut]
    hidden_rows = rows[cut:]

    _write_jsonl(Path(args.dev_output), dev_rows)
    _write_jsonl(Path(args.hidden_output), hidden_rows)

    print(f"Prepared splits: dev={len(dev_rows)}, hidden={len(hidden_rows)}")


if __name__ == "__main__":
    main()
