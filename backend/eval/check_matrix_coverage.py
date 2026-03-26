import argparse
import json
from collections import Counter
from pathlib import Path


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Check dataset bucket coverage against target matrix")
    parser.add_argument("--dataset", default="backend/eval/dataset_dev.jsonl", help="Dataset JSONL")
    parser.add_argument("--target", default="backend/eval/eval_matrix_target.json", help="Target matrix JSON")
    args = parser.parse_args()

    rows = _load_rows(Path(args.dataset))
    target = json.loads(Path(args.target).read_text(encoding="utf-8"))

    bucket_counter = Counter(str(r.get("bucket", "unknown")) for r in rows)
    abstain_count = sum(1 for r in rows if bool(r.get("should_abstain", False)))
    abstain_ratio = (abstain_count / len(rows)) if rows else 0.0

    print(f"Dataset rows: {len(rows)}")
    print(f"Abstain-required: {abstain_count} ({abstain_ratio:.2%})")
    print("Bucket coverage:")

    for bucket, required in target.get("buckets", {}).items():
        current = bucket_counter.get(bucket, 0)
        delta = required - current
        status = "OK" if delta <= 0 else f"MISSING {delta}"
        print(f"- {bucket}: current={current}, target={required} -> {status}")

    min_ratio = float(target.get("abstain_required_min_ratio", 0.15))
    if abstain_ratio < min_ratio:
        print(f"Abstain ratio below target: {abstain_ratio:.2%} < {min_ratio:.2%}")
    else:
        print(f"Abstain ratio meets target: {abstain_ratio:.2%}")


if __name__ == "__main__":
    main()
