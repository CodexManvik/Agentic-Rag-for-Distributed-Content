import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import chromadb


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "item"


def _bucket_for_question(question: str) -> str:
    q = question.lower()
    if any(word in q for word in ["compare", "difference", "versus", "vs"]):
        return "comparison_questions"
    if any(word in q for word in ["how", "steps", "procedure", "setup"]):
        return "procedure_how_to"
    return "fact_lookup"


def _question_variants(title: str, section: str) -> list[str]:
    base = section or title or "this section"
    return [
        f"What does {base} explain?",
        f"How is {base} described in the documentation?",
        f"Summarize the key points of {base} with citations.",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate candidate eval rows from ingested chunk metadata")
    parser.add_argument("--chroma-dir", default="./chroma_data", help="Chroma persist directory")
    parser.add_argument("--collection", default="knowledge_base", help="Chroma collection name")
    parser.add_argument("--output", default="backend/eval/candidate_dataset.jsonl", help="Output JSONL")
    parser.add_argument("--max-sections", type=int, default=80, help="Max unique sections to sample")
    args = parser.parse_args()

    client = chromadb.PersistentClient(path=args.chroma_dir)
    collection = client.get_collection(args.collection)
    payload = collection.get(include=cast(Any, ["metadatas"]))
    metadatas = payload.get("metadatas") or []

    grouped: dict[str, dict[str, Any]] = defaultdict(dict)
    for item in metadatas:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("source") or "unknown")
        section = str(item.get("section") or "overview")
        source_type = str(item.get("source_type") or "web")
        url = str(item.get("url") or item.get("path") or "")
        key = f"{title}::{section}"
        if key not in grouped:
            grouped[key] = {
                "title": title,
                "section": section,
                "source_type": source_type,
                "url": url,
            }

    rows: list[dict[str, Any]] = []
    section_items = list(grouped.values())[: args.max_sections]
    for idx, sec in enumerate(section_items, start=1):
        title = sec["title"]
        section = sec["section"]
        source_type = sec["source_type"]
        must_cite = [title.lower()]
        for variant_idx, q in enumerate(_question_variants(title, section), start=1):
            bucket = _bucket_for_question(q)
            row = {
                "id": f"cand-{idx:04d}-{variant_idx}",
                "query": q,
                "expected_answer": "",
                "must_cite_sources": must_cite,
                "difficulty": "medium",
                "requires_multi_hop": bucket == "comparison_questions",
                "should_abstain": False,
                "reason_if_abstain": "",
                "tags": [source_type, bucket, _slug(section)],
                "bucket": bucket,
                "manual_review_required": True,
                "review_notes": "Validate expected_answer and must_cite_sources before using as gold eval row.",
            }
            rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Generated {len(rows)} candidate rows -> {output}")


if __name__ == "__main__":
    main()
