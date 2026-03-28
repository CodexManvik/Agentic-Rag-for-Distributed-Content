import json
from pathlib import Path

# Configuration
SOURCE_FILE = Path('dataset_dev.jsonl')
OUTPUT_DIR = Path('backend/eval')
OUTPUT_FILE = OUTPUT_DIR / 'dataset_smoke.jsonl'

TARGETS = {
    'fact_lookup': 3,
    'procedure_how_to': 3,
    'unanswerable_out_of_scope': 3,
    'adversarial_noisy': 3,
    'comparison_questions': 3,
    'multi_hop_synthesis': 3,
    'edge_ambiguity': 2,
}

def generate_smoke_test():
    if not SOURCE_FILE.exists():
        print(f"Error: Source file {SOURCE_FILE} not found.")
        return

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load and parse lines
    lines = SOURCE_FILE.read_text().splitlines()
    rows = [json.loads(l) for l in lines if l.strip()]

    # Group by bucket while tracking original indices
    by_bucket = {}
    for i, r in enumerate(rows):
        bucket_name = r.get('bucket', '?')
        by_bucket.setdefault(bucket_name, []).append((i, r))

    # Sample based on targets
    selected_indices = []
    print(f"{'INDEX':<10} | {'BUCKET':<28} | {'ABSTAIN':<8} | {'QUERY SNIPPET'}")
    print("-" * 80)

    for bucket, limit in TARGETS.items():
        candidates = by_bucket.get(bucket, [])[:limit]
        for idx, row in candidates:
            selected_indices.append(idx)
            
            # Print status for visibility
            abstain = row.get('should_abstain', False)
            query = row.get('query', '')[:60].replace('\n', ' ')
            print(f"row {idx:7d} | {bucket:<28} | {str(abstain):<8} | {query}")

    # Sort to maintain original dataset order
    selected_indices.sort()
    smoke_dataset = [rows[i] for i in selected_indices]

    # Write output
    output_content = '\n'.join(json.dumps(r) for r in smoke_dataset)
    OUTPUT_FILE.write_text(output_content)

    print(f"\nSmoke test set: {len(smoke_dataset)} rows")
    print(f"Wrote to: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_smoke_test()