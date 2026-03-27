# Eval Report

Dataset size: 120
Dataset path: backend\eval\dataset_dev.jsonl

## Hardware + Runtime Profile

- Python: 3.11.0
- Platform: Windows-10-10.0.26200-SP0
- Chat model: qwen3.5:0.8b
- Embedding model: nomic-embed-text:latest

## Profile Comparison

| Metric | balanced | low_latency |
|---|---:|---:|
| Retrieval Hit@k | 0.383 | 0.417 |
| MRR | 0.115 | 0.129 |
| Citation precision | 0.309 | 0.177 |
| Support coverage | 0.117 | 0.146 |
| Abstain precision | 0.345 | 0.580 |
| Abstain recall | 1.000 | 0.967 |
| Adversarial abstain rate | 1.000 | 1.000 |
| Latency P50 (ms) | 40276.7 | 40383.7 |
| Latency P95 (ms) | 75953.6 | 49155.1 |

## Per-Bucket Retrieval Hit@k

| Bucket | balanced | low_latency |
|---|---:|---:|
| adversarial_noisy | n/a | n/a |
| comparison_questions | 0.000 | 0.000 |
| edge_ambiguity | 0.000 | 0.000 |
| fact_lookup | 0.350 | 0.500 |
| multi_hop_synthesis | 0.000 | 0.000 |
| procedure_how_to | 0.600 | 0.667 |
| unanswerable_out_of_scope | n/a | 1.000 |

## Abstain Subset

- balanced: {'required_count': 30, 'precision': 0.3448275862068966, 'recall': 1.0, 'tp': 30, 'fp': 57, 'fn': 0}
- low_latency: {'required_count': 30, 'precision': 0.58, 'recall': 0.9666666666666667, 'tp': 29, 'fp': 21, 'fn': 1}

## Citation Precision by Source Type

- balanced: {'project_doc': 0.0, 'pdf': 0.0, 'web': 0.7435897435897436, 'confluence': 0.0}
- low_latency: {'project_doc': 0.0, 'pdf': 0.0, 'web': 0.6507936507936508, 'confluence': 0.0}

## Citation Scoring Coverage

- balanced: {'scored_rows': 33, 'skipped_rows': 87, 'manual_review_rows': 0}
- low_latency: {'scored_rows': 69, 'skipped_rows': 51, 'manual_review_rows': 0}

## Top False-Abstain Reasons

- balanced: [{'reason': 'Evidence is insufficient or unverifiable', 'count': 33}, {'reason': 'Synthesis parse failure after strict retry', 'count': 24}]
- low_latency: [{'reason': 'Evidence is insufficient or unverifiable', 'count': 21}]