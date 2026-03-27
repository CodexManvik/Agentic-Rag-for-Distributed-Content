# Eval Report

Dataset size: 5
Dataset path: backend\eval\dataset_dev.jsonl

## Hardware + Runtime Profile

- Python: 3.11.0
- Platform: Windows-10-10.0.26200-SP0
- Chat model: qwen3.5:0.8b
- Embedding model: nomic-embed-text:latest

## Profile Comparison

| Metric | balanced | low_latency |
|---|---:|---:|
| Hit@k | 0.400 | 0.400 |
| MRR | 0.200 | 0.200 |
| Citation precision | 0.062 | 0.083 |
| Support coverage | 0.250 | 0.250 |
| Abstain precision | 1.000 | 1.000 |
| Abstain recall | 1.000 | 1.000 |
| Adversarial abstain rate | 0.000 | 0.000 |
| Latency P50 (ms) | 45320.8 | 22431.6 |
| Latency P95 (ms) | 50844.8 | 27219.9 |

## Per-Bucket Hit@k

| Bucket | balanced | low_latency |
|---|---:|---:|
| fact_lookup | 0.500 | 0.500 |
| procedure_how_to | 0.000 | 0.000 |
| unanswerable_out_of_scope | n/a | n/a |

## Abstain Subset

- balanced: {'required_count': 1, 'precision': 1.0, 'recall': 1.0, 'tp': 1, 'fp': 0, 'fn': 0}
- low_latency: {'required_count': 1, 'precision': 1.0, 'recall': 1.0, 'tp': 1, 'fp': 0, 'fn': 0}

## Citation Precision by Source Type

- balanced: {'pdf': 0.0, 'web': 1.0}
- low_latency: {'pdf': 0.0, 'web': 1.0}