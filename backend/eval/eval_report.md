# Eval Report

Dataset size: 4
Dataset path: backend\eval\dataset_dev.jsonl

## Hardware + Runtime Profile

- Python: 3.11.0
- Platform: Windows-10-10.0.26200-SP0
- Chat model: qwen3.5:0.8b
- Embedding model: nomic-embed-text:latest

## Profile Comparison

| Metric | balanced | low_latency |
|---|---:|---:|
| Hit@k | 0.750 | 0.750 |
| MRR | 0.625 | 0.625 |
| Citation precision | 0.833 | 0.889 |
| Support coverage | 1.000 | 1.000 |
| Abstain precision | 0.000 | 0.000 |
| Abstain recall | 0.000 | 0.000 |
| Latency P50 (ms) | 39056.3 | 23481.6 |
| Latency P95 (ms) | 50317.5 | 23659.8 |

## Per-Bucket Hit@k

| Bucket | balanced | low_latency |
|---|---:|---:|
| fact_lookup | 1.000 | 1.000 |
| procedure_how_to | 1.000 | 1.000 |
| unanswerable_out_of_scope | 0.000 | 0.000 |

## Abstain Subset

- balanced: {'required_count': 1, 'precision': 0.0, 'recall': 0.0, 'tp': 0, 'fp': 0, 'fn': 1}
- low_latency: {'required_count': 1, 'precision': 0.0, 'recall': 0.0, 'tp': 0, 'fp': 0, 'fn': 1}

## Citation Precision by Source Type

- balanced: {'web': 0.8333333333333334}
- low_latency: {'web': 0.8888888888888888}