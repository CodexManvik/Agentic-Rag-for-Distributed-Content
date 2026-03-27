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
| Hit@k | 0.733 | 0.725 |
| MRR | 0.690 | 0.675 |
| Citation precision | 0.775 | 0.763 |
| Support coverage | 0.642 | 0.635 |
| Abstain precision | 1.000 | 0.297 |
| Abstain recall | 0.967 | 1.000 |
| Latency P50 (ms) | 69101.9 | 28120.2 |
| Latency P95 (ms) | 80571.0 | 28773.6 |

## Per-Bucket Hit@k

| Bucket | balanced | low_latency |
|---|---:|---:|
| adversarial_noisy | 0.000 | 0.000 |
| comparison_questions | 1.000 | 1.000 |
| edge_ambiguity | 1.000 | 1.000 |
| fact_lookup | 0.950 | 0.900 |
| multi_hop_synthesis | 1.000 | 1.000 |
| procedure_how_to | 0.933 | 0.933 |
| unanswerable_out_of_scope | 0.000 | 0.000 |

## Abstain Subset

- balanced: {'required_count': 30, 'precision': 1.0, 'recall': 0.9666666666666667, 'tp': 29, 'fp': 0, 'fn': 1}
- low_latency: {'required_count': 30, 'precision': 0.297029702970297, 'recall': 1.0, 'tp': 30, 'fp': 71, 'fn': 0}

## Citation Precision by Source Type

- balanced: {'web': 0.775}
- low_latency: {'web': 0.762962962962963}