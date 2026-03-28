# Eval Report

Dataset size: 35
Dataset path: dataset_dev.jsonl

## Hardware + Runtime Profile

- Python: 3.11.0
- Platform: Windows-10-10.0.26200-SP0
- Chat model: qwen3.5:2b
- Embedding model: nomic-embed-text:latest

## Profile Comparison

| Metric | low_latency | low_latency |
|---|---:|---:|
| Retrieval Hit@k | 0.143 | 0.143 |
| MRR | 0.120 | 0.120 |
| Citation precision | 0.071 | 0.071 |
| Support coverage | 0.143 | 0.143 |
| Abstain precision | 0.000 | 0.000 |
| Abstain recall | 0.000 | 0.000 |
| Adversarial abstain rate | 0.000 | 0.000 |
| Latency P50 (ms) | 70463.1 | 70463.1 |
| Latency P95 (ms) | 71106.3 | 71106.3 |

## Per-Bucket Retrieval Hit@k

| Bucket | low_latency | low_latency |
|---|---:|---:|
| fact_lookup | 0.033 | 0.033 |
| procedure_how_to | 0.800 | 0.800 |

## Abstain Performance Analysis

### Global Metrics

- low_latency: TP=0, FP=0, FN=0 | Prec=0.000 | Recall=0.000
- low_latency: TP=0, FP=0, FN=0 | Prec=0.000 | Recall=0.000

### Per-Bucket Abstain Metrics (BUG #8 fix)

#### LOW_LATENCY

| Bucket | Total | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| fact_lookup | 30 | 0 | 0 | 0 | 0.000 | 0.000 |
| procedure_how_to | 5 | 0 | 0 | 0 | 0.000 | 0.000 |

#### LOW_LATENCY

| Bucket | Total | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| fact_lookup | 30 | 0 | 0 | 0 | 0.000 | 0.000 |
| procedure_how_to | 5 | 0 | 0 | 0 | 0.000 | 0.000 |

## Citation Precision by Source Type

- low_latency: {'web': 1.0, 'pdf': 0.0, 'project_doc': 0.0}
- low_latency: {'web': 1.0, 'pdf': 0.0, 'project_doc': 0.0}

## Citation Scoring Coverage

- low_latency: {'scored_rows': 35, 'skipped_rows': 0, 'manual_review_rows': 0}
- low_latency: {'scored_rows': 35, 'skipped_rows': 0, 'manual_review_rows': 0}

## Top False-Abstain Reasons

- low_latency: []
- low_latency: []