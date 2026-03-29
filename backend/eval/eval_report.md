# Eval Report

Dataset size: 40
Dataset path: dataset_dev.jsonl

## Hardware + Runtime Profile

- Python: 3.11.0
- Platform: Windows-10-10.0.26200-SP0
- Chat model: llama3.2:3b
- Embedding model: nomic-embed-text:latest

## Profile Comparison

| Metric | balanced | low_latency |
|---|---:|---:|
| Retrieval Hit@k | 0.483 | 0.690 |
| MRR | 0.359 | 0.549 |
| Citation precision | 0.167 | 0.176 |
| Support coverage | 0.136 | 0.295 |
| Abstain precision | 0.625 | 0.778 |
| Abstain recall | 0.909 | 0.636 |
| Adversarial abstain rate | 1.000 | 1.000 |
| Latency P50 (ms) | 32487.2 | 28845.6 |
| Latency P95 (ms) | 50313.9 | 35680.1 |

## Per-Bucket Retrieval Hit@k

| Bucket | balanced | low_latency |
|---|---:|---:|
| adversarial_noisy | n/a | n/a |
| comparison_questions | 0.667 | 0.667 |
| fact_lookup | 0.444 | 0.722 |
| multi_hop_synthesis | 0.667 | 0.667 |
| procedure_how_to | 0.400 | 0.600 |
| unanswerable_out_of_scope | 0.000 | 0.000 |

## Abstain Performance Analysis

### Global Metrics

- balanced: TP=10, FP=6, FN=1 | Prec=0.625 | Recall=0.909
- low_latency: TP=7, FP=2, FN=4 | Prec=0.778 | Recall=0.636

### Per-Bucket Abstain Metrics (BUG #8 fix)

#### BALANCED

| Bucket | Total | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| adversarial_noisy | 3 | 3 | 0 | 0 | 1.000 | 1.000 |
| comparison_questions | 3 | 0 | 1 | 0 | 0.000 | 0.000 |
| fact_lookup | 18 | 0 | 4 | 0 | 0.000 | 0.000 |
| multi_hop_synthesis | 3 | 0 | 0 | 0 | 0.000 | 0.000 |
| procedure_how_to | 5 | 0 | 1 | 0 | 0.000 | 0.000 |
| unanswerable_out_of_scope | 8 | 7 | 0 | 1 | 1.000 | 0.875 |

#### LOW_LATENCY

| Bucket | Total | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| adversarial_noisy | 3 | 3 | 0 | 0 | 1.000 | 1.000 |
| comparison_questions | 3 | 0 | 0 | 0 | 0.000 | 0.000 |
| fact_lookup | 18 | 0 | 2 | 0 | 0.000 | 0.000 |
| multi_hop_synthesis | 3 | 0 | 0 | 0 | 0.000 | 0.000 |
| procedure_how_to | 5 | 0 | 0 | 0 | 0.000 | 0.000 |
| unanswerable_out_of_scope | 8 | 4 | 0 | 4 | 1.000 | 0.500 |

## Citation Precision by Source Type

- balanced: {'pdf': 0.0, 'project_doc': 0.0, 'web': 0.8181818181818182, 'confluence': 1.0}
- low_latency: {'pdf': 0.0, 'project_doc': 0.0, 'web': 0.6111111111111112, 'confluence': 1.0}

## Citation Scoring Coverage

- balanced: {'scored_rows': 23, 'skipped_rows': 17, 'manual_review_rows': 0}
- low_latency: {'scored_rows': 27, 'skipped_rows': 13, 'manual_review_rows': 0}

## Top False-Abstain Reasons

- balanced: [{'reason': 'Evidence is insufficient or unverifiable', 'count': 5}, {'reason': 'Citation validation failed', 'count': 1}]
- low_latency: [{'reason': 'Citation validation failed', 'count': 2}]