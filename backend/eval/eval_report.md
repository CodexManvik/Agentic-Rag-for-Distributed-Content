# Eval Report

Dataset size: 20

## Hardware + Runtime Profile

- Python: 3.11.0
- Platform: Windows-10-10.0.26200-SP0
- Chat model: qwen3.5:0.8b
- Embedding model: nomic-embed-text:latest

## Profile Comparison

| Metric | balanced | low_latency |
|---|---:|---:|
| Hit@k | 0.700 | 0.700 |
| MRR | 0.650 | 0.650 |
| Citation precision | 0.850 | 0.833 |
| Support coverage | 0.824 | 0.824 |
| Abstain precision | 1.000 | 0.500 |
| Abstain recall | 0.200 | 0.200 |
| Latency P50 (ms) | 35814.4 | 21554.9 |
| Latency P95 (ms) | 44514.4 | 24369.6 |