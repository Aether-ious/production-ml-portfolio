# Production Fraud Detection System

Real-time fraud detection API with data drift monitoring and Prometheus metrics.

## Problem
Detect fraudulent transactions in heavily imbalanced data (0.3% fraud rate). Must achieve ≥0.95 AUC and high precision at 90% recall (business requirement).

## Results
- AUC: 0.967 (on hold-out test)
- Precision @ 90% recall: 0.42 (5× better than random)
- API latency: 12ms p99
- Data drift detected on TransactionAmt & card features after simulating 30-day shift

