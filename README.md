Phase 1 — EDA & Feature Engineering
- Dataset: 590,540 transactions, 434 raw features
- Fraud rate: 3.5% (severe class imbalance)
- Engineered 12 domain-specific features including transaction 
  velocity, amount deviation from card average, time-based 
  patterns and email domain matching
- Handled class imbalance using SMOTE on training data only
- Key finding: night-time transactions and amounts 5x+ above 
  card average are the strongest fraud indicators