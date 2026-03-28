Phase 1 — EDA & Feature Engineering
- Dataset: 590,540 transactions, 434 raw features
- Fraud rate: 3.5% (severe class imbalance)
- Engineered 12 domain-specific features including transaction 
  velocity, amount deviation from card average, time-based 
  patterns and email domain matching
- Handled class imbalance using SMOTE on training data only
- Key finding: night-time transactions and amounts 5x+ above 
  card average are the strongest fraud indicators



  Phase 2 - Isolation Forest (Unsupervised Anomaly Detection)
- Trained on 472K transactions with no fraud labels
- Contamination tuned to match actual fraud rate (3.5%)
- AUC: ~0.78 | Catches fraud patterns without supervision
- Key insight: highest anomaly scores cluster around 
  night-time transactions with amounts 5x+ above card average
- Anomaly scores saved for ensemble combination with XGBoost


Phase 3 — LSTM Sequential Fraud Detection
- Reshaped 590K transactions into sliding windows 
  of 5 consecutive transactions per sequence
- Input shape: (samples, 5 timesteps, 10 features)
- Architecture: 2-layer LSTM (64→32 units) + Dense layers
- Class weights applied to handle 96.5/3.5 imbalance
- EarlyStopping on validation AUC to prevent overfitting
- Key advantage: catches sequential fraud patterns 
  (card testing, velocity abuse) that XGBoost misses