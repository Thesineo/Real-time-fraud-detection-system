# Real-time Financial Fraud Detection System
### End-to-end ML pipeline — Anomaly Detection · Deep Learning · LLM Explainability · Live Dashboard

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-red?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple?style=flat-square)
![OpenAI](https://img.shields.io/badge/GPT--4o--mini-LLM-412991?style=flat-square&logo=openai)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=flat-square&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

---

## Live Demo

**[View the live fraud detection dashboard →](https://real-time-fraud-detection-system-3cm2aesobqngve365zm5dh.streamlit.app/)**

---

## Overview

This project builds a production-grade fraud detection pipeline on 590,540 real financial transactions from the IEEE-CIS Fraud Detection dataset — the same dataset used in a real Kaggle competition sponsored by Vesta Corporation, a global payments company.

The system combines three complementary detection approaches — unsupervised anomaly detection, supervised classification, and sequential deep learning — then uses SHAP and GPT-4o-mini to auto-generate plain English fraud alert reports that a non-technical analyst can act on immediately.

---

## The Business Problem

Financial fraud costs the global economy over **$32 billion annually**. Traditional rule-based systems miss sophisticated fraud patterns and flag too many legitimate transactions, damaging customer experience. This project demonstrates how a layered ML approach catches fraud that single models miss — and explains every decision in plain English so fraud analysts can act without needing to understand the underlying models.

**Key questions this system answers:**
- Which transactions are anomalous compared to normal behaviour patterns?
- Which specific customers are at highest risk right now?
- Why did the model flag this transaction — in plain English?
- What sequential transaction patterns indicate card testing or account takeover?

---

## Project Structure

```
real-time-fraud-detection-system/
├── notebooks/
│   ├── 01_eda_features.ipynb        # EDA and feature engineering
│   ├── 02_classical_ml.ipynb        # Isolation Forest + XGBoost
│   ├── 03_deep_learning.ipynb       # LSTM sequential model (Google Colab)
│   └── 04_llm_explanations.ipynb    # SHAP + GPT fraud reports
├── app/
│   └── dashboard.py                 # Streamlit live dashboard
├── models/
│   ├── shap_summary.png             # SHAP feature importance plot
│   └── shap_waterfall.png           # Per-transaction SHAP waterfall
├── requirements.txt
└── README.md
```

---

## Dataset

**IEEE-CIS Fraud Detection** — Kaggle Competition Dataset (Vesta Corporation)

- 590,540 real anonymised transactions
- 434 raw features across transaction and identity tables
- Fraud rate: **3.5%** (severe class imbalance)
- Merged from two tables: `train_transaction.csv` + `train_identity.csv`

[Download from Kaggle →](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

Place files in the `data/` folder:
```
data/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
└── test_identity.csv
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data processing & EDA | Python, pandas, numpy, matplotlib, seaborn |
| Anomaly detection | Isolation Forest (unsupervised) |
| Classification | XGBoost, LightGBM |
| Deep learning | TensorFlow/Keras, LSTM, Autoencoder |
| Explainability | SHAP (TreeExplainer, waterfall plots) |
| LLM integration | OpenAI GPT-4o-mini |
| Dashboard | Streamlit, Plotly |
| Environment | Google Colab (deep learning), VS Code (ML + EDA) |
| Version control | Git, GitHub |

---

## Project Phases

### Phase 1 — EDA & Feature Engineering
- Merged transaction and identity tables into 590,540 × 434 feature matrix
- Engineered 12 domain-specific fraud signals:
  - `amt_vs_card_avg` — how much this transaction deviates from card's historical average
  - `isNightTime` — transactions between 10pm–5am flagged
  - `card1_transaction_count` — velocity of card activity
  - `P_email_domain_match` — billing vs shipping email mismatch
  - `TransactionAmt_decimal` — round amounts as fraud signal
- Dropped columns with >50% missing values
- Handled class imbalance using `class_weight` (avoids memory issues of SMOTE at scale)

**Key finding:** Fraudulent transactions are on average **4.2x the card's normal spend** and cluster heavily between 11pm–3am.

---

### Phase 2 — Classical ML: Anomaly Detection + Classification

**Isolation Forest (Unsupervised)**
- Trained on 472K transactions with zero fraud labels
- Contamination tuned to match actual fraud rate (3.5%)
- Catches fraud patterns without supervision — powerful for detecting novel fraud types
- AUC: **0.782**

**XGBoost Classifier (Supervised)**
- `scale_pos_weight` set to 27.6 to handle class imbalance
- Hyperparameter tuning: depth, learning rate, subsample
- SHAP values computed for every prediction
- AUC: **0.924**

**Model comparison:**

| Model | AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Isolation Forest | 0.782 | 0.41 | 0.68 | 0.51 |
| XGBoost | 0.924 | 0.87 | 0.82 | 0.84 |
| LSTM | 0.891 | 0.79 | 0.85 | 0.82 |

---

### Phase 3 — Deep Learning: LSTM Sequential Fraud Detection
*(Trained on Google Colab with T4 GPU)*

The key insight: fraud often follows a **sequence pattern** that single-transaction models miss. Card testing involves many small transactions before one large fraudulent one. Account takeover shows sudden category changes. A standard XGBoost model sees each transaction in isolation — an LSTM sees the last 5 transactions together.

- Reshaped 590K transactions into sliding windows of **5 consecutive transactions**
- Input shape: `(samples, 5 timesteps, 10 features)`
- Architecture: 2-layer LSTM (64→32 units) + BatchNormalization + Dropout + Dense
- Class weights applied to handle 96.5/3.5 imbalance
- EarlyStopping on validation AUC — prevents overfitting
- AUC: **0.891**

**What the LSTM catches that XGBoost misses:**
- Card testing: $5 → $8 → $12 → $850 (velocity escalation)
- Geographic impossibility: transactions in two countries 30 minutes apart
- Merchant category switching: groceries → electronics → international wire

---

### Phase 4 — LLM Integration: Plain English Fraud Explanations

The most unique phase of this project. After flagging transactions, a fraud analyst still needs to understand *why* the model flagged it before taking action. This phase translates SHAP values into readable fraud reports using GPT-4o-mini.

**Pipeline:**
1. SHAP `TreeExplainer` computes feature contributions for every flagged transaction
2. Top 5 SHAP drivers extracted per transaction with direction and magnitude
3. Structured prompt built combining transaction details + SHAP reasons
4. GPT-4o-mini generates a 3-section analyst report: Summary → Risk Factors → Action

**Sample auto-generated fraud alert:**

> **SUMMARY:** This transaction has been flagged as HIGH risk with a fraud probability of 91.3%.
>
> **KEY RISK FACTORS:**
> - The transaction amount of $2,847 is 8.4x higher than this card's average spend of $338
> - The purchase occurred at 2:14 AM, outside the cardholder's normal usage hours
> - The billing and shipping email domains do not match — a common indicator of account compromise
> - This card has made 14 transactions in the last 6 hours, far above its normal daily average of 2
>
> **RECOMMENDED ACTION:** Place an immediate hold on this transaction and contact the cardholder directly for verification before processing.

---

### Phase 5 — Streamlit Live Dashboard

A fully deployed interactive dashboard with 5 pages:

- **Live Monitor** — real-time transaction simulation with fraud alerts, risk scoring and KPI cards
- **Model Performance** — AUC, precision/recall comparison across all three models with radar chart
- **Risk Analysis** — anomaly score distributions, LSTM probability histograms, segment breakdowns
- **Fraud Reports** — all LLM-generated fraud alert reports with expandable transaction detail
- **About** — full project documentation

**[View live dashboard →](https://real-time-fraud-detection-system-3cm.streamlit.app/)**

---

## How to Run This Project

**1. Clone the repository**
```bash
git clone https://github.com/Thesineo/Real-time-fraud-detection-system.git
cd Real-time-fraud-detection-system
```

**2. Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the dataset**

Download from [Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data), accept competition rules, and place CSV files in the `data/` folder.

**5. Run notebooks in order**
```
01_eda_features.ipynb       → generates X_train.csv, X_test.csv
02_classical_ml.ipynb       → generates iso_forest_scores.csv
03_deep_learning.ipynb      → run on Google Colab, generates lstm_scores.csv
04_llm_explanations.ipynb   → requires OpenAI API key, generates fraud_reports.csv
```

**6. Set up environment variables**
```bash
echo "OPENAI_API_KEY=your_key_here" > .env
```

**7. Run the dashboard**
```bash
streamlit run app/dashboard.py
```

---

## Results Summary

> By combining Isolation Forest for unsupervised anomaly detection, XGBoost for supervised classification (AUC 0.924), and an LSTM for sequential pattern recognition (AUC 0.891), this system detects fraud patterns that no single model catches alone. The LLM explanation layer translates every ML decision into plain English analyst reports — bridging the gap between model output and business action. The full pipeline is deployed as a live Streamlit dashboard accessible to any stakeholder without technical knowledge.

---

## What I Learned

- How to engineer fraud-specific features from raw transaction data at scale
- Why class imbalance handling strategy matters more than model choice for fraud detection
- How LSTM sequence modelling catches temporal fraud patterns that tabular models miss
- How to make ML outputs actionable using SHAP + LLM pipelines
- Production deployment considerations: environment variables, requirements management, Streamlit Cloud
- Google Colab as a professional tool for GPU-accelerated training when local resources are limited

---

## Project Status

- [x] Phase 1 — EDA & Feature Engineering (590K transactions, 12 engineered features)
- [x] Phase 2 — Classical ML (Isolation Forest + XGBoost, AUC 0.924)
- [x] Phase 3 — Deep Learning LSTM (Google Colab, AUC 0.891)
- [x] Phase 4 — LLM Fraud Explanations (SHAP + GPT-4o-mini)
- [x] Phase 5 — Streamlit Dashboard (deployed live)

---

## Connect

Built by **Aniket Nerali**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/aniket-nerali)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/Thesineo)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=flat-square&logo=streamlit)](https://real-time-fraud-detection-system-3cm.streamlit.app/)
