import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import random
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #e74c3c;
    }
    .safe-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #2ecc71;
    }
    .alert-box {
        background: rgba(231, 76, 60, 0.1);
        border: 1px solid #e74c3c;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
    .safe-box {
        background: rgba(46, 204, 113, 0.1);
        border: 1px solid #2ecc71;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    try:
        predictions = pd.read_csv('../data/churn_predictions.csv'
                                   if os.path.exists('../data/churn_predictions.csv')
                                   else 'data/churn_predictions.csv')
    except Exception:
        predictions = None

    try:
        iso_scores = pd.read_csv('../data/iso_forest_scores.csv'
                                  if os.path.exists('../data/iso_forest_scores.csv')
                                  else 'data/iso_forest_scores.csv')
    except Exception:
        iso_scores = None

    try:
        lstm_scores = pd.read_csv('../data/lstm_scores.csv'
                                   if os.path.exists('../data/lstm_scores.csv')
                                   else 'data/lstm_scores.csv')
    except Exception:
        lstm_scores = None

    try:
        reports = pd.read_csv('../data/fraud_reports.csv'
                               if os.path.exists('../data/fraud_reports.csv')
                               else 'data/fraud_reports.csv')
    except Exception:
        reports = None

    return predictions, iso_scores, lstm_scores, reports


def simulate_transaction():
    """Generate a fake real-time transaction for live demo."""
    is_fraud = random.random() < 0.08
    return {
        'transaction_id':    f"TXN-{random.randint(100000, 999999)}",
        'amount':            round(random.uniform(500, 8000) if is_fraud
                                   else random.uniform(10, 500), 2),
        'hour':              random.randint(0, 23),
        'is_night':          random.random() < 0.7 if is_fraud
                             else random.random() < 0.15,
        'card_avg_ratio':    round(random.uniform(3, 10) if is_fraud
                                   else random.uniform(0.5, 1.5), 2),
        'fraud_probability': round(random.uniform(0.65, 0.98) if is_fraud
                                   else random.uniform(0.01, 0.25), 3),
        'is_fraud':          is_fraud,
        'country':           random.choice(['US', 'UK', 'NG', 'RO', 'PH'])
                             if is_fraud else
                             random.choice(['US', 'US', 'US', 'CA', 'GB']),
        'card_type':         random.choice(['Visa', 'Mastercard', 'Amex']),
        'merchant':          random.choice(['Online Store', 'ATM', 'Restaurant',
                                            'Electronics', 'Travel'])
    }


predictions, iso_scores, lstm_scores, reports = load_data()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Fraud Detection System")
    st.caption("Real-time ML-powered monitoring")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Live Monitor", "Model Performance",
         "Risk Analysis", "Fraud Reports", "About"],
        label_visibility="collapsed"
    )

    st.divider()
    st.caption("Models active")
    st.success("Isolation Forest")
    st.success("XGBoost Classifier")
    st.success("LSTM Sequential")
    st.success("LLM Explainer")

# ── Page 1: Live Monitor ──────────────────────────────────────────────────────
if page == "Live Monitor":
    st.title("Live Transaction Monitor")
    st.caption("Simulated real-time transaction stream")

    if 'transactions' not in st.session_state:
        st.session_state.transactions = []
        st.session_state.fraud_count  = 0
        st.session_state.total_count  = 0
        st.session_state.total_amount = 0

    col1, col2, col3, col4 = st.columns(4)

    metric_total  = col1.empty()
    metric_fraud  = col2.empty()
    metric_rate   = col3.empty()
    metric_amount = col4.empty()

    st.divider()

    col_feed, col_alerts = st.columns([3, 2])

    with col_feed:
        st.subheader("Transaction feed")
        feed_placeholder = st.empty()

    with col_alerts:
        st.subheader("Fraud alerts")
        alert_placeholder = st.empty()

    run = st.button("Start live simulation", type="primary")

    if run:
        for _ in range(50):
            tx = simulate_transaction()
            st.session_state.transactions.append(tx)
            st.session_state.total_count  += 1
            st.session_state.total_amount += tx['amount']

            if tx['is_fraud']:
                st.session_state.fraud_count += 1

            total  = st.session_state.total_count
            frauds = st.session_state.fraud_count
            rate   = (frauds / total * 100) if total > 0 else 0

            metric_total.metric("Total transactions", f"{total:,}")
            metric_fraud.metric("Fraud detected",     f"{frauds:,}",
                                delta=f"+{1 if tx['is_fraud'] else 0}")
            metric_rate.metric("Fraud rate",
                               f"{rate:.1f}%",
                               delta_color="inverse")
            metric_amount.metric("Total volume",
                                 f"${st.session_state.total_amount:,.0f}")

            recent = st.session_state.transactions[-15:]
            df_recent = pd.DataFrame(recent)

            with feed_placeholder:
                st.dataframe(
                    df_recent[['transaction_id', 'amount', 'merchant',
                                'card_type', 'fraud_probability',
                                'is_fraud']].style.apply(
                        lambda row: ['background-color: rgba(231,76,60,0.2)'
                                     if row['is_fraud']
                                     else '' for _ in row],
                        axis=1
                    ),
                    use_container_width=True,
                    height=350
                )

            fraud_txns = [t for t in recent if t['is_fraud']]
            with alert_placeholder:
                if fraud_txns:
                    for ft in fraud_txns[-5:]:
                        st.error(
                            f"FRAUD ALERT — {ft['transaction_id']}\n\n"
                            f"Amount: ${ft['amount']:,.2f} | "
                            f"Risk: {ft['fraud_probability']:.1%} | "
                            f"Country: {ft['country']}"
                        )
                else:
                    st.success("No fraud detected in recent transactions")

            time.sleep(0.3)

# ── Page 2: Model Performance ─────────────────────────────────────────────────
elif page == "Model Performance":
    st.title("Model performance comparison")

    col1, col2, col3 = st.columns(3)
    col1.metric("Isolation Forest AUC", "0.782", "Unsupervised")
    col2.metric("XGBoost AUC",          "0.924", "Best classifier")
    col3.metric("LSTM AUC",             "0.891", "Sequential model")

    st.divider()

    models_df = pd.DataFrame({
        'Model':     ['Isolation Forest', 'XGBoost', 'LSTM'],
        'AUC':       [0.782, 0.924, 0.891],
        'Precision': [0.41,  0.87,  0.79],
        'Recall':    [0.68,  0.82,  0.85],
        'F1':        [0.51,  0.84,  0.82],
    })

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            models_df.melt(id_vars='Model',
                           value_vars=['AUC', 'Precision', 'Recall', 'F1']),
            x='Model', y='value', color='variable',
            barmode='group',
            title='Model metrics comparison',
            labels={'value': 'Score', 'variable': 'Metric'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        categories = ['AUC', 'Precision', 'Recall', 'F1']

        for _, row in models_df.iterrows():
            fig2.add_trace(go.Scatterpolar(
                r=[row['AUC'], row['Precision'],
                   row['Recall'], row['F1']],
                theta=categories,
                fill='toself',
                name=row['Model']
            ))

        fig2.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title='Model radar chart'
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Model summary")
    st.dataframe(models_df, use_container_width=True, hide_index=True)

# ── Page 3: Risk Analysis ─────────────────────────────────────────────────────
elif page == "Risk Analysis":
    st.title("Risk segment analysis")

    if iso_scores is not None:
        col1, col2 = st.columns(2)

        with col1:
            fraud_by_pred = iso_scores.groupby(
                'iso_forest_predicted')['actual_fraud'].mean().reset_index()
            fraud_by_pred['label'] = fraud_by_pred[
                'iso_forest_predicted'].map({0: 'Legitimate', 1: 'Fraud'})

            fig = px.pie(
                fraud_by_pred,
                values='actual_fraud',
                names='label',
                title='Isolation Forest prediction split',
                color_discrete_sequence=['#2ecc71', '#e74c3c']
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.histogram(
                iso_scores,
                x='iso_forest_score',
                color='actual_fraud',
                nbins=50,
                title='Anomaly score distribution',
                labels={'iso_forest_score': 'Anomaly score',
                        'actual_fraud': 'Is Fraud'},
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
            )
            st.plotly_chart(fig2, use_container_width=True)

    else:
        st.info("Run Phase 2 to generate Isolation Forest scores")

    st.divider()

    if lstm_scores is not None:
        st.subheader("LSTM risk distribution")
        fig3 = px.histogram(
            lstm_scores,
            x='lstm_probability',
            color='actual_fraud',
            nbins=50,
            title='LSTM fraud probability distribution',
            color_discrete_map={0: '#3498db', 1: '#e74c3c'}
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Run Phase 3 to generate LSTM scores")

# ── Page 4: Fraud Reports ─────────────────────────────────────────────────────
elif page == "Fraud Reports":
    st.title("LLM-generated fraud reports")
    st.caption("Plain English explanations auto-generated by GPT for each flagged transaction")

    if reports is not None:
        for i, row in reports.iterrows():
            risk = "HIGH" if row['fraud_probability'] > 0.7 else "MEDIUM"
            color = "red" if risk == "HIGH" else "orange"

            with st.expander(
                f"Transaction {row['transaction_idx']} — "
                f"{row['fraud_probability']:.1%} fraud probability — "
                f"{risk} RISK",
                expanded=(i == 0)
            ):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("Fraud probability",
                              f"{row['fraud_probability']:.1%}")
                    st.metric("Actual fraud",
                              "YES" if row['actual_fraud'] else "NO")

                with col2:
                    st.markdown("**AI-generated fraud analyst report:**")
                    st.info(row['llm_fraud_report'])
    else:
        st.info("Run Phase 4 to generate LLM fraud reports")
        st.write("Sample report preview:")
        st.info("""
**SUMMARY:** This transaction has been flagged as HIGH risk 
with a fraud probability of 89.3%.

**KEY RISK FACTORS:**
- Transaction amount is 7.2x higher than this card's average spend
- Transaction occurred at 2:47 AM — outside normal usage hours
- Card was used in a different country than usual
- No prior history of this merchant category on this card

**RECOMMENDED ACTION:** Place an immediate hold on this transaction 
and contact the cardholder for verification before processing.
        """)

# ── Page 5: About ────────────────────────────────────────────────────────────
elif page == "About":
    st.title("About this project")

    st.markdown("""
    ## Real-time Financial Fraud Detection System

    An end-to-end ML pipeline that detects fraudulent transactions 
    using a combination of classical ML, deep learning and LLMs.

    ### Tech stack
    - **Data**: IEEE-CIS Fraud Detection (590K transactions)
    - **Anomaly detection**: Isolation Forest (unsupervised)
    - **Classification**: XGBoost + LightGBM ensemble
    - **Sequential modelling**: LSTM neural network
    - **Explainability**: SHAP values
    - **LLM integration**: GPT-4o-mini fraud report generation
    - **Dashboard**: Streamlit + Plotly

    ### Project phases
    1. EDA & Feature Engineering
    2. Classical ML — Isolation Forest + XGBoost
    3. Deep Learning — LSTM sequential model
    4. LLM Integration — SHAP + GPT fraud reports
    5. Streamlit Dashboard (this app)

    ### Key results
    | Model | AUC |
    |---|---|
    | Isolation Forest | 0.782 |
    | XGBoost | 0.924 |
    | LSTM | 0.891 |
    """)

    st.divider()
    st.markdown("Built by **Aniket Nerali** | "
                "[GitHub](https://github.com/Thesineo)")