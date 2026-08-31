"""
Explainable Health Insurance Claim Fraud Detector — Dashboard

Run with:
    streamlit run dashboard.py

Two tabs:
  1. Browse Providers  - search/filter existing providers, see fraud score + SHAP reasons
  2. Simulate New Claim - enter feature values manually, get a live fraud score + explanation
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

st.set_page_config(page_title="Claim Fraud Detector", layout="wide")

# ---------------------------------------------------------------------------
# Load model + data (cached so it only loads once)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    # Loaded from XGBoost's portable JSON format (not pickle) so it works
    # regardless of which xgboost version is installed locally.
    model = XGBClassifier()
    model.load_model("xgb_model.json")
    feature_names = joblib.load("feature_names.pkl")
    explainer = shap.TreeExplainer(model)
    return model, feature_names, explainer

@st.cache_data
def load_data():
    df = pd.read_csv("provider_features.csv")
    results = pd.read_csv("provider_fraud_explanations.csv")
    test_results = None
    try:
        test_results = pd.read_csv("test_set_predictions.csv")
    except FileNotFoundError:
        pass
    return df, results, test_results

model, feature_names, explainer = load_model()
provider_features, results, test_results = load_data()

FEATURE_DESCRIPTIONS = {
    "total_claims": "Total number of claims filed",
    "num_inpatient_claims": "Number of inpatient claims",
    "num_outpatient_claims": "Number of outpatient claims",
    "total_claim_amount": "Total amount reimbursed across all claims ($)",
    "avg_claim_amount": "Average reimbursed amount per claim ($)",
    "max_claim_amount": "Largest single claim amount ($)",
    "std_claim_amount": "Variability (std dev) in claim amounts",
    "unique_patients": "Number of distinct patients served",
    "unique_physicians": "Number of distinct physicians involved",
    "avg_claim_duration": "Average claim duration (days)",
    "avg_patient_age": "Average age of patients",
    "avg_num_diagnosis_codes": "Average number of diagnosis codes per claim",
    "avg_num_procedure_codes": "Average number of procedure codes per claim",
    "total_deductible_paid": "Total deductible amount paid ($)",
    "claims_per_patient": "Claims filed per unique patient",
    "claims_per_physician": "Claims filed per unique physician",
    "inpatient_ratio": "Fraction of claims that are inpatient",
}


def explain_prediction(feature_vector_df):
    """Return fraud probability + top 3 reasons for a single-row feature dataframe."""
    proba = model.predict_proba(feature_vector_df)[:, 1][0]
    shap_vals = explainer.shap_values(feature_vector_df)[0]
    contrib = pd.Series(shap_vals, index=feature_names)
    top_positive = contrib[contrib > 0].sort_values(ascending=False).head(3)

    reasons = []
    for feat, sv in top_positive.items():
        val = feature_vector_df.iloc[0][feat]
        desc = FEATURE_DESCRIPTIONS.get(feat, feat)
        reasons.append(f"**{desc}** = {val:,.2f}  →  pushed fraud score up by {sv:.3f}")
    if not reasons:
        reasons = ["No individual feature strongly pushed this toward fraud."]
    return proba, reasons, contrib


st.title("🩺 Explainable Health Insurance Claim Fraud Detector")
st.caption("Provider-level fraud scoring with SHAP-based explanations, trained on the Kaggle Healthcare Provider Fraud Detection dataset.")

tab1, tab2, tab3 = st.tabs(["📋 Browse Providers", "🧮 Simulate New Claim", "🧪 Unseen Test Set"])

# ---------------------------------------------------------------------------
# TAB 1: Browse existing providers
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Provider Risk Review Queue")

    col1, col2, col3 = st.columns(3)
    with col1:
        min_prob = st.slider("Minimum fraud probability", 0.0, 1.0, 0.5, 0.05)
    with col2:
        actual_filter = st.selectbox("Filter by actual label", ["All", "Fraud", "No Fraud"])
    with col3:
        search_id = st.text_input("Search Provider ID (optional)", "")

    filtered = results[results["FraudProbability"] >= min_prob]
    if actual_filter != "All":
        filtered = filtered[filtered["ActualLabel"] == actual_filter]
    if search_id:
        filtered = filtered[filtered["Provider"].str.contains(search_id.upper())]

    st.write(f"Showing **{len(filtered)}** providers")
    st.dataframe(
        filtered[["Provider", "FraudProbability", "ActualLabel", "RecommendedAction", "TopReasons"]],
        use_container_width=True,
        height=400,
    )

    st.divider()
    st.subheader("Inspect a single provider")
    selected_provider = st.selectbox("Choose a Provider ID", results["Provider"].tolist())

    if selected_provider:
        row = results[results["Provider"] == selected_provider].iloc[0]
        feat_row = provider_features[provider_features["Provider"] == selected_provider][feature_names]

        c1, c2, c3 = st.columns(3)
        c1.metric("Fraud Probability", f"{row['FraudProbability']:.1%}")
        c2.metric("Actual Label", row["ActualLabel"])
        c3.metric("Recommendation", row["RecommendedAction"])

        proba, reasons, contrib = explain_prediction(feat_row)
        st.markdown("**Top reasons for this score:**")
        for r in reasons:
            st.markdown(f"- {r}")

        fig, ax = plt.subplots(figsize=(8, 4))
        top_feats = contrib.abs().sort_values(ascending=False).head(8).index
        colors = ["#d62728" if contrib[f] > 0 else "#2ca02c" for f in top_feats]
        ax.barh(top_feats[::-1], contrib[top_feats][::-1], color=colors[::-1])
        ax.set_xlabel("SHAP contribution (red = pushes toward fraud, green = pushes away)")
        ax.set_title(f"Feature contributions for {selected_provider}")
        st.pyplot(fig)

# ---------------------------------------------------------------------------
# TAB 2: Simulate a new claim / provider profile
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Enter a provider profile to get a live fraud score")
    st.caption("Fill in aggregate stats for a hypothetical or new provider. Defaults are set to dataset medians.")

    defaults = provider_features[feature_names].median()

    with st.form("simulate_form"):
        cols = st.columns(3)
        user_input = {}
        for i, feat in enumerate(feature_names):
            col = cols[i % 3]
            label = FEATURE_DESCRIPTIONS.get(feat, feat)
            user_input[feat] = col.number_input(
                label, value=float(defaults[feat]), format="%.2f", key=feat
            )
        submitted = st.form_submit_button("Score this provider")

    if submitted:
        input_df = pd.DataFrame([user_input])[feature_names]
        proba, reasons, contrib = explain_prediction(input_df)

        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Predicted Fraud Probability", f"{proba:.1%}")
        c2.metric("Recommendation", "🚩 Flag for manual review" if proba > 0.5 else "✅ No action needed")

        st.markdown("**Top reasons for this score:**")
        for r in reasons:
            st.markdown(f"- {r}")

        fig, ax = plt.subplots(figsize=(8, 4))
        top_feats = contrib.abs().sort_values(ascending=False).head(8).index
        colors = ["#d62728" if contrib[f] > 0 else "#2ca02c" for f in top_feats]
        ax.barh(top_feats[::-1], contrib[top_feats][::-1], color=colors[::-1])
        ax.set_xlabel("SHAP contribution (red = pushes toward fraud, green = pushes away)")
        ax.set_title("Feature contributions for this simulated provider")
        st.pyplot(fig)

# ---------------------------------------------------------------------------
# TAB 3: Unseen test set results (true holdout, no labels available)
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Scoring on the unseen Test set")
    st.caption(
        "These providers were never used in training. There are no ground-truth "
        "labels for them (this is a true holdout, matching the original Kaggle "
        "competition setup) — scores here demonstrate how the model would behave "
        "in real deployment."
    )

    if test_results is None:
        st.warning(
            "test_set_predictions.csv not found. Run 04_score_test_set.py first "
            "to generate test-set predictions."
        )
    else:
        n_flagged = (test_results["FraudProbability"] > 0.5).sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total test providers", len(test_results))
        c2.metric("Flagged for review", n_flagged)
        c3.metric("Flag rate", f"{n_flagged/len(test_results):.1%}")

        min_prob_test = st.slider("Minimum fraud probability", 0.0, 1.0, 0.5, 0.05, key="test_slider")
        filtered_test = test_results[test_results["FraudProbability"] >= min_prob_test]
        st.write(f"Showing **{len(filtered_test)}** test providers")
        st.dataframe(
            filtered_test[["Provider", "FraudProbability", "RecommendedAction", "TopReasons"]],
            use_container_width=True,
            height=400,
        )

        st.divider()
        st.subheader("Inspect a single test-set provider")
        selected_test_provider = st.selectbox(
            "Choose a Provider ID", test_results["Provider"].tolist(), key="test_provider_select"
        )
        if selected_test_provider:
            row = test_results[test_results["Provider"] == selected_test_provider].iloc[0]
            c1, c2 = st.columns(2)
            c1.metric("Fraud Probability", f"{row['FraudProbability']:.1%}")
            c2.metric("Recommendation", row["RecommendedAction"])
            st.markdown("**Top reasons for this score:**")
            for reason in row["TopReasons"].split(" | "):
                st.markdown(f"- {reason}")
