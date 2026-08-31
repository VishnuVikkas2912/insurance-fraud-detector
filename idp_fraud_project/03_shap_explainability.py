"""
Step 3: Explainability layer.
Load the trained XGBoost model, compute SHAP values for every provider,
and produce:
  1. A global feature importance plot (what matters overall)
  2. Per-provider explanations (why THIS provider was flagged)
  3. A results table: Provider, fraud_probability, top 3 reasons (human-readable)
"""
import pandas as pd
import numpy as np
import joblib
import shap
from xgboost import XGBClassifier
import matplotlib
matplotlib.use("Agg")  # no display, just save files
import matplotlib.pyplot as plt

df = pd.read_csv("/home/claude/idp/provider_features.csv")
feature_names = joblib.load("/home/claude/idp/feature_names.pkl")
model = XGBClassifier()
model.load_model("/home/claude/idp/xgb_model.json")

X = df[feature_names]
y = df["PotentialFraud"]

# ---------- 1. Compute SHAP values ----------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)  # shape: (n_providers, n_features)

# ---------- 2. Global feature importance plot ----------
plt.figure()
shap.summary_plot(shap_values, X, show=False, plot_type="bar")
plt.tight_layout()
plt.savefig("/home/claude/idp/shap_global_importance.png", dpi=150)
plt.close()

plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("/home/claude/idp/shap_summary_beeswarm.png", dpi=150)
plt.close()
print("Saved shap_global_importance.png and shap_summary_beeswarm.png")

# ---------- 3. Per-provider explanation: top 3 reasons in plain language ----------
def explain_row(i):
    """Return top 3 features pushing this provider's prediction toward fraud."""
    row_shap = shap_values[i]
    row_vals = X.iloc[i]
    # sort by absolute SHAP contribution, take top 3 POSITIVE (fraud-pushing) ones
    contrib = pd.Series(row_shap, index=feature_names)
    top_positive = contrib[contrib > 0].sort_values(ascending=False).head(3)

    reasons = []
    for feat, shap_val in top_positive.items():
        actual_val = row_vals[feat]
        reasons.append(f"{feat} = {round(actual_val, 2)} (contributed +{round(shap_val, 3)} to fraud score)")
    if not reasons:
        reasons = ["No strong fraud-pushing features; flagged mainly by baseline risk level."]
    return " | ".join(reasons)


fraud_proba = model.predict_proba(X)[:, 1]

results = pd.DataFrame({
    "Provider": df["Provider"],
    "ActualLabel": y.map({1: "Fraud", 0: "No Fraud"}),
    "FraudProbability": fraud_proba.round(4),
    "RecommendedAction": np.where(fraud_proba > 0.5, "Flag for manual review", "No action needed"),
})
results["TopReasons"] = [explain_row(i) for i in range(len(df))]

results = results.sort_values("FraudProbability", ascending=False)
results.to_csv("/home/claude/idp/provider_fraud_explanations.csv", index=False)

print("\nSaved provider_fraud_explanations.csv")
print("\nTop 5 highest-risk providers:")
for _, r in results.head(5).iterrows():
    print(f"\n{r['Provider']}  |  Fraud probability: {r['FraudProbability']}  |  Actual: {r['ActualLabel']}")
    print(f"  Reasons: {r['TopReasons']}")
