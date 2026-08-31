"""
Step 4: Apply the trained pipeline to the TEST set.
The test set has no fraud labels (this is the genuine unseen holdout),
so this script produces fraud scores + SHAP explanations for providers
the model has never seen labeled data for -- a realistic "deployment" demo.
"""
import pandas as pd
import numpy as np
import joblib
import shap
from xgboost import XGBClassifier

DATA_DIR = "/mnt/user-data/uploads/"

# ---------- 1. Load raw TEST files ----------
test_providers = pd.read_csv(DATA_DIR + "1784957585017_Test-1542969243754.csv")
beneficiary = pd.read_csv(DATA_DIR + "1784957585016_Test_Beneficiarydata-1542969243754.csv")
inpatient = pd.read_csv(DATA_DIR + "1784957585016_Test_Inpatientdata-1542969243754.csv")
outpatient = pd.read_csv(DATA_DIR + "1784957585017_Test_Outpatientdata-1542969243754.csv")

print("test providers:", test_providers.shape)
print("beneficiary:", beneficiary.shape)
print("inpatient:", inpatient.shape)
print("outpatient:", outpatient.shape)

# ---------- 2. Same merge/feature logic as training (01_build_features.py) ----------
inpatient["ClaimType"] = "Inpatient"
outpatient["ClaimType"] = "Outpatient"

common_cols = list(set(inpatient.columns) & set(outpatient.columns))
claims = pd.concat([inpatient[common_cols], outpatient[common_cols]], ignore_index=True)

claims = claims.merge(beneficiary, on="BeneID", how="left")

for col in ["ClaimStartDt", "ClaimEndDt"]:
    claims[col] = pd.to_datetime(claims[col], errors="coerce")
claims["ClaimDuration"] = (claims["ClaimEndDt"] - claims["ClaimStartDt"]).dt.days

beneficiary["DOB"] = pd.to_datetime(beneficiary["DOB"], errors="coerce")
beneficiary["Age"] = 2009 - beneficiary["DOB"].dt.year
claims = claims.merge(beneficiary[["BeneID", "Age"]], on="BeneID", how="left")

diag_cols = [c for c in claims.columns if c.startswith("ClmDiagnosisCode_")]
claims["NumDiagnosisCodes"] = claims[diag_cols].notna().sum(axis=1)

proc_cols = [c for c in claims.columns if c.startswith("ClmProcedureCode_")]
claims["NumProcedureCodes"] = claims[proc_cols].notna().sum(axis=1)

# ---------- 3. Aggregate to provider level (identical feature set as training) ----------
provider_features_test = claims.groupby("Provider").agg(
    total_claims=("ClaimID", "count"),
    num_inpatient_claims=("ClaimType", lambda x: (x == "Inpatient").sum()),
    num_outpatient_claims=("ClaimType", lambda x: (x == "Outpatient").sum()),
    total_claim_amount=("InscClaimAmtReimbursed", "sum"),
    avg_claim_amount=("InscClaimAmtReimbursed", "mean"),
    max_claim_amount=("InscClaimAmtReimbursed", "max"),
    std_claim_amount=("InscClaimAmtReimbursed", "std"),
    unique_patients=("BeneID", "nunique"),
    unique_physicians=("AttendingPhysician", "nunique"),
    avg_claim_duration=("ClaimDuration", "mean"),
    avg_patient_age=("Age", "mean"),
    avg_num_diagnosis_codes=("NumDiagnosisCodes", "mean"),
    avg_num_procedure_codes=("NumProcedureCodes", "mean"),
    total_deductible_paid=("DeductibleAmtPaid", "sum"),
).reset_index()

provider_features_test["claims_per_patient"] = (
    provider_features_test["total_claims"] / provider_features_test["unique_patients"]
)
provider_features_test["claims_per_physician"] = (
    provider_features_test["total_claims"] / provider_features_test["unique_physicians"].replace(0, np.nan)
)
provider_features_test["inpatient_ratio"] = (
    provider_features_test["num_inpatient_claims"] / provider_features_test["total_claims"]
)
provider_features_test = provider_features_test.fillna(0)

# Keep only providers that are actually in the official Test.csv list
provider_features_test = provider_features_test.merge(
    test_providers, on="Provider", how="inner"
)
print("\nTest provider-level feature table:", provider_features_test.shape)

# ---------- 4. Load the trained model and score the test providers ----------
feature_names = joblib.load("/home/claude/idp/feature_names.pkl")
model = XGBClassifier()
model.load_model("/home/claude/idp/xgb_model.json")

X_test_final = provider_features_test[feature_names]
fraud_proba = model.predict_proba(X_test_final)[:, 1]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_final)


def explain_row(i):
    contrib = pd.Series(shap_values[i], index=feature_names)
    top_positive = contrib[contrib > 0].sort_values(ascending=False).head(3)
    reasons = []
    for feat, sv in top_positive.items():
        val = X_test_final.iloc[i][feat]
        reasons.append(f"{feat} = {round(val, 2)} (contributed +{round(sv, 3)} to fraud score)")
    if not reasons:
        reasons = ["No strong fraud-pushing features; flagged mainly by baseline risk level."]
    return " | ".join(reasons)


test_results = pd.DataFrame({
    "Provider": provider_features_test["Provider"],
    "FraudProbability": fraud_proba.round(4),
    "RecommendedAction": np.where(fraud_proba > 0.5, "Flag for manual review", "No action needed"),
})
test_results["TopReasons"] = [explain_row(i) for i in range(len(provider_features_test))]
test_results = test_results.sort_values("FraudProbability", ascending=False)

# ---------- 5. Save outputs ----------
provider_features_test.to_csv("/home/claude/idp/provider_features_test.csv", index=False)
test_results.to_csv("/home/claude/idp/test_set_predictions.csv", index=False)

n_flagged = (fraud_proba > 0.5).sum()
print(f"\nScored {len(test_results)} unseen (test) providers")
print(f"Flagged for manual review: {n_flagged} ({n_flagged/len(test_results):.1%})")
print("\nTop 10 highest-risk providers in the TEST set (never seen during training):")
for _, r in test_results.head(10).iterrows():
    print(f"\n{r['Provider']}  |  Fraud probability: {r['FraudProbability']}")
    print(f"  Reasons: {r['TopReasons']}")
