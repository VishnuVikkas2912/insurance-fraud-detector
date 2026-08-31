"""
Step 1: Load raw Kaggle files, merge claims + beneficiary data,
engineer provider-level features, and save a clean feature table.
"""
import pandas as pd
import numpy as np

DATA_DIR = "/mnt/user-data/uploads/"

# ---------- 1. Load raw files ----------
train = pd.read_csv(DATA_DIR + "1784957585019_Train-1542865627584.csv")
beneficiary = pd.read_csv(DATA_DIR + "1784957585018_Train_Beneficiarydata-1542865627584.csv")
inpatient = pd.read_csv(DATA_DIR + "1784957585018_Train_Inpatientdata-1542865627584.csv")
outpatient = pd.read_csv(DATA_DIR + "1784957585019_Train_Outpatientdata-1542865627584.csv")

print("train:", train.shape)
print("beneficiary:", beneficiary.shape)
print("inpatient:", inpatient.shape)
print("outpatient:", outpatient.shape)

# ---------- 2. Tag claim type and stack inpatient + outpatient ----------
inpatient["ClaimType"] = "Inpatient"
outpatient["ClaimType"] = "Outpatient"

common_cols = list(set(inpatient.columns) & set(outpatient.columns))
claims = pd.concat([inpatient[common_cols], outpatient[common_cols]], ignore_index=True)
print("combined claims:", claims.shape)

# ---------- 3. Merge beneficiary demographic/chronic condition info ----------
claims = claims.merge(beneficiary, on="BeneID", how="left")

# ---------- 4. Basic cleaning ----------
# Convert date columns
for col in ["ClaimStartDt", "ClaimEndDt"]:
    claims[col] = pd.to_datetime(claims[col], errors="coerce")
claims["ClaimDuration"] = (claims["ClaimEndDt"] - claims["ClaimStartDt"]).dt.days

beneficiary["DOB"] = pd.to_datetime(beneficiary["DOB"], errors="coerce")
beneficiary["Age"] = 2009 - beneficiary["DOB"].dt.year  # dataset is from 2009

claims = claims.merge(beneficiary[["BeneID", "Age"]], on="BeneID", how="left")

# Count how many diagnosis codes were filled in per claim (0-10)
diag_cols = [c for c in claims.columns if c.startswith("ClmDiagnosisCode_")]
claims["NumDiagnosisCodes"] = claims[diag_cols].notna().sum(axis=1)

proc_cols = [c for c in claims.columns if c.startswith("ClmProcedureCode_")]
claims["NumProcedureCodes"] = claims[proc_cols].notna().sum(axis=1)

# ---------- 5. Aggregate claims to PROVIDER level ----------
# (fraud label is per-provider, not per-claim, so features must be too)
provider_features = claims.groupby("Provider").agg(
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

# Derived ratio features (often strong fraud signals)
provider_features["claims_per_patient"] = (
    provider_features["total_claims"] / provider_features["unique_patients"]
)
provider_features["claims_per_physician"] = (
    provider_features["total_claims"] / provider_features["unique_physicians"].replace(0, np.nan)
)
provider_features["inpatient_ratio"] = (
    provider_features["num_inpatient_claims"] / provider_features["total_claims"]
)

# fill any remaining NaNs (e.g. std of a single claim, division edge cases)
provider_features = provider_features.fillna(0)

# ---------- 6. Attach fraud label ----------
train["PotentialFraud"] = train["PotentialFraud"].map({"Yes": 1, "No": 0})
provider_features = provider_features.merge(train, on="Provider", how="inner")

print("\nFinal provider-level feature table:", provider_features.shape)
print(provider_features["PotentialFraud"].value_counts())

provider_features.to_csv("/home/claude/idp/provider_features.csv", index=False)
print("\nSaved to provider_features.csv")
print(provider_features.head())
