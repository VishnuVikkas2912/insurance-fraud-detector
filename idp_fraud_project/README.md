# Explainable Health Insurance Claim Fraud Detector

An end-to-end pipeline that flags healthcare **providers** likely to be committing
insurance fraud, using aggregated claim/beneficiary data, an XGBoost classifier,
and SHAP-based explainability — plus an interactive Streamlit dashboard for
browsing results and simulating new claims.

Built on the [Kaggle Healthcare Provider Fraud Detection](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis) dataset.

## Project structure

```
IDP/
├── Dataset/                          # Raw Kaggle CSVs (train + test)
│   ├── Train-*.csv                   # Provider-level fraud labels (train)
│   ├── Train_Beneficiarydata-*.csv   # Patient demographics & chronic conditions
│   ├── Train_Inpatientdata-*.csv     # Inpatient claims
│   ├── Train_Outpatientdata-*.csv    # Outpatient claims
│   ├── Test-*.csv                    # Unseen provider list (no labels)
│   ├── Test_Beneficiarydata-*.csv
│   ├── Test_Inpatientdata-*.csv
│   └── Test_Outpatientdata-*.csv
│
└── idp_fraud_project/
    ├── 01_build_features.py          # Merge raw data → provider-level features
    ├── 02_baseline_model.py          # Train LR / RF / XGBoost, evaluate, save model
    ├── 03_shap_explainability.py     # SHAP values, global importance, per-provider explanations
    ├── 04_score_test_set.py          # Apply pipeline + model to the unseen test set
    ├── dashboard.py                  # Streamlit app (3 tabs, see below)
    ├── xgb_model.json                # Trained XGBoost model (portable format)
    ├── feature_names.pkl             # Feature list used at inference time
    ├── provider_features.csv         # Engineered train features
    ├── provider_features_test.csv    # Engineered test features
    ├── provider_fraud_explanations.csv
    ├── test_set_predictions.csv
    ├── shap_global_importance.png
    ├── shap_summary_beeswarm.png
    └── train_vs_test_distribution.png
```

## What it does

1. **Feature engineering** — raw per-claim records (inpatient + outpatient +
   beneficiary demographics) are aggregated up to one row per **provider**:
   total claim amount, claims per patient, deductible paid, average claim
   duration, diagnosis/procedure code counts, chronic condition rates, etc.
2. **Modeling** — Logistic Regression, Random Forest, and XGBoost are trained
   and compared on ROC-AUC, PR-AUC, precision, and recall. XGBoost is saved
   as the production model (`xgb_model.json`).
3. **Explainability** — SHAP values explain *why* each provider is flagged,
   both globally (which features matter most overall) and per-provider
   (which specific factors drove that provider's score).
4. **Holdout evaluation** — the exact same feature pipeline and trained
   model are applied to a completely unseen test set of providers (no
   labels) as a true generalization check.
5. **Dashboard** — a Streamlit app to explore it all interactively.

## Results

- **Train set**: 5,410 providers, 9.35% labeled fraudulent
- **Best models**: ROC-AUC ≈ 0.95 across Logistic Regression, Random Forest, and XGBoost
- **Top fraud signals**: total claim amount, claims per patient, total deductible paid
- **Test set** (unseen, 1,353 providers): 16.2% flagged for review, closely
  matching the train set's 15.4% flag rate — suggesting the model
  generalizes rather than overfitting to training-specific patterns (see
  `train_vs_test_distribution.png`)

## Setup

```bash
pip install pandas numpy scikit-learn xgboost shap streamlit joblib matplotlib
```

## Running the dashboard (quick start)

All the pipeline outputs (`provider_features.csv`, `xgb_model.json`,
SHAP explanations, test predictions) are already included in this repo,
so you can jump straight to the dashboard without re-running the pipeline:

```bash
cd idp_fraud_project
streamlit run dashboard.py
```

This opens a browser tab (usually `http://localhost:8501`) with three tabs:

1. **Browse Providers** — train set, with actual fraud labels
2. **Simulate New Claim** — manual input form with live scoring
3. **Unseen Test Set** — holdout providers, no labels, true generalization demo

## Re-running the full pipeline from scratch

If you change the data or feature logic, re-run the scripts in order:

```
01_build_features.py       → provider_features.csv
02_baseline_model.py       → xgb_model.json, feature_names.pkl
03_shap_explainability.py  → provider_fraud_explanations.csv, SHAP plots
04_score_test_set.py       → provider_features_test.csv, test_set_predictions.csv
streamlit run dashboard.py
```

**⚠️ Before running the pipeline scripts, update the file paths.** They
currently point to the original development environment's file locations
(`/mnt/user-data/uploads/...` and `/home/claude/idp/...`). Update
`DATA_DIR` in `01_build_features.py` and `04_score_test_set.py` to point
to the local `Dataset/` folder, and update the output paths (e.g.
`provider_features.to_csv(...)`) to save into `idp_fraud_project/`
instead, or wherever you prefer.

## Notes

- The model is saved in XGBoost's native JSON format (not pickle) to avoid
  version-mismatch errors across machines.
- Dataset redistribution: check [Kaggle's terms](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis)
  before making this repo public if you're including the raw `Dataset/` CSVs.
