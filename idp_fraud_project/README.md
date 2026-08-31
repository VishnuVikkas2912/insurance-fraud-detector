# Explainable Health Insurance Claim Fraud Detector

## Setup
```
pip install pandas numpy scikit-learn xgboost shap streamlit joblib matplotlib
```

## Files
- `01_build_features.py` — loads raw Kaggle Train CSVs, merges claims + beneficiary data, builds provider-level feature table
- `02_baseline_model.py` — trains Logistic Regression, Random Forest, XGBoost; evaluates with ROC-AUC/PR-AUC/precision/recall; saves xgb_model.json
- `03_shap_explainability.py` — computes SHAP values on the TRAIN set, saves global importance plots and a per-provider explanation CSV
- `04_score_test_set.py` — applies the same feature pipeline + trained model to the unseen TEST set (no labels — a true holdout demo)
- `dashboard.py` — Streamlit app with three tabs:
  1. Browse Providers (train set, with actual labels)
  2. Simulate New Claim (manual input form, live scoring)
  3. Unseen Test Set (holdout providers, no labels, true generalization demo)

Model is saved as `xgb_model.json` (XGBoost's own portable format), not pickle,
to avoid version-mismatch errors across machines.

## Run the dashboard
```
streamlit run dashboard.py
```
(Run from inside this folder — it loads all CSV/model files from the current directory.)

## Pipeline order
1. `01_build_features.py` → provider_features.csv
2. `02_baseline_model.py` → xgb_model.json, feature_names.pkl
3. `03_shap_explainability.py` → provider_fraud_explanations.csv, SHAP plots
4. `04_score_test_set.py` → provider_features_test.csv, test_set_predictions.csv
5. `streamlit run dashboard.py`

## Results
- Train set: 5,410 providers, 9.35% labeled fraudulent
- Best models: ROC-AUC ~0.95 across Logistic Regression, Random Forest, XGBoost
- Top fraud signals: total_claim_amount, claims_per_patient, total_deductible_paid
- Test set (unseen, 1,353 providers): 16.2% flagged for review, closely matching
  the train set's 15.4% flag rate — indicates the model generalizes well and
  isn't overfit to training-specific patterns (see train_vs_test_distribution.png)
