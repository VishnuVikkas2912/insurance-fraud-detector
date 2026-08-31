# Explainable Health Insurance Claim Fraud Detector

An end-to-end machine learning pipeline that flags healthcare **providers**
(not individual claims) likely to be committing insurance fraud — built on
claims + beneficiary data, an XGBoost classifier, and SHAP-based
explainability, wrapped in an interactive Streamlit dashboard.

Built on the [Kaggle Healthcare Provider Fraud Detection](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis) dataset.

## Why this project exists

Health insurance fraud is typically hidden inside enormous volumes of
individually-legitimate-looking claims — no single claim looks suspicious,
but a *provider's overall pattern* (unusually high claim volumes, repeat
billing on the same patients, inflated amounts) can reveal fraud. This
project:

1. Turns thousands of raw, patient-level claim records into a small set of
   **provider-level risk features**.
2. Trains a classifier to separate fraudulent from legitimate providers.
3. Explains **why** each provider is flagged, in plain language — not just
   a black-box score — using [SHAP](https://shap.readthedocs.io/) (SHapley
   Additive exPlanations).
4. Validates that the model actually generalizes, by scoring a completely
   unseen holdout set of providers.
5. Puts all of this behind a no-code dashboard that a non-technical
   reviewer (e.g. a claims analyst) could use.

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

## How it works

### 1. Feature engineering (`01_build_features.py`)

The raw Kaggle data is split across several files at the **claim** level
(one row per inpatient or outpatient visit) and the **beneficiary** level
(one row per patient, with demographics and chronic conditions). Since the
fraud label only exists at the **provider** level, everything gets rolled
up:

- Inpatient and outpatient claims are combined and tagged by type.
- Beneficiary demographics (age, computed from date of birth) are merged in.
- Diagnosis/procedure code counts are derived per claim.
- Everything is grouped by `Provider` and aggregated into **17 features**:

| Feature | Meaning |
|---|---|
| `total_claims` | Total number of claims filed |
| `num_inpatient_claims` / `num_outpatient_claims` | Claim counts by type |
| `total_claim_amount` | Total amount reimbursed across all claims ($) |
| `avg_claim_amount` | Average reimbursed amount per claim ($) |
| `max_claim_amount` | Largest single claim amount ($) |
| `std_claim_amount` | Variability (std dev) in claim amounts |
| `unique_patients` | Number of distinct patients served |
| `unique_physicians` | Number of distinct physicians involved |
| `avg_claim_duration` | Average claim duration (days) |
| `avg_patient_age` | Average age of patients treated |
| `avg_num_diagnosis_codes` | Average diagnosis codes listed per claim |
| `avg_num_procedure_codes` | Average procedure codes listed per claim |
| `total_deductible_paid` | Total deductible amount paid ($) |
| `claims_per_patient` | Claims filed per unique patient — high values can indicate repeat/duplicate billing |
| `claims_per_physician` | Claims filed per unique physician |
| `inpatient_ratio` | Fraction of a provider's claims that are inpatient |

Output: `provider_features.csv` — one row per provider, labeled
`PotentialFraud` (Yes/No from the original data, encoded as 1/0).

### 2. Modeling (`02_baseline_model.py`)

Three classifiers are trained on an 80/20 stratified train/test split and
compared head-to-head:

| Model | Handling class imbalance | Why included |
|---|---|---|
| Logistic Regression | `class_weight="balanced"` | Simple, interpretable baseline |
| Random Forest | `class_weight="balanced"`, 300 trees | Non-linear baseline, robust to outliers |
| **XGBoost** (production model) | `scale_pos_weight` | Best accuracy/speed trade-off, native SHAP support |

Fraud is a minority class (~9.35% of providers), so all three models
correct for imbalance rather than defaulting to "predict no fraud."
Each model is scored on **ROC-AUC**, **PR-AUC (average precision)**,
precision/recall, and a confusion matrix. XGBoost is selected as the
production model and saved to `xgb_model.json`.

> **Why JSON instead of pickle?** `xgb.save_model(...)` uses XGBoost's own
> portable format. A pickled model embeds a version-specific binary blob
> that can silently break (or throw obscure errors) when loaded with a
> different XGBoost version — e.g. the version on your laptop vs. the
> training machine. JSON avoids that entirely.

### 3. Explainability (`03_shap_explainability.py`)

Using `shap.TreeExplainer` on the trained XGBoost model:

- **Global importance** (`shap_global_importance.png`) — a bar chart of
  which features matter most across *all* providers.
- **Beeswarm summary** (`shap_summary_beeswarm.png`) — shows both feature
  importance and the direction/spread of each feature's effect.
- **Per-provider explanations** (`provider_fraud_explanations.csv`) — for
  every provider, the top 3 features that pushed its score toward "fraud,"
  written out in plain language, e.g.:
  ```
  total_claim_amount = 184320.0 (contributed +0.412 to fraud score)
  claims_per_patient = 3.8 (contributed +0.201 to fraud score)
  ```
  This file also includes a `RecommendedAction` column
  (`Flag for manual review` if fraud probability > 0.5, else `No action needed`).

### 4. Holdout evaluation (`04_score_test_set.py`)

The *exact same* feature-engineering logic and trained model are applied
to the Kaggle **test set** — a separate group of providers with no fraud
labels at all. This is the closest thing to "real-world" evaluation
available here: if the test set's flag rate and feature distributions
look similar to the training set's, that's evidence the model learned
genuine fraud patterns rather than memorizing training-set quirks.
Output: `provider_features_test.csv` and `test_set_predictions.csv`.

### 5. Dashboard (`dashboard.py`)

A Streamlit app with three tabs:

1. **📋 Browse Providers** — the labeled training set as a searchable/
   sortable risk review queue; click into any provider to see its SHAP
   explanation.
2. **🧮 Simulate New Claim** — manually enter feature values (claim counts,
   amounts, patient counts, etc.) for a hypothetical provider and get a
   live fraud probability + explanation, without touching any code.
3. **🧪 Unseen Test Set** — browse the holdout providers with no ground-
   truth labels, to see how the model behaves on data it never trained on.

The model, SHAP explainer, and CSVs are loaded once and cached
(`@st.cache_resource` / `@st.cache_data`) so the app stays responsive.

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

## Interpreting the results

- A **fraud probability** close to 1.0 means the model is highly confident
  this provider's claim pattern resembles known fraud cases.
- The **top reasons** are not proof of fraud — they're the statistical
  features that most influenced the score. E.g. a legitimate large clinic
  will naturally have high `total_claim_amount`; the model weighs this
  alongside other signals like `claims_per_patient` and `inpatient_ratio`.
- This is a **screening tool**, not an automated fraud determination.
  `RecommendedAction = "Flag for manual review"` means exactly that — send
  it to a human investigator, not an auto-reject.

## Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| `FileNotFoundError` when running a script | Scripts must be run from inside `idp_fraud_project/`. |
| `streamlit: command not found` | Activate your virtual environment, or run `python -m streamlit run dashboard.py`. |
| XGBoost model fails to load | Make sure you're loading `xgb_model.json` with `model.load_model(...)`, not `pickle.load(...)` — see the JSON note above. |
| Dashboard shows stale data after editing a CSV | Streamlit caches data/model loads; restart the app or clear cache from Streamlit's menu (⋮ → Clear cache). |

## Notes

- The model is saved in XGBoost's native JSON format (not pickle) to avoid
  version-mismatch errors across machines.
- Patient age is computed as `2009 - birth_year`, since this dataset's
  claims are all from 2009.
- Dataset redistribution: check [Kaggle's terms](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis)
  before making this repo public if you're including the raw `Dataset/` CSVs.

## Tech stack

`pandas` · `numpy` · `scikit-learn` · `xgboost` · `shap` · `streamlit` · `matplotlib` · `joblib`
