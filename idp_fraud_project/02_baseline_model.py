"""
Step 2: Train baseline models (Logistic Regression, Random Forest, XGBoost)
on the provider-level feature table, with proper handling of class imbalance.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix
)
from xgboost import XGBClassifier

df = pd.read_csv("/home/claude/idp/provider_features.csv")

X = df.drop(columns=["Provider", "PotentialFraud"])
y = df["PotentialFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Train:", X_train.shape, "Test:", X_test.shape)
print("Train fraud rate:", y_train.mean().round(3), "| Test fraud rate:", y_test.mean().round(3))


def evaluate(name, model, X_te, y_te):
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    print(f"\n===== {name} =====")
    print(classification_report(y_te, y_pred, target_names=["No Fraud", "Fraud"]))
    print("ROC-AUC:      ", round(roc_auc_score(y_te, y_proba), 4))
    print("PR-AUC (AP):  ", round(average_precision_score(y_te, y_proba), 4))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_te, y_pred))


# ---------- 1. Logistic Regression (interpretable baseline) ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
logreg.fit(X_train_scaled, y_train)
evaluate("Logistic Regression (class_weight=balanced)", logreg, X_test_scaled, y_test)

# ---------- 2. Random Forest ----------
rf = RandomForestClassifier(
    n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
evaluate("Random Forest (class_weight=balanced)", rf, X_test, y_test)

# ---------- 3. XGBoost ----------
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    scale_pos_weight=scale_pos_weight, eval_metric="logloss",
    random_state=42
)
xgb.fit(X_train, y_train)
evaluate("XGBoost (scale_pos_weight)", xgb, X_test, y_test)

# ---------- Feature importance from XGBoost (quick sanity check) ----------
importances = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 features (XGBoost importance):")
print(importances.head(10))

import joblib
# Save in XGBoost's own portable format (JSON) instead of pickle.
# Pickle embeds a version-specific binary blob that breaks when loaded
# with a different xgboost version (e.g. training env vs. your laptop).
xgb.save_model("/home/claude/idp/xgb_model.json")
joblib.dump(list(X.columns), "/home/claude/idp/feature_names.pkl")
print("\nSaved xgb_model.json and feature_names.pkl")
