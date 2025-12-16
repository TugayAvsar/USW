"""
02_xgBoost_1min.py
------------------
XGBoost Pipeline für TSLA auf 1-Minuten-Basis.

- Baseline (No Features): Majority Class
- XGBoost Grid
- Confusion Matrix / ROC / Feature Importances
- Speichert bestes Modell

Outputs:
- images/baseline_no_features_confusion_matrix_val_1min_xgb.png
- images/xgb_confusion_matrix_val_1min.png
- images/xgb_roc_curve_val_1min.png
- images/xgb_feature_importances_1min.png
- models/tsla_xgb_best_1min.pkl
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

# -------------------------------------------------------
# Pfade
# -------------------------------------------------------
DATA_DIR  = "../../data/processed"
MODEL_DIR = "../../models"
IMG_DIR   = "../../images"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(DATA_DIR, "tsla_train_1min.parquet")
VAL_FILE   = os.path.join(DATA_DIR, "tsla_val_1min.parquet")
TEST_FILE  = os.path.join(DATA_DIR, "tsla_test_1min.parquet")

# -------------------------------------------------------
# 1) Daten laden
# -------------------------------------------------------
train_df = pd.read_parquet(TRAIN_FILE)
val_df   = pd.read_parquet(VAL_FILE)
test_df  = pd.read_parquet(TEST_FILE)

TARGET = "target"
FEATURES = [c for c in train_df.columns if c not in ["datetime", TARGET]]

# -------------------------------------------------------
# Feature-Liste für Deployment speichern
# -------------------------------------------------------
FEATURE_FILE = os.path.join(MODEL_DIR, "xgb_features_1min.txt")

with open(FEATURE_FILE, "w") as f:
    for col in FEATURES:
        f.write(col + "\n")

print(f"Saved feature list → {FEATURE_FILE}")


# NaNs -> median train
median_values = train_df[FEATURES].median(numeric_only=True)
train_df[FEATURES] = train_df[FEATURES].fillna(median_values)
val_df[FEATURES]   = val_df[FEATURES].fillna(median_values)
test_df[FEATURES]  = test_df[FEATURES].fillna(median_values)

X_train, y_train = train_df[FEATURES].values, train_df[TARGET].values
X_val,   y_val   = val_df[FEATURES].values,   val_df[TARGET].values
X_test,  y_test  = test_df[FEATURES].values,  test_df[TARGET].values

print("Loaded data shapes (1-min):")
print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
print("#Features:", len(FEATURES))

# -------------------------------------------------------
# 2) Baseline (No Features)
# -------------------------------------------------------
print("\n===== BASELINE (NO FEATURES): Majority Class =====\n")
majority_class = int(pd.Series(y_train).mode().iloc[0])
print("Majority class (train):", majority_class)

y_val_base = np.full_like(y_val, fill_value=majority_class)
print("Val Accuracy (No-Feature Baseline):", round(accuracy_score(y_val, y_val_base), 3))
print("\nClassification Report (Val, No-Feature):\n",
      classification_report(y_val, y_val_base, zero_division=0))

disp = ConfusionMatrixDisplay.from_predictions(y_val, y_val_base)
plt.title("No-Feature Baseline – Confusion Matrix (Validation, 1-min, XGB script)")
plt.tight_layout()
cm_base_path = os.path.join(IMG_DIR, "baseline_no_features_confusion_matrix_val_1min_xgb.png")
plt.savefig(cm_base_path, dpi=300)
plt.close()
print("Saved baseline confusion matrix →", cm_base_path)

# -------------------------------------------------------
# 3) XGBoost – Parameter Grid
# -------------------------------------------------------
param_grid = [
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.9},
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9},
    {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8},
]

best_model = None
best_val_acc = -1
best_params = None

print("\n===== XGBoost Training (1-min) =====")

for params in param_grid:
    print(f"\nTesting params: {params}")

    xgb = XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    xgb.fit(X_train, y_train)

    val_pred = xgb.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    print(f"Validation Accuracy: {val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = xgb
        best_params = params

print("\n===== BEST XGBOOST MODEL (1-min) =====")
print("Best Params:", best_params)
print("Best Validation Accuracy:", round(best_val_acc, 3))

# -------------------------------------------------------
# 4) Finale Evaluation
# -------------------------------------------------------
y_train_pred = best_model.predict(X_train)
y_val_pred   = best_model.predict(X_val)
y_test_pred  = best_model.predict(X_test)

print("\nTrain Accuracy:", round(accuracy_score(y_train, y_train_pred), 3))
print("Val Accuracy:  ", round(accuracy_score(y_val,   y_val_pred),   3))
print("Test Accuracy: ", round(accuracy_score(y_test,  y_test_pred),  3))

print("\nClassification Report (Val):")
print(classification_report(y_val, y_val_pred, zero_division=0))

# -------------------------------------------------------
# 5) Confusion Matrix (Val)
# -------------------------------------------------------
disp = ConfusionMatrixDisplay.from_predictions(y_val, y_val_pred)
plt.title("XGBoost – Confusion Matrix (Validation, 1-min)")
plt.tight_layout()
cm_path = os.path.join(IMG_DIR, "xgb_confusion_matrix_val_1min.png")
plt.savefig(cm_path, dpi=300)
plt.close()
print("Saved confusion matrix →", cm_path)

# -------------------------------------------------------
# 6) ROC Curve (Val)
# -------------------------------------------------------
y_val_prob = best_model.predict_proba(X_val)[:, 1]
fpr, tpr, _ = roc_curve(y_val, y_val_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], "--", color="gray")
plt.title("XGBoost – ROC Curve (Validation, 1-min)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
roc_path = os.path.join(IMG_DIR, "xgb_roc_curve_val_1min.png")
plt.savefig(roc_path, dpi=300)
plt.close()
print("Saved ROC curve →", roc_path)

# -------------------------------------------------------
# 7) Feature Importances
# -------------------------------------------------------
importances = best_model.feature_importances_
imp_series = pd.Series(importances, index=FEATURES).sort_values(ascending=False)

plt.figure(figsize=(8,6))
imp_series.head(15).plot(kind="bar")
plt.title("Top 15 Feature Importances – XGBoost (1-min)")
plt.tight_layout()
fi_path = os.path.join(IMG_DIR, "xgb_feature_importances_1min.png")
plt.savefig(fi_path, dpi=300)
plt.close()
print("Saved feature importances →", fi_path)

# -------------------------------------------------------
# 8) Modell speichern
# -------------------------------------------------------
model_path = os.path.join(MODEL_DIR, "tsla_xgb_best_1min.pkl")
joblib.dump(best_model, model_path)
print("\nSaved XGBoost model →", model_path)

print("\n✅ XGBOOST MODEL TRAINING COMPLETED (1-min).")
