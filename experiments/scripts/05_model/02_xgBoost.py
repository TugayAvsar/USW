"""
02_xgBoost.py
-----------------------------
XGBoost Modell für TSLA (30-Minuten-Basis) – kompatible Version ohne Early-Stopping
(weil deine xgboost-Version weder early_stopping_rounds noch callbacks in .fit() unterstützt)

Enthält:
0) No-Feature Baseline (Majority Class, leak-free)
1) XGBoost Grid (ohne early stopping)
2) Overfitting-Check über Train/Val/Test AUC
3) Confusion Matrix, ROC Curve, Feature Importances
4) Modell-Speicherung
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    roc_auc_score
)

# -------------------------------------------------------
# Pfade
# -------------------------------------------------------
DATA_DIR  = "../../data/processed"
MODEL_DIR = "../../models"
IMG_DIR   = "../../images"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(DATA_DIR, "tsla_train.parquet")
VAL_FILE   = os.path.join(DATA_DIR, "tsla_val.parquet")
TEST_FILE  = os.path.join(DATA_DIR, "tsla_test.parquet")

# -------------------------------------------------------
# 1) Daten laden
# -------------------------------------------------------
train_df = pd.read_parquet(TRAIN_FILE)
val_df   = pd.read_parquet(VAL_FILE)
test_df  = pd.read_parquet(TEST_FILE)

TARGET = "target"
FEATURES = [c for c in train_df.columns if c not in ["datetime", TARGET]]

# NaNs -> Median vom Train (leak-free)
median_vals = train_df[FEATURES].median(numeric_only=True)
train_df[FEATURES] = train_df[FEATURES].fillna(median_vals)
val_df[FEATURES]   = val_df[FEATURES].fillna(median_vals)
test_df[FEATURES]  = test_df[FEATURES].fillna(median_vals)

X_train, y_train = train_df[FEATURES].values, train_df[TARGET].values
X_val,   y_val   = val_df[FEATURES].values,   val_df[TARGET].values
X_test,  y_test  = test_df[FEATURES].values,  test_df[TARGET].values

print("Loaded data:")
print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
print("#Features:", len(FEATURES))

# -------------------------------------------------------
# 2) BASELINE – NO FEATURES (Majority Class)
# -------------------------------------------------------
print("\n===== BASELINE (NO FEATURES): Majority Class =====\n")

majority_class = int(pd.Series(y_train).mode().iloc[0])
print("Majority class (train):", majority_class)

y_train_base = np.full_like(y_train, majority_class)
y_val_base   = np.full_like(y_val, majority_class)
y_test_base  = np.full_like(y_test, majority_class)

print("Train Accuracy (Baseline):", round(accuracy_score(y_train, y_train_base), 3))
print("Val   Accuracy (Baseline):", round(accuracy_score(y_val,   y_val_base),   3))
print("Test  Accuracy (Baseline):", round(accuracy_score(y_test,  y_test_base),  3))

print("\nClassification Report (Val, Baseline):\n",
      classification_report(y_val, y_val_base, zero_division=0))

disp = ConfusionMatrixDisplay.from_predictions(y_val, y_val_base)
plt.title("No-Feature Baseline – Confusion Matrix (Validation)")
plt.tight_layout()
baseline_cm_path = os.path.join(IMG_DIR, "baseline_no_features_confusion_matrix_val_xgb.png")
plt.savefig(baseline_cm_path, dpi=300)
plt.close()
print("Saved baseline confusion matrix →", baseline_cm_path)

# -------------------------------------------------------
# 3) XGBoost – Grid (OHNE early stopping)
# -------------------------------------------------------
# Tipp: Damit es nicht ewig dauert, bleib bei moderaten n_estimators.
param_grid = [
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.10, "subsample": 0.9},
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.9},
    {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8},
]

best_model = None
best_val_auc = -1
best_params = None

print("\n===== XGBOOST TRAINING (no early stopping - compatible) =====")

for params in param_grid:
    print(f"\nTesting params: {params}")

    xgb = XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",  # kompatibel: AUC berechnen wir manuell
        random_state=42,
        n_jobs=-1
    )

    xgb.fit(X_train, y_train)

    val_prob = xgb.predict_proba(X_val)[:, 1]
    val_auc  = roc_auc_score(y_val, val_prob)

    print(f"Validation AUC: {val_auc:.3f}")

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_model = xgb
        best_params = params

print("\n===== BEST XGBOOST MODEL =====")
print("Best Params:", best_params)
print("Best Validation AUC:", round(best_val_auc, 3))

# -------------------------------------------------------
# 4) Overfitting-Check (Train / Val / Test AUC)
# -------------------------------------------------------
train_prob = best_model.predict_proba(X_train)[:, 1]
val_prob   = best_model.predict_proba(X_val)[:, 1]
test_prob  = best_model.predict_proba(X_test)[:, 1]

train_auc = roc_auc_score(y_train, train_prob)
val_auc   = roc_auc_score(y_val,   val_prob)
test_auc  = roc_auc_score(y_test,  test_prob)

print("\nAUC Scores:")
print(f"Train AUC: {train_auc:.3f}")
print(f"Val   AUC: {val_auc:.3f}")
print(f"Test  AUC: {test_auc:.3f}")

# einfache Overfitting-Heuristik
gap = train_auc - val_auc
print(f"\nOverfitting Gap (TrainAUC - ValAUC): {gap:.3f}")
if gap > 0.05:
    print("⚠️ Hinweis: Gap > 0.05 → mögliches Overfitting.")
else:
    print("✅ Gap klein → kein starkes Overfitting-Indiz (nach AUC).")

# -------------------------------------------------------
# 5) Confusion Matrix (Validation)
# -------------------------------------------------------
y_val_pred = best_model.predict(X_val)

disp = ConfusionMatrixDisplay.from_predictions(y_val, y_val_pred)
plt.title("XGBoost – Confusion Matrix (Validation)")
plt.tight_layout()
cm_path = os.path.join(IMG_DIR, "xgb_confusion_matrix_val.png")
plt.savefig(cm_path, dpi=300)
plt.close()
print("Saved confusion matrix →", cm_path)

print("\nClassification Report (Val, XGB):\n",
      classification_report(y_val, y_val_pred, zero_division=0))

# -------------------------------------------------------
# 6) ROC Curve (Validation)
# -------------------------------------------------------
fpr, tpr, _ = roc_curve(y_val, val_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.title("XGBoost – ROC Curve (Validation)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
roc_path = os.path.join(IMG_DIR, "xgb_roc_curve_val.png")
plt.savefig(roc_path, dpi=300)
plt.close()
print("Saved ROC curve →", roc_path)

# -------------------------------------------------------
# 7) Feature Importances
# -------------------------------------------------------
importances = best_model.feature_importances_
imp_series = pd.Series(importances, index=FEATURES).sort_values(ascending=False)

plt.figure(figsize=(8, 6))
imp_series.head(15).plot(kind="bar")
plt.title("Top 15 Feature Importances – XGBoost")
plt.tight_layout()
fi_path = os.path.join(IMG_DIR, "xgb_feature_importances.png")
plt.savefig(fi_path, dpi=300)
plt.close()
print("Saved feature importances →", fi_path)

# -------------------------------------------------------
# 8) Modell speichern
# -------------------------------------------------------
model_path = os.path.join(MODEL_DIR, "tsla_xgb_best.pkl")
joblib.dump(best_model, model_path)
print("\nSaved model →", model_path)

print("\n✅ XGBOOST TRAINING + OVERFITTING CHECK COMPLETED (compatible).")
