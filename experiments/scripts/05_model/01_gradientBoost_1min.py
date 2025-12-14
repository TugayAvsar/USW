"""
01_gradientBoost_1min.py
------------------------
Modeling Pipeline für TSLA auf 1-Minuten-Basis (leak-free splits).

Modelle:
A) Baseline – Majority Class (No Features)
B) Baseline – Logistic Regression (mit Features)
C) Gradient Boosting Classifier (Parametervergleich)

Outputs:
- images/baseline_no_features_confusion_matrix_val_1min.png
- images/logreg_confusion_matrix_val_1min.png
- images/gb_confusion_matrix_val_1min.png
- images/gb_roc_curve_val_1min.png
- images/gb_feature_importances_1min.png
- models/tsla_gb_best_1min.pkl
- models/tsla_logreg_1min.pkl
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)

# ---------------------------------------------------
# Pfade
# ---------------------------------------------------
DATA_DIR  = "../../data/processed"
MODEL_DIR = "../../models"
IMG_DIR   = "../../images"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(DATA_DIR, "tsla_train_1min.parquet")
VAL_FILE   = os.path.join(DATA_DIR, "tsla_val_1min.parquet")
TEST_FILE  = os.path.join(DATA_DIR, "tsla_test_1min.parquet")

# ---------------------------------------------------
# 1) Daten laden
# ---------------------------------------------------
train_df = pd.read_parquet(TRAIN_FILE)
val_df   = pd.read_parquet(VAL_FILE)
test_df  = pd.read_parquet(TEST_FILE)

TARGET = "target"
FEATURES = [c for c in train_df.columns if c not in ["datetime", TARGET]]

X_train, y_train = train_df[FEATURES].values, train_df[TARGET].values
X_val,   y_val   = val_df[FEATURES].values,   val_df[TARGET].values
X_test,  y_test  = test_df[FEATURES].values,  test_df[TARGET].values

print("Loaded data shapes (1-min):")
print("X_train:", X_train.shape, " y_train:", y_train.shape)
print("X_val:  ", X_val.shape,   " y_val:",   y_val.shape)
print("X_test: ", X_test.shape,  " y_test:",  y_test.shape)
print("#Features:", len(FEATURES))

# ---------------------------------------------------
# 1b) NaN Handling (Indikatoren starten mit NaN) -> median from train
# ---------------------------------------------------
median_values = train_df[FEATURES].median(numeric_only=True)
train_df[FEATURES] = train_df[FEATURES].fillna(median_values)
val_df[FEATURES]   = val_df[FEATURES].fillna(median_values)
test_df[FEATURES]  = test_df[FEATURES].fillna(median_values)

X_train, y_train = train_df[FEATURES].values, train_df[TARGET].values
X_val,   y_val   = val_df[FEATURES].values,   val_df[TARGET].values
X_test,  y_test  = test_df[FEATURES].values,  test_df[TARGET].values

# ---------------------------------------------------
# 2) Baseline A: No-Feature Baseline (Majority Class)
# ---------------------------------------------------
print("\n===== BASELINE (NO FEATURES): Majority Class =====\n")
majority_class = int(pd.Series(y_train).mode().iloc[0])
print("Majority class (train):", majority_class)

y_train_base = np.full_like(y_train, fill_value=majority_class)
y_val_base   = np.full_like(y_val,   fill_value=majority_class)
y_test_base  = np.full_like(y_test,  fill_value=majority_class)

print("Train Accuracy (No-Feature Baseline):", round(accuracy_score(y_train, y_train_base), 3))
print("Val Accuracy   (No-Feature Baseline):", round(accuracy_score(y_val,   y_val_base),   3))
print("Test Accuracy  (No-Feature Baseline):", round(accuracy_score(y_test,  y_test_base),  3))

print("\nClassification Report (Validation, No-Feature Baseline):\n",
      classification_report(y_val, y_val_base, zero_division=0))

disp = ConfusionMatrixDisplay.from_predictions(y_val, y_val_base)
plt.title("No-Feature Baseline – Confusion Matrix (Validation, 1-min)")
plt.tight_layout()
cm_path = os.path.join(IMG_DIR, "baseline_no_features_confusion_matrix_val_1min.png")
plt.savefig(cm_path, dpi=300)
plt.close()
print("Saved no-feature baseline confusion matrix →", cm_path)

# ---------------------------------------------------
# 3) Baseline B: Logistic Regression (mit Features)
# ---------------------------------------------------
print("\n===== BASELINE MODEL: Logistic Regression (with features) =====\n")
logreg = LogisticRegression(max_iter=2000)
logreg.fit(X_train, y_train)

y_train_lr = logreg.predict(X_train)
y_val_lr   = logreg.predict(X_val)

print("Train Accuracy (LogReg):", round(accuracy_score(y_train, y_train_lr), 3))
print("Val Accuracy   (LogReg):", round(accuracy_score(y_val,   y_val_lr),   3))
print("\nClassification Report (Validation, LogReg):\n",
      classification_report(y_val, y_val_lr, zero_division=0))

disp = ConfusionMatrixDisplay.from_predictions(y_val, y_val_lr)
plt.title("Logistic Regression – Confusion Matrix (Validation, 1-min)")
plt.tight_layout()
lr_cm_path = os.path.join(IMG_DIR, "logreg_confusion_matrix_val_1min.png")
plt.savefig(lr_cm_path, dpi=300)
plt.close()
print("Saved logreg confusion matrix →", lr_cm_path)

logreg_path = os.path.join(MODEL_DIR, "tsla_logreg_1min.pkl")
joblib.dump(logreg, logreg_path)
print("Saved logreg model →", logreg_path)

# ---------------------------------------------------
# 4) Gradient Boosting – Parameter Grid
# ---------------------------------------------------
print("\n===== GRADIENT BOOSTING MODEL (1-min) =====\n")

param_grid = [
    {"n_estimators": 100, "learning_rate": 0.1,  "max_depth": 2},
    {"n_estimators": 200, "learning_rate": 0.1,  "max_depth": 2},
    {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3},
    {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3},
]

best_model = None
best_val_acc = -1
best_params = None

for params in param_grid:
    print(f"Testing params: {params}")

    gb = GradientBoostingClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        random_state=42,
    )

    gb.fit(X_train, y_train)

    val_pred = gb.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    print(f"Validation Accuracy: {val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = gb
        best_params = params

print("\n===== BEST GRADIENT BOOSTING MODEL (1-min) =====")
print("Best Params:", best_params)
print("Best Validation Accuracy:", round(best_val_acc, 3))

# ---------------------------------------------------
# 5) Finale Evaluation (Train, Val, Test)
# ---------------------------------------------------
y_train_pred = best_model.predict(X_train)
y_val_pred   = best_model.predict(X_val)
y_test_pred  = best_model.predict(X_test)

print("\nFinal GB Train Accuracy:", round(accuracy_score(y_train, y_train_pred), 3))
print("Final GB Val Accuracy:  ", round(accuracy_score(y_val,   y_val_pred),   3))
print("Final GB Test Accuracy: ", round(accuracy_score(y_test,  y_test_pred),  3))

print("\nClassification Report (Validation, GB):\n",
      classification_report(y_val, y_val_pred, zero_division=0))

# ---------------------------------------------------
# 6) Confusion Matrix (Val)
# ---------------------------------------------------
disp = ConfusionMatrixDisplay.from_predictions(y_val, y_val_pred)
plt.title("Gradient Boosting – Confusion Matrix (Validation, 1-min)")
plt.tight_layout()
cm_path = os.path.join(IMG_DIR, "gb_confusion_matrix_val_1min.png")
plt.savefig(cm_path, dpi=300)
plt.close()
print("Saved confusion matrix →", cm_path)

# ---------------------------------------------------
# 7) ROC Curve (Val)
# ---------------------------------------------------
y_val_prob = best_model.predict_proba(X_val)[:, 1]
fpr, tpr, _ = roc_curve(y_val, y_val_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], "k--")
plt.title("ROC Curve – Gradient Boosting (Validation, 1-min)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
roc_path = os.path.join(IMG_DIR, "gb_roc_curve_val_1min.png")
plt.savefig(roc_path, dpi=300)
plt.close()
print("Saved ROC curve →", roc_path)

# ---------------------------------------------------
# 8) Feature Importances
# ---------------------------------------------------
importances = best_model.feature_importances_
imp_series = pd.Series(importances, index=FEATURES).sort_values(ascending=False)

plt.figure(figsize=(8,6))
imp_series.head(15).plot(kind="bar")
plt.title("Top 15 Feature Importances – Gradient Boosting (1-min)")
plt.tight_layout()
fi_path = os.path.join(IMG_DIR, "gb_feature_importances_1min.png")
plt.savefig(fi_path, dpi=300)
plt.close()
print("Saved feature importances →", fi_path)

# ---------------------------------------------------
# 9) Modell speichern
# ---------------------------------------------------
model_path = os.path.join(MODEL_DIR, "tsla_gb_best_1min.pkl")
joblib.dump(best_model, model_path)
print("\nSaved model →", model_path)

print("\n✅ MODEL TRAINING COMPLETED (1-min).")
