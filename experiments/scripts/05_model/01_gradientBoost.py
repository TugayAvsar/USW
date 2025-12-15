"""
01_GradientBoost.py
--------------------
Modeling Pipeline für TSLA:

Baselines + Modelle:
0) No-Feature Baseline – Majority Class (leak-free, nur Target-Verteilung aus Train)
1) Baseline – Logistic Regression
2) Gradient Boosting Classifier (mehrere Parameter-Kombinationen)

Speichert:
- Confusion Matrix (No-Feature Baseline)
- Bestes Gradient-Boosting Modell
- Confusion Matrix
- ROC Curve
- Feature Importances
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
DATA_DIR = "../../data/processed"
MODEL_DIR = "../../models"
IMG_DIR = "../../images"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(DATA_DIR, "tsla_train.parquet")
VAL_FILE = os.path.join(DATA_DIR, "tsla_val.parquet")
TEST_FILE = os.path.join(DATA_DIR, "tsla_test.parquet")

# ---------------------------------------------------
# 1) Daten laden
# ---------------------------------------------------
train_df = pd.read_parquet(TRAIN_FILE)
val_df = pd.read_parquet(VAL_FILE)
test_df = pd.read_parquet(TEST_FILE)

# Remove NaN values (Median nur aus Train -> leak-free)
median_values = train_df.median(numeric_only=True)
train_df = train_df.fillna(median_values)
val_df = val_df.fillna(median_values)
test_df = test_df.fillna(median_values)

TARGET = "target"
FEATURES = [c for c in train_df.columns if c not in ["datetime", TARGET]]

X_train, y_train = train_df[FEATURES].values, train_df[TARGET].values
X_val, y_val = val_df[FEATURES].values, val_df[TARGET].values
X_test, y_test = test_df[FEATURES].values, test_df[TARGET].values

print("Loaded data shapes:")
print("X_train:", X_train.shape, " y_train:", y_train.shape)
print("X_val:  ", X_val.shape, " y_val:", y_val.shape)
print("X_test: ", X_test.shape, " y_test:", y_test.shape)

# ---------------------------------------------------
# 1.5) BASELINE OHNE FEATURES (Majority-Class)
# ---------------------------------------------------
print("\n===== BASELINE (NO FEATURES): Majority Class =====\n")

# Majority class wird NUR aus Train bestimmt (leak-free)
majority_class = int(pd.Series(y_train).mode()[0])

# Vorhersage: immer die Mehrheitsklasse
y_train_base_nf = np.full(len(y_train), majority_class)
y_val_base_nf = np.full(len(y_val), majority_class)
y_test_base_nf = np.full(len(y_test), majority_class)

print(f"Majority class (train): {majority_class}")

train_acc_nf = accuracy_score(y_train, y_train_base_nf)
val_acc_nf = accuracy_score(y_val, y_val_base_nf)
test_acc_nf = accuracy_score(y_test, y_test_base_nf)

print("Train Accuracy (No-Feature Baseline):", round(train_acc_nf, 3))
print("Val Accuracy   (No-Feature Baseline):", round(val_acc_nf, 3))
print("Test Accuracy  (No-Feature Baseline):", round(test_acc_nf, 3))

print("\nClassification Report (Validation, No-Feature Baseline):\n",
      classification_report(y_val, y_val_base_nf))

# Confusion Matrix speichern (Validation)
disp = ConfusionMatrixDisplay.from_predictions(y_val, y_val_base_nf)
plt.title("No-Feature Baseline - Confusion Matrix (Validation)")
plt.tight_layout()
cm_path = os.path.join(IMG_DIR, "baseline_no_features_confusion_matrix_val.png")
plt.savefig(cm_path, dpi=300)
plt.close()
print("Saved no-feature baseline confusion matrix →", cm_path)

# ---------------------------------------------------
# 2) Baseline Logistic Regression (mit Features)
# ---------------------------------------------------
print("\n===== BASELINE MODEL: Logistic Regression (with features) =====\n")

baseline = LogisticRegression(max_iter=2000)
baseline.fit(X_train, y_train)

y_train_base = baseline.predict(X_train)
y_val_base = baseline.predict(X_val)

print("Train Accuracy (LogReg):", round(accuracy_score(y_train, y_train_base), 3))
print("Val Accuracy   (LogReg):", round(accuracy_score(y_val, y_val_base), 3))
print("\nClassification Report (Validation, LogReg):\n",
      classification_report(y_val, y_val_base))

# ---------------------------------------------------
# 3) Gradient Boosting – Parameter Grid
# ---------------------------------------------------
print("\n===== GRADIENT BOOSTING MODEL =====\n")

param_grid = [
    {
        "n_estimators": 80,
        "learning_rate": 0.05,
        "max_depth": 2,
        "subsample": 0.7,
        "min_samples_leaf": 50,
        "min_samples_split": 100,
    },
    {
        "n_estimators": 120,
        "learning_rate": 0.03,
        "max_depth": 2,
        "subsample": 0.7,
        "min_samples_leaf": 100,
        "min_samples_split": 200,
    },
    {
        "n_estimators": 150,
        "learning_rate": 0.03,
        "max_depth": 3,
        "subsample": 0.6,
        "min_samples_leaf": 150,
        "min_samples_split": 300,
    },
]


best_model = None
best_val_acc = -1
best_params = None

for params in param_grid:
    print(f"Testing params: {params}")

    gb = GradientBoostingClassifier(
        random_state=42,
        **params
    )

    gb.fit(X_train, y_train)

    val_pred = gb.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    print(f"Validation Accuracy: {val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = gb
        best_params = params

print("\n===== BEST GRADIENT BOOSTING MODEL =====")
print("Best Params:", best_params)
print("Best Validation Accuracy:", round(best_val_acc, 3))

# ---------------------------------------------------
# 4) Finale Evaluation (Train, Val, Test)
# ---------------------------------------------------
y_train_pred = best_model.predict(X_train)
y_val_pred = best_model.predict(X_val)
y_test_pred = best_model.predict(X_test)

print("\nFinal GB Train Accuracy:", round(accuracy_score(y_train, y_train_pred), 3))
print("Final GB Val Accuracy:  ", round(accuracy_score(y_val, y_val_pred), 3))
print("Final GB Test Accuracy: ", round(accuracy_score(y_test, y_test_pred), 3))

print("\nClassification Report (Validation, GB):\n",
      classification_report(y_val, y_val_pred))

# ---------------------------------------------------
# 5) Confusion Matrix (GB)
# ---------------------------------------------------
disp = ConfusionMatrixDisplay.from_predictions(y_val, y_val_pred)
plt.title("Gradient Boosting - Confusion Matrix (Validation)")
plt.tight_layout()
cm_path = os.path.join(IMG_DIR, "gb_confusion_matrix_val.png")
plt.savefig(cm_path, dpi=300)
plt.close()
print("Saved confusion matrix →", cm_path)

# ---------------------------------------------------
# 6) ROC Curve (GB)
# ---------------------------------------------------
y_val_prob = best_model.predict_proba(X_val)[:, 1]
fpr, tpr, _ = roc_curve(y_val, y_val_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--")
plt.title("ROC Curve – Gradient Boosting (Validation)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
roc_path = os.path.join(IMG_DIR, "gb_roc_curve_val.png")
plt.savefig(roc_path, dpi=300)
plt.close()
print("Saved ROC curve →", roc_path)

# ---------------------------------------------------
# 7) Feature Importances (GB)
# ---------------------------------------------------
importances = best_model.feature_importances_
imp_series = pd.Series(importances, index=FEATURES).sort_values(ascending=False)

plt.figure(figsize=(8, 6))
imp_series.head(15).plot(kind="bar")
plt.title("Top 15 Feature Importances – Gradient Boosting")
plt.tight_layout()
fi_path = os.path.join(IMG_DIR, "gb_feature_importances.png")
plt.savefig(fi_path, dpi=300)
plt.close()
print("Saved feature importances →", fi_path)

# ---------------------------------------------------
# 8) Modell speichern
# ---------------------------------------------------
model_path = os.path.join(MODEL_DIR, "tsla_gb_best.pkl")
joblib.dump(best_model, model_path)
print("\nSaved model →", model_path)

print("\n✅ MODEL TRAINING COMPLETED.")
