"""
03_compare_30min_vs_1min.py
---------------------------
Vergleicht Modell-Ergebnisse zwischen 30-min und 1-min Pipeline.

Liest die processed-files und trainiert die Modelle "quick" nochmal,
um Accuracy (Train/Val/Test) für beide Granularitäten tabellarisch
zu vergleichen.

Output:
- images/model_comparison_table.png (optional)
- data/processed/model_comparison_30min_vs_1min.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

BASE_DATA = "../../data/processed"
OUT_CSV = os.path.join(BASE_DATA, "model_comparison_30min_vs_1min.csv")
IMG_DIR = "../../images"
os.makedirs(IMG_DIR, exist_ok=True)

def load_split(gran: str):
    """gran: '30min' oder '1min'"""
    if gran == "30min":
        train = pd.read_parquet(os.path.join(BASE_DATA, "tsla_train.parquet"))
        val   = pd.read_parquet(os.path.join(BASE_DATA, "tsla_val.parquet"))
        test  = pd.read_parquet(os.path.join(BASE_DATA, "tsla_test.parquet"))
    elif gran == "1min":
        train = pd.read_parquet(os.path.join(BASE_DATA, "tsla_train_1min.parquet"))
        val   = pd.read_parquet(os.path.join(BASE_DATA, "tsla_val_1min.parquet"))
        test  = pd.read_parquet(os.path.join(BASE_DATA, "tsla_test_1min.parquet"))
    else:
        raise ValueError("gran must be '30min' or '1min'")
    return train, val, test

def prep_xy(df):
    TARGET = "target"
    FEATURES = [c for c in df.columns if c not in ["datetime", TARGET]]
    # NaNs -> median per df (wir nehmen train median später)
    return FEATURES

def eval_models(gran: str):
    train_df, val_df, test_df = load_split(gran)
    TARGET = "target"
    FEATURES = [c for c in train_df.columns if c not in ["datetime", TARGET]]

    # NaNs -> median train
    med = train_df[FEATURES].median(numeric_only=True)
    for d in (train_df, val_df, test_df):
        d[FEATURES] = d[FEATURES].fillna(med)

    X_train, y_train = train_df[FEATURES].values, train_df[TARGET].values
    X_val,   y_val   = val_df[FEATURES].values,   val_df[TARGET].values
    X_test,  y_test  = test_df[FEATURES].values,  test_df[TARGET].values

    results = []

    # A) No-feature baseline
    maj = int(pd.Series(y_train).mode().iloc[0])
    for split_name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        y_pred = np.full_like(y, fill_value=maj)
        acc = accuracy_score(y, y_pred)
        results.append((gran, "Baseline(NoFeatures)", split_name, acc))

    # B) Logistic Regression
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_train, y_train)
    for split_name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        acc = accuracy_score(y, lr.predict(X))
        results.append((gran, "LogReg", split_name, acc))

    # C) Gradient Boosting (fix params wie best guess)
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=2, random_state=42)
    gb.fit(X_train, y_train)
    for split_name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        acc = accuracy_score(y, gb.predict(X))
        results.append((gran, "GradBoost", split_name, acc))

    # D) XGBoost (optional)
    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=-1
        )
        xgb.fit(X_train, y_train)
        for split_name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
            acc = accuracy_score(y, xgb.predict(X))
            results.append((gran, "XGBoost", split_name, acc))
    else:
        results.append((gran, "XGBoost", "val", np.nan))

    return results

all_rows = []
for gran in ["30min", "1min"]:
    all_rows.extend(eval_models(gran))

res = pd.DataFrame(all_rows, columns=["granularity", "model", "split", "accuracy"])
res.to_csv(OUT_CSV, index=False)
print("✅ Saved comparison CSV →", OUT_CSV)

# Pivot für schnelle Übersicht
pivot = res.pivot_table(index=["model", "split"], columns="granularity", values="accuracy")
print("\n=== Comparison (Accuracy) ===")
print(pivot)

# Optional: simple plot (bar-like table visualization)
# (als Bild ist nice für README)
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis("off")
tbl = ax.table(
    cellText=np.round(pivot.values, 3),
    rowLabels=[f"{m} | {s}" for m, s in pivot.index],
    colLabels=list(pivot.columns),
    loc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.3)

img_path = os.path.join(IMG_DIR, "model_comparison_30min_vs_1min.png")
plt.tight_layout()
plt.savefig(img_path, dpi=300)
plt.close()
print("✅ Saved comparison image →", img_path)
