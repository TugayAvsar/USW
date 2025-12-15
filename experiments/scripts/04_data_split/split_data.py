"""
04_split_data.py (Leak-Free Version)
------------------------------------
Post-Split Data Preparation für TSLA (30-Minuten-Features).

WICHTIGE ÄNDERUNGEN:
- 'close_next' wird entfernt (Data Leakage Fix!)
- Features und Target sind strikt getrennt
- StandardScaler wird NUR auf Train gefittet
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------
# Pfade
# ---------------------------------------------------
BASE_DATA = "../../data"
FEAT_FILE = os.path.join(BASE_DATA, "features", "tsla_features_30min.parquet")
PROC_DIR = os.path.join(BASE_DATA, "processed")
os.makedirs(PROC_DIR, exist_ok=True)

# ---------------------------------------------------
# 1) Daten laden
# ---------------------------------------------------
df = pd.read_parquet(FEAT_FILE)

# sicherstellen, dass datetime-Spalte existiert
if "datetime" in df.columns:
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
else:
    df = df.reset_index().rename(columns={"index": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

# 🧨 LEAK FIX: close_next entfernen
if "close_next" in df.columns:
    print("⚠️ Entferne 'close_next' (Leak-Fix)")
    df = df.drop(columns=["close_next"])

df = df.sort_values("datetime").reset_index(drop=True)

print("\n✅ Geladene Daten (nach Leak-Fix):")
print(df.head())

print("\nZeitraum:", df["datetime"].min(), "→", df["datetime"].max())
print("Spalten:", df.columns.tolist())

# ---------------------------------------------------
# 2) Zeitbasierter Split
# ---------------------------------------------------
TRAIN_END = pd.Timestamp("2023-12-31", tz="UTC")
VAL_END   = pd.Timestamp("2024-12-31", tz="UTC")

train_mask = df["datetime"] <= TRAIN_END
val_mask   = (df["datetime"] > TRAIN_END) & (df["datetime"] <= VAL_END)
test_mask  = df["datetime"] > VAL_END

train_df = df[train_mask].copy()
val_df   = df[val_mask].copy()
test_df  = df[test_mask].copy()

print("\nSplit sizes:")
print(f"Train: {len(train_df)}")
print(f"Val:   {len(val_df)}")
print(f"Test:  {len(test_df)}")

# ---------------------------------------------------
# 3) Features & Target definieren
# ---------------------------------------------------
TARGET_COL = "target"

# 🧹 Entferne Target + Zeitspalte
feature_cols = [
    c for c in df.columns
    if c not in ["datetime", TARGET_COL]
]

print("\nFeature-Spalten:")
print(feature_cols)

X_train = train_df[feature_cols].values
y_train = train_df[TARGET_COL].values

X_val = val_df[feature_cols].values
y_val = val_df[TARGET_COL].values

X_test = test_df[feature_cols].values
y_test = test_df[TARGET_COL].values

# ---------------------------------------------------
# 4) Skalierung (nur TRAIN fit!)
# ---------------------------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# ---------------------------------------------------
# 5) Zurückschreiben als DataFrame
# ---------------------------------------------------
train_df_scaled = train_df.copy()
val_df_scaled   = val_df.copy()
test_df_scaled  = test_df.copy()

train_df_scaled[feature_cols] = X_train_scaled
val_df_scaled[feature_cols]   = X_val_scaled
test_df_scaled[feature_cols]  = X_test_scaled

# ---------------------------------------------------
# 6) Speichern
# ---------------------------------------------------
train_path = os.path.join(PROC_DIR, "tsla_train.parquet")
val_path   = os.path.join(PROC_DIR, "tsla_val.parquet")
test_path  = os.path.join(PROC_DIR, "tsla_test.parquet")

train_df_scaled.to_parquet(train_path, index=False)
val_df_scaled.to_parquet(val_path, index=False)
test_df_scaled.to_parquet(test_path, index=False)

print(f"\n💾 Saved Train → {train_path}")
print(f"💾 Saved Val   → {val_path}")
print(f"💾 Saved Test  → {test_path}")

print("\nBeispiel Daten:")
print(train_df_scaled.head()[feature_cols + [TARGET_COL]])

print("\n🚀 Split + Scaling abgeschlossen (Leak-Free).")