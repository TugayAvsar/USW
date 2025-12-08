"""
04_split_data.py
----------------
Post-Split Data Preparation für TSLA (30-Minuten-Features).

- Lädt tsla_features_30min.parquet
- Definiert Train / Validation / Test per Datum
- Skaliert Features mit StandardScaler (nur auf Train gefittet)
- Speichert:
    data/processed/tsla_train.parquet
    data/processed/tsla_val.parquet
    data/processed/tsla_test.parquet
"""

import os
import pandas as pd
import numpy as np
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


df = df.sort_values("datetime").reset_index(drop=True)

print("✅ Geladene Daten:")
print(df.head(5))
print("\nZeitraum:", df["datetime"].min(), "→", df["datetime"].max())
print("Spalten:", list(df.columns))

# ---------------------------------------------------
# 2) Train / Val / Test Splits per Datum
#    -> kannst du bei Bedarf anpassen
# ---------------------------------------------------
TRAIN_END = pd.Timestamp("2024-12-31", tz="UTC")
VAL_END   = pd.Timestamp("2025-06-30",  tz="UTC")


train_mask = df["datetime"] <= TRAIN_END
val_mask   = (df["datetime"] > TRAIN_END) & (df["datetime"] <= VAL_END)
test_mask  = df["datetime"] > VAL_END

train_df = df[train_mask].copy()
val_df   = df[val_mask].copy()
test_df  = df[test_mask].copy()

print(f"\nSplit sizes:")
print(f"Train: {len(train_df)}")
print(f"Val:   {len(val_df)}")
print(f"Test:  {len(test_df)}")

# ---------------------------------------------------
# 3) Features & Target definieren
# ---------------------------------------------------
TARGET_COL = "target"

feature_cols = [c for c in df.columns
                if c not in [TARGET_COL, "datetime"]]

print("\nVerwendete Feature-Spalten:")
print(feature_cols)

X_train = train_df[feature_cols].values
y_train = train_df[TARGET_COL].values

X_val = val_df[feature_cols].values
y_val = val_df[TARGET_COL].values

X_test = test_df[feature_cols].values
y_test = test_df[TARGET_COL].values

# ---------------------------------------------------
# 4) Skalierung
# ---------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# Zur Kontrolle wieder in DataFrames packen
train_df_scaled = train_df.copy()
val_df_scaled   = val_df.copy()
test_df_scaled  = test_df.copy()

train_df_scaled[feature_cols] = X_train_scaled
val_df_scaled[feature_cols]   = X_val_scaled
test_df_scaled[feature_cols]  = X_test_scaled

# ---------------------------------------------------
# 5) Speichern
# ---------------------------------------------------
train_path = os.path.join(PROC_DIR, "tsla_train.parquet")
val_path   = os.path.join(PROC_DIR, "tsla_val.parquet")
test_path  = os.path.join(PROC_DIR, "tsla_test.parquet")

train_df_scaled.to_parquet(train_path, index=False)
val_df_scaled.to_parquet(val_path, index=False)
test_df_scaled.to_parquet(test_path, index=False)

print(f"\n💾 Saved: {train_path}")
print(f"💾 Saved: {val_path}")
print(f"💾 Saved: {test_path}")

print("\nBeispiel Train-Daten (Features + Target):")
print(train_df_scaled.head(5)[feature_cols + [TARGET_COL]])

print("\n✅ Post-Split Data Preparation abgeschlossen.")