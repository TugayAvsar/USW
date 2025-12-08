"""
04_split_data.py
----------------
Data Preparation **POST-SPLIT** für das TSLA-Modell.

Dieses Skript macht NICHT mehr neues Feature Engineering, sondern:
    1) Lädt den vorbereiteten Feature-Datensatz (tsla_features_30min.parquet)
    2) Trennt in Features (X) und Zielvariable (y)
    3) Führt einen ZEITBASIERTEN Train/Validation/Test-Split durch
    4) Skaliert die Features mit StandardScaler (nur auf Train gefittet)
    5) Speichert die vorbereiteten Arrays für das Modeling

Zielvariable:
    target = 1, wenn close[t+1] > close[t], sonst 0
"""

import os
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# ---------------------------------------------------------
# 0) Konfiguration & Pfade
# ---------------------------------------------------------

# Projekt-Config laden (DATA_PATH etc.)
params = yaml.safe_load(open("../../conf/params.yaml"))
DATA_PATH = params["DATA_ACQUISITION"]["DATA_PATH"]

FEATURE_PATH = os.path.join(DATA_PATH, "features")
PREPARED_PATH = os.path.join(DATA_PATH, "prepared")
os.makedirs(PREPARED_PATH, exist_ok=True)

# Train/Val/Test-Anteile (kannst du in der Präsi erwähnen)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15    # Rest geht automatisch in den Test-Split


# ---------------------------------------------------------
# 1) Feature-Datensatz laden
# ---------------------------------------------------------

feat_file = os.path.join(FEATURE_PATH, "tsla_features_30min.parquet")
df = pd.read_parquet(feat_file)

print(f"Loaded feature file: {feat_file}")
print(f"Shape: {df.shape}")

# Zeitspalte erkennen & sortieren (wichtig für Zeitreihen!)
time_col = None
for candidate in ["datetime", "date"]:
    if candidate in df.columns:
        time_col = candidate
        df[candidate] = pd.to_datetime(df[candidate])
        df = df.sort_values(candidate)
        print(f"Using time column for sorting: {candidate}")
        break

if time_col is None:
    print("⚠️ Keine explizite Zeitspalte gefunden – Reihenfolge wird wie gespeichert verwendet.")
else:
    print(f"Zeitspanne: {df[time_col].min()} → {df[time_col].max()}")


# ---------------------------------------------------------
# 2) X (Features) und y (Target) definieren
# ---------------------------------------------------------

if "target" not in df.columns:
    raise ValueError("Spalte 'target' nicht im Feature-Datensatz gefunden!")

# Spalten, die NICHT in X bleiben sollen
drop_cols = ["target"]

# Hilfsspalte aus 01er-Skript (falls noch vorhanden)
if "close_next" in df.columns:
    drop_cols.append("close_next")

# Zeitstempel auch entfernen (nicht als numerisches Feature verwenden)
if time_col is not None:
    drop_cols.append(time_col)

feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols].copy()
y = df["target"].astype(int).copy()

print("\nVerwendete Feature-Spalten (X):")
print(feature_cols)
print(f"\nX shape: {X.shape}, y shape: {y.shape}")


# ---------------------------------------------------------
# 3) Zeitbasierter Train/Validation/Test-Split
# ---------------------------------------------------------
# WICHTIG: KEIN Shuffle, Reihenfolge bleibt chronologisch.

n = len(df)
train_end = int(n * TRAIN_RATIO)
val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val = X.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]

X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]

print("\nSplit sizes:")
print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Val:   {X_val.shape}, {y_val.shape}")
print(f"Test:  {X_test.shape}, {y_test.shape}")

if time_col is not None:
    print("\nZeitbereiche pro Split:")
    print(f"Train: {df[time_col].iloc[0]}  → {df[time_col].iloc[train_end-1]}")
    print(f"Val:   {df[time_col].iloc[train_end]}  → {df[time_col].iloc[val_end-1]}")
    print(f"Test:  {df[time_col].iloc[val_end]}  → {df[time_col].iloc[-1]}")


# ---------------------------------------------------------
# 4) Skalierung der Features (StandardScaler)
# ---------------------------------------------------------
# Nur auf Trainingsdaten fitten → kein Data Leakage!

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Kleine Kontrolle: Beispiel-Features ausgeben
print("\n📋 Beispiel-Features (skaliertes X_train):")
print(pd.DataFrame(X_train_scaled, columns=feature_cols).head())

print("\n📋 Beispiel-Targets (y_train):")
print(y_train.head())


# ---------------------------------------------------------
# 5) Arrays & Scaler speichern
# ---------------------------------------------------------

np.save(os.path.join(PREPARED_PATH, "X_train.npy"), X_train_scaled)
np.save(os.path.join(PREPARED_PATH, "X_val.npy"), X_val_scaled)
np.save(os.path.join(PREPARED_PATH, "X_test.npy"), X_test_scaled)

np.save(os.path.join(PREPARED_PATH, "y_train.npy"), y_train.values)
np.save(os.path.join(PREPARED_PATH, "y_val.npy"), y_val.values)
np.save(os.path.join(PREPARED_PATH, "y_test.npy"), y_test.values)

scaler_path = os.path.join(PREPARED_PATH, "scaler_tsla_std.pkl")
joblib.dump(scaler, scaler_path)

print(f"\n✅ Data split & scaling finished.")
print(f"Prepared arrays saved to: {PREPARED_PATH}")
print(f"Scaler saved to: {scaler_path}")
