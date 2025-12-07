"""
03_data_preparation_pre_split.py
---------------------------------
Feature Engineering (vor Train/Test-Split)
für Tesla (Hauptsymbol) mit Einflussparametern von Ford & GM.

Zielvariable:
    target = 1, wenn Tesla in der nächsten 30-Minuten-Periode steigt.
"""

import pandas as pd
import numpy as np
import yaml, os

# ---------------- Config laden ----------------
params = yaml.safe_load(open("../../conf/params.yaml"))
DATA_PATH = params["DATA_ACQUISITION"]["DATA_PATH"]
SYMBOLS = params["DATA_ACQUISITION"].get("SYMBOLS", ["TSLA", "F", "GM"])
RAW_PATH = os.path.join(DATA_PATH, "raw")
FEAT_PATH = os.path.join(DATA_PATH, "features")
os.makedirs(FEAT_PATH, exist_ok=True)

# ---------------- Helper-Funktionen ----------------
def ema(series, span): return series.ewm(span=span, adjust=False).mean()
def rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ema = pd.Series(gain, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    loss_ema = pd.Series(loss, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = gain_ema / loss_ema
    return 100 - (100 / (1 + rs))

# ---------------- Hauptdaten (Tesla) ----------------
main_symbol = "TSLA"
tsla = pd.read_parquet(os.path.join(RAW_PATH, f"{main_symbol.lower()}_30min.parquet"))
tsla["datetime"] = pd.to_datetime(tsla["datetime"])
tsla = tsla.set_index("datetime").sort_index()

# ---------------- Vergleichsaktien laden ----------------
comparisons = {}
for sym in SYMBOLS:
    if sym != main_symbol:
        path = os.path.join(RAW_PATH, f"{sym.lower()}_30min.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime").sort_index()
            comparisons[sym] = df
            print(f"[OK] {sym} geladen ({len(df)} Zeilen).")

# ---------------- Features: Tesla ----------------
df = tsla.copy()
df["return"] = df["close"].pct_change()
df["ema_5"] = ema(df["close"], 5)
df["ema_10"] = ema(df["close"], 10)
df["rsi_14"] = rsi(df["close"], 14)
df["vwap_diff"] = (df["close"] - df["vwap"]) / df["vwap"]
df["volume_change"] = df["volume"].pct_change()

# ---------------- Cross-Stock-Features ----------------
for sym, comp in comparisons.items():
    comp_ret = comp["close"].pct_change()
    df[f"corr_{sym.lower()}"] = df["return"].rolling(12).corr(comp_ret)
    df[f"spread_{sym.lower()}"] = (df["close"] - comp["close"]) / comp["close"]

# ---------------- Zielvariable ----------------
df["target"] = (df["close"].shift(-1) > df["close"]).astype("Int8")
df = df.dropna().copy()

# ---------------- Speichern ----------------
OUT_PATH = os.path.join(FEAT_PATH, f"tsla_features_30min.parquet")
df.to_parquet(OUT_PATH)
print(f"\n✅ Features gespeichert: {OUT_PATH}")

# ---------------- Beispiele & Statistik ----------------
print("\n📋 Beispiel-Daten:")
print(df.head(5)[["close", "return", "ema_5", "ema_10", "rsi_14", "corr_f", "corr_gm", "target"]])

print("\n📊 Deskriptive Statistik (Auszug):")
print(df[["return", "ema_5", "ema_10", "rsi_14", "vwap_diff", "volume_change"]].describe().round(3))

# ---------------- Visualisierung ----------------
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,4))
sns.histplot(df["rsi_14"], bins=50, kde=True)
plt.title("Verteilung RSI (TSLA)")
plt.tight_layout()
plt.savefig(os.path.join(FEAT_PATH, "prep_rsi_dist.png"))
plt.close()

plt.figure(figsize=(6,4))
sns.heatmap(df.corr(numeric_only=True)[["target"]].sort_values("target", ascending=False), annot=True, cmap="coolwarm")
plt.title("Feature-Korrelation mit Target (TSLA)")
plt.tight_layout()
plt.savefig(os.path.join(FEAT_PATH, "prep_feature_corr.png"))
plt.close()

print("\n✅ Data Preparation abgeschlossen – Features & Plots erstellt.")