"""
03_data_preparation_pre_split.py
---------------------------------
Feature Engineering (vor Train/Test-Split)
für Tesla (Hauptsymbol TSLA) mit Einflussparametern von Ford & GM.

WICHTIG:
- Dieses Skript baut auf den bereits in 01_data_acquisition erzeugten
  Features auf:
    experiments/data/features/tsla_features_30min.parquet

- In dieser Datei sind bereits enthalten:
    * 30-Minuten-Bars (open, high, low, close, volume, vwap, ...)
    * technische Basis-Features (close_pct_change, sma_5, ema_5, rsi_14, ...)
    * Zielvariable target (Up/Down nächste 30-Minuten-Periode)
    * News-Features (nur TSLA):
        - news_count
        - avg_headline_len
        - avg_summary_len
        - avg_sentiment (VADER)

Ziel dieses Skripts:
    1) TSLA-Feature-Datensatz laden (inkl. News-Features)
    2) Zusätzliche Features berechnen (return, vwap_diff, volume_change)
    3) Cross-Stock-Features mit F & GM ergänzen:
         - rollende Korrelation (corr_f, corr_gm)
         - relative Spreads (spread_f, spread_gm)
    4) Zielvariable target konsistent definieren
    5) Übersichtliche Statistiken und Korrelationen ausgeben
    6) Den finalen Feature-Datensatz für 04_split_data.py speichern

Zielvariable:
    target = 1, wenn close[t+1] > close[t], sonst 0
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ---------------- Config laden ----------------
BASE_DIR = Path(__file__).resolve().parents[2]  # experiments/scripts/01_data_acquisition
CONF_DIR = BASE_DIR / "conf"

params = yaml.safe_load(open(CONF_DIR / "params.yaml"))
DATA_PATH = params["DATA_ACQUISITION"]["DATA_PATH"]
SYMBOLS = params["DATA_ACQUISITION"].get("SYMBOLS", ["TSLA", "F", "GM"])

RAW_PATH  = os.path.join(DATA_PATH, "raw")      # 30-Minuten-RAW-Daten
FEAT_PATH = os.path.join(DATA_PATH, "features") # Feature-Datensätze
os.makedirs(FEAT_PATH, exist_ok=True)

# ---------------- Helper-Funktionen ----------------
def ema(series, span):
    """Exponentiell gewichteter gleitender Durchschnitt."""
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    """Standard-RSI-Berechnung."""
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ema = pd.Series(gain, index=series.index).ewm(alpha=1/period,
                                                       adjust=False).mean()
    loss_ema = pd.Series(loss, index=series.index).ewm(alpha=1/period,
                                                       adjust=False).mean()
    rs = gain_ema / loss_ema.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------- Hauptdaten (Tesla) laden ----------------
main_symbol = "TSLA"

# WICHTIG: Wir laden HIER den bereits vorbereiteten TSLA-Features-Datensatz
# aus 01_data_acquisition (inklusive News-Features etc.).
tsla_feat_path = os.path.join(FEAT_PATH, f"{main_symbol.lower()}_features_30min.parquet")
tsla = pd.read_parquet(tsla_feat_path)

# Zeitspalte vereinheitlichen & Index setzen
from pandas.api.types import is_datetime64_any_dtype as is_datetime

if "datetime" in tsla.columns:
    tsla["datetime"] = pd.to_datetime(tsla["datetime"])
    tsla = tsla.set_index("datetime").sort_index()
elif "date" in tsla.columns:
    tsla["date"] = pd.to_datetime(tsla["date"])
    tsla = tsla.set_index("date").sort_index()
else:
    # Fall: Zeitstempel steckt schon im Index (z. B. aus 01er Skript)
    if is_datetime(tsla.index):
        tsla = tsla.sort_index()
        print("[INFO] Keine 'datetime'/'date'-Spalte, aber DatetimeIndex gefunden – Index wird verwendet.")
    else:
        print("Spalten im TSLA-Features-Datensatz:")
        print(tsla.columns)
        raise ValueError("TSLA-Features enthalten weder 'datetime'/'date' Spalte noch einen DatetimeIndex!")


# ---------------- Vergleichsaktien F & GM laden (RAW 30min) ----------------
comparisons = {}
for sym in SYMBOLS:
    if sym != main_symbol:
        path = os.path.join(RAW_PATH, f"{sym.lower()}_30min.parquet")
        if os.path.exists(path):
            df_c = pd.read_parquet(path)
            # Zeitspalte vereinheitlichen
            if "datetime" in df_c.columns:
                df_c["datetime"] = pd.to_datetime(df_c["datetime"])
                df_c = df_c.set_index("datetime").sort_index()
            elif "date" in df_c.columns:
                df_c["date"] = pd.to_datetime(df_c["date"])
                df_c = df_c.set_index("date").sort_index()
            else:
                raise ValueError(f"{sym}: Keine Zeitspalte gefunden!")

            comparisons[sym] = df_c
            print(f"[OK] {sym} RAW 30min geladen ({len(df_c)} Zeilen).")
        else:
            print(f"[WARN] Kein RAW-30min-File für {sym} gefunden unter {path}.")


# ---------------- Features: Tesla erweitern ----------------
df = tsla.copy()

# 1) Returns
if "close_pct_change" in df.columns:
    df["return"] = df["close_pct_change"]
else:
    df["return"] = df["close"].pct_change()

# 2) Trend & Momentum
df["ema_5"] = ema(df["close"], 5)
df["ema_10"] = ema(df["close"], 10)
df["ema_diff"] = df["ema_5"] - df["ema_10"]
df["ema_diff_change"] = df["ema_diff"].diff()

df["rsi_14"] = rsi(df["close"], 14)
df["rsi_change"] = df["rsi_14"].diff()

# 3) Volatilität
df["vol_6"] = df["return"].rolling(6).std()
df["vol_12"] = df["return"].rolling(12).std()
df["vol_ratio"] = df["vol_6"] / df["vol_12"]

# 4) Preisbeschleunigung
df["return_acc"] = df["return"].diff()

# 5) Volumen & VWAP
df["vwap_diff"] = (df["close"] - df["vwap"]) / df["vwap"]
df["volume_change"] = df["volume"].pct_change()

# 6) News-Dynamik (falls vorhanden)
if "avg_sentiment" in df.columns:
    df["news_sentiment_change"] = df["avg_sentiment"].diff()
    df["news_active"] = (df["news_count"] > 0).astype(int)


# ---------------- Cross-Stock-Features (F & GM) ----------------
for sym, comp in comparisons.items():
    # sicherstellen, dass Indizes kompatibel sind (gleiche Zeitzone / Timestamps)
    comp_aligned = comp.reindex(df.index)  # auf TSLA-Zeitstempel ausrichten
    comp_ret = comp_aligned["close"].pct_change()

    # rollende 30-Min-Korrelation (~ 6 * 5min, hier 12 * 30min ???)
    # Hier: Rolling-Fenster 12 * 30min = 6 Stunden
    df[f"corr_{sym.lower()}"] = df["return"].rolling(12).corr(comp_ret)

    # relativer Preisabstand
    df[f"spread_{sym.lower()}"] = (df["close"] - comp_aligned["close"]) / comp_aligned["close"]

print("\n[INFO] Cross-Stock-Features hinzugefügt (corr_*, spread_*).")


# ---------------- Zielvariable (target) konsistent setzen ----------------
# Falls aus 01 bereits target vorhanden ist, wird es hier überschrieben (gleiches Konzept).
df["target"] = (df["close"].shift(-1) > df["close"]).astype("Int8")

# Letzte Zeilen mit NaNs entfernen (durch shift, rolling etc.)
df = df.dropna().copy()

print(f"\n[INFO] Finaler Feature-Datensatz: {df.shape[0]} Zeilen, {df.shape[1]} Spalten.")


# ---------------- Speichern ----------------
OUT_PATH = os.path.join(FEAT_PATH, f"{main_symbol.lower()}_features_30min.parquet")
df.to_parquet(OUT_PATH)
print(f"\n✅ Features gespeichert (inkl. News & Cross-Stock): {OUT_PATH}")

# ---------------- Beispiele & Statistik ----------------
print("\n📋 Beispiel-Daten (Auszug):")
example_cols = [
    "close", "return", "ema_5", "ema_10", "rsi_14",
]

# Cross-Stock-Spalten (falls vorhanden) anhängen
for sym in comparisons.keys():
    example_cols.append(f"corr_{sym.lower()}")
    example_cols.append(f"spread_{sym.lower()}")

# News-Features (falls vorhanden) anhängen
for c in ["news_count", "avg_headline_len", "avg_summary_len", "avg_sentiment"]:
    if c in df.columns:
        example_cols.append(c)

example_cols.append("target")

print(df[example_cols].head(5))

print("\n📊 Deskriptive Statistik (Auszug technischer Features):")
print(
    df[["return", "ema_5", "ema_10", "rsi_14", "vwap_diff", "volume_change"]]
    .describe()
    .round(3)
)

if "avg_sentiment" in df.columns:
    print("\n📊 Deskriptive Statistik (News-Sentiment):")
    print(df[["news_count", "avg_sentiment"]].describe().round(3))

# ---------------- Visualisierung ----------------

# Output-Folder für Plots
IMG_PATH = FEAT_PATH  # direkt im features-Ordner speichern
os.makedirs(IMG_PATH, exist_ok=True)

# 1) RSI-Verteilung
plt.figure(figsize=(6, 4))
sns.histplot(df["rsi_14"], bins=50, kde=True)
plt.title("Verteilung RSI (TSLA)")
plt.tight_layout()
plt.savefig(os.path.join(IMG_PATH, "prep_rsi_dist.png"))
plt.close()

# 2) Korrelation aller numerischen Features mit dem Target
plt.figure(figsize=(6, max(6, len(df.columns) * 0.25)))
corr_target = df.corr(numeric_only=True)[["target"]].sort_values("target", ascending=False)
sns.heatmap(corr_target, annot=True, cmap="coolwarm")
plt.title("Feature-Korrelation mit Target (TSLA)")
plt.tight_layout()
plt.savefig(os.path.join(IMG_PATH, "prep_feature_corr.png"))
plt.close()

print("\n✅ Data Preparation (pre-split) abgeschlossen – Features & Plots erstellt.")

