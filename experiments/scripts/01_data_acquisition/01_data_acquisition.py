# Python
"""
Dieses Skript lädt historische **Daily**-Bar-Daten für **Tesla (TSLA)** über die Alpaca Market Data API,
berechnet ein Ziel-Label (Up/Down für den nächsten Handelstag) sowie einfache technische Indikatoren
(SMA, EMA, RSI, %Change), und speichert getrennt **Rohdaten** und **Feature-Daten**.

Problemfrage:
- Binary Classification: Steigt der Schlusskurs morgen? target=1 wenn Close[t+1] > Close[t], sonst 0.

Inputs:
- YAML-Configs:
    ../../conf/keys.yaml        (API Keys)
    ../../conf/params.yaml      (Datenpfad, Datumsspanne, Symbol)
- (optional) Verzeichnis-Struktur wird bei Bedarf angelegt

Outputs:
- <DATA_PATH>/raw/tsla_daily.parquet        (Rohdaten: OHLCV)
- <DATA_PATH>/raw/tsla_daily.csv
- <DATA_PATH>/features/tsla_features.parquet (Features + target)
- <DATA_PATH>/features/tsla_features.csv

Requirements (pip):
- alpaca-py, pandas, numpy, pyyaml, pyarrow
"""

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment
import pandas as pd
import numpy as np
from datetime import datetime
import yaml
import os

# ---------- Helper: simple technical indicators ----------
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain_ema = pd.Series(gain, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    loss_ema = pd.Series(loss, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = gain_ema / (loss_ema.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

# ---------- Load configs ----------
keys = yaml.safe_load(open("../../conf/keys.yaml"))
API_KEY = keys["KEYS"]["APCA-API-KEY-ID-Data"]
SECRET_KEY = keys["KEYS"]["APCA-API-SECRET-KEY-Data"]

params = yaml.safe_load(open("../../conf/params.yaml"))
DATA_PATH   = params["DATA_ACQUISITION"]["DATA_PATH"]      # z.B. "../../data"
START_DATE  = datetime.strptime(params["DATA_ACQUISITION"]["START_DATE"], "%Y-%m-%d")
END_DATE    = datetime.strptime(params["DATA_ACQUISITION"]["END_DATE"], "%Y-%m-%d")
SYMBOL      = params["DATA_ACQUISITION"].get("SYMBOL", "TSLA")

# ---------- Prepare output dirs ----------
raw_dir = os.path.join(DATA_PATH, "raw")
feat_dir = os.path.join(DATA_PATH, "features")
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(feat_dir, exist_ok=True)

# ---------- Init client ----------
client = StockHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)

# ---------- Request daily bars ----------
print(f"Fetching Daily bars for {SYMBOL} from {START_DATE.date()} to {END_DATE.date()}...")
req = StockBarsRequest(
    symbol_or_symbols=SYMBOL,
    timeframe=TimeFrame.Day,
    adjustment=Adjustment.ALL,   # Splits/Dividenden adjustiert
    start=START_DATE,
    end=END_DATE
)

bars = client.get_stock_bars(req)
df = bars.df.reset_index()

# Wenn Multi-Symbol-Index vorhanden, 'symbol' entfernen
if "symbol" in df.columns:
    df = df.drop(columns=["symbol"])

# Einheitliche Spaltennamen
df = df.rename(columns={
    "timestamp": "date",
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "trade_count": "trade_count",
    "vwap": "vwap"
})

# Sortierung & Datentypen
df = df.sort_values("date").reset_index(drop=True)
df["date"] = pd.to_datetime(df["date"])

# ---------- Save RAW ----------
raw_parquet = os.path.join(raw_dir, "tsla_daily.parquet")
raw_csv     = os.path.join(raw_dir, "tsla_daily.csv")
df.to_parquet(raw_parquet, index=False)
df.to_csv(raw_csv, index=False)
print(f"Saved RAW to:\n  {raw_parquet}\n  {raw_csv}")

# ---------- Feature Engineering ----------
feat = df.copy()

# Prozentuale Veränderung Close (heute vs. gestern)
feat["close_pct_change"] = feat["close"].pct_change()

# Trendindikatoren
feat["sma_5"]  = sma(feat["close"], 5)
feat["sma_10"] = sma(feat["close"], 10)
feat["ema_5"]  = ema(feat["close"], 5)
feat["ema_10"] = ema(feat["close"], 10)

# Momentum
feat["rsi_14"] = rsi(feat["close"], 14)

# Target: steigt der nächste Schlusskurs?
feat["close_next"] = feat["close"].shift(-1)
feat["target"] = (feat["close_next"] > feat["close"]).astype("Int64")

# Letzte Zeile hat kein next Close -> entfernen
feat = feat.dropna(subset=["target"]).reset_index(drop=True)

# Auswahl & Reihenfolge
cols = [
    "date", "open", "high", "low", "close", "volume",
    "close_pct_change", "sma_5", "sma_10", "ema_5", "ema_10", "rsi_14",
    "target"
]
feat = feat[cols]

# ---------- Save FEATURES ----------
feat_parquet = os.path.join(feat_dir, "tsla_features.parquet")
feat_csv     = os.path.join(feat_dir, "tsla_features.csv")
feat.to_parquet(feat_parquet, index=False)
feat.to_csv(feat_csv, index=False)
print(f"Saved FEATURES to:\n  {feat_parquet}\n  {feat_csv}")

# ---------- Quick preview ----------
print("\nRAW head():")
print(df.head(3))
print("\nFEATURES head():")
print(feat.head(3))
print("\nLabel balance (target):")
print(feat["target"].value_counts(dropna=False))