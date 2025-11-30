"""
Dieses Skript lädt historische **30-Minuten-Bar-Daten** für mehrere Aktien (z. B. Tesla, Ford, GM)
über die Alpaca Market Data API. Für jede Aktie werden Rohdaten und einfache Feature-Daten
inkl. Zielvariable (Up/Down für die nächste 30-Minuten-Periode) erzeugt und gespeichert.

Zielvariable:
    target = 1, wenn close[t+1] > close[t], sonst 0

Inputs:
    ../../conf/keys.yaml   -> API Keys
    ../../conf/params.yaml -> Datenpfad, Zeitraum, Symbol-Liste

Outputs pro Symbol:
    <DATA_PATH>/raw/<symbol>_30min.parquet
    <DATA_PATH>/raw/<symbol>_30min.csv
    <DATA_PATH>/features/<symbol>_features_30min.parquet
    <DATA_PATH>/features/<symbol>_features_30min.csv
"""

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import Adjustment
import pandas as pd
import numpy as np
from datetime import datetime
import yaml, os

# ---------- Helper functions ----------
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
DATA_PATH   = params["DATA_ACQUISITION"]["DATA_PATH"]
START_DATE  = datetime.strptime(params["DATA_ACQUISITION"]["START_DATE"], "%Y-%m-%d")
END_DATE    = datetime.strptime(params["DATA_ACQUISITION"]["END_DATE"], "%Y-%m-%d")
SYMBOLS     = params["DATA_ACQUISITION"].get("SYMBOLS", ["TSLA"])

# ---------- Prepare output dirs ----------
raw_dir = os.path.join(DATA_PATH, "raw")
feat_dir = os.path.join(DATA_PATH, "features")
os.makedirs(raw_dir, exist_ok=True)
os.makedirs(feat_dir, exist_ok=True)

# ---------- Init Alpaca client ----------
client = StockHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)

# ---------- Loop über alle Symbole ----------
for symbol in SYMBOLS   :
    print(f"\nFetching 30-MINUTE bars for {symbol} from {START_DATE.date()} to {END_DATE.date()}...")

    # Hier der entscheidende Unterschied:
    # TimeFrame(unit=TimeFrameUnit.Minute, amount=30) liefert 30-Minuten-Balken
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(amount=30, unit=TimeFrameUnit.Minute),  # 👈 30-Minuten-Auflösung
        adjustment=Adjustment.ALL,
        start=START_DATE,
        end=END_DATE
    )

    bars = client.get_stock_bars(req)
    df = bars.df.reset_index()

    if "symbol" in df.columns:
        df = df.drop(columns=["symbol"])

    df = df.rename(columns={
        "timestamp": "datetime",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "trade_count": "trade_count",
        "vwap": "vwap"
    })
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # ---------- Save RAW ----------
    raw_parquet = os.path.join(raw_dir, f"{symbol.lower()}_30min.parquet")
    raw_csv     = os.path.join(raw_dir, f"{symbol.lower()}_30min.csv")
    df.to_parquet(raw_parquet, index=False)
    df.to_csv(raw_csv, index=False)
    print(f"Saved RAW → {raw_parquet}")

    # ---------- Feature Engineering ----------
    feat = df.copy()
    feat["close_pct_change"] = feat["close"].pct_change()
    feat["sma_5"]  = sma(feat["close"], 5)
    feat["sma_10"] = sma(feat["close"], 10)
    feat["ema_5"]  = ema(feat["close"], 5)
    feat["ema_10"] = ema(feat["close"], 10)
    feat["rsi_14"] = rsi(feat["close"], 14)

    # Target: nächste 30-Minuten-Periode
    feat["close_next"] = feat["close"].shift(-1)
    feat["target"] = (feat["close_next"] > feat["close"]).astype("Int64")
    feat = feat.dropna(subset=["target"]).reset_index(drop=True)

    # ---------- Save FEATURES ----------
    feat_parquet = os.path.join(feat_dir, f"{symbol.lower()}_features_30min.parquet")
    feat_csv     = os.path.join(feat_dir, f"{symbol.lower()}_features_30min.csv")
    feat.to_parquet(feat_parquet, index=False)
    feat.to_csv(feat_csv, index=False)
    print(f"Saved FEATURES → {feat_parquet}")
    print(f"{symbol}: {len(feat)} rows")

print("\n✅ DONE: Alle Symbole (30-Minuten) verarbeitet.")