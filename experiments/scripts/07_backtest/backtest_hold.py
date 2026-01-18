import os
import math
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
import yaml

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# -------------------
# CONFIG (wie in 06_deployment.py)
# -------------------
BASE_DIR = os.path.dirname(os.path.abspath(_file_))
EXP_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
CONF_DIR = os.path.join(EXP_DIR, "conf")
MODEL_DIR = os.path.join(EXP_DIR, "models")

FEATURE_FILE = os.path.join(MODEL_DIR, "xgb_features_1min.txt")
MODEL_PATH = os.path.join(MODEL_DIR, "tsla_xgb_best_1min.pkl")

SYMBOL = "TSLA"
PROB_THRESHOLD = 0.55

# Hier testest du HOLD-Werte:
HOLD_MINUTES_LIST = [1, 5, 10]

# Backtest-Zeitraum (UTC, Format: YYYY-MM-DD)
START = "2026-01-05"
END   = "2026-01-10"

# -------------------
# Features (identisch)
# -------------------
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain_ema = pd.Series(gain, index=series.index).ewm(alpha=1 / period, adjust=False).mean()
    loss_ema = pd.Series(loss, index=series.index).ewm(alpha=1 / period, adjust=False).mean()

    rs = gain_ema / (loss_ema.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def build_features(df: pd.DataFrame, FEATURES: list[str]) -> pd.DataFrame:
    d = df.copy()

    for c in ["open", "high", "low", "close", "volume", "trade_count", "vwap"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d["datetime"] = pd.to_datetime(d["datetime"], utc=True, errors="coerce")
    d = d.sort_values("datetime")

    d["close_pct_change"] = d["close"].pct_change()
    d["sma_5"] = sma(d["close"], 5)
    d["sma_10"] = sma(d["close"], 10)
    d["ema_5"] = ema(d["close"], 5)
    d["ema_10"] = ema(d["close"], 10)
    d["rsi_14"] = rsi(d["close"], 14)

    d = d.dropna(subset=FEATURES)
    return d

# -------------------
# Data: Alpaca historical bars (stabil für Backtest)
# -------------------
def load_alpaca_bars() -> pd.DataFrame:
    with open(os.path.join(CONF_DIR, "keys.yaml")) as f:
        keys = yaml.safe_load(f)

    API_KEY = keys["KEYS"]["APCA-API-KEY-ID-Data"]
    API_SECRET = keys["KEYS"]["APCA-API-SECRET-KEY-Data"]

    data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

    req = StockBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=TimeFrame.Minute,
        start=pd.Timestamp(START, tz="UTC"),
        end=pd.Timestamp(END, tz="UTC"),
        limit=10000
    )

    bars = data_client.get_stock_bars(req).df
    if bars is None or bars.empty:
        return pd.DataFrame()

    bars = bars.reset_index()
    bars = bars.rename(columns={"timestamp": "datetime"})

    if "symbol" in bars.columns:
        bars = bars.drop(columns=["symbol"])

    cols = [c for c in ["datetime","open","high","low","close","volume","trade_count","vwap"] if c in bars.columns]
    bars = bars[cols].copy()
    bars["datetime"] = pd.to_datetime(bars["datetime"], utc=True, errors="coerce").dt.floor("min")
    bars = bars.dropna(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return bars

# -------------------
# Backtest
# -------------------
def run_backtest(df_raw: pd.DataFrame, model, FEATURES: list[str], hold_minutes: int):
    df_feat = build_features(df_raw, FEATURES)

    in_pos = False
    entry_time = None
    entry_price = None

    trades = []
    equity = 1.0  # normiert, wir tracken prozentual

    # wir iterieren über Feature-DF (hat schon DropNa)
    for i in range(len(df_feat)):
        row = df_feat.iloc[i]
        t = row["datetime"]
        price = float(row["close"])

        # EXIT: time-based
        if in_pos and (t - entry_time) >= pd.Timedelta(minutes=hold_minutes):
            ret = (price / entry_price) - 1.0
            equity *= (1.0 + ret)
            trades.append(ret)
            in_pos = False
            entry_time = None
            entry_price = None

        # ENTRY nur wenn flat
        if not in_pos:
            X = df_feat[FEATURES].iloc[[i]].values
            prob = float(model.predict_proba(X)[0, 1])

            if prob > PROB_THRESHOLD:
                in_pos = True
                entry_time = t
                entry_price = price

    # falls Position am Ende offen: glattstellen am letzten Preis
    if in_pos:
        price = float(df_feat["close"].iloc[-1])
        ret = (price / entry_price) - 1.0
        equity *= (1.0 + ret)
        trades.append(ret)

    if len(trades) == 0:
        return {
            "hold": hold_minutes,
            "trades": 0,
            "avg_trade": 0.0,
            "winrate": 0.0,
            "equity_mult": equity
        }

    trades_arr = np.array(trades, dtype=float)
    winrate = float((trades_arr > 0).mean())
    return {
        "hold": hold_minutes,
        "trades": int(len(trades)),
        "avg_trade": float(trades_arr.mean()),
        "winrate": winrate,
        "equity_mult": float(equity)
    }

def main():
    if not os.path.exists(FEATURE_FILE):
        raise FileNotFoundError(f"Missing feature list: {FEATURE_FILE}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")

    with open(FEATURE_FILE) as f:
        FEATURES = [line.strip() for line in f.readlines() if line.strip()]

    model = joblib.load(MODEL_PATH)

    df_raw = load_alpaca_bars()
    if df_raw is None or df_raw.empty:
        print("[ERROR] No historical bars loaded. Check keys.yaml + date range.")
        return

    print(f"[info] Bars loaded: {len(df_raw)} from {df_raw['datetime'].min()} to {df_raw['datetime'].max()}")

    results = []
    for hold in HOLD_MINUTES_LIST:
        res = run_backtest(df_raw, model, FEATURES, hold)
        results.append(res)

    out = pd.DataFrame(results).sort_values("hold")
    print("\n=== HOLD comparison ===")
    print(out.to_string(index=False))

if _name_ == "_main_":
    main()