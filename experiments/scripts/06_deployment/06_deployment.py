"""
main.py – Minimal Deployment Script (Team 16)

- Lädt trainiertes XGBoost 1-Minuten-Modell
- Holt aktuelle 1-Minuten-Daten von Alpaca (Paper Trading)
- Berechnet dieselben Features wie im Training
- Leitet einfache Entry-/Exit-Regeln ab
- Platziert Market Orders via Alpaca Paper API

Run as one-shot (z. B. alle 5 Minuten via cron oder tmux-loop).
"""

import os
import time
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import yaml

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


# =====================================================
# CONFIG
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR  = os.path.abspath(os.path.join(BASE_DIR, "..",".."))

CONF_DIR   = os.path.join(EXP_DIR, "conf")
MODEL_DIR  = os.path.join(EXP_DIR, "models")

FEATURE_FILE = os.path.join(MODEL_DIR, "xgb_features_1min.txt")

with open(FEATURE_FILE) as f:
    FEATURES = [line.strip() for line in f.readlines()]

print(f"[init] Loaded {len(FEATURES)} features for XGBoost")

SYMBOL = "TSLA"
QTY = 1

PROB_THRESHOLD = 0.55        # Entry-Schwelle
HOLD_MINUTES   = 30          # Exit nach 30 Minuten


# =====================================================
# LOAD KEYS
# =====================================================
with open(os.path.join(CONF_DIR, "keys.yaml")) as f:
    keys = yaml.safe_load(f)

API_KEY    = keys["KEYS"]["APCA-API-KEY-ID-Data"]
API_SECRET = keys["KEYS"]["APCA-API-SECRET-KEY-Data"]

# =====================================================
# CLIENTS
# =====================================================
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
trade_client = TradingClient(API_KEY, API_SECRET, paper=True)


# =====================================================
# LOAD MODEL
# =====================================================
model_path = os.path.join(MODEL_DIR, "tsla_xgb_best_1min.pkl")
model = joblib.load(model_path)


# =====================================================
# FEATURE ENGINEERING (IDENTISCH ZUM TRAINING)
# =====================================================
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ema = pd.Series(gain, index=series.index).ewm(alpha=1/period).mean()
    loss_ema = pd.Series(loss, index=series.index).ewm(alpha=1/period).mean()
    rs = gain_ema / loss_ema.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["ema_5"] = ema(df["close"], 5)
    df["ema_10"] = ema(df["close"], 10)
    df["rsi_14"] = rsi(df["close"], 14)
    df["volume_change"] = df["volume"].pct_change()
    df["vwap_diff"] = (df["close"] - df["vwap"]) / df["vwap"]
    return df.dropna()


# =====================================================
# GET LATEST DATA
# =====================================================
def get_latest_data():
    req = StockBarsRequest(
        symbol_or_symbols=SYMBOL,
        timeframe=TimeFrame.Minute,
        limit=120
    )
    bars = data_client.get_stock_bars(req).df
    bars = bars.reset_index()
    bars = bars.rename(columns={"timestamp": "datetime"})
    bars = bars.sort_values("datetime")
    return bars


# =====================================================
# ENTRY DECISION
# =====================================================
def check_entry():
    df = get_latest_data()
    df_feat = build_features(df)

    X = df_feat.iloc[-1:][[
        "return",
        "ema_5",
        "ema_10",
        "rsi_14",
        "volume_change",
        "vwap_diff",
    ]]

    prob = model.predict_proba(X)[0, 1]
    print(f"[{datetime.utcnow()}] P(up) = {prob:.3f}")

    return prob > PROB_THRESHOLD


# =====================================================
# EXIT LOGIC
# =====================================================
def close_old_positions():
    positions = trade_client.get_all_positions()
    now = datetime.now(timezone.utc)

    for pos in positions:
        filled_at = datetime.fromisoformat(pos.exchange_opened_at.replace("Z", "+00:00"))
        if now - filled_at > timedelta(minutes=HOLD_MINUTES):
            print(f"Closing position {pos.symbol}")
            order = MarketOrderRequest(
                symbol=pos.symbol,
                qty=abs(int(float(pos.qty))),
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            trade_client.submit_order(order)


# =====================================================
# MAIN
# =====================================================
def main():
    print("=== XGBoost Paper Trading Bot (1-Min) ===")

    if check_entry():
        print("BUY signal detected")
        order = MarketOrderRequest(
            symbol=SYMBOL,
            qty=QTY,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        trade_client.submit_order(order)
    else:
        print("No entry signal")

    close_old_positions()


if __name__ == "__main__":
    main()
