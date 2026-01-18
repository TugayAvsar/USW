"""
06_deployment.py – Paper Trading Bot (Team 16) – HYBRID DATA (yfinance + Alpaca)
-------------------------------------------------------------------------------
Überarbeitete Version basierend auf:
- Orders weiterhin über Alpaca Paper API
- Option A Exit: Time-based Exit (HOLD = 5 Minuten)
- Risk/Execution Fixes:
  (1) Position-/Order-Guard: kein Buy wenn Position existiert oder offene Order existiert
  (2) Position Sizing Cap: max. X% Equity pro Trade
  (3) Buying Power Check + Buffer
- Robust: Entry-Zeit lokal in bot_state.json speichern (nicht exchange_opened_at)

Aufruf:
- DRY_RUN (optional):
    export DRY_RUN=true   (Linux)
    $env:DRY_RUN="true"   (PowerShell)

- Start (one-shot):
    python 06_deployment.py
"""

import os
import math
import json
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import yaml
import yfinance as yf

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# =====================================================
# PATHS & CONFIG
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXP_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))  # experiments/
CONF_DIR = os.path.join(EXP_DIR, "conf")
MODEL_DIR = os.path.join(EXP_DIR, "models")

FEATURE_FILE = os.path.join(MODEL_DIR, "xgb_features_1min.txt")
MODEL_PATH = os.path.join(MODEL_DIR, "tsla_xgb_best_1min.pkl")
STATE_FILE = os.path.join(EXP_DIR, "bot_state.json")

SYMBOL = "TSLA"

# Entry / Exit
PROB_THRESHOLD = 0.55  # <- stellt das wieder "normal" ein (nicht 0.4)
HOLD_MINUTES = 5       # <- Option A, Prof-Feedback: hold im Backtest testen

# Optional Safety Layer (kannst du auch auf None setzen)
STOP_PCT = None        # z.B. -0.003 für -0.3%
TAKE_PCT = None        # z.B.  0.003 für +0.3%

# Risk / Execution
MAX_ALLOC = 0.10       # max 10% vom Equity pro Trade
MIN_BP_BUFFER = 50.0   # kleiner Puffer, damit BP nicht "auf Kante" läuft

# Market Order Quantity hard cap (zusätzlicher Schutz)
MAX_QTY_CAP = 50

# Data
YF_LOOKBACK_MINUTES = 180  # genug für SMA_10/RSI + DropNa
ALPACA_ENRICH_LIMIT = 200  # für vwap/trade_count enrichment

# DRY RUN: keine Orders schicken
DRY_RUN = str(os.getenv("DRY_RUN", "false")).lower() in ("1", "true", "yes", "y")
print(f"[init] DRY_RUN={DRY_RUN}")

# =====================================================
# STATE (Entry tracking)
# =====================================================
def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

def set_entry(symbol: str, ts_iso: str, entry_price: float | None = None) -> None:
    state = load_state()
    state[symbol] = {"entry_time": ts_iso, "entry_price": entry_price}
    save_state(state)

def clear_entry(symbol: str) -> None:
    state = load_state()
    if symbol in state:
        del state[symbol]
        save_state(state)

def get_entry(symbol: str) -> dict | None:
    state = load_state()
    return state.get(symbol)

# =====================================================
# LOAD FEATURES LIST
# =====================================================
if not os.path.exists(FEATURE_FILE):
    raise FileNotFoundError(f"Feature list not found: {FEATURE_FILE} (run training script that saves it)")

with open(FEATURE_FILE) as f:
    FEATURES = [line.strip() for line in f.readlines() if line.strip()]

print(f"[init] Loaded {len(FEATURES)} features for XGBoost")

# =====================================================
# LOAD KEYS
# =====================================================
with open(os.path.join(CONF_DIR, "keys.yaml")) as f:
    keys = yaml.safe_load(f)

API_KEY = keys["KEYS"]["APCA-API-KEY-ID-Data"]
API_SECRET = keys["KEYS"]["APCA-API-SECRET-KEY-Data"]

# =====================================================
# CLIENTS
# =====================================================
data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
trade_client = TradingClient(API_KEY, API_SECRET, paper=True)

# =====================================================
# LOAD MODEL
# =====================================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
print(f"[init] Loaded model: {type(model)} from {MODEL_PATH}")

# =====================================================
# HELPERS: Broker Guards / Risk
# =====================================================
def get_open_orders_for_symbol(symbol: str):
    try:
        orders = trade_client.get_orders()  # usually open orders
        return [o for o in orders if getattr(o, "symbol", None) == symbol]
    except Exception as e:
        print(f"[WARN] Could not fetch open orders: {e}")
        return []

def has_position_in_symbol(symbol: str) -> bool:
    try:
        positions = trade_client.get_all_positions()
        return any(p.symbol == symbol and float(p.qty) != 0.0 for p in positions)
    except Exception as e:
        print(f"[WARN] Could not fetch positions: {e}")
        return False

def compute_qty(latest_price: float) -> int:
    account = trade_client.get_account()
    equity = float(account.equity)
    buying_power = float(account.buying_power)

    # Budget: min( MAX_ALLOC*equity, buying_power - buffer )
    budget = min(equity * MAX_ALLOC, max(0.0, buying_power - MIN_BP_BUFFER))
    qty = math.floor(budget / latest_price)

    if qty > MAX_QTY_CAP:
        qty = MAX_QTY_CAP

    return max(0, qty)

# =====================================================
# FEATURE ENGINEERING (MATCH TRAINING FEATURES)
# =====================================================
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

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Erwartet Spalten: datetime, open, high, low, close, volume, trade_count, vwap
    Erzeugt: close_pct_change, sma_5, sma_10, ema_5, ema_10, rsi_14
    """
    d = df.copy()

    # Ensure numeric
    for c in ["open", "high", "low", "close", "volume", "trade_count", "vwap"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # If datetime exists, sort by it
    if "datetime" in d.columns:
        d["datetime"] = pd.to_datetime(d["datetime"], utc=True, errors="coerce")
        d = d.sort_values("datetime")

    d["close_pct_change"] = d["close"].pct_change()
    d["sma_5"] = sma(d["close"], 5)
    d["sma_10"] = sma(d["close"], 10)
    d["ema_5"] = ema(d["close"], 5)
    d["ema_10"] = ema(d["close"], 10)
    d["rsi_14"] = rsi(d["close"], 14)

    # keep only what we need
    # drop rows where required features are nan
    d = d.dropna(subset=FEATURES)

    return d

# =====================================================
# DATA FETCH (HYBRID)
# =====================================================
def get_latest_data_yfinance() -> pd.DataFrame:
    """
    Holt 1-min TSLA Bars via yfinance.
    Returns columns: datetime, open, high, low, close, volume
    """
    try:
        ticker = yf.Ticker(SYMBOL)
        # yfinance: period small; interval 1m
        # NOTE: Yahoo provides limited intraday history per request; 1d is typical
        hist = ticker.history(period="1d", interval="1m")

        if hist is None or hist.empty:
            return pd.DataFrame()

        hist = hist.reset_index()
        # column name can be "Datetime" or "Date"
        dt_col = "Datetime" if "Datetime" in hist.columns else ("Date" if "Date" in hist.columns else None)
        if dt_col is None:
            return pd.DataFrame()

        out = pd.DataFrame({
            "datetime": pd.to_datetime(hist[dt_col], utc=True, errors="coerce").dt.floor("min"),
            "open": hist["Open"],
            "high": hist["High"],
            "low": hist["Low"],
            "close": hist["Close"],
            "volume": hist["Volume"],
        }).dropna(subset=["datetime"]).sort_values("datetime")

        # keep last N minutes
        out = out.tail(YF_LOOKBACK_MINUTES).reset_index(drop=True)
        return out
    except Exception as e:
        print(f"[WARN] yfinance fetch failed: {e}")
        return pd.DataFrame()

def get_latest_data_alpaca() -> pd.DataFrame:
    """
    Holt 1-min Bars via Alpaca (fallback / enrichment).
    Returns columns: datetime, open, high, low, close, volume, trade_count, vwap
    """
    try:
        req = StockBarsRequest(
            symbol_or_symbols=SYMBOL,
            timeframe=TimeFrame.Minute,
            limit=ALPACA_ENRICH_LIMIT
        )
        bars = data_client.get_stock_bars(req).df

        if bars is None or bars.empty:
            return pd.DataFrame()

        bars = bars.reset_index()
        bars = bars.rename(columns={"timestamp": "datetime"})
        bars["datetime"] = pd.to_datetime(bars["datetime"], utc=True, errors="coerce").dt.floor("min")
        bars = bars.sort_values("datetime")

        # Some alpaca dfs include symbol column; drop it
        if "symbol" in bars.columns:
            bars = bars.drop(columns=["symbol"])

        cols = [c for c in ["datetime", "open", "high", "low", "close", "volume", "trade_count", "vwap"] if c in bars.columns]
        return bars[cols].dropna(subset=["datetime"]).reset_index(drop=True)
    except Exception as e:
        print(f"[WARN] Alpaca fetch failed: {e}")
        return pd.DataFrame()

def get_latest_data() -> pd.DataFrame:
    """
    Hybrid approach:
    - Primary: yfinance OHLCV
    - Enrich: Alpaca vwap/trade_count when possible
    - Fallback: Alpaca full bars if yfinance empty
    """
    yf_df = get_latest_data_yfinance()
    if yf_df is None or yf_df.empty:
        print("[warn] yfinance returned no bars, falling back to Alpaca.")
        return get_latest_data_alpaca()

    alp_df = get_latest_data_alpaca()
    if alp_df is not None and not alp_df.empty:
        a = alp_df[["datetime", "vwap", "trade_count"]].copy()
        y = yf_df.copy()

        a["datetime"] = pd.to_datetime(a["datetime"], utc=True, errors="coerce").dt.floor("min")
        y["datetime"] = pd.to_datetime(y["datetime"], utc=True, errors="coerce").dt.floor("min")

        merged = y.merge(a, on="datetime", how="left")

        merged["trade_count"] = pd.to_numeric(merged.get("trade_count"), errors="coerce").fillna(0.0)
        merged["vwap"] = pd.to_numeric(merged.get("vwap"), errors="coerce")
        merged["vwap"] = merged["vwap"].fillna((merged["high"] + merged["low"] + merged["close"]) / 3.0)

        merged = merged.sort_values("datetime").reset_index(drop=True)
        return merged

    # If Alpaca not available: approximate vwap/trade_count
    yf_df = yf_df.copy()
    yf_df["trade_count"] = 0.0
    yf_df["vwap"] = (yf_df["high"] + yf_df["low"] + yf_df["close"]) / 3.0
    yf_df = yf_df.sort_values("datetime").reset_index(drop=True)
    return yf_df

# =====================================================
# ENTRY / EXIT
# =====================================================
def compute_signal_probability(feat_df: pd.DataFrame) -> float:
    """
    Predict P(up) using the last row of the feature dataframe, respecting exact feature order.
    """
    X = feat_df[FEATURES].tail(1).values
    prob = float(model.predict_proba(X)[0, 1])
    return prob

def close_old_positions(latest_price: float | None = None):
    """
    Time-based exit (HOLD_MINUTES) + optional stop/take.
    Entry times tracked in STATE_FILE.
    """
    now = datetime.now(timezone.utc)

    try:
        positions = trade_client.get_all_positions()
    except Exception as e:
        print(f"[WARN] Could not fetch positions for exit: {e}")
        return

    for pos in positions:
        sym = pos.symbol

        # Guard: don't spam sells if open order exists
        if get_open_orders_for_symbol(sym):
            print(f"[skip] Open order exists for {sym}, not closing")
            continue

        entry = get_entry(sym)
        if not entry or "entry_time" not in entry:
            # If no entry tracking exists, do nothing (safe)
            continue

        opened_at = datetime.fromisoformat(entry["entry_time"])

        # Optional Stop/Take
        if latest_price is not None:
            try:
                entry_price = float(entry.get("entry_price") or pos.avg_entry_price)
                pnl_pct = (float(latest_price) / entry_price) - 1.0

                if STOP_PCT is not None and pnl_pct <= STOP_PCT:
                    qty = abs(int(float(pos.qty)))
                    print(f"[exit] Stop-loss SELL {sym} qty={qty} pnl={pnl_pct:.4f}")
                    if not DRY_RUN:
                        trade_client.submit_order(MarketOrderRequest(
                            symbol=sym, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY
                        ))
                    clear_entry(sym)
                    continue

                if TAKE_PCT is not None and pnl_pct >= TAKE_PCT:
                    qty = abs(int(float(pos.qty)))
                    print(f"[exit] Take-profit SELL {sym} qty={qty} pnl={pnl_pct:.4f}")
                    if not DRY_RUN:
                        trade_client.submit_order(MarketOrderRequest(
                            symbol=sym, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY
                        ))
                    clear_entry(sym)
                    continue
            except Exception as e:
                print(f"[warn] Stop/Take check failed for {sym}: {e}")

        # Time-based Exit
        if now - opened_at >= timedelta(minutes=HOLD_MINUTES):
            qty = abs(int(float(pos.qty)))
            print(f"[exit] Time exit SELL {sym} qty={qty} (held >= {HOLD_MINUTES}m)")
            if not DRY_RUN:
                trade_client.submit_order(MarketOrderRequest(
                    symbol=sym, qty=qty, side=OrderSide.SELL, time_in_force=TimeInForce.DAY
                ))
            clear_entry(sym)

# =====================================================
# MAIN
# =====================================================
def main():
    print("=== XGBoost Paper Trading Bot (1-Min) – HYBRID DATA ===")

    raw_df = get_latest_data()
    if raw_df is None or raw_df.empty:
        print("[warn] No bars returned. Skipping run.")
        return

    # ensure required raw cols exist
    needed_raw = {"datetime", "open", "high", "low", "close", "volume", "trade_count", "vwap"}
    missing_raw = [c for c in needed_raw if c not in raw_df.columns]
    if missing_raw:
        print(f"[warn] Missing raw cols: {missing_raw}. Columns: {list(raw_df.columns)}")
        return

    latest_price = float(pd.to_numeric(raw_df["close"].iloc[-1], errors="coerce"))
    if not np.isfinite(latest_price) or latest_price <= 0:
        print("[warn] latest_price invalid. Skipping.")
        return

    # Build features
    feat_df = build_features(raw_df)
    if feat_df is None or feat_df.empty:
        print("[warn] Not enough data after feature build (dropna).")
        return

    # Sanity check: feature coverage
    missing_features = [f for f in FEATURES if f not in feat_df.columns]
    if missing_features:
        print(f"[warn] Missing FEATURES in feat_df: {missing_features}")
        return

    # Predict
    prob = compute_signal_probability(feat_df)
    print(f"[model] P(up)={prob:.3f}  threshold={PROB_THRESHOLD:.2f}")

    # Always try exit logic (so positions get closed even if no entry)
    close_old_positions(latest_price=latest_price)

    # Entry decision
    if prob <= PROB_THRESHOLD:
        print("[entry] No buy signal.")
        return

    # Guard 1: open order
    if get_open_orders_for_symbol(SYMBOL):
        print(f"[entry-skip] Open order exists for {SYMBOL}.")
        return

    # Guard 2: existing position
    if has_position_in_symbol(SYMBOL):
        print(f"[entry-skip] Position already exists for {SYMBOL}.")
        return

    qty = compute_qty(latest_price)
    if qty <= 0:
        print("[entry-skip] Qty computed as 0 (insufficient buying power / buffer).")
        return

    print(f"[entry] BUY {SYMBOL} qty={qty} @ ~{latest_price:.2f} (max_alloc={MAX_ALLOC:.0%})")

    if not DRY_RUN:
        trade_client.submit_order(MarketOrderRequest(
            symbol=SYMBOL,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        ))

    # Track entry time locally
    set_entry(SYMBOL, datetime.now(timezone.utc).isoformat(), entry_price=latest_price)

if __name__ == "__main__":
    main()