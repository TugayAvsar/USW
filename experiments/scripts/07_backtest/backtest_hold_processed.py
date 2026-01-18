import os
import numpy as np
import pandas as pd
import joblib

# ---------------- CONFIG ----------------
MODEL_PATH = "experiments/models/tsla_xgb_best_1min.pkl"
FEATURE_FILE = "experiments/models/xgb_features_1min.txt"
DATA_PATH = "experiments/data/processed/tsla_val_1min.parquet"

PROB_THRESHOLD = 0.55
HOLD_LIST = [1, 5, 10]

# ---------------- LOAD ----------------
model = joblib.load(MODEL_PATH)

with open(FEATURE_FILE) as f:
    FEATURES = [x.strip() for x in f if x.strip()]

df = pd.read_parquet(DATA_PATH)
df = df.sort_values("datetime").reset_index(drop=True)

# ---------------- BACKTEST ----------------
def run_backtest(hold_minutes):
    in_pos = False
    entry_price = None
    entry_time = None
    trades = []

    for i in range(len(df)):
        row = df.iloc[i]
        t = row["datetime"]
        price = row["close"]

        # EXIT
        if in_pos and (t - entry_time).total_seconds() >= hold_minutes * 60:
            ret = (price / entry_price) - 1
            trades.append(ret)
            in_pos = False

        # ENTRY
        if not in_pos:
            X = row[FEATURES].values.reshape(1, -1)
            prob = model.predict_proba(X)[0, 1]
            if prob > PROB_THRESHOLD:
                in_pos = True
                entry_price = price
                entry_time = t

    if in_pos:
        ret = (df.iloc[-1]["close"] / entry_price) - 1
        trades.append(ret)

    if len(trades) == 0:
        return {
            "hold": hold_minutes,
            "trades": 0,
            "avg_trade": 0.0,
            "winrate": 0.0,
            "equity_mult": 1.0
        }

    arr = np.array(trades)
    return {
        "hold": hold_minutes,
        "trades": len(arr),
        "avg_trade": arr.mean(),
        "winrate": (arr > 0).mean(),
        "equity_mult": float(np.prod(1 + arr))
    }

# ---------------- RUN ----------------
results = [run_backtest(h) for h in HOLD_LIST]
out = pd.DataFrame(results)
print("\n=== HOLD BACKTEST (processed features) ===")
print(out.to_string(index=False))