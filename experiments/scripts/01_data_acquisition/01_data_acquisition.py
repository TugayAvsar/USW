"""
01_data_acquisition.py
----------------------
Dieses Skript lädt historische **1-Minuten-Bar-Daten** für mehrere Aktien (z. B. TSLA, F, GM)
über die Alpaca Market Data API, speichert die **1-Minuten-Rohdaten**, aggregiert diese auf
**30-Minuten-Bars**, erzeugt darauf basierend **Features + Zielvariable** für:

A) 1-Minuten-Basis
B) 30-Minuten-Basis

WICHTIG (Anforderung vom Dozenten):
- Daten werden minütlich (TimeFrame.Minute) von der API geladen
- Für Analysen/Modeling kann man sowohl 1-Min als auch 30-Min nutzen

Zusätzlich (nur für TSLA):
- Abruf von News über Alpaca News API
- Speicherung der Roh-News unter <DATA_PATH>/news/tsla_news.parquet/csv
- VADER Sentiment auf Headline
- Aggregation auf 30-Minuten-Buckets → Features: news_count, avg_headline_len, avg_summary_len, avg_sentiment
- Join dieser News-Features in TSLA 30-Min Features

Zielvariable (für beide Granularitäten):
    target = 1, wenn close[t+1] > close[t], sonst 0
"""

from alpaca.data.historical import StockHistoricalDataClient, NewsClient
from alpaca.data.requests import StockBarsRequest, NewsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment

import pandas as pd
import numpy as np
from datetime import datetime
import yaml
import os
from pathlib import Path


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


# -------------------- Technical indicator helpers --------------------
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


# -------------------- Load configs --------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # experiments/scripts/01_data_acquisition
CONF_DIR = BASE_DIR / "conf"
keys = yaml.safe_load(open(CONF_DIR / "keys.yaml"))
API_KEY = keys["KEYS"]["APCA-API-KEY-ID-Data"]
SECRET_KEY = keys["KEYS"]["APCA-API-SECRET-KEY-Data"]

params = yaml.safe_load(open(CONF_DIR / "params.yaml"))
DATA_PATH   = params["DATA_ACQUISITION"]["DATA_PATH"]
START_DATE  = datetime.strptime(params["DATA_ACQUISITION"]["START_DATE"], "%Y-%m-%d")
END_DATE    = datetime.strptime(params["DATA_ACQUISITION"]["END_DATE"], "%Y-%m-%d")
SYMBOLS     = params["DATA_ACQUISITION"].get("SYMBOLS", ["TSLA"])


# -------------------- Prepare output directories --------------------
raw_dir       = os.path.join(DATA_PATH, "raw")        # 30-min raw
feat_dir      = os.path.join(DATA_PATH, "features")   # features (1min & 30min)
raw_1min_dir  = os.path.join(DATA_PATH, "raw_1min")   # 1-min raw
news_dir      = os.path.join(DATA_PATH, "news")       # raw news (TSLA)

os.makedirs(raw_dir, exist_ok=True)
os.makedirs(feat_dir, exist_ok=True)
os.makedirs(raw_1min_dir, exist_ok=True)
os.makedirs(news_dir, exist_ok=True)


# -------------------- Clients --------------------
bars_client = StockHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)
news_client = NewsClient(api_key=API_KEY, secret_key=SECRET_KEY)


# -------------------- Helper: resample 1min -> 30min --------------------
def resample_to_30min(df_1min: pd.DataFrame) -> pd.DataFrame:
    """
    df_1min: DataFrame mit DatetimeIndex (1-min bars)
    Aggregation auf 30-min:
      open: first, high: max, low: min, close: last,
      volume: sum, trade_count: sum, vwap: mean
    """
    df_30 = df_1min.resample("30min").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "trade_count": "sum",
        "vwap": "mean",
    })
    return df_30.dropna(subset=["close"])


# -------------------- Helper: news features for TSLA (30min buckets) --------------------
def build_news_features_30min(symbol: str) -> pd.DataFrame:
    """
    Holt News (Alpaca News) für Symbol, berechnet VADER-Sentiment auf Headline,
    speichert Roh-News (nur TSLA) und aggregiert auf 30-min Buckets:
      - news_count
      - avg_headline_len
      - avg_summary_len
      - avg_sentiment
    """
    print(f"Fetching NEWS for {symbol} from {START_DATE.date()} to {END_DATE.date()}...")

    req = NewsRequest(
        symbols=symbol,
        start=START_DATE,
        end=END_DATE,
        limit=5000,  # wenn zu wenig: paging wäre nötig
    )

    news_set = news_client.get_news(req)
    news_df = news_set.df

    if news_df is None or news_df.empty:
        print(f"[WARN] No news found for {symbol}.")
        return pd.DataFrame(columns=["news_count", "avg_headline_len", "avg_summary_len", "avg_sentiment"])

    news_df["created_at"] = pd.to_datetime(news_df["created_at"])
    news_df = news_df.sort_values("created_at")

    # VADER sentiment auf headline
    news_df["headline"] = news_df["headline"].fillna("").astype(str)
    news_df["sentiment"] = news_df["headline"].apply(lambda x: analyzer.polarity_scores(x)["compound"])

    # speichern roh-news nur für TSLA
    if symbol.upper() == "TSLA":
        pqt = os.path.join(news_dir, "tsla_news.parquet")
        csv = os.path.join(news_dir, "tsla_news.csv")
        news_df.to_parquet(pqt, index=False)
        news_df.to_csv(csv, index=False)
        print(f"Saved RAW NEWS TSLA → {pqt} ({len(news_df)} rows)")

    # bucket features
    news_df = news_df.set_index("created_at")
    news_df["headline_len"] = news_df["headline"].str.len()
    news_df["summary_len"] = news_df["summary"].fillna("").astype(str).str.len()

    news_30 = news_df.resample("30min").agg({
        "headline": "count",
        "headline_len": "mean",
        "summary_len": "mean",
        "sentiment": "mean",
    }).rename(columns={
        "headline": "news_count",
        "headline_len": "avg_headline_len",
        "summary_len": "avg_summary_len",
        "sentiment": "avg_sentiment",
    })

    return news_30


# -------------------- MAIN LOOP --------------------
for symbol in SYMBOLS:
    print(f"\nFetching 1-MINUTE bars for {symbol} from {START_DATE.date()} to {END_DATE.date()}...")

    bars_req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        adjustment=Adjustment.ALL,
        start=START_DATE,
        end=END_DATE,
    )

    bars = bars_client.get_stock_bars(bars_req)
    df_1min = bars.df.reset_index()

    # symbol column entfernen (falls MultiIndex)
    if "symbol" in df_1min.columns:
        df_1min = df_1min.drop(columns=["symbol"])

    df_1min = df_1min.rename(columns={"timestamp": "datetime"})
    df_1min["datetime"] = pd.to_datetime(df_1min["datetime"])
    df_1min = df_1min.sort_values("datetime").set_index("datetime")

    # -------------------- save RAW 1-min --------------------
    raw_1min_parquet = os.path.join(raw_1min_dir, f"{symbol.lower()}_1min.parquet")
    raw_1min_csv     = os.path.join(raw_1min_dir, f"{symbol.lower()}_1min.csv")
    df_1min.to_parquet(raw_1min_parquet)
    df_1min.to_csv(raw_1min_csv)
    print(f"Saved RAW 1-MIN → {raw_1min_parquet}")

    # ======================================================
    # 1) FEATURES auf 1-Minuten-Basis erzeugen + speichern
    # ======================================================
    feat_1min = df_1min.copy()
    feat_1min["close_pct_change"] = feat_1min["close"].pct_change()
    feat_1min["sma_5"]  = sma(feat_1min["close"], 5)
    feat_1min["sma_10"] = sma(feat_1min["close"], 10)
    feat_1min["ema_5"]  = ema(feat_1min["close"], 5)
    feat_1min["ema_10"] = ema(feat_1min["close"], 10)
    feat_1min["rsi_14"] = rsi(feat_1min["close"], 14)

    feat_1min["close_next"] = feat_1min["close"].shift(-1)
    feat_1min["target"] = (feat_1min["close_next"] > feat_1min["close"]).astype("Int64")
    feat_1min = feat_1min.dropna(subset=["target"]).copy()

    feat_1min_out = feat_1min.reset_index()

    feat_1min_parquet = os.path.join(feat_dir, f"{symbol.lower()}_features_1min.parquet")
    feat_1min_csv     = os.path.join(feat_dir, f"{symbol.lower()}_features_1min.csv")
    feat_1min_out.to_parquet(feat_1min_parquet, index=False)
    feat_1min_out.to_csv(feat_1min_csv, index=False)

    print(f"Saved FEATURES 1-MIN → {feat_1min_parquet} ({len(feat_1min_out)} rows)")

    # ======================================================
    # 2) 1min -> 30min resample, RAW speichern
    # ======================================================
    df_30 = resample_to_30min(df_1min).reset_index()  # datetime wieder als Spalte

    raw_30_parquet = os.path.join(raw_dir, f"{symbol.lower()}_30min.parquet")
    raw_30_csv     = os.path.join(raw_dir, f"{symbol.lower()}_30min.csv")
    df_30.to_parquet(raw_30_parquet, index=False)
    df_30.to_csv(raw_30_csv, index=False)
    print(f"Saved RAW 30-MIN → {raw_30_parquet}")

    # ======================================================
    # 3) FEATURES auf 30-Minuten-Basis erzeugen
    # ======================================================
    feat_30 = df_30.copy()
    feat_30["close_pct_change"] = feat_30["close"].pct_change()
    feat_30["sma_5"]  = sma(feat_30["close"], 5)
    feat_30["sma_10"] = sma(feat_30["close"], 10)
    feat_30["ema_5"]  = ema(feat_30["close"], 5)
    feat_30["ema_10"] = ema(feat_30["close"], 10)
    feat_30["rsi_14"] = rsi(feat_30["close"], 14)

    feat_30 = feat_30.set_index("datetime")

    # News features nur TSLA (auf 30-min) joinen
    if symbol.upper() == "TSLA":
        news_30 = build_news_features_30min(symbol)
        if not news_30.empty:
            feat_30 = feat_30.join(news_30, how="left")
            for col in ["news_count", "avg_headline_len", "avg_summary_len", "avg_sentiment"]:
                if col in feat_30.columns:
                    feat_30[col] = feat_30[col].fillna(0)
            print("✅ TSLA news features joined (30-min).")
        else:
            print("⚠️ No TSLA news features joined (empty).")

    # Target: nächste 30-min Periode
    feat_30["close_next"] = feat_30["close"].shift(-1)
    feat_30["target"] = (feat_30["close_next"] > feat_30["close"]).astype("Int64")
    feat_30 = feat_30.dropna(subset=["target"]).reset_index()

    feat_30_parquet = os.path.join(feat_dir, f"{symbol.lower()}_features_30min.parquet")
    feat_30_csv     = os.path.join(feat_dir, f"{symbol.lower()}_features_30min.csv")
    feat_30.to_parquet(feat_30_parquet, index=False)
    feat_30.to_csv(feat_30_csv, index=False)

    print(f"Saved FEATURES 30-MIN → {feat_30_parquet} ({len(feat_30)} rows)")

print("\n✅ DONE: 1-min download + 1-min features + 30-min features for all symbols (TSLA includes news+sentiment).")
