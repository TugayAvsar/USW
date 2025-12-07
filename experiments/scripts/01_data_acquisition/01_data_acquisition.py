"""
Dieses Skript lädt historische **1-Minuten-Bar-Daten** für mehrere Aktien (z. B. Tesla, Ford, GM)
über die Alpaca Market Data API, aggregiert sie anschließend auf **30-Minuten-Bars** und erzeugt
darauf basierend einfache Feature-Daten inkl. Zielvariable (Up/Down für die nächste 30-Minuten-Periode).

WICHTIG (Anforderung vom Dozenten):
- Die Daten werden **minütlich** von der API gezogen (TimeFrame.Minute),
- Für Analysen, Plots und Modeling arbeiten wir weiterhin mit **30-Minuten-Bars**.

Zusätzlich (nur für TSLA):
- Abruf von News über Alpaca News API
- Speicherung der kompletten Roh-News (headline, summary, source, url, sentiment, ...)
  unter   <DATA_PATH>/news/tsla_news.parquet  und  tsla_news.csv
- Aggregation der News auf 30-Minuten-Buckets und Join als zusätzliche Features
  (z. B. news_count, avg_headline_len, avg_summary_len, avg_sentiment) in tsla_features_30min.*

Zielvariable:
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

# --- VADER Sentiment (vorher: pip install vaderSentiment) ---
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# ---------- Helper-Funktionen für technische Indikatoren ----------
def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window, min_periods=window).mean()

def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponentiell gewichteter gleitender Durchschnitt"""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI) nach Standardformel"""
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    gain_ema = (
        pd.Series(gain, index=series.index)
        .ewm(alpha=1 / period, adjust=False)
        .mean()
    )
    loss_ema = (
        pd.Series(loss, index=series.index)
        .ewm(alpha=1 / period, adjust=False)
        .mean()
    )

    rs = gain_ema / (loss_ema.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val


# ---------- Config laden ----------
keys = yaml.safe_load(open("../../conf/keys.yaml"))
API_KEY = keys["KEYS"]["APCA-API-KEY-ID-Data"]
SECRET_KEY = keys["KEYS"]["APCA-API-SECRET-KEY-Data"]

params = yaml.safe_load(open("../../conf/params.yaml"))
DATA_PATH   = params["DATA_ACQUISITION"]["DATA_PATH"]
START_DATE  = datetime.strptime(params["DATA_ACQUISITION"]["START_DATE"], "%Y-%m-%d")
END_DATE    = datetime.strptime(params["DATA_ACQUISITION"]["END_DATE"], "%Y-%m-%d")
SYMBOLS     = params["DATA_ACQUISITION"].get("SYMBOLS", ["TSLA"])

# ---------- Output-Verzeichnisse vorbereiten ----------
raw_dir      = os.path.join(DATA_PATH, "raw")        # 30-Minuten-Bars
feat_dir     = os.path.join(DATA_PATH, "features")   # Feature-Datensätze
raw_1min_dir = os.path.join(DATA_PATH, "raw_1min")   # 1-Minuten-Rohdaten
news_dir     = os.path.join(DATA_PATH, "news")       # Roh-News (nur TSLA)

os.makedirs(raw_dir, exist_ok=True)
os.makedirs(feat_dir, exist_ok=True)
os.makedirs(raw_1min_dir, exist_ok=True)
os.makedirs(news_dir, exist_ok=True)

# ---------- Alpaca Clients initialisieren ----------
bars_client = StockHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)
news_client = NewsClient(api_key=API_KEY, secret_key=SECRET_KEY)


# ---------- Helper: 1-Minuten → 30-Minuten resamplen ----------
def resample_to_30min(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nimmt 1-Minuten-Bars mit Index = datetime und aggregiert auf 30-Minuten-Bars.
    """
    df_30 = df.resample("30T").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "trade_count": "sum",
            "vwap": "mean",
        }
    )
    df_30 = df_30.dropna(subset=["close"])
    return df_30


# ---------- Helper: News für TSLA holen & als Raw + 30-Min-Features speichern ----------
def build_news_features_30min(symbol: str) -> pd.DataFrame:
    """
    Holt News von Alpaca für das gegebene Symbol.

    Schritte:
      1. News als Rohdaten laden (headline, summary, source, url, sentiment, ...)
      2. Für TSLA: komplette Roh-News unter DATA_PATH/news/tsla_news.* speichern
      3. News auf 30-Minuten-Buckets aggregieren → Features:
         - news_count
         - avg_headline_len
         - avg_summary_len
         - avg_sentiment (VADER-Score der Headlines)
    """
    print(f"Fetching NEWS for {symbol} from {START_DATE.date()} to {END_DATE.date()}...")

    news_req = NewsRequest(
        symbols=symbol,
        start=START_DATE,
        end=END_DATE,
        limit=5000,  # reicht in der Regel; sonst Paging notwendig
    )

    news_set = news_client.get_news(news_req)
    news_df = news_set.df  # direkt als DataFrame

    if news_df.empty:
        print(f"No news found for {symbol} in given period.")
        return pd.DataFrame(
            columns=["news_count", "avg_headline_len", "avg_summary_len", "avg_sentiment"]
        )

    # created_at als Timestamp
    news_df["created_at"] = pd.to_datetime(news_df["created_at"])
    news_df = news_df.sort_values("created_at")

    # --- VADER Sentiment auf Headline berechnen ---
    news_df["headline"] = news_df["headline"].fillna("")
    news_df["sentiment"] = news_df["headline"].apply(
        lambda x: analyzer.polarity_scores(str(x))["compound"]
    )

    # ----- 1) Roh-News für TSLA speichern -----
    if symbol.upper() == "TSLA":
        raw_news_parquet = os.path.join(news_dir, "tsla_news.parquet")
        raw_news_csv     = os.path.join(news_dir, "tsla_news.csv")
        news_df.to_parquet(raw_news_parquet, index=False)
        news_df.to_csv(raw_news_csv, index=False)
        print(f"Saved RAW NEWS for TSLA → {raw_news_parquet}")
        print(f"Rows: {len(news_df)}")

    # ----- 2) Für Aggregation Index auf created_at setzen -----
    news_df = news_df.set_index("created_at")

    # einfache Textlängen-Features
    news_df["headline_len"] = news_df["headline"].fillna("").str.len()
    news_df["summary_len"] = news_df["summary"].fillna("").str.len()

    # Aggregation auf 30-Minuten-Buckets (inkl. Sentiment-Mittelwert)
    news_30 = news_df.resample("30T").agg(
        {
            "headline": "count",       # Anzahl Artikel
            "headline_len": "mean",    # mittlere Headline-Länge
            "summary_len": "mean",     # mittlere Summary-Länge
            "sentiment": "mean",       # mittleres Sentiment der Headlines
        }
    )

    news_30 = news_30.rename(
        columns={
            "headline": "news_count",
            "headline_len": "avg_headline_len",
            "summary_len": "avg_summary_len",
            "sentiment": "avg_sentiment",
        }
    )

    return news_30


# ---------- Hauptschleife über alle Symbole ----------
for symbol in SYMBOLS:
    print(
        f"\nFetching **1-MINUTE** bars for {symbol} "
        f"from {START_DATE.date()} to {END_DATE.date()}..."
    )

    # 1-Minuten-Bars von Alpaca holen
    bars_req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,        # 1-Minuten-Auflösung
        adjustment=Adjustment.ALL,
        start=START_DATE,
        end=END_DATE,
    )

    bars = bars_client.get_stock_bars(bars_req)
    df_1min = bars.df.reset_index()

    # Multi-Index (symbol, timestamp) auflösen und nur Timestamp behalten
    if "symbol" in df_1min.columns:
        df_1min = df_1min.drop(columns=["symbol"])

    df_1min = df_1min.rename(columns={"timestamp": "datetime"})
    df_1min["datetime"] = pd.to_datetime(df_1min["datetime"])
    df_1min = df_1min.sort_values("datetime").set_index("datetime")

    # ---------- 1-Minuten-Rohdaten speichern ----------
    raw_1min_parquet = os.path.join(raw_1min_dir, f"{symbol.lower()}_1min.parquet")
    raw_1min_csv     = os.path.join(raw_1min_dir, f"{symbol.lower()}_1min.csv")
    df_1min.to_parquet(raw_1min_parquet)
    df_1min.to_csv(raw_1min_csv)
    print(f"Saved RAW 1-MIN → {raw_1min_parquet}")

    # ---------- 1-Minuten-Bars → 30-Minuten-Bars aggregieren ----------
    df_30 = resample_to_30min(df_1min).reset_index()  # Index zurück in 'datetime'

    # ---------- 30-Minuten-RAW speichern ----------
    raw_parquet = os.path.join(raw_dir, f"{symbol.lower()}_30min.parquet")
    raw_csv     = os.path.join(raw_dir, f"{symbol.lower()}_30min.csv")
    df_30.to_parquet(raw_parquet, index=False)
    df_30.to_csv(raw_csv, index=False)
    print(f"Saved RAW 30-MIN → {raw_parquet}")

    # ---------- Feature Engineering auf 30-Minuten-Basis ----------
    feat = df_30.copy()

    feat["close_pct_change"] = feat["close"].pct_change()
    feat["sma_5"]  = sma(feat["close"], 5)
    feat["sma_10"] = sma(feat["close"], 10)
    feat["ema_5"]  = ema(feat["close"], 5)
    feat["ema_10"] = ema(feat["close"], 10)
    feat["rsi_14"] = rsi(feat["close"], 14)

    feat = feat.set_index("datetime")

    # ---------- News-Features nur für TSLA joinen ----------
    if symbol.upper() == "TSLA":
        news_30 = build_news_features_30min(symbol)

        if not news_30.empty:
            feat = feat.join(news_30, how="left")
            # fehlende News-Werte = 0
            for col in ["news_count", "avg_headline_len", "avg_summary_len", "avg_sentiment"]:
                if col in feat.columns:
                    feat[col] = feat[col].fillna(0)
            print("✅ News features (inkl. avg_sentiment) joined for TSLA.")
        else:
            print("⚠️ No news features joined (empty news dataframe).")

    # ---------- Zielvariable: nächste 30-Minuten-Periode ----------
    feat["close_next"] = feat["close"].shift(-1)
    feat["target"] = (feat["close_next"] > feat["close"]).astype("Int64")

    feat = feat.dropna(subset=["target"]).reset_index()

    # ---------- FEATURES speichern ----------
    feat_parquet = os.path.join(feat_dir, f"{symbol.lower()}_features_30min.parquet")
    feat_csv     = os.path.join(feat_dir, f"{symbol.lower()}_features_30min.csv")
    feat.to_parquet(feat_parquet, index=False)
    feat.to_csv(feat_csv, index=False)

    print(f"Saved FEATURES → {feat_parquet}")
    print(f"{symbol}: {len(feat)} rows (30-MIN features mit Target)")

print("\n✅ DONE: Alle Symbole mit 1-Minuten-Download → 30-Minuten-Features verarbeitet (inkl. TSLA-News + Sentiment in /news & als Features).")
