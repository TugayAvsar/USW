"""
DATA UNDERSTANDING

Dieses Skript analysiert die vorbereiteten Feature-Datensätze
für Tesla (TSLA), Ford (F) und General Motors (GM).

Ziele:
1) Relevante Daten-Spalten erklären (inkl. News-Features für TSLA)
2) Deskriptive Statistik der Variablen anzeigen
3) Relevante Plots der wichtigsten Variablen erzeugen
4) Extra: News-Korrelation – Wie hängen TSLA-News, Sentiment und Price Movements zusammen?
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Pfade
DATA_FEATURES_PATH = "../../data/features"
DATA_NEWS_PATH = "../../data/news"
SYMBOLS = ["tsla", "f", "gm"]

dfs = {}

# ---------------------------------------------------------
# 1) Datensätze laden
# ---------------------------------------------------------
for symbol in SYMBOLS:
    path = os.path.join(DATA_FEATURES_PATH, f"{symbol}_features_30min.csv")
    df = pd.read_csv(path)

    # Zeitspalte vereinheitlichen
    if "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"])
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        raise ValueError(f"{symbol}: Keine Zeitspalte gefunden!")

    dfs[symbol] = df
    print(f"Loaded {symbol.upper()}: {df.shape[0]} rows")


# ---------------------------------------------------------
# 2) Spalten erklären
# ---------------------------------------------------------
print("\nCOLUMN EXPLANATION:")
print("""
Gemeinsame Spalten:
date                = 30-Minuten Zeitstempel
open, high, low     = Preisbereiche der Periode
close               = Schlusskurs dieser Periode
volume              = gehandeltes Volumen
close_pct_change    = relative Veränderung gegenüber letzter Periode
sma_5, sma_10       = Simple Moving Averages (Trendindikatoren)
ema_5, ema_10       = Exponentielle Moving Averages (Trendindikatoren)
rsi_14              = Momentumindikator
target              = 1 wenn nächster Schlusskurs höher, sonst 0

TSLA zusätzliche Spalten (News):
news_count          = Anzahl News im 30-Minuten-Fenster
avg_headline_len    = Durchschnittliche Headline-Länge
avg_summary_len     = Durchschnittliche Summary-Länge
avg_sentiment       = Durchschnittlicher VADER-Sentimentwert (-1 bis +1)
""")

# ---------------------------------------------------------
# 3) Deskriptive Statistik
# ---------------------------------------------------------
for symbol, df in dfs.items():
    print(f"\n===== DESCRIPTIVE STATISTICS FOR {symbol.upper()} =====")
    print(df.describe().round(3))
    print("\nTarget distribution:")
    print(df["target"].value_counts(normalize=True).rename("proportion").round(3))


# ---------------------------------------------------------
# 4) Plots (save + show)
# ---------------------------------------------------------
IMG_PATH = "../../images"
os.makedirs(IMG_PATH, exist_ok=True)

def save_and_show(filename: str):
    filepath = os.path.join(IMG_PATH, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    print(f"📁 Saved plot → {filepath}")
    plt.show()

# ---------------------------------------------------------
# 4.1 Cumulative Return Comparison
# ---------------------------------------------------------
plt.figure(figsize=(14, 6))
for symbol, df in dfs.items():
    df["cum_return"] = (1 + df["close_pct_change"].fillna(0)).cumprod() - 1
    plt.plot(df["date"], df["cum_return"], label=symbol.upper(), linewidth=1.5)

plt.title("Cumulative Return (30min) – TSLA vs F vs GM")
plt.xlabel("Time")
plt.ylabel("Cumulative Return")
plt.legend()
save_and_show("cum_return_comparison.png")

# ---------------------------------------------------------
# 4.2 RSI Comparison
# ---------------------------------------------------------
plt.figure(figsize=(14, 6))
for symbol, df in dfs.items():
    plt.plot(df["date"], df["rsi_14"], alpha=0.6, label=symbol.upper())
plt.axhline(70, linestyle="--", color="red")
plt.axhline(30, linestyle="--", color="green")
plt.title("RSI(14) over Time – TSLA vs F vs GM")
plt.xlabel("Time")
plt.ylabel("RSI")
plt.legend()
save_and_show("rsi_comparison.png")

# ---------------------------------------------------------
# 4.3 Distribution of Returns
# ---------------------------------------------------------
plt.figure(figsize=(14, 5))
for symbol, df in dfs.items():
    sns.histplot(
        df["close_pct_change"].dropna(),
        kde=True,
        label=symbol.upper(),
        element="step",
        stat="density",
        alpha=0.4,
    )
plt.title("Distribution of Close Percentage Change (30min)")
plt.xlabel("Percentage Change")
plt.ylabel("Density")
plt.legend()
save_and_show("pct_change_distribution.png")

# ---------------------------------------------------------
# 4.4 Target Distribution
# ---------------------------------------------------------
target_df = pd.DataFrame(
    {sym.upper(): dfs[sym]["target"].value_counts(normalize=True) for sym in dfs}
).T

target_df.plot(kind="bar", figsize=(8, 5),
               title="Target Distribution (Up vs Down) – TSLA vs F vs GM")
plt.ylabel("Proportion")
save_and_show("target_distribution.png")

# ---------------------------------------------------------
# 5) NEWS ↔ TSLA PRICE MOVEMENT
# ---------------------------------------------------------
tsla = dfs["tsla"].copy()

if "news_count" in tsla.columns:

    # ------------------------------
    # 5a) Korrelationen (inkl. Sentiment)
    # ------------------------------
    corr_cols = ["close_pct_change", "news_count", "avg_headline_len", "avg_summary_len"]
    if "avg_sentiment" in tsla.columns:
        corr_cols.append("avg_sentiment")

    corr_df = tsla[corr_cols].corr()

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Between TSLA News Features and Price Movement")
    save_and_show("tsla_news_correlation.png")

    # ------------------------------
    # 5b) Scatterplot: News Count vs Price Change
    # ------------------------------
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=tsla, x="news_count", y="close_pct_change", alpha=0.3)
    plt.title("TSLA: News Count vs Price Change")
    plt.xlabel("News Count")
    plt.ylabel("Price Change")
    save_and_show("tsla_news_vs_pricechange.png")

    # ------------------------------
    # 5c) Returns nach News-Sentiment-Bucket (nur Zeitfenster mit News)
    # ------------------------------
    if "avg_sentiment" in tsla.columns:
        tsla_sent = tsla[tsla["news_count"] > 0].copy()

        def sentiment_bucket(x):
            if x <= -0.2:
                return "negative"
            elif x >= 0.2:
                return "positive"
            else:
                return "neutral"

        tsla_sent["sentiment_bucket"] = tsla_sent["avg_sentiment"].apply(sentiment_bucket)

        # Boxplot der Returns pro Sentiment-Bucket
        plt.figure(figsize=(8, 5))
        sns.boxplot(
            data=tsla_sent,
            x="sentiment_bucket",
            y="close_pct_change",
            order=["negative", "neutral", "positive"]
        )
        plt.title("TSLA: Price Change by News Sentiment Category")
        plt.xlabel("News Sentiment")
        plt.ylabel("Price Change")
        save_and_show("tsla_returns_by_sentiment_bucket_box.png")

        # ------------------------------
        # 5d) Violinplot – Verteilung der Returns pro Sentiment-Bucket
        # ------------------------------
        plt.figure(figsize=(8, 5))
        sns.violinplot(
            data=tsla_sent,
            x="sentiment_bucket",
            y="close_pct_change",
            order=["negative", "neutral", "positive"]
        )
        plt.title("TSLA: Return Distribution by News Sentiment")
        plt.xlabel("News Sentiment")
        plt.ylabel("Price Change")
        save_and_show("tsla_returns_by_sentiment_bucket_violin.png")

        # ------------------------------
        # 5e) Scatter + Regression: Sentiment vs. Price Change
        # ------------------------------
        plt.figure(figsize=(7, 5))
        sns.regplot(
            data=tsla_sent,
            x="avg_sentiment",
            y="close_pct_change",
            scatter_kws={'alpha': 0.4}
        )
        plt.title("TSLA: Sentiment vs Price Change (Regression)")
        plt.xlabel("Average Sentiment (30min)")
        plt.ylabel("Price Change")
        save_and_show("tsla_sentiment_regression.png")

        # ------------------------------
        # 5f) Durchschnittliche Returns pro Sentiment-Gruppe
        # ------------------------------
        mean_returns = tsla_sent.groupby("sentiment_bucket")["close_pct_change"].mean()
        mean_returns = mean_returns.reindex(["negative", "neutral", "positive"])

        plt.figure(figsize=(7, 5))
        mean_returns.plot(kind="bar")
        plt.title("TSLA: Average Price Change by News Sentiment")
        plt.xlabel("News Sentiment")
        plt.ylabel("Average Price Change")
        save_and_show("tsla_avg_return_by_sentiment.png")

print("\n✅ DATA UNDERSTANDING FINISHED.")
