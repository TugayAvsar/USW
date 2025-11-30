"""
DATA UNDERSTANDING

Dieses Skript analysiert die vorbereiteten Feature-Datensätze
für Tesla (TSLA), Ford (F) und General Motors (GM).

Ziele:
1) Relevante Daten-Spalten erklären
2) Deskriptive Statistik der Variablen anzeigen
3) Relevante Plots der wichtigsten Variablen erzeugen
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "../../data/features"
SYMBOLS = ["tsla", "f", "gm"]  # alle drei Aktien

dfs = {}  # DataFrames je Symbol

# ---------------------------------------------------------
# 1) Datensätze laden
# ---------------------------------------------------------
for symbol in SYMBOLS:
    path = os.path.join(DATA_PATH, f"{symbol}_features_30min.csv")
    df = pd.read_csv(path)

    # Sicherstellen, dass wir eine Zeitspalte 'date' haben
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.rename(columns={"datetime": "date"})
    else:
        raise ValueError(f"{symbol}: Weder 'date' noch 'datetime' Spalte gefunden!")

    dfs[symbol] = df
    print(f"Loaded {symbol}: {df.shape[0]} rows")

# ---------------------------------------------------------
# 2) Spalten erklären
# ---------------------------------------------------------
print("\nCOLUMN EXPLANATION:")
print("""
date                = 30-Minuten Zeitstempel
open, high, low     = Preisbereiche der Periode
close               = Schlusskurs dieser Periode
volume              = gehandeltes Volumen
close_pct_change    = relative Veränderung gegenüber letzter Periode
sma_5, sma_10       = Trendindikatoren über 5/10 Perioden (simple)
ema_5, ema_10       = Trendindikatoren (exponentiell, stärker gewichtet)
rsi_14              = Momentum, >70 überkauft, <30 überverkauft
target              = 1 wenn nächster Schlusskurs höher, sonst 0
(ggf. close_next    = Hilfsspalte für Target-Berechnung)
""")

# ---------------------------------------------------------
# 3) Deskriptive Statistik
# ---------------------------------------------------------
for symbol, df in dfs.items():
    print(f"\n===== DESCRIPTIVE STATISTICS FOR {symbol.upper()} =====")
    print(df.describe())
    print("\nTarget distribution:")
    print(
        df["target"]
        .value_counts(normalize=True)
        .rename("proportion")
        .round(3)
    )

# ---------------------------------------------------------
# 4) Plots (save + show)
# ---------------------------------------------------------

IMG_PATH = "../../images"
os.makedirs(IMG_PATH, exist_ok=True)

def save_and_show(filename: str):
    """Speichert aktuelle Figure und zeigt sie an."""
    filepath = os.path.join(IMG_PATH, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    print(f"📁 Saved plot → {filepath}")
    plt.show()


# 4.1 Cumulative Percentage Return Comparison
plt.figure(figsize=(14, 6))

for symbol, df in dfs.items():
    df["cum_return"] = (1 + df["close_pct_change"].fillna(0)).cumprod() - 1
    plt.plot(df["date"], df["cum_return"], label=symbol.upper(), linewidth=1.5)

plt.title("Cumulative Return (30min) – TSLA vs F vs GM")
plt.xlabel("Time")
plt.ylabel("Cumulative Return (%)")
plt.legend()
save_and_show("cum_return_comparison.png")


# 4.2 RSI comparison
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


# 4.3 Distribution of percentage changes
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
plt.title("Distribution of Close Percentage Change (30min Returns)")
plt.xlabel("Percentage Change")
plt.ylabel("Density")
plt.legend()
save_and_show("pct_change_distribution.png")


# 4.4 Target distribution comparison
target_df = pd.DataFrame(
    {sym.upper(): dfs[sym]["target"].value_counts(normalize=True)
     for sym in dfs}
).T

target_df.plot(kind="bar", figsize=(8, 5),
               title="Target Distribution (Up vs Down) – TSLA vs F vs GM")
plt.ylabel("Proportion")
save_and_show("target_distribution.png")

print("\n✅ DATA UNDERSTANDING FINISHED. Plots saved in /images")
