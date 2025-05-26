import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import yfinance as yf

from src.get_macro_data import download_macro_data

import matplotlib.pyplot as plt
import pandas as pd

from src.features import add_technical_indicators, add_macro_features
from src.get_macro_data import download_macro_data
from src.regime_models import cluster_kmeans, cluster_gmm, cluster_hmm, auto_label_regimes
from src.strategy import train_regime_models, predict_signals, apply_strategy
from src.backtest import calculate_performance

# Sidebar Controls 

st.sidebar.title("📊 Strategy Configuration")

ticker = st.sidebar.selectbox(
    "Choose a stock:",
    ["SPY", "AAPL", "QQQ", "DIA", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "VTI"]
)

clustering_method = st.sidebar.selectbox(
    "Clustering Method:",
    ["KMeans", "GMM", "HMM"]
)

show_metrics = st.sidebar.checkbox("Show Performance Table", value=True)

# Load & Preprocess Data 

import os
import pandas as pd

def load_data(ticker):
    filepath = f"data/{ticker}.csv"

    if os.path.exists(filepath):
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        st.sidebar.success(f"✅ Loaded {ticker} from disk.")
    else:
        df = yf.download(ticker, start="2010-01-01", auto_adjust=False)

        df.columns = df.columns.get_level_values(0)
        df.columns.name = None

        df.to_csv(filepath)
        st.sidebar.success(f"📥 Downloaded and saved {ticker} to data/{ticker}.csv")

    df = add_technical_indicators(df)
    return df

load = st.sidebar.button("📥 Load Data")

if load:
    
    df = load_data(ticker)
    macro = download_macro_data()
    df = add_macro_features(df, macro)

else:
    st.warning("⚠️ Click 'Load Data' to begin.")
    st.stop()

# Regime Detection

if clustering_method == "KMeans":
    df, _ = cluster_kmeans(df)
    df, _ = auto_label_regimes(df, "regime_kmeans")
elif clustering_method == "GMM":
    df, _ = cluster_gmm(df)
    df, _ = auto_label_regimes(df, "regime_gmm")
elif clustering_method == "HMM":
    df, _ = cluster_hmm(df)
    df, _ = auto_label_regimes(df, "regime_hmm")

# Strategy Logic

features = ["volatility", "rsi", "macd", "cpi", "fed_funds", "yield_spread"]
df["target"] = (df["returns"].shift(-1) > 0).astype(int)

models, scores = train_regime_models(df, features)
df = predict_signals(df, features, models)
df = apply_strategy(df)

# Plot Cumulative Return

st.title("📈 Alpha Regime Dashboard")
st.subheader(f"Cumulative Returns — {ticker} ({clustering_method})")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df.index, df["cumulative_market"], label="Buy & Hold", color="gray")
ax.plot(df.index, df["cumulative_strategy_net"], label="Strategy (Net)", color="green")
ax.set_ylabel("Cumulative Return")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Performance Metrics

if show_metrics:
    st.subheader("📊 Performance Metrics")
    metrics = calculate_performance(df)
    col1, col2, col3 = st.columns(3)
    col1.metric("CAGR", f"{metrics['strategy_cagr']:.2%}")
    col2.metric("Sharpe Ratio", f"{metrics['strategy_sharpe']:.2f}")
    col3.metric("Sortino Ratio", f"{metrics['strategy_sortino']:.2f}")

    col1.metric("Max Drawdown", f"{metrics['strategy_max_drawdown']:.2%}")
    col2.metric("Win Rate", f"{metrics['win_rate']:.2%}")
    col3.metric("Trades Taken", int(metrics['trades']))