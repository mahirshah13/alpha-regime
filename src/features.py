import pandas as pd
import numpy as np


def compute_returns_volatility(df, window=10):
    df["returns"] = df["Adj Close"].pct_change()
    df["volatility"] = df["returns"].rolling(window).std()
    return df

def compute_rsi(df, window=14):
    delta = df["Adj Close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    ma_up = up.rolling(window=window).mean()
    ma_down = down.rolling(window=window).mean()

    rs = ma_up / (ma_down + 1e-6)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df

def compute_macd(df, short=12, long=26, signal=9):
    ema_short = df["Adj Close"].ewm(span=short, adjust=False).mean()
    ema_long = df["Adj Close"].ewm(span=long, adjust=False).mean()

    df["macd"] = ema_short - ema_long
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    return df

def compute_bollinger_bands(df, window=20, num_std=2):
    rolling_mean = df["Adj Close"].rolling(window).mean()
    rolling_std = df["Adj Close"].rolling(window).std()

    df["bollinger_upper"] = rolling_mean + num_std * rolling_std
    df["bollinger_lower"] = rolling_mean - num_std * rolling_std
    return df

def add_technical_indicators(df):
    df = compute_returns_volatility(df)
    df = compute_rsi(df)
    df = compute_macd(df)
    df = compute_bollinger_bands(df)
    df = df.dropna()
    return df

def add_macro_features(df, macro_df):
    df = df.copy()
    macro_df = macro_df.copy()

    df.index = pd.to_datetime(df.index)
    macro_df.index = pd.to_datetime(macro_df.index)

    macro_df = macro_df.resample("D").ffill()

    df = df.loc[~df.index.duplicated()]
    macro_df = macro_df.loc[~macro_df.index.duplicated()]

    df = df.join(macro_df, how="left")
    df = df.fillna(method="ffill")

    return df
