import numpy as np
import pandas as pd


def calculate_performance(df, strategy_col="strategy_return_net", market_col="returns"):
    perf = {}

    strategy_returns = df[strategy_col].dropna()
    market_returns = df[market_col].dropna()

    trading_days = 252

    # Annualized returns
    perf["strategy_cagr"] = (1 + strategy_returns.mean())**trading_days - 1
    perf["market_cagr"] = (1 + market_returns.mean())**trading_days - 1

    # Sharpe ratio
    perf["strategy_sharpe"] = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(trading_days)
    perf["market_sharpe"] = (market_returns.mean() / market_returns.std()) * np.sqrt(trading_days)

    # Sortino ratio
    downside_std = strategy_returns[strategy_returns < 0].std()
    perf["strategy_sortino"] = (strategy_returns.mean() / downside_std) * np.sqrt(trading_days)

    # Max drawdown
    cum = (1 + strategy_returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = cum / rolling_max - 1
    perf["strategy_max_drawdown"] = drawdown.min()

    # Win rate
    perf["win_rate"] = (strategy_returns > 0).mean()

    # Trade count (days with position)
    perf["trades"] = (df[strategy_col] != 0).sum()

    return perf