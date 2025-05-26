import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

def train_regime_models(df, features, target_col="target", regime_col="regime_label"):
    models = {}
    regime_scores = {}

    for regime in df[regime_col].dropna().unique():
        subset = df[df[regime_col] == regime].dropna(subset=features + [target_col])
        X = subset[features]
        y = subset[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = xgb.XGBClassifier(eval_metric="logloss")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        regime_scores[regime] = acc
        models[regime] = model

    return models, regime_scores


def predict_signals(df, features, models, regime_col="regime_label"):
    df = df.copy()
    df["model_signal"] = np.nan

    for regime, model in models.items():
        subset = df[df[regime_col] == regime].dropna(subset=features)
        X = subset[features]
        preds = model.predict(X)
        df.loc[subset.index, "model_signal"] = preds

    return df


def apply_strategy(df, txn_cost=0.001):
    df = df.copy()
    df["strategy_return"] = 0

    trade_days = (df["regime_label"] == "Bull") & (df["model_signal"] == 1)
    df.loc[trade_days, "strategy_return"] = df["returns"]

    # transaction cost
    df["strategy_return_net"] = df["strategy_return"]
    df.loc[trade_days, "strategy_return_net"] -= txn_cost

    # cumulative returns
    df["cumulative_market"] = (1 + df["returns"]).cumprod()
    df["cumulative_strategy_net"] = (1 + df["strategy_return_net"]).cumprod()

    # Drawdown
    df["rolling_max"] = df["cumulative_strategy_net"].cummax()
    df["drawdown"] = df["cumulative_strategy_net"] / df["rolling_max"] - 1

    return df
