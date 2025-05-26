import pandas_datareader.data as web
import pandas as pd

def download_macro_data():
    indicators = {
        "cpi": "CPIAUCSL",
        "fed_funds": "FEDFUNDS",
        "unemployment": "UNRATE",
        "gs10": "GS10",
        "gs2": "GS2"
    }

    df = pd.DataFrame()
    for name, code in indicators.items():
        series = web.DataReader(code, "fred", start="2000-01-01")
        df[name] = series[code]

    df["yield_spread"] = df["gs10"] - df["gs2"]
    df = df.fillna(method="ffill")

    return df
