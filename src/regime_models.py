import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import ruptures as rpt


def cluster_kmeans(df, n_clusters=3):
    df = df.copy()
    features = ["returns", "volatility"]
    X = df[features].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    regimes = kmeans.fit_predict(X_scaled)

    df.loc[X.index, "regime_kmeans"] = regimes
    return df, kmeans


def cluster_gmm(df, n_components=3):
    df = df.copy()
    features = ["returns", "volatility"]
    X = df[features].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=n_components, random_state=42)
    regimes = gmm.fit_predict(X_scaled)

    df.loc[X.index, "regime_gmm"] = regimes
    return df, gmm


def cluster_hmm(df, n_states=3):
    df = df.copy()
    features = ["returns", "volatility"]
    X = df[features].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    hmm = GaussianHMM(n_components=n_states, covariance_type="full", random_state=42)
    hmm.fit(X_scaled)
    regimes = hmm.predict(X_scaled)

    df.loc[X.index, "regime_hmm"] = regimes
    return df, hmm


def cluster_spectral(df, n_clusters=3):
    df = df.copy()
    features = ["returns", "volatility"]
    X = df[features].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
    regimes = spectral.fit_predict(X_scaled)

    df.loc[X.index, "regime_spectral"] = regimes
    return df, spectral


def detect_changepoints(df, model="rbf", n_bkps=5):
    df = df.copy()
    signal = df["returns"].dropna().values.reshape(-1, 1)

    algo = rpt.Binseg(model=model).fit(signal)
    breakpoints = algo.predict(n_bkps=n_bkps)

    cp_series = np.zeros(len(signal))
    for bp in breakpoints[:-1]:
        cp_series[bp:] += 1

    cp_index = df["returns"].dropna().index
    df.loc[cp_index, "regime_cp"] = cp_series
    return df, breakpoints

def auto_label_regimes(df, regime_col: str = "regime_kmeans"):
    """
    Map numeric regime clusters to interpretable labels based on average return and volatility.
    Example output: Bull, Bear, Volatile.
    """
    df = df.copy()
    stats = df.groupby(regime_col)["returns"].mean().sort_values(ascending=False)

    if len(stats) < 3:
        raise ValueError("Need at least 3 regimes to label as Bull/Bear/Volatile")

    # Map clusters by sorted average return
    sorted_clusters = stats.index.tolist()
    label_map = {
        sorted_clusters[0]: "Bull",
        sorted_clusters[1]: "Volatile",
        sorted_clusters[2]: "Bear"
    }

    df["regime_label"] = df[regime_col].map(label_map)
    return df, label_map
