import numpy as np

from streamlit_app import load_next_csv, pca_svd


def test_btc_pca_shapes():
    df = load_next_csv("bitcoin_trading_ml/data/BitstampData.csv")
    df = df.dropna().head(200)
    feats = ["Open", "High", "Low", "Close", "Volume_(BTC)"]
    X = df[feats].values
    comps, scores, explained = pca_svd(X, n_components=3)
    assert comps.shape == (3, len(feats))
    assert scores.shape[0] == X.shape[0]
    assert np.isclose(explained.sum(), 1.0, atol=1e-6)
