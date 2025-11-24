import numpy as np

from streamlit_app import load_next_csv


def test_eigen_portfolio_weights_sum():
    df = load_next_csv("portfolio_eigen/data/Dow_adjcloses.csv", parse_dates=["Date"]).set_index("Date")
    ret = df.pct_change().dropna()
    cov = ret.cov()
    vals, vecs = np.linalg.eigh(cov.values)
    idx = vals.argsort()[::-1]
    top_vec = vecs[:, idx[0]]
    weights = top_vec / np.sum(np.abs(top_vec))
    assert np.isclose(np.sum(np.abs(weights)), 1.0)
