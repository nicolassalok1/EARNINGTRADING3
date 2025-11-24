import numpy as np
from streamlit_app import load_next_csv, inverse_variance_weights


def test_crypto_allocation_weights_sum():
    df_crypto = load_next_csv("crypto_portfolio.csv", parse_dates=["Date"]).set_index("Date")
    ret = df_crypto.pct_change().dropna()
    cov = ret.cov()
    w = inverse_variance_weights(cov.values)
    assert np.isclose(w.sum(), 1.0)
