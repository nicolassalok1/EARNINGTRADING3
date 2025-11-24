import numpy as np

from streamlit_app import simulate_gbm, option_replication


def test_replication_has_expected_length():
    steps = 30
    path = simulate_gbm(S0=100, T=1.0, r=0.01, sigma=0.2, steps=steps)
    rep = option_replication(path, K=100, T=1.0, r=0.01, sigma=0.2)
    assert len(rep) == steps - 1
    assert not rep.isna().any().any()
