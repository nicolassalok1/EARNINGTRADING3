import numpy as np

from streamlit_app import simulate_gbm, option_replication, bsm_call_value


def test_simulate_gbm_length_and_positive():
    path = simulate_gbm(S0=100, T=1.0, r=0.01, sigma=0.2, steps=12)
    assert len(path) == 13
    assert np.all(path > 0)


def test_option_replication_outputs_columns():
    path = simulate_gbm(S0=100, T=1.0, r=0.01, sigma=0.2, steps=12)
    rep = option_replication(path, K=100, T=1.0, r=0.01, sigma=0.2)
    for col in ["St", "C", "V", "s", "b"]:
        assert col in rep.columns


def test_bsm_call_value_monotonic_strike():
    low = bsm_call_value(100, 90, 1.0, 0, 0.01, 0.2)
    high = bsm_call_value(100, 110, 1.0, 0, 0.01, 0.2)
    assert low > high
