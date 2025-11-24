import numpy as np

from streamlit_app import bsm_call_value, bsm_put_value


def test_bsm_prices_positive():
    call = bsm_call_value(100, 100, 1.0, 0, 0.01, 0.2)
    put = bsm_put_value(100, 100, 1.0, 0, 0.01, 0.2)
    assert call > 0
    assert put > 0


def test_call_vs_put_parity_direction():
    call = bsm_call_value(100, 90, 1.0, 0, 0.01, 0.2)
    put = bsm_put_value(100, 90, 1.0, 0, 0.01, 0.2)
    assert call > put
