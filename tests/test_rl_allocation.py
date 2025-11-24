import numpy as np

from streamlit_app import inverse_variance_weights


def test_inverse_variance_weights_normalized():
    cov = np.array([[0.1, 0.0], [0.0, 0.2]])
    w = inverse_variance_weights(cov)
    assert w.shape == (2,)
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w > 0)
