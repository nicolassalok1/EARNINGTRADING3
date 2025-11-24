import numpy as np

from streamlit_app import train_test_split_series, load_raw_data, set_global_seed


def test_train_test_split_series_shapes():
    series = load_raw_data().iloc[:, 0].dropna().head(50)
    train, test = train_test_split_series(series, train_ratio=0.8)
    assert len(train) + len(test) == len(series)
    assert len(train) >= 5


def test_set_global_seed_reproducible():
    set_global_seed(123)
    a = np.random.rand(3)
    set_global_seed(123)
    b = np.random.rand(3)
    assert np.allclose(a, b)
