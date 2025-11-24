import pandas as pd

from streamlit_app import load_next_csv, train_test_split_series, polyfit_predict


def test_stock_prediction_dataset_loads():
    df = load_next_csv("stock_return_prediction/data/data.csv", parse_dates=["Date"])
    assert not df.empty
    assert "Close" in df.columns


def test_stock_prediction_polyfit_shapes():
    df = load_next_csv("stock_return_prediction/data/data.csv", parse_dates=["Date"])
    series = df.set_index("Date")["Close"].dropna().head(100)
    train, test = train_test_split_series(series, 0.8)
    preds_train, next_pred = polyfit_predict(train, deg=2)
    assert len(preds_train) == len(train)
    assert isinstance(next_pred, float)
