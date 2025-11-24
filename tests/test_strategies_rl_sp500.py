import pandas as pd

from streamlit_app import load_next_csv


def test_sp500_dataset_head():
    df = load_next_csv("reinforcement_trading_strategy/data/SP500.csv", parse_dates=["Date"])
    assert not df.empty
    assert {"Close", "Open", "Volume"}.issubset(df.columns)
