import pandas as pd

from streamlit_app import load_next_csv


def test_btc_ma_strategy_perf_available():
    df_btc = load_next_csv("bitcoin_trading_strategy/data/BitstampData.csv")
    df_btc = df_btc.dropna().copy()
    df_btc["Date"] = pd.to_datetime(df_btc["Timestamp"], unit="s")
    df_btc = df_btc.set_index("Date").sort_index()
    short = 5
    long = 20
    df_btc["ret"] = df_btc["Close"].pct_change()
    df_btc["sma_s"] = df_btc["Close"].rolling(short).mean()
    df_btc["sma_l"] = df_btc["Close"].rolling(long).mean()
    df_btc["signal"] = (df_btc["sma_s"] - df_btc["sma_l"]).apply(lambda x: 1 if x > 0 else -1)
    df_btc["strat_ret"] = df_btc["signal"].shift(1) * df_btc["ret"]
    perf = (1 + df_btc[["ret", "strat_ret"]].dropna()).cumprod()
    assert "strat_ret" in perf.columns
