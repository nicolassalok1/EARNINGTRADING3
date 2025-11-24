import pandas as pd

from streamlit_app import load_next_csv


def test_nlp_sentiment_files_loadable():
    files = [
        "strategies_nlp_trading/data/Step4_DataWithSentimentsResults.csv",
        "strategies_nlp_trading/data/Step3_NewsAndReturnData.csv",
        "strategies_nlp_trading/data/Step2.2_ReturnData.csv",
        "strategies_nlp_trading/data/LexiconData.csv",
        "strategies_nlp_trading/data/LabelledNewsData.csv",
    ]
    for path in files:
        df = load_next_csv(path)
        assert not df.empty
