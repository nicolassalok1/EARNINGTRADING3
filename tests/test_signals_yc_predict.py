from streamlit_app import load_next_csv


def test_yield_curve_forecast_has_shape():
    yc = load_next_csv("yield_curve_prediction/data/DownloadedData.csv", parse_dates=["DATE"]).set_index("DATE")
    forecast = yc.ewm(alpha=0.2).mean().iloc[-1]
    assert forecast.shape[0] == yc.shape[1]
