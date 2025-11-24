from streamlit_app import load_next_csv, pca_svd


def test_yield_curve_build_pca_shapes():
    yc = load_next_csv("yield_curve_construction/data/DownloadedData.csv", parse_dates=["DATE"]).set_index("DATE")
    comps, scores, explained = pca_svd(yc.values, n_components=3)
    assert comps.shape[0] == 3
    assert scores.shape[0] == yc.shape[0]
    assert len(explained) == 3
