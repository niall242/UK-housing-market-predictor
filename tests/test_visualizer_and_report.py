# tests/test_visualizer_and_report.py
from pathlib import Path
import pandas as pd
from matplotlib.figure import Figure

from modules.visualizer import Visualizer
from modules.report_generator import ReportGenerator


def make_demo_df():
    # Two regions × three months
    data = {
        "Date": pd.date_range("2024-01-01", periods=3, freq="MS").tolist() * 2,
        "Region": ["London"] * 3 + ["Manchester"] * 3,
        "Price": [500000, 505000, 510000, 250000, 255000, 260000],
    }
    return pd.DataFrame(data)


def test_trend_line_returns_figure_object():
    df = make_demo_df()
    vis = Visualizer()
    fig = vis.trend_line(df, "Trend test")
    assert isinstance(fig, Figure)
    # Should have at least one line drawn
    assert len(fig.axes[0].lines) > 0


def test_bar_growth_returns_figure_object():
    df = make_demo_df()
    vis = Visualizer()
    fig = vis.bar_growth(df, "Bar test")
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].patches) > 0  # bars drawn


def test_report_generator_creates_files(tmp_path: Path):
    rep = ReportGenerator()
    summary = {"rows_loaded": 3, "region": "London"}
    dummy_fig = Visualizer().trend_line(make_demo_df(), "Dummy")
    metrics = {"MAE": 123.4, "RMSE": 456.7}
    forecasts = pd.DataFrame({
        "Date": pd.date_range("2024-04-01", periods=2, freq="MS"),
        "Region": ["London", "London"],
        "PropertyType": ["Detached", "Detached"],
        "PredictedPrice": [520000, 530000],
    })

    rep.compile(summary, [dummy_fig], metrics, forecasts)
    html_path = rep.export(str(tmp_path / "out.html"))
    csv_path = rep.export_csv(forecasts, str(tmp_path / "out.csv"))


    # Both files should exist and contain something
    assert Path(html_path).exists()
    assert Path(csv_path).exists()
    assert Path(html_path).stat().st_size > 0
    assert Path(csv_path).stat().st_size > 0
