# app_controller.py
from pathlib import Path
import pandas as pd
from modules.contracts import IDataLoader, IAnalyzer, IPredictor, IVisualizer, IReportGenerator
from matplotlib.figure import Figure
from typing import Optional

class AppController:
    def __init__(
        self,
        loader: IDataLoader,
        analyzer: IAnalyzer,
        predictor: IPredictor,
        visualizer: IVisualizer,
        reporter: IReportGenerator,
        artifacts_dir: str = "artifacts",
    ):
        self.loader = loader
        self.analyzer = analyzer
        self.predictor = predictor
        self.visualizer = visualizer
        self.reporter = reporter
        self.artifacts = Path(artifacts_dir)
        self.artifacts.mkdir(parents=True, exist_ok=True)

    def run(self, region: str, property_type: Optional[str] = None, forecast_months: int = 6) -> None:
        # 1) Load
        raw = self.loader.load_data()

        # 2) Clean + features
        clean = self.analyzer.clean(raw)
        feats = self.analyzer.features(clean)

        '''# 3) Filter series for chosen region + property type
        series = feats[(feats["Region"] == region) & (feats["PropertyType"] == property_type)].copy()
        if series.empty:
            raise ValueError(f"No rows for region={region} & property_type={property_type}")'''
        
        # 3) Filter series for chosen region (+ optional property type)
        if property_type:
            long_df = self.analyzer.melt_property_prices(feats)
            series = long_df[(long_df["RegionName"] == region) & (long_df["PropertyType"] == property_type)].copy()
        else:
            # Use overall average price when no property type is selected
            series = feats[feats["RegionName"] == region][["Date", "RegionName", "AveragePrice"]].copy()
            series = series.rename(columns={"AveragePrice": "Price"})

        if series.empty:
            raise ValueError(f"No rows for region={region} & property_type={property_type}")

        # 4) Prepare X, y (simple baseline – time index → price)
        series = series.sort_values("Date")
        series["t"] = range(len(series))
        X = series[["t"]].values
        y = series["Price"].values

        # split last 6 points for validation if available
        split = max(len(series) - 6, 1)
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]

        # 5) Train + validate
        self.predictor.train(X_train, y_train)
        metrics = self.predictor.validate(X_val, y_val) if len(X_val) else {"MAE": None, "RMSE": None}

        # 6) Forecast
        last_t = int(series["t"].iloc[-1])
        X_future = [[t] for t in range(last_t + 1, last_t + 1 + forecast_months)]
        y_future = self.predictor.forecast(X_future)

        future_dates = pd.date_range(series["Date"].iloc[-1] + pd.offsets.MonthBegin(), periods=forecast_months, freq="MS")
        forecasts = pd.DataFrame({
            "Date": future_dates,
            "Region": region,
            "PropertyType": property_type,
            "PredictedPrice": y_future,
        })

        # 7) Visuals
        plot_df = series.rename(columns={"RegionName": "Region"})  # visualizer groups by 'Region'
        trend_fig: Figure = self.visualizer.trend_line(
            plot_df[["Date", "Region", "Price"]],
            f"Trend – {region}" + (f" / {property_type}" if property_type else "")
        )
        trend_path = self.artifacts / "trend.png"
        trend_fig.savefig(trend_path, bbox_inches="tight")


        # 8) Summary
        summary = {
            "rows_loaded": len(raw),
            "rows_clean": len(clean),
            "region": region,
            "property_type": property_type,
        }

        # 9) Report
        self.reporter.compile(summary=summary, figures=[trend_fig], metrics=metrics, forecasts=forecasts)
        report_path = self.reporter.export(str(self.artifacts / "report.html"))
        csv_path = self.reporter.export_csv(forecasts, str(self.artifacts / "forecasts.csv"))


        print(f"Saved: {trend_path}")
        print(f"Saved: {report_path}")
        print(f"Saved: {csv_path}")
