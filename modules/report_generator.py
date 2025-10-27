# modules/report_generator.py
from typing import List, Dict
import pandas as pd
from matplotlib.figure import Figure
from .contracts import IReportGenerator

class ReportGenerator(IReportGenerator):
    def __init__(self):
        self._sections: list[str] = []

    def compile(self, summary: Dict, figures: List[Figure], metrics: Dict, forecasts: pd.DataFrame) -> None:
        self._sections.append("<h1>UK Housing Market – Summary</h1>")
        self._sections.append("<h2>Key Metrics</h2>")
        self._sections.append("<pre>" + repr(metrics) + "</pre>")
        self._sections.append("<h2>Top-line Stats</h2>")
        self._sections.append("<pre>" + repr(summary) + "</pre>")
        # Figures will be saved by controller – here we just reference filenames.
        self._sections.append("<h2>Figures</h2><p>See exported PNGs.</p>")
        self._sections.append("<h2>Forecasts</h2>")
        self._sections.append(forecasts.head(10).to_html(index=False))

    def export(self, path: str = "report.html") -> str:
        html = "<html><body>" + "\n".join(self._sections) + "</body></html>"
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return path

    def export_csv(self, forecasts: pd.DataFrame, path: str = "forecasts.csv") -> str:
        forecasts.to_csv(path, index=False)
        return path
