# modules/visualizer.py

import matplotlib
matplotlib.use("Agg")  # use non-GUI backend for headless/test environments

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from .contracts import IVisualizer

class Visualizer(IVisualizer):
    def trend_line(self, df: pd.DataFrame, title: str) -> Figure:
        fig, ax = plt.subplots()
        for key, sub in df.groupby("Region"):
            sub = sub.sort_values("Date")
            ax.plot(sub["Date"], sub["Price"], label=key)
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (£)")
        ax.legend(loc="best")
        return fig

    def bar_growth(self, df: pd.DataFrame, title: str) -> Figure:
        agg = df.groupby("Region")["Price"].mean().sort_values(ascending=False).head(10)
        labels = agg.index.astype(str).tolist()
        heights = agg.to_numpy(dtype=float)

        fig, ax = plt.subplots()
        ax.bar(labels, heights)
        ax.set_title(title)
        ax.set_xlabel("Region")
        ax.set_ylabel("Avg Price (£)")
        ax.tick_params(axis="x", rotation=45)
        return fig

