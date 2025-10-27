# modules/analyzer.py
import pandas as pd
from .contracts import IAnalyzer

class HousingAnalyzer(IAnalyzer):
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Use the real UK-HPI columns (no PropertyType/Price at this stage)
        required = ["Date", "RegionName", "AveragePrice"]
        out = out.dropna(subset=required)

        # Tidy strings
        out["RegionName"] = out["RegionName"].astype(str).str.strip()

        return out


    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["Year"] = out["Date"].dt.year
        out["Month"] = out["Date"].dt.month
        return out

    def average_price_by_region(self, df: pd.DataFrame) -> pd.Series:
        # OLD
        # return df.groupby("Region")["Price"].mean().sort_values(ascending=False)

        # NEW
        return df.groupby("RegionName")["AveragePrice"].mean().sort_values(ascending=False)


    def rolling_mean(self, df: pd.DataFrame, window: int = 12) -> pd.DataFrame:
        out = df.sort_values("Date").copy()
        value_col = "Price" if "Price" in out.columns else "AveragePrice"
        group_keys = ["Region", "PropertyType"] if "PropertyType" in out.columns else ["RegionName"]
        out["RollingMean12"] = out.groupby(group_keys)[value_col].transform(
            lambda s: s.rolling(window=window, min_periods=1).mean()
        )
        return out



    def melt_property_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        price_cols = ["DetachedPrice", "SemiDetachedPrice", "TerracedPrice", "FlatPrice"]
        present = [c for c in price_cols if c in out.columns]
        long_df = out.melt(
            id_vars=["Date", "RegionName"],
            value_vars=present,
            var_name="PropertyTypeCol",
            value_name="Price"
        ).dropna(subset=["Price"])
        mapping = {
            "DetachedPrice": "Detached",
            "SemiDetachedPrice": "SemiDetached",
            "TerracedPrice": "Terraced",
            "FlatPrice": "Flat",
        }
        long_df["PropertyType"] = long_df["PropertyTypeCol"].map(mapping).fillna(long_df["PropertyTypeCol"])
        return long_df[["Date", "RegionName", "PropertyType", "Price"]]
