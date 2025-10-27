# tests/test_analyzer.py
import pandas as pd
import numpy as np
from modules.analyzer import HousingAnalyzer


def make_base_df():
    # Minimal realistic rows like the UK-HPI headers
    return pd.DataFrame({
        "Date": pd.to_datetime(
            pd.Series(["2024-01-01", "2024-02-01", None], dtype="object"),
            errors="coerce",
        ),
        "RegionName": ["London", " London ", "Manchester"],
        "AveragePrice": [500000, 505000, np.nan],
        # Per–property-type columns (wide format)
        "DetachedPrice": [800000, 810000, 400000],
        "SemiDetachedPrice": [600000, 605000, 300000],
        "TerracedPrice": [550000, 552000, 250000],
        "FlatPrice": [450000, 452000, 200000],
    })


def test_clean_drops_nans_and_strips_whitespace():
    df = make_base_df()
    analyzer = HousingAnalyzer()

    clean = analyzer.clean(df)

    # Third row has NaN AveragePrice and None Date → should be dropped
    assert len(clean) == 2

    # RegionName should be stripped
    assert list(clean["RegionName"]) == ["London", "London"]

    # Required columns still present
    for col in ["Date", "RegionName", "AveragePrice"]:
        assert col in clean.columns


def test_features_adds_year_month():
    df = make_base_df()
    analyzer = HousingAnalyzer()

    clean = analyzer.clean(df)
    feats = analyzer.features(clean)

    assert "Year" in feats.columns
    assert "Month" in feats.columns
    assert feats.loc[feats.index[0], "Year"] == 2024
    assert feats.loc[feats.index[0], "Month"] == 1


def test_melt_property_prices_produces_long_tidy_table():
    df = make_base_df()
    analyzer = HousingAnalyzer()

    long_df = analyzer.melt_property_prices(df)

    # Expected long columns
    assert set(["Date", "RegionName", "PropertyType", "Price"]).issubset(long_df.columns)

    # Two non-NaN base rows × 4 property types = 8 long rows
    # (the third base row has NaNs in AveragePrice but typed prices are present;
    # our melt drops NaN "Price" rows, so we count only non-NaN)
    # Here, all typed prices exist → expect 3 rows × 4 = 12
    assert len(long_df) == 12

    # PropertyType labels mapped correctly
    assert set(long_df["PropertyType"].unique()) == {"Detached", "SemiDetached", "Terraced", "Flat"}

    # Sample numeric check
    sample = long_df[(long_df["RegionName"] == "London") & (long_df["PropertyType"] == "Detached")]
    assert not sample.empty
    assert float(sample["Price"].iloc[0]) == 800000.0
