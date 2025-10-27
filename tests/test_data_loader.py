# tests/test_data_loader.py
import textwrap
from pathlib import Path
import pandas as pd
import pytest

from modules.data_loader import DataLoader
from modules.contracts import DataLoadError


def write_csv(p: Path, content: str) -> Path:
    p.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    return p


def test_load_data_ok(tmp_path: Path):
    # minimal valid CSV with required columns
    csv_path = write_csv(
        tmp_path / "ok.csv",
        """
        Date,RegionName,AveragePrice
        2024-01-01,London,500000
        2024-02-01,London,505000
        """
    )
    loader = DataLoader(str(csv_path))
    df = loader.load_data()
    assert not df.empty
    assert set(["Date", "RegionName", "AveragePrice"]).issubset(df.columns)
    # Ensure date parsed
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])


def test_load_data_missing_columns_raises(tmp_path: Path):
    # Missing AveragePrice on purpose
    csv_path = write_csv(
        tmp_path / "missing_cols.csv",
        """
        Date,RegionName
        2024-01-01,London
        """
    )
    loader = DataLoader(str(csv_path))
    with pytest.raises(DataLoadError) as exc:
        loader.load_data()
    assert "Missing required columns" in str(exc.value)


def test_load_data_bad_date_raises(tmp_path: Path):
    # Bad/non-coercible date on purpose
    csv_path = write_csv(
        tmp_path / "bad_date.csv",
        """
        Date,RegionName,AveragePrice
        NOT_A_DATE,London,500000
        """
    )
    loader = DataLoader(str(csv_path))
    with pytest.raises(DataLoadError) as exc:
        loader.load_data()
    assert "Invalid dates detected" in str(exc.value)
