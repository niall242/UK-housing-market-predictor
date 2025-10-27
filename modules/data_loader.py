# modules/data_loader.py
import pandas as pd
from .contracts import IDataLoader, DataLoadError

# OLD
# REQUIRED_COLS = {"Date", "Region", "PropertyType", "Price"}

# NEW
REQUIRED_COLS = {"Date", "RegionName", "AveragePrice"}

class DataLoader(IDataLoader):
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.filepath)
        except FileNotFoundError as e:
            raise DataLoadError(f"File not found: {self.filepath}") from e
        self.validate_schema(df)
        return df

    def validate_schema(self, df: pd.DataFrame) -> None:
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise DataLoadError(f"Missing required columns: {sorted(missing)}")
        # light dtype normalisation
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        if df["Date"].isna().any():
            raise DataLoadError("Invalid dates detected.")

        # Normalise the columns we’ll use later
        df.rename(columns={"RegionName": "RegionName", "AveragePrice": "AveragePrice"}, inplace=True)