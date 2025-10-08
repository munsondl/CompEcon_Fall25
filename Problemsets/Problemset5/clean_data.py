"""
clean_data.py

Contains the function `clean_data` that downloads (or reads cached) FRED series,
performs basic transformations, aligns frequencies, and returns a cleaned DataFrame.

Saves a CSV copy to ../data/fred_data.csv for reproducibility.
"""

from pathlib import Path
import pandas as pd
import pandas_datareader.data as web
import os
from datetime import datetime

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)

def clean_data(start="2003-01-01", end=None, use_cache=True):
    """
    Download and clean FRED data for the project.

    Args:
        start (str): start date (YYYY-MM-DD).
        end (str or None): end date (YYYY-MM-DD). If None, uses today.
        use_cache (bool): if True and data file exists, read from cache.

    Returns:
        pd.DataFrame: monthly-aligned DataFrame containing:
            - FEDFUNDS (Effective Federal Funds Rate)
            - T5YIE (5-year breakeven inflation)
            - CPIAUCSL (CPI All Urban Consumers)
            - UNRATE (Unemployment rate)
    Side effects:
        - Saves the cleaned data to data/fred_data.csv
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    cache_path = DATA_DIR / "fred_data.csv"
    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=True, index_col=0)
        return df

    series = {
        "FEDFUNDS": "FEDFUNDS",
        "T5YIE": "T5YIE",
        "CPIAUCSL": "CPIAUCSL",
        "UNRATE": "UNRATE"
    }

    # Fetch via pandas_datareader (FRED)
    df = web.DataReader(list(series.values()), "fred", start, end)
    # pandas_datareader returns wide DataFrame with same column names
    # Resample/align to monthly end (use .resample('M').last() for daily series like T5YIE)
    df = df.resample("M").last()

    # Basic transforms: e.g., compute inflation (12-month % change) if needed
    df["CPI_YoY"] = df["CPIAUCSL"].pct_change(12) * 100  # percent change

    # Drop rows with missing T5YIE or FEDFUNDS
    df = df.dropna(subset=["T5YIE", "FEDFUNDS"])

    # Save cache
    df.to_csv(cache_path)
    return df

if __name__ == "__main__":
    # allows running from command line
    df = clean_data()
    print("Data cleaned. Rows:", len(df))
    print(df.tail())
