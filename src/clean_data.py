"""
clean_data.py
Cleaning and feature engineering for the CarDekho car price dataset.

Running this directly produces data/processed/car_clean.csv from
data/raw/car_details_v3.csv
"""

import re
import pandas as pd
import numpy as np


def _to_float(value):
    """Extracts the first float from a string like '23.4 kmpl' or '1248 CC'."""
    if pd.isna(value):
        return np.nan
    match = re.search(r"[\d.]+", str(value))
    return float(match.group()) if match else np.nan


def load_raw(path="data/raw/car_details_v3.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame, current_year: int = 2026) -> pd.DataFrame:
    df = df.copy()

    # --- Rows with no target are dropped ---
    df = df.dropna(subset=["selling_price"])

    # --- Units are stripped from numeric-looking text columns ---
    df["mileage_kmpl"] = df["mileage"].apply(_to_float)
    df["engine_cc"] = df["engine"].apply(_to_float)
    df["max_power_bhp"] = df["max_power"].apply(_to_float)

    # --- Brand is extracted from the free-text 'name' column ---
    df["brand"] = df["name"].str.split().str[0].str.title()

    # --- Feature engineering ---
    df["car_age"] = current_year - df["year"]

    # --- Columns no longer needed in raw form are dropped ---
    df = df.drop(columns=["name", "mileage", "engine", "max_power", "torque"])

    # --- Missing values ---
    # Numeric columns are filled with the median (robust to outliers)
    for col in ["mileage_kmpl", "engine_cc", "max_power_bhp", "seats"]:
        df[col] = df[col].fillna(df[col].median())

    # --- Outliers ---
    # km_driven / selling_price are capped at the 99th percentile
    for col in ["km_driven", "selling_price"]:
        cap = df[col].quantile(0.99)
        df[col] = np.where(df[col] > cap, cap, df[col])

    # --- 'owner' is simplified into an ordinal scale ---
    owner_map = {
        "First Owner": 1,
        "Second Owner": 2,
        "Third Owner": 3,
        "Fourth & Above Owner": 4,
        "Test Drive Car": 0,
    }
    df["owner_num"] = df["owner"].map(owner_map)

    return df.reset_index(drop=True)


def save_processed(df: pd.DataFrame, path="data/processed/car_clean.csv"):
    df.to_csv(path, index=False)
    print(f"Cleaned dataset exported: {df.shape[0]} rows, {df.shape[1]} cols -> {path}")


if __name__ == "__main__":
    raw = load_raw()
    cleaned = clean(raw)
    save_processed(cleaned)
