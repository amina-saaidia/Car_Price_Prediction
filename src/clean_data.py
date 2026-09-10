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

    # --- FIX: Drop exact duplicate rows to prevent target memorization ---
    df = df.drop_duplicates()

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

    # --- 'owner' is simplified into an ordinal scale ---
    owner_map = {
        "First Owner": 1,
        "Second Owner": 2,
        "Third Owner": 3,
        "Fourth & Above Owner": 4,
        "Test Drive Car": 0,
    }
    df["owner_num"] = df["owner"].map(owner_map)

    # NOTE: Global fillna(median) and quantile(0.99) capping were removed 
    # to prevent data leakage across train/test sets. Imputation is now 
    # handled safely inside train_model.py.

    return df.reset_index(drop=True)


def save_processed(df: pd.DataFrame, path="data/processed/car_clean.csv"):
    df.to_csv(path, index=False)
    print(f"Cleaned dataset exported: {df.shape[0]} rows, {df.shape[1]} cols -> {path}")


if __name__ == "__main__":
    raw = load_raw()
    cleaned = clean(raw)
    save_processed(cleaned)