"""
train_model.py
Trains and compares Linear Regression, Random Forest, and Gradient Boosting
on the cleaned CarDekho dataset. Saves the best model and a results table.

Run: python src/train_model.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


NUMERIC_FEATURES = [
    "car_age", "km_driven", "mileage_kmpl", "engine_cc",
    "max_power_bhp", "seats", "owner_num",
]
CATEGORICAL_FEATURES = ["brand", "fuel", "seller_type", "transmission"]
TARGET = "selling_price"


def load_clean(path="data/processed/car_clean.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def build_pipeline(model) -> Pipeline:
    """Wrap a model in a preprocessing pipeline: handle missing numeric values 
    with SimpleImputer (leak-free), one-hot encode categoricals."""
    
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def evaluate(name, pipeline, X_test, y_test) -> dict:
    preds = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


def main():
    df = load_clean()

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.08, random_state=42
        ),
    }

    results = []
    fitted = {}

    for name, model in candidates.items():
        pipe = build_pipeline(model)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        metrics = evaluate(name, pipe, X_test, y_test)
        results.append(metrics)
        print(f"{name:20s} RMSE={metrics['rmse']:,.0f}  MAE={metrics['mae']:,.0f}  R2={metrics['r2']:.4f}")

    results_df = pd.DataFrame(results).sort_values("rmse")
    results_df.to_csv("models/results.csv", index=False)
    print("\nSaved comparison table -> models/results.csv")

    best_name = results_df.iloc[0]["model"]
    best_pipe = fitted[best_name]
    print(f"\nBest model: {best_name}")

    joblib.dump(best_pipe, "models/best_model.pkl")
    print("Saved best model -> models/best_model.pkl")

    with open("models/best_model_name.json", "w") as f:
        json.dump({"best_model": best_name}, f)

    # --- Feature importance for tree models ---
    if best_name in ("Random Forest", "Gradient Boosting"):
        ohe = best_pipe.named_steps["preprocess"].named_transformers_["cat"]
        cat_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES)
        all_names = NUMERIC_FEATURES + list(cat_names)

        importances = best_pipe.named_steps["model"].feature_importances_
        imp_df = pd.DataFrame({"feature": all_names, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False).head(15)
        imp_df.to_csv("models/feature_importance.csv", index=False)

        plt.figure(figsize=(8, 6))
        plt.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
        plt.title(f"Top 15 Feature Importances ({best_name})")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig("visuals/feature_importance.png", dpi=120)
        print("Saved feature importance chart -> visuals/feature_importance.png")

    # --- Predicted vs. actual scatter for the best model ---
    preds = best_pipe.predict(X_test)
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, preds, alpha=0.3)
    lims = [0, max(y_test.max(), preds.max())]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("Actual Selling Price")
    plt.ylabel("Predicted Selling Price")
    plt.title(f"Predicted vs Actual ({best_name})")
    plt.tight_layout()
    plt.savefig("visuals/predicted_vs_actual.png", dpi=120)
    print("Saved predicted vs actual plot -> visuals/predicted_vs_actual.png")

    return results_df, best_name


if __name__ == "__main__":
    main()