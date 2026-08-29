# Car Price Prediction 🚗

Predicting the resale price of used cars using regression models, built as
part of the EXPS Nexus Data Science Internship (Algeria Pilot Cohort 01).

## Overview

This project uses the **CarDekho Vehicle Dataset** (8,128 real used-car
listings from India) to predict `selling_price` from features like car age,
kilometers driven, fuel type, transmission, engine size, and horsepower.

**Task requirements covered:**
- Car-related features including **brand** (a proxy for brand goodwill/reputation
  — one-hot encoded so the model learns a price effect per brand), **horsepower**
  (`max_power_bhp`), and **mileage** (`mileage_kmpl`), alongside age, engine
  size, transmission, fuel type, and ownership history
- A **regression model** (not classification) predicting a continuous price value
- Full **data preprocessing**, **feature engineering**, and **model evaluation**
- Built entirely with **Pandas**, **Scikit-learn**, and **Matplotlib/Seaborn**

## Dataset

- **Source:** [Vehicle Dataset from CarDekho](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) (Kaggle)
- 8,128 rows × 13 original columns
- Key features: `name`, `year`, `km_driven`, `fuel`, `seller_type`,
  `transmission`, `owner`, `mileage`, `engine`, `max_power`, `torque`, `seats`
- Target: `selling_price` (INR)

## Methodology

1. **Cleaning & feature engineering** (`src/clean_data.py`)
   - Parsed numeric values out of unit-laden text fields (`"23.4 kmpl"` → `23.4`)
   - Extracted `brand` from the car name
   - Engineered `car_age` from `year`
   - Imputed missing values with column medians
   - Capped extreme outliers in `km_driven` / `selling_price` at the 99th percentile
   - Converted `owner` to an ordinal scale

2. **EDA** — distribution of price, price vs. age/mileage, price by fuel type
   and transmission, correlation heatmap. See `visuals/`.

3. **Modeling** (`src/train_model.py`) — Linear Regression baseline,
   Random Forest, and Gradient Boosting, compared on held-out test data.

4. **Evaluation** — RMSE, MAE, R² across all models, plus feature importance
   from the winning model.

## Results

| Model | RMSE (₹) | MAE (₹) | R² |
|---|---|---|---|
| Linear Regression | 422,638 | 257,199 | 0.7149 |
| Gradient Boosting | 138,496 | 73,689 | 0.9694 |
| **Random Forest** | **125,184** | **68,223** | **0.9750** |

**Best model:** Random Forest — saved to `models/best_model.pkl`

Both tree-based models dramatically outperform the linear baseline, confirming
that price depends on features (especially age and power) in a non-linear
way that a straight line can't capture.

## Key findings

- `max_power_bhp` is by far the strongest predictor of price, followed by `car_age`.
- Feature importance from the trained model matches the correlation patterns
  seen in EDA — a good consistency check that the model learned real signal.
- Car age is the strongest negative predictor of price.
- Automatic-transmission cars sell at a clear premium over manual.
- Diesel cars have a higher median resale price than petrol/CNG/LPG.
- Predicted vs. actual prices track the diagonal closely across the full
  price range, with no systematic bias.

## How to run locally

```bash
git clone https://github.com/<your-username>/car-price-prediction.git
cd car-price-prediction
pip install -r requirements.txt

# Reproduce the cleaned dataset
python src/clean_data.py

# Train and evaluate models
python src/train_model.py

# Or open the full walkthrough
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```

## Project structure

```
car-price-prediction/
├── data/
│   ├── raw/car_details_v3.csv
│   └── processed/car_clean.csv
├── notebooks/
│   ├── 01_eda_and_modeling.ipynb
│   └── 01_eda_and_modeling.py       # plain-script source of the notebook
├── src/
│   ├── clean_data.py                # Cleaning & feature engineering
│   ├── train_model.py               # Trains & compares 3 models
│   └── predict.py                   # inference on a new car
├── models/
│   ├── best_model.pkl
│   ├── best_model_name.json
│   ├── results.csv
│   └── feature_importance.csv
├── visuals/
│   ├── eda_overview.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   └── predicted_vs_actual.png
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
```

## Real-world applications

Accurate car price prediction has direct use cases on both sides of the
used-car market:
- **Buyers** get an objective reference point to check whether an asking
  price is fair before negotiating.
- **Sellers/dealers** can price inventory competitively based on the actual
  drivers of value (age, power, brand) rather than guesswork.
- **Online marketplaces** (like CarDekho itself) can use models like this
  to auto-suggest listing prices or flag suspiciously over/under-priced ads.
- The same regression + feature-engineering approach generalizes to pricing
  any depreciating asset — real estate, machinery, electronics resale, etc.

## Sample visuals

**EDA overview** — price vs. age, price vs. km driven, price by fuel/transmission
`visuals/eda_overview.png`

**Feature importance (Random Forest)**
`visuals/feature_importance.png`

**Predicted vs. actual price**
`visuals/predicted_vs_actual.png`

## Author

Built as Task 3 of the EXPS Nexus Data Science Internship.
