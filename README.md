# 🚗 EXPS - Car Price Prediction

Predicts the resale price of used cars using regression models, built as
Task 3 of the EXPS Nexus Data Science Internship (Algeria Pilot Cohort 01).

## Overview

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
- 8,128 rows × 13 original columns, **6,926 rows after removing 1,202 exact
  duplicate listings** (about 15% of the raw data — see Methodology below)
- Key features: `name`, `year`, `km_driven`, `fuel`, `seller_type`,
  `transmission`, `owner`, `mileage`, `engine`, `max_power`, `torque`, `seats`
- Target: `selling_price` (INR)

## Methodology

1. **Cleaning & feature engineering** (`src/clean_data.py`)
   - Removed exact duplicate rows, to prevent the same car listing from
     appearing in both the training and test split
   - Parsed numeric values out of unit-laden text fields (`"23.4 kmpl"` → `23.4`)
   - Extracted `brand` from the car name
   - Engineered `car_age` from `year`
   - Converted `owner` to an ordinal scale

2. **EDA** — distribution of price, price vs. age/mileage, price by fuel type,
   transmission, and seller type, correlation heatmap. See `visuals/`.

3. **Modeling** (`src/train_model.py`) — Linear Regression baseline,
   Random Forest, and Gradient Boosting, compared on held-out test data.
   Missing values are filled in and categoricals are encoded **inside the
   model pipeline**, fit only on the training split — see the note below.

4. **Evaluation** — RMSE, MAE, R² across all models, plus feature importance
   from the winning model.

### A note on a leakage fix

An earlier version of this pipeline filled in missing values and capped
extreme outliers on the *entire* dataset, before splitting into train and
test. That's a subtle form of data leakage: those fill-in and cutoff values
were calculated using information from the test set too, which the model
isn't supposed to see ahead of time. It also didn't remove duplicate rows,
which meant the same car could appear in both the training and test data.

Testing each issue separately showed that removing the duplicate rows was
responsible for almost the entire change in score — the leak-free
imputation fix, on its own, barely moved the numbers. Both issues are now
fixed: duplicates are removed in `clean_data.py`, and missing-value filling
happens inside the training pipeline using only the training data.

## Results

| Model | RMSE (₹) | MAE (₹) | R² |
|---|---|---|---|
| Linear Regression | 300,052 | 168,911 | 0.5895 |
| Random Forest | 127,043 | 72,182 | 0.9264 |
| **Gradient Boosting** | **117,695** | **72,035** | **0.9368** |

**Best model:** Gradient Boosting — saved to `models/best_model.pkl`

Both tree-based models clearly outperform the linear baseline, confirming
that price depends on features (especially age and power) in a non-linear
way that a straight line can't capture. These scores are lower than an
earlier version of this project (which reached R²=0.975), because that
earlier version had duplicate listings leaking between the training and
test sets. A lower score from a correct, leak-free setup is more
trustworthy than a higher score that isn't.

## Key Findings

- `max_power_bhp` is by far the strongest predictor of price, followed by `car_age`.
- Feature importance from the trained model matches the correlation patterns
  seen in EDA — a good consistency check that the model learned real signal.
- Car age is the strongest negative predictor of price.
- Automatic-transmission cars sell at a clear premium over manual.
- Diesel cars have a higher median resale price than petrol/CNG/LPG.
- Dealer listings have a higher median price than individual-seller
  listings, likely reflecting reconditioning and warranties.
- Predicted vs. actual prices track the diagonal closely across the full
  price range, with no systematic bias.

## Live Demo

Try the interactive Streamlit app:
**https://carpriceprediction-vlzxbrkgnwc5lyjwqqmxdc.streamlit.app/**

## How to Run Locally

```bash
git clone https://github.com/amina-saaidia/Car_Price_Prediction.git
cd Car_Price_Prediction
pip install -r requirements.txt

# Reproduce the cleaned dataset
python src/clean_data.py

# Train and evaluate models
python src/train_model.py

# Try a prediction on a new car
python src/predict.py

# Or launch the interactive app
streamlit run app.py

# Or open the full walkthrough
jupyter notebook notebooks/eda.ipynb
```

## Project Structure

```text
Car_Price_Prediction/
├── data/
│   ├── raw/car_details_v3.csv
│   └── processed/car_clean.csv
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── clean_data.py
│   ├── train_model.py
│   └── predict.py
├── models/
│   └── best_model.pkl
├── visuals/
│   ├── eda_overview.png
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   └── predicted_vs_actual.png
├── app.py                           # Streamlit interactive demo
├── .streamlit/
│   └── config.toml
├── requirements.txt
├── .gitignore
└── README.md
```

## Real-World Applications

Accurate car price prediction has direct use cases on both sides of the
used-car market, particularly within the Indian market this model was
trained on:
- **Buyers** get an objective reference point to check whether an asking
  price is fair before negotiating.
- **Sellers** can price their car competitively based on the actual
  drivers of value (age, power, brand) rather than guesswork.
- **Platforms like CarDekho itself** (the source of this dataset) could
  use a model like this to auto-suggest listing prices or flag
  suspiciously over/under-priced ads.
- The same regression and feature-engineering approach could be adapted
  to other markets or asset types (e.g. used cars in a different country,
  or other depreciating assets like electronics) — but would need to be
  retrained on data from that specific market, since pricing patterns
  don't transfer directly across regions.

## Scope and Limitations

- This model only reflects the **Indian used-car market** it was trained
  on. Brand availability, import taxes, and buyer preferences differ
  significantly by country.
- The optional currency conversions shown in `predict.py` and `app.py` are
  simple unit conversions of the Indian price (checked against a currency
  converter) — **not** predictions of what an equivalent car would sell
  for in another country's market. Import taxes in particular (very
  significant in Algeria, for example) are not captured by this model.
- The dataset covers a specific time period and may not reflect current
  market prices.

## Author

Built as Task 3 of the EXPS Nexus Data Science Internship (Algeria Pilot
Cohort 01).
