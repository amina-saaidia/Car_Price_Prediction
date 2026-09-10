"""
predict.py
Loads the saved model and predicts the price of a new car.

Run: python src/predict.py
The `new_car` dictionary below can be edited with a different car's details.
"""

import joblib
import pandas as pd

# --- The trained model is loaded ---
model = joblib.load("models/best_model.pkl")

# --- The car to be priced ---
new_car = {
    "car_age": 5,            # years old
    "km_driven": 40000,      # kilometers driven
    "mileage_kmpl": 18.5,    # fuel efficiency (km per liter)
    "engine_cc": 1200,       # engine size in CC
    "max_power_bhp": 85,     # horsepower
    "seats": 5,
    "owner_num": 1,          # 1 = first owner, 2 = second owner, etc.
    "brand": "Maruti",
    "fuel": "Petrol",
    "seller_type": "Individual",
    "transmission": "Manual",
}

# --- It's converted into a one-row table, since the model expects a DataFrame ---
new_car_df = pd.DataFrame([new_car])

# --- Prediction ---
predicted_price_inr = model.predict(new_car_df)[0]

# Approximate conversions for reference
INR_TO_USD = 0.01058668
INR_TO_DZD = 1.4031

predicted_price_usd = predicted_price_inr * INR_TO_USD
predicted_price_dzd = predicted_price_inr * INR_TO_DZD

print(f"Predicted selling price: ₹{predicted_price_inr:,.0f} INR")
print(f"  ≈ ${predicted_price_usd:,.0f} USD")
print(f"  ≈ {predicted_price_dzd:,.0f} DZD")