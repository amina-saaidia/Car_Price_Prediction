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
predicted_price = model.predict(new_car_df)[0]

print(f"Predicted selling price: {predicted_price:,.0f}")
