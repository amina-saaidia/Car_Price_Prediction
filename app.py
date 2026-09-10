"""
app.py
Interactive web app for the car price prediction model.

Run locally:
    streamlit run app.py
"""

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")


model = load_model()


# ============================================================
# CONSTANTS
# ============================================================

BRANDS = [
    "Ambassador", "Ashok", "Audi", "Bmw", "Chevrolet", "Daewoo",
    "Datsun", "Fiat", "Force", "Ford", "Honda", "Hyundai", "Isuzu",
    "Jaguar", "Jeep", "Kia", "Land", "Lexus", "Mahindra", "Maruti",
    "Mercedes-Benz", "Mg", "Mitsubishi", "Nissan", "Opel", "Peugeot",
    "Renault", "Skoda", "Tata", "Toyota", "Volkswagen", "Volvo"
]

FUEL_TYPES = ["CNG", "Diesel", "LPG", "Petrol"]
SELLER_TYPES = ["Dealer", "Individual", "Trustmark Dealer"]
TRANSMISSIONS = ["Automatic", "Manual"]


# ============================================================
# HEADER SECTION
# ============================================================

st.title("🚗 Car Price Predictor")

st.caption("MACHINE LEARNING · USED CAR VALUATION")

st.write(
    "Estimate the selling price of a used car using a machine learning model "
    "trained on 6,926 real-world vehicle listings."
)

st.divider()


# ============================================================
# INPUT FORM
# ============================================================

with st.form("car_form"):

    # 01. Vehicle Information
    st.subheader("01. Vehicle Information")

    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox(
            "Brand",
            BRANDS,
            index=BRANDS.index("Maruti")
        )

        car_age = st.number_input(
            "Car age (years)",
            min_value=0,
            max_value=40,
            value=5
        )

        km_driven = st.number_input(
            "Kilometers driven",
            min_value=0,
            value=40000,
            step=1000
        )

        seats = st.selectbox(
            "Seats",
            [2, 4, 5, 6, 7, 8, 9, 10],
            index=2
        )

    with col2:
        fuel = st.selectbox(
            "Fuel type",
            FUEL_TYPES,
            index=FUEL_TYPES.index("Petrol")
        )

        seller_type = st.selectbox(
            "Seller type",
            SELLER_TYPES,
            index=SELLER_TYPES.index("Individual")
        )

        transmission = st.selectbox(
            "Transmission",
            TRANSMISSIONS,
            index=TRANSMISSIONS.index("Manual")
        )

        owner_num = st.selectbox(
            "Ownership",
            [0, 1, 2, 3, 4],
            index=1,
            format_func=lambda x: {
                0: "Test drive car",
                1: "First owner",
                2: "Second owner",
                3: "Third owner",
                4: "Fourth & above owner",
            }[x],
        )

    # 02. Engine & Performance
    st.subheader("02. Engine & Performance")

    col3, col4, col5 = st.columns(3)

    with col3:
        engine_cc = st.number_input(
            "Engine size (CC)",
            min_value=600,
            max_value=6000,
            value=1200
        )

    with col4:
        max_power_bhp = st.number_input(
            "Max power (bhp)",
            min_value=20,
            max_value=500,
            value=85
        )

    with col5:
        mileage_kmpl = st.number_input(
            "Mileage (km/L)",
            min_value=5.0,
            max_value=40.0,
            value=18.5,
            step=0.1
        )

    # Submit Button
    submitted = st.form_submit_button(
        "Predict Estimated Price",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# PREDICTION & RESULTS
# ============================================================

if submitted:

    new_car = pd.DataFrame(
        [{
            "car_age": car_age,
            "km_driven": km_driven,
            "mileage_kmpl": mileage_kmpl,
            "engine_cc": engine_cc,
            "max_power_bhp": max_power_bhp,
            "seats": seats,
            "owner_num": owner_num,
            "brand": brand,
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
        }]
    )

    predicted_price_inr = model.predict(new_car)[0]

    st.divider()

    st.success("Prediction Complete!")

    st.markdown("**ESTIMATED SELLING PRICE (INR)**")

    # Highlight the predicted price in orange/gold
    st.markdown(
        f"""
        <h1 style="color:#F59E0B; font-weight:800; font-size:48px;">
            ₹{predicted_price_inr:,.0f}
        </h1>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================
    # CURRENCY CONVERSIONS
    # ========================================================

    st.divider()

    st.subheader("🌍 Approximate Currency Conversion")

    INR_TO_USD = 0.01058668
    INR_TO_DZD = 1.4031

    predicted_price_usd = predicted_price_inr * INR_TO_USD
    predicted_price_dzd = predicted_price_inr * INR_TO_DZD

    c1, c2 = st.columns(2)

    c1.metric(
        "US Dollars",
        f"${predicted_price_usd:,.0f}"
    )

    c2.metric(
        "Algerian Dinar",
        f"{predicted_price_dzd:,.0f} DZD"
    )

    st.warning(
        "These values are simple currency conversions of the Indian price — "
        "they are not predictions of what the car would sell for in another "
        "country's market. The model was trained only on Indian used-car listings."
    )


# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

st.subheader("🧠 How the prediction works")

w1, w2, w3 = st.columns(3)

w1.info(
    "1️⃣ **Vehicle Details**\n"
    "Enter parameters into the input fields."
)

w2.info(
    "2️⃣ **Machine Learning**\n"
    "Gradient Boosting model processes all inputs."
)

w3.info(
    "3️⃣ **Estimated Price**\n"
    "Instantaneous valuation calculated."
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Car Price Predictor · Gradient Boosting Regressor · R² = 0.937 · "
    "CarDekho Vehicle Dataset"
)