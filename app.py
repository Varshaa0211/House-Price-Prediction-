import streamlit as st
import pandas as pd
import joblib

# Load model and feature columns
model = joblib.load('house_price_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')

st.title("🏠 House Price Prediction App")

# Input fields
bedrooms = st.number_input('Bedrooms', min_value=1, max_value=10, value=3)
bathrooms = st.number_input('Bathrooms', min_value=1.0, max_value=10.0, value=2.0)
sqft_living = st.number_input('Living Area (sqft)', min_value=300, max_value=10000, value=1500)
sqft_lot = st.number_input('Lot Size (sqft)', min_value=300, max_value=50000, value=5000)
floors = st.number_input('Floors', min_value=1.0, max_value=3.5, value=1.0)
waterfront = st.selectbox('Waterfront (0 = No, 1 = Yes)', [0, 1])
view = st.number_input('View (0-4)', min_value=0, max_value=4, value=0)
condition = st.number_input('Condition (1-5)', min_value=1, max_value=5, value=3)
grade = st.number_input('Grade (1-13)', min_value=1, max_value=13, value=7)
sqft_above = st.number_input('Sqft Above Ground', min_value=300, max_value=8000, value=1500)
sqft_basement = st.number_input('Sqft Basement', min_value=0, max_value=5000, value=0)
yr_built = st.number_input('Year Built', min_value=1900, max_value=2025, value=1990)
yr_renovated = st.number_input('Year Renovated', min_value=0, max_value=2025, value=0)
zipcode = st.number_input('Zipcode', min_value=98000, max_value=99999, value=98178)
lat = st.number_input('Latitude', min_value=47.0, max_value=48.0, value=47.5112)
long = st.number_input('Longitude', min_value=-123.0, max_value=-121.0, value=-122.257)

# Prediction
if st.button("Predict Price"):
    # Create input dictionary
    input_dict = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft_living': sqft_living,
        'sqft_lot': sqft_lot,
        'floors': floors,
        'waterfront': waterfront,
        'view': view,
        'condition': condition,
        'grade': grade,
        'sqft_above': sqft_above,
        'sqft_basement': sqft_basement,
        'yr_built': yr_built,
        'yr_renovated': yr_renovated,
        'zipcode': zipcode,
        'lat': lat,
        'long': long
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # 🔥 Ensure same columns as training
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # fill missing features with 0

    input_df = input_df[feature_columns]  # reorder columns

    # Prediction
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted House Price: ${prediction:,.2f}")
