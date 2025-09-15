import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model
model = joblib.load('house_price_model.pkl')

st.title('House Price Prediction')

# Input fields
OverallQual = st.number_input('Overall Quality (1-10)', min_value=1, max_value=10, value=5)
GrLivArea = st.number_input('Above ground living area (sq ft)', value=1500)
GarageCars = st.number_input('Garage Cars', min_value=0, max_value=5, value=2)
TotalBsmtSF = st.number_input('Total Basement SF', value=800)

# Prediction button
if st.button('Predict Price'):
    input_data = np.array([[OverallQual, GrLivArea, GarageCars, TotalBsmtSF]])
    predicted_price = model.predict(input_data)[0]
    st.success(f'Predicted House Price: ${predicted_price:,.2f}')
