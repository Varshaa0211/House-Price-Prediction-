import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('house_price_model.pkl')

st.title('House Price Prediction')

sqft_living = st.number_input('Enter sqft_living')
bedrooms = st.number_input('Enter bedrooms')
bathrooms = st.number_input('Enter bathrooms')
floors = st.number_input('Enter floors')
feature_5 = st.number_input('Enter feature_5')
feature_18 = st.number_input('Enter feature_18')

if st.button('Predict'):
    features = np.array([[sqft_living, bedrooms, bathrooms, floors, feature_5, ..., feature_18]])
    prediction = model.predict(features)[0]
    st.success(f'Predicted Price: ${prediction:,.2f}')



