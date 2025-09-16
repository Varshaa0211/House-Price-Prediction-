import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('house_price_model.pkl')

st.title('House Price Prediction')

sqft = st.number_input('Enter sqft_living')
bedrooms = st.number_input('Enter number of bedrooms')
bathrooms = st.number_input('Enter number of bathrooms')
floors = st.number_input('Enter number of floors')

if st.button('Predict Price'):
    features = np.array([[sqft, bedrooms, bathrooms, floors]])
    price = model.predict(features)[0]
    st.success(f'Predicted House Price: ${price:,.2f}')
