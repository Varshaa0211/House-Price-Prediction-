

import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

@st.cache
def load_data():
    data = pd.read_csv('/content/house_data.csv')
    return data[['sqft_living', 'price']]

data = load_data()

X = data[['sqft_living']]
y = data['price']

model = LinearRegression()
model.fit(X, y)

st.title("🏠 House Price Prediction App")

if st.checkbox("Show Sample Data"):
    st.write(data.head())

sqft = st.number_input("Enter Square Feet:", min_value=500, max_value=10000, value=1000)

if st.button('Predict Price'):
    price = model.predict([[sqft]])
    st.success(f"Predicted House Price: ${price[0]:,.2f}")
