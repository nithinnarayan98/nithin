import streamlit as st
import pandas as pd
import pickle
import numpy as np

# App title
st.title('Car Price Prediction App')
st.write('This app predicts the *selling price of used cars* based on the input features.')

# Load the trained model and feature columns
with open('m-columns.pkl', 'rb') as file:
    columns = pickle.load(file)

with open('m-lr.pkl', 'rb') as file:
    model = pickle.load(file)

# User input fields
brand = st.selectbox('Car Brand', ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Ford', 'Others'])
year = st.number_input('Year of Manufacture', 1990, 2025, 2015, 1)
km_driven = st.number_input('Kilometers Driven', 0, 500000, 50000)
fuel = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
owner = st.selectbox('Number of Previous Owners', ['First', 'Second', 'Third', 'Fourth & Above'])

# Calculate car age
car_age = 2025 - year

# Create DataFrame with user input
user_data = pd.DataFrame({
    'brand': [brand],
    'km_driven': [np.log1p(km_driven)],  # Log transformation
    'fuel': [fuel],
    'owner': [owner],
    'car_age': [car_age]
})

# One-hot encode input to match model training data
user_data_encoded = pd.get_dummies(user_data).reindex(columns=columns, fill_value=0)

# Predict selling price
if st.button('Predict Selling Price'):
    prediction = model.predict(user_data_encoded)[0]
    predicted_price = np.expm1(prediction)  # Reverse log transformation
    st.write(f'### Predicted Selling Price: ₹{predicted_price:,.2f}')
