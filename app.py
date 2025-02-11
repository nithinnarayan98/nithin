import streamlit as st
import pandas as pd
import pickle
import numpy as np
import sklearn

# App title
st.title('Car Price Prediction App')
st.write('This app predicts the *selling price of used cars* based on the input features.')

# Load the trained model and feature columns
columns = pickle.load(open('m-columns.pkl', 'rb'))
model = pickle.load(open('m-lr.pkl', 'rb'))

# Get input from the user
brand = st.selectbox('Car Brand', ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Ford', 'Others'])
year = st.number_input('Year of Manufacture', min_value=1990, max_value=2025, value=2015, step=1)
km_driven = st.number_input('Kilometers Driven', min_value=0, max_value=500000, value=50000)
fuel = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
owner = st.selectbox('Number of Previous Owners', ['First', 'Second', 'Third', 'Fourth & Above'])

# Feature engineering (convert inputs to match the model’s format)
car_age = 2025 - year

# Convert user input to a DataFrame
user_data = pd.DataFrame({
    'brand': [brand],
    'km_driven': [np.log1p(km_driven)],  # Apply log transformation
    'fuel': [fuel],
    'owner': [owner],
    'car_age': [car_age]
})

# One-hot encode user input to match training data
user_data_encoded = pd.get_dummies(user_data)

# Ensure that input columns match the model's training columns
user_data_encoded = user_data_encoded.reindex(columns=columns, fill_value=0)

# Predict the selling price
if st.button('Predict Selling Price'):
    prediction = model.predict(user_data_encoded)
    predicted_price = np.expm1(prediction[0])  # Reverse log transformation
    st.write(f'### Predicted Selling Price: ₹{predicted_price:,.2f}')