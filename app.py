import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("best_xgboost_model.pkl")

# Title
st.title("🏡 California House Price Predictor")

st.write("Enter housing details below to predict the median house price.")

# Input fields for the user
MedInc = st.number_input("Median Income ($1000s)", min_value=0.0, max_value=20.0, value=5.0)
HouseAge = st.number_input("House Age (years)", min_value=0, max_value=100, value=20)
AveRooms = st.number_input("Average Rooms per Household", min_value=1.0, max_value=10.0, value=5.0)
AveBedrms = st.number_input("Average Bedrooms per Household", min_value=0.5, max_value=5.0, value=1.0)
Population = st.number_input("Population in Block Group", min_value=100, max_value=4000, value=1000)
AveOccup = st.number_input("Average Occupants per Household", min_value=1.0, max_value=10.0, value=3.0)
Latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=37.0)
Longitude = st.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-121.0)

# Predict button
if st.button("Predict Price"):
    # Convert input data to a NumPy array for prediction
    user_input = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])

    # Make prediction
    prediction = model.predict(user_input)[0] * 100000  # Convert to actual dollar value
    st.success(f"🏠 Estimated House Price: **${prediction:,.2f}**")
