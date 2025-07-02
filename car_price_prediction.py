import streamlit as st
import pandas as pd
from joblib import dump, load

# Load the trained model and encoders using joblib
model = load("random_forest_model.joblib")
label_encoders = load("label_encoders.joblib")

# Function to encode input
def encode_input(user_input, encoders):
    input_df = pd.DataFrame([user_input])
    for col in ['brand', 'model', 'fuel_type', 'accident']:
        le = encoders[col]
        if input_df[col].iloc[0] not in le.classes_:
            input_df[col] = -1  # unknown class
        else:
            input_df[col] = le.transform(input_df[col])
    return input_df

# Streamlit UI
st.title("🚗 Used Car Price Predictor")

brand = st.text_input("Brand", "Toyota")
model_name = st.text_input("Model", "Corolla")
milage = st.number_input("Mileage (in KM)", value=60000)
fuel_type = st.selectbox("Fuel Type", ["gasoline", "diesel", "hybrid", "e85 flex fuel"])
accident = st.selectbox("Accident History", ["reported", "none reported"])
car_age = st.number_input("Car Age (in years)", value=5)

if st.button("Predict Price"):
    user_input = {
        "brand": brand,
        "model": model_name,
        "milage": milage,
        "fuel_type": fuel_type,
        "accident": accident,
        "car_age": car_age
    }

    try:
        processed_input = encode_input(user_input, label_encoders)
        predicted_price = model.predict(processed_input)[0]
        st.success(f"💰 Estimated Price: ₹ {predicted_price:,.2f}")
    except Exception as e:
        st.error(f"❌ Error in prediction: {str(e)}")
