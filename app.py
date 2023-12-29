import streamlit as st
import joblib
import numpy as np

model = joblib.load('clf_gini.pkl')

def predict(input_data):
    prediction = model.predict(input_data)
    return prediction

st.title("Netflix prediction")

rt = st.text_input("Enter ratings: ")

# Check if the entered value is non-empty and not just spaces
if rt.strip():
    # Create an array with the entered value
    arr = np.array([[rt]], dtype=np.object_)
    
    if st.button("Predict"):
        result = predict(arr)
        st.success(f"Prediction: {result[0]}")
