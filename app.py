'''Creating a single page UI to track YOE & Test Scores to predict salary'''
import streamlit as st
import requests

# Streamlit UI
st.title("Salary Prediction")

# Years of Experience
exp = st.number_input("Enter years of experience",
                      min_value=0, max_value=50, step=1)

# Test Scores
test_score = st.number_input(
    "Enter test scores", min_value=0, max_value=100, step=1)

# Generate Salary Button
if st.button("Generate Salary"):
    # API Request to Flask Server
    response = requests.post(
        "http://localhost:5000/predict", json={"exp": exp, "test_score": test_score})

    # Show Predicted Salary
    predicted_salary = response.json()["predicted_salary"]
    st.success(f"Predicted Salary: ${predicted_salary:.2f}")
