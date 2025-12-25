import streamlit as st
import numpy as np
import pickle

# Load trained model
with open("my-cricket.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Cricket Career Runs Predictor")

st.title("🏏 Cricket Career Runs Predictor")
st.write("Predict a cricketer's total career runs using Machine Learning")

st.sidebar.header("Input Player Statistics")

# Inputs
matches = st.sidebar.number_input("Mat", min_value=1, step=1)
innings = st.sidebar.number_input("Inns", min_value=1, step=1)
not_outs = st.sidebar.number_input("NO", min_value=0, step=1)
balls_faced = st.sidebar.number_input("BF", min_value=1, step=1)

# Prediction
if st.button("Predict Career Runs"):
    features = np.array([[Mat, Inns, NO, BF]])
    prediction = model.predict(features)

    st.success(f"🏆 Predicted Career Runs: **{int(prediction[0])}**")

st.markdown("---")
st.markdown("""
### Model Details
- Algorithm: Linear Regression
- Features: Matches, Innings, Not Outs, Balls Faced
- Test R² Score: ~0.96
- RMSE: ~770 runs
""")
