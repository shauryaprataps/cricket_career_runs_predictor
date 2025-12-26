import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load trained model
with open("my-cricket.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="ODI Career Runs Predictor")

st.title("🏏 ODI Career Runs Predictor")
st.write("Predict a cricketer's total ODI runs using")

st.sidebar.header("Input Player Statistics")

matches = st.sidebar.number_input("Mat", min_value=1, step=1)
innings = st.sidebar.number_input("Inns", min_value=1, step=1)
not_outs = st.sidebar.number_input("NO", min_value=0, step=1)
balls_faced = st.sidebar.number_input("BF", min_value=1, step=1)

if st.button("Predict ODI Career Runs"):
    input_df = pd.DataFrame(
        [[matches, innings, not_outs, balls_faced]],
        columns=['Mat', 'Inns', 'NO', 'BF']
    )

    prediction = model.predict(input_df)
    st.success(f"🏆 Predicted Career Runs: {int(prediction[0])}")

st.markdown("---")
st.markdown("""

