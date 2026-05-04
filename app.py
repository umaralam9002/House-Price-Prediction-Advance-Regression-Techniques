import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("model.pkl", "rb"))

st.title("🏠 House Price Predictor")

st.write("Enter house details:")

overall_qual = st.slider("Overall Quality", 1, 10, 5)
gr_liv_area = st.number_input("Living Area (sq ft)", 500, 5000, 1500)
garage_cars = st.slider("Garage Cars", 0, 4, 1)

if st.button("Predict"):
    input_df = pd.DataFrame({
        "OverallQual": [overall_qual],
        "GrLivArea": [gr_liv_area],
        "GarageCars": [garage_cars]
    })

    prediction = model.predict(input_df)
    st.success(f"Estimated Price: ${prediction[0]:,.2f}")