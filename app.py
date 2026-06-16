import streamlit as st
import pandas as pd
import pickle


model = pickle.load(open("model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Predictor")
st.write("Enter house details to estimate the selling price.")


overall_qual = st.slider(
    "Overall Quality",
    min_value=1,
    max_value=10,
    value=5
)

gr_liv_area = st.number_input(
    "Living Area (sq ft)",
    min_value=500,
    max_value=5000,
    value=1500
)

garage_cars = st.slider(
    "Garage Cars",
    min_value=0,
    max_value=4,
    value=1
)

garage_area = st.number_input(
    "Garage Area (sq ft)",
    min_value=0,
    max_value=1500,
    value=400
)

total_bsmt_sf = st.number_input(
    "Total Basement Area (sq ft)",
    min_value=0,
    max_value=3000,
    value=800
)

first_flr_sf = st.number_input(
    "1st Floor Area (sq ft)",
    min_value=300,
    max_value=4000,
    value=1200
)

full_bath = st.slider(
    "Full Bathrooms",
    min_value=0,
    max_value=5,
    value=2
)

tot_rms = st.slider(
    "Total Rooms Above Ground",
    min_value=2,
    max_value=15,
    value=6
)

year_built = st.number_input(
    "Year Built",
    min_value=1900,
    max_value=2025,
    value=2000
)


if st.button("Predict Price"):

    input_df = pd.DataFrame(
        0,
        index=[0],
        columns=features
    )

    input_df["OverallQual"] = overall_qual
    input_df["GrLivArea"] = gr_liv_area
    input_df["GarageCars"] = garage_cars
    input_df["GarageArea"] = garage_area
    input_df["TotalBsmtSF"] = total_bsmt_sf
    input_df["1stFlrSF"] = first_flr_sf
    input_df["FullBath"] = full_bath
    input_df["TotRmsAbvGrd"] = tot_rms
    input_df["YearBuilt"] = year_built

    prediction = model.predict(input_df)

    st.success(
        f"Estimated House Price: ${prediction[0]:,.2f}"
    )