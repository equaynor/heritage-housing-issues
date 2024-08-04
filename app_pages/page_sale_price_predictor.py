import streamlit as st
import pandas as pd
from src.data_management import load_house_prices_data, load_pkl_file
from src.machine_learning.predictive_analysis_ui import predict_sale_price

def page_sale_price_predictor_body():

    st.write("# House Price Predictor")
    
    st.info(
        "This tool allows you to predict house prices in Ames, Iowa based on key features. "
        "It addresses Business Requirement 2: predicting sale prices for inherited houses "
        "and other properties in the area."
    )

    # Load prediction model and features
    model = load_pkl_file("outputs/ml_pipeline/predict_price/regression_model.pkl")
    features = pd.read_csv("outputs/ml_pipeline/predict_price/model_features.csv").columns.tolist()

    # User input section
    st.write("## Enter House Details")
    user_input = get_user_input(features)

    if st.button("Predict Price"):
        prediction = predict_sale_price(user_input, features, model)
        st.success(f"Predicted Sale Price: ${prediction:,.2f}")

    st.write("---")

    # Inherited houses section
    st.write("## Inherited Houses Valuation")
    inherited_data = load_house_prices_data().filter(features)
    st.write(inherited_data)

    if st.button("Value Inherited Houses"):
        total_value = sum(predict_sale_price(inherited_data.iloc[[i]], features, model) for i in range(len(inherited_data)))
        st.success(f"Total Value of Inherited Houses: ${total_value:,.2f}")
