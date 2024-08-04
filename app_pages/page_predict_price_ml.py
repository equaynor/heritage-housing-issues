import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_management import load_house_prices_data, load_pkl_file
from src.machine_learning.evaluate_reg import (
    regression_performance,
    regression_evaluation_plots)


def page_predict_price_ml_body():

    version = 'v1'
    # load needed files
    sale_price_pipe = load_pkl_file(
        f"outputs/ml_pipeline/predict_sale_price/{version}/regression_pipeline.pkl"
    )
    sale_price_feat_importance = plt.imread(
        f"outputs/ml_pipeline/predict_sale_price/{version}/features_importance.png"
    )
    X_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_sale_price/{version}/X_train.csv")
    X_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_sale_price/{version}/X_test.csv")
    y_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_sale_price/{version}/y_train.csv")
    y_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_sale_price/{version}/y_test.csv")

    st.write("### ML Pipeline: Predict Property Sale Price")
    # display pipeline training summary conclusions
    st.success(
        f"* We successfully trained a Regressor model to predict the sale price of "
        f"properties in Ames, Iowa, meeting our project requirement (BR2) of an R2 Score "
        f"of 0.8 or better.\n"
        f"* Through a hyperparameter search and feature engineering process, "
        f"we achieved an R2 Score of 0.871 on the train set and 0.809 on the test set, "
        f"demonstrating the model's strong predictive power.\n"
        f"* Our model identified the most important features contributing to the sale "
        f"price, and we present the pipeline steps, feature importance plot, and "
        f"performance reports below for further insight."
        )

    # show pipelines
    st.write("---")
    st.write("#### There are 2 ML Pipelines arranged in series.")

    st.write(" * The first is responsible for data cleaning and feature engineering.")
    st.write(churn_pipe_dc_fe)

    st.write("* The second is for feature scaling and modelling.")
    st.write(churn_pipe_model)

    # show feature importance plot
    st.write("---")
    st.write("* The features the model was trained and their importance.")
    st.write(X_train.columns.to_list())
    st.image(churn_feat_importance)

    # We don't need to apply dc_fe pipeline, since X_train and X_test
    # were already transformed in the jupyter notebook (Predict Customer Churn.ipynb)

    # evaluate performance on train and test set
    st.write("---")
    st.write("### Pipeline Performance")
    clf_performance(X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test,
                    pipeline=churn_pipe_model,
                    label_map=["No Churn", "Yes Churn"])