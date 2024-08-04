import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_management import load_house_prices_data, load_pkl_file
from src.machine_learning.evaluate_reg import (
    regression_performance,
    regression_evaluation_plots)


def page_predict_price_ml_body():
    """
    Displays  ML pipeline information.
    """

    vsn = 'v1'
    # load needed files
    sale_price_pipe = load_pkl_file(
        f"outputs/ml_pipeline/predict_sale_price/{vsn}/regression_pipeline.pkl"
    )
    sale_price_feat_importance = plt.imread(
        f"outputs/ml_pipeline/predict_sale_price/{vsn}/features_importance.png"
    )
    X_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_sale_price/{vsn}/X_train.csv")
    X_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_sale_price/{vsn}/X_test.csv")
    y_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_sale_price/{vsn}/y_train.csv").squeeze()
    y_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_sale_price/{vsn}/y_test.csv").squeeze()

    st.write("### ML Pipeline: Predict Property Sale Price")
    # display pipeline training summary conclusions
    st.success(
        f"* We successfully trained a Regressor model to predict"
        f"the sale price of "
        f"properties in Ames, Iowa, meeting our project requirement"
        f"(BR2) of an R2 Score "
        f"of 0.8 or better.\n"
        f"* Through a hyperparameter search and feature engineering process, "
        f"we achieved an R2 Score of 0.871 on the train set"
        f"and 0.809 on the test set, "
        f"demonstrating the model's strong predictive power.\n"
        f"* Our model identified the most important features"
        f"contributing to the sale "
        f"price, and we present the pipeline steps,"
        f"feature importance plot, and"
        f"performance reports below for further insight."
        )

    # show pipelines
    st.write("---")
    st.write("**ML pipeline to predict property sale prices.**")
    st.code(sale_price_pipe)

    # show feature importance plot
    st.write("---")
    st.write("**The features the model was trained on and their importance.**")
    st.write(X_train.columns.to_list())
    st.image(sale_price_feat_importance)

    # evaluate performance on train and test set
    st.write("---")
    st.write("### Pipeline Performance")
    regression_performance(X_train=X_train, y_train=y_train,
                           X_test=X_test, y_test=y_test,
                           pipeline=sale_price_pipe)

    st.write("**Regression Performance Plots**")
    regression_evaluation_plots(X_train=X_train, y_train=y_train,
                                X_test=X_test,
                                y_test=y_test, pipeline=sale_price_pipe,
                                alpha_scatter=0.5)
