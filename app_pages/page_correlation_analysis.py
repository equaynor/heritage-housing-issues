import plotly.express as px
import numpy as np
from feature_engine.discretisation import ArbitraryDiscretiser
import streamlit as st
from src.data_management import load_cleaned_house_prices_data

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import ppscore as pps

from feature_engine.encoding import OneHotEncoder


def page_correlation_analysis_body():

    # load data
    df = load_cleaned_house_prices_data()

    # from correlation study notebook
    vars_to_study = ['1stFlrSF',
                     'GarageArea',
                     'GrLivArea',
                     'KitchenQual_TA',
                     'OverallQual',
                     'TotalBsmtSF',
                     'YearBuilt',
                     'YearRemodAdd']

    st.write("### Property Sale Price Correlation Analysis")
    st.success(
        f"* The client is interested in discovering\
            how the house attributes correlate with the sale price.\
            Therefore, the client expects data visualisations of the correlated variables\
            against the sale price to show that."
    )

    # inspect data
    if st.checkbox("Inspect Sale Price Dataset"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, "
            f"find below the first 10 rows.")

        st.write(df.head(10))

    st.write("---")

    # Correlation Study Summary
    st.write(
        f"* A correlation study was conducted in the notebook to better understand how "
        f"the variables are correlated to the house sale prices. \n"
        f"The most correlated variable are: **{vars_to_study}**"
    )

    # Text based on "02 - Churned Customer Study" notebook - "Conclusions and Next steps" section
    # st.info(
    #     f"The correlation indications and plots below interpretation converge. "
    #     f"It is indicated that: \n"
    #     f"* A churned customer typically has a month-to-month contract \n"
    #     f"* A churned customer typically has fibre optic. \n"
    #     f"* A churned customer typically doesn't have tech support. \n"
    #     f"* A churned customer doesn't have online security. \n"
    #     f"* A churned customer typically has low tenure levels. \n"
    # )

    encoder = OneHotEncoder(variables=df.columns[df.dtypes=='object'].to_list(), drop_last=False)
    df_ohe = encoder.fit_transform(df)
    # Code copied from "03_correlation_study" notebook - "EDA of chosen variables" section
    df_eda = df_ohe.filter(list(vars_to_study) + ['SalePrice'])
    target_var = 'SalePrice'

    st.write("#### Data Visualizations \n")
    # Distribution of target variable
    if st.checkbox("Distribution of Sale Prices"):
        plot_target_hist(df_eda, target_var) 

    # Individual plots per variable
    if st.checkbox("Sale Price per Correlating Variable"):
        sale_price_per_variable(df_eda)

    # Parallel plot
    # if st.checkbox("Parallel Plot"):
    #     st.write(
    #         f"* Information in yellow indicates the profile from a churned customer")
    #     parallel_plot_churn(df_eda)

def plot_target_hist(df, target_var):
  """
  Function to plot a histogram of the target and
  save the figure to folder.
  """
  fig, axes = plt.subplots(figsize=(12, 5))
  sns.histplot(data=df, x=target_var, kde=True)
  plt.title(f"Distribution of {target_var}", fontsize=20)
  st.pyplot(fig)  # st.pyplot() renders image, in notebook is plt.show()


# function created using "03_correlation_study" notebook code - "Distribution of SalePrice" section
def sale_price_per_variable(df_eda):
    target_var = 'SalePrice'
    time = ['YearBuilt', 'YearRemodAdd']
    # from correlation study notebook
    vars_to_study = ['1stFlrSF',
                     'GarageArea',
                     'GrLivArea',
                     'KitchenQual_TA',
                     'OverallQual',
                     'TotalBsmtSF',
                     'YearBuilt',
                     'YearRemodAdd']

    for col in vars_to_study:
        if len(df_eda[col].unique()) <= 10:
            corr_box_plot(df_eda, col, target_var)
            print("\n\n")
        else:
            if col in time:
                corr_line_plot(df_eda, col, target_var)
                print("\n\n")
            else:
                corr_lm_plot(df_eda, col, target_var)
                print("\n\n")


def corr_line_plot(df, col, target_var):
  """
  Line plots of target variable vs time variables (years)
  """
  fig, axes = plt.subplots(figsize=(10, 5))
  sns.lineplot(data=df, x=col, y=target_var)
  plt.title(f"{col}", fontsize=20, y=1.05)
  st.pyplot(fig)  # st.pyplot() renders image, in notebook is plt.show()


def corr_box_plot(df, col, target_var):
  """
  Box plots of target variable vs categorical variables
  """
  fig, axes = plt.subplots(figsize=(10, 5))
  sns.boxplot(data=df, x=col, y=target_var) 
  plt.title(f"{col}", fontsize=20, y=1.05)
  st.pyplot(fig)  # st.pyplot() renders image, in notebook is plt.show()


def corr_lm_plot(df, col, target_var):
  """
  Linear regression plots of target variable vs continuous features"
  """
  fig, axes = plt.subplots(figsize=(10, 5))
  sns.lmplot(data=df, x=col, y=target_var, height=6, aspect=1.5)
  plt.title(f"{col}", fontsize=20, y=1.05)
  st.pyplot(fig)  # st.pyplot() renders image, in notebook is plt.show()
