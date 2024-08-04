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
    """
    Displays a Streamlit page for correlation analysis of house sale prices.
    Provides interactive visualizations to explore relationships between variables and sale price.
    """
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

    # Pearson and Spearman Correlations
    if st.checkbox("Pearson Correlation"):
        calc_display_pearson_corr_heat(df_eda)
        calc_display_pearson_corr_bar(df_eda)

    st.info(
    "*** Heatmap and Barplot: Pearson Correlation ***\n\n"
    "The Pearson Correlation measures the strength and direction of the linear relationship "
    "between two continuous variables. It assesses how well the correlation between the "
    "variables can be represented by a straight line.\n"
    "The heatmap highlights the variables on the x-axis that have a strong linear "
    "correlation with the Sale Price, with a correlation coefficient of up to almost 0.8. "
    "The most correlating variables are then visualized in a bar plot to provide a clearer "
    "understanding of their relationships with the Sale Price."
    )

    if st.checkbox("Spearman Correlation"):
        calc_display_spearman_corr_heat(df_eda)
        calc_display_spearman_corr_bar(df_eda)

    st.info(
    "*** Heatmap and Barplot: Spearman Correlation ***\n\n"
    "The Spearman correlation evaluates the monotonic relationship between two variables, "
    "assessing how well the variables behave similarly but not necessarily linearly.\n"
    "The heatmap highlights the variables on the x-axis that have a strong monotonic "
    "correlation with the Sale Price, with a correlation coefficient of up to almost 0.8. "
    "The most correlating variables are then visualized in a bar plot to provide a clearer "
    "understanding of their relationships with the Sale Price."
    )

    if st.checkbox("Predictive Power Score"):
        calc_display_pps_matrix(df)

    st.info(
    "*** Heatmap and Barplot: Predictive Power Score (PPS) ***\n\n"
    "The Predictive Power Score (PPS) detects both linear and non-linear relationships "
    "between two variables, providing a measure of their predictive power.\n"
    "The PPS score ranges from 0 (no predictive power) to 1 (perfect predictive power). "
    "To interpret the plot, locate the 'SalePrice' row on the y-axis and examine the "
    "variables on the x-axis with a PPS score above 0.2. "
    "Notably, Overall Quality (OverallQual) exhibits the highest predictive power for the Sale Price target."
    )

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
    """
    Plots the relationship between each variable and the sale price.
    """
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
  fig = sns.lmplot(data=df, x=col, y=target_var, height=6, aspect=1.5)
  plt.title(f"{col}", fontsize=20, y=1.05)
  st.pyplot(fig)  # st.pyplot() renders image, in notebook is plt.show()


# Heatmaps

def calc_display_pearson_corr_heat(df):
    """ Calcuate and display Pearson Correlation """
    df_corr_pearson = df.corr(method="pearson")
    heatmap_corr(df=df_corr_pearson, threshold=0.4,
                 figsize=(12, 10), font_annot=10)


def calc_display_spearman_corr_heat(df):
    """ Calcuate and display Spearman Correlation """
    df_corr_spearman = df.corr(method="spearman")
    heatmap_corr(df=df_corr_spearman, threshold=0.4,
                 figsize=(12, 10), font_annot=10)


def heatmap_corr(df,threshold, figsize=(20,12), font_annot = 8):
  """
  Function to create heatmap using correlations.
  """
  if len(df.columns) > 1:
    mask = np.zeros_like(df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    mask[abs(df) < threshold] = True

    fig, axes = plt.subplots(figsize=figsize)
    sns.heatmap(df, annot=True, xticklabels=True, yticklabels=True,
                mask=mask, cmap='viridis', annot_kws={"size": font_annot}, ax=axes,
                linewidth=0.5
                     )
    axes.set_yticklabels(df.columns, rotation = 0)
    plt.ylim(len(df.columns),0)
    st.pyplot(fig)


def heatmap_pps(df, threshold, figsize=(20, 12), font_annot=8):
    """ Heatmap for predictive power score from CI template"""
    if len(df.columns) > 1:
        mask = np.zeros_like(df, dtype=bool)
        mask[abs(df) < threshold] = True
        fig, axes = plt.subplots(figsize=figsize)
        ax = sns.heatmap(df, annot=True, xticklabels=True, yticklabels=True,
                           mask=mask, cmap='rocket_r',
                           annot_kws={"size": font_annot},
                           linewidth=0.05, linecolor='grey')
        plt.ylim(len(df.columns), 0)
        st.pyplot(fig)


# Barplots

def calc_display_pearson_corr_bar(df):
    """ Calcuate and display Pearson Correlation """
    corr_pearson = df.corr(method='pearson')['SalePrice'].sort_values(
        key=abs, ascending=False)[1:]
    fig, axes = plt.subplots(figsize=(6, 3))
    plt.bar(x=corr_pearson[:5].index, height=corr_pearson[:5])
    plt.title("Pearson Correlation with Sale Price", fontsize=14, y=1.05)
    st.pyplot(fig)


def calc_display_spearman_corr_bar(df):
    """ Calcuate and display Spearman Correlation """
    corr_spearman = df.corr(method='spearman')['SalePrice'].sort_values(
        key=abs, ascending=False)[1:]
    fig, axes = plt.subplots(figsize=(6, 3))
    plt.bar(x=corr_spearman[:5].index, height=corr_spearman[:5])
    plt.title("Spearman Correlation with Sale Price", fontsize=14, y=1.05)
    st.pyplot(fig)



def calc_display_pps_matrix(df):
    """ Calcuate and display Predictive Power Score """
    pps_matrix_raw = pps.matrix(df)
    pps_matrix = pps_matrix_raw.filter(['x', 'y', 'ppscore']).pivot(
        columns='x', index='y', values='ppscore')
    heatmap_pps(df=pps_matrix, threshold=0.2, figsize=(12, 10), font_annot=10)
