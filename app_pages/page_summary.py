import streamlit as st
import pandas as pd


def page_summary_body():
    """
    Displays contents of the project summary page.
    """
    st.write("# Project Overview")

    st.info(
        "**Project Background**\n\n"
        "Our client, a resident of Belgium, has inherited\
        four houses in Ames, Iowa, "
        "and seeks our assistance in maximizing their sales price.\
        To achieve this, "
        "we'll utilize a Machine Learning model and regression\
        algorithms to provide "
        "accurate property valuations."
    )

    st.write("# Project Dataset")

    st.info(
        "**Dataset Source**\n"
        "The dataset is sourced from Kaggle,\
        containing approximately 1,500 records "
        "of housing sales in Ames, Iowa.\
        Each record includes 23 features describing "
        "the house profile, such as floor area, basement,\
        garage, and sale price."
    )

    st.write("# Business Requirements")

    st.success(
        "**Project Objectives**\n\n"
        "The project has two primary objectives:\n"
        "* **BR1**: Analyze the correlation between house attributes\
        and sale prices, providing data visualizations to\
        illustrate the relationships.\n"
        "* **BR2**: Develop a predictive model to estimate the sale prices\
        of the inherited houses and other properties in Ames, Iowa."
    )

    st.write("# Additional Information")

    st.write(
        "* For more information on this project, please visit the "
        "[README file](https://github.com/equaynor/heritage-housing-issues)."
    )

    d = {'lat': [42.0308], 'lon': [-93.6319]}
    df_ames = pd.DataFrame(data=d)
    st.map(data=df_ames, zoom=11)


page_summary_body()
