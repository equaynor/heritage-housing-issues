import streamlit as st

def page_project_hypothesis_body():
    """
    Displays a Streamlit page for project hypotheses and their validation.
    """
    st.write("### Project Hypotheses and Validation")

    # conclusions taken from "03 - Correlation_Study" notebook 
    st.success(
       f"**H1 - Size Hypothesis:** We hypothesize that larger propeties tend to have higher sale prices.\n"
        f"* **Correct.** Our correlation study revealed that property size-related features exhibited a\
            moderate positive correlation with sale price..\n\n"
       
       f"**H2 - Quality Hypothesis:** Higher quality ratings are associated with higher sale prices.\n"
        f"* **Correct.**  We leveraged the correlation between sale price and kitchen quality and\
            overall quality ratings to demonstrate this relationship.\n\n"
       
       f"**H3 - Accuracy Hypothesis:** We hypothesize that our model will achieve\
         a sale price prediction accuracy with an R2 value of at least 0.8.\n"
        f"* **Correct.** The R2 analysis on the train and test sets confirms this.\n"
       )
        