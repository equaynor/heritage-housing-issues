import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_correlation_analysis import page_correlation_analysis_body
from app_pages.page_sale_price_predictor import page_sale_price_predictor_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body

# Create an instance of the app
app = MultiPage(app_name="Heritage Housing Sale Price Predictor")

# Add your app pages here using .add_page()
app.add_page("Project Overview", page_summary_body)
app.add_page("Correlation Analysis", page_correlation_analysis_body)
app.add_page("Sale Price Prediction", page_sale_price_predictor_body)
app.add_page("Hypothesis Validation", page_project_hypothesis_body)


app.run()  # Run the  app