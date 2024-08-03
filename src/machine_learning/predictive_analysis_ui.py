import streamlit as st


def predict_sale_price(X_live, house_features, sale_price_pipeline):

    # from live data, subset features related to this pipeline
    X_live_sale_price = X_live.filter(house_features)

    # predict
    sale_price_prediction = sale_price_pipeline.predict(X_live_sale_price)

    # Create a logic to display the results
    statement = (
        f"* Given the values provided for the property features, the model has "
        f"  predicted a sale value of:"
    )

    st.write(statement)

    return sale_price_prediction