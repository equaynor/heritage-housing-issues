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

    statement_inheritance = (
        f"* Given the values of the inherited properties features, the model has "
        f"  predicted a sale value of: {round(sale_price_prediction[0])}"
    )

    if len(sale_price_prediction) == 1:
        st.write(statement)
    else:
        st.write(statement_inheritance)
    

    return sale_price_prediction