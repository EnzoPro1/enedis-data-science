import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Title of the app
st.title("Dashboard - Electricity Consumption")

st.write("AI Project: RS (Secondary Residence) vs RP (Primary Residence)")

# Dropdown to choose client type
client_type = st.selectbox("Select client type", ["RP", "RS"])

# Simple function to generate consumption curves
def generate_curve(client_type="RP", n=48):
    x = np.arange(n)

    # Different patterns for RP and RS
    if client_type == "RP":
        curve = 1.5 + 0.8*np.sin(2*np.pi*(x-10)/24) + 0.4*np.sin(2*np.pi*x/12)
    else:
        curve = 0.8 + 0.3*np.sin(2*np.pi*(x-12)/24)

    # Add random noise
    noise = np.random.normal(0, 0.1, n)
    curve = np.maximum(curve + noise, 0)

    # Return as DataFrame
    return pd.DataFrame({
        "time": x,
        "consumption": curve
    })

# Button to generate a new curve
if st.button("Generate curve"):
    df = generate_curve(client_type)

    # Plot the curve
    fig = px.line(df, x="time", y="consumption",
                  title=f"Generated curve - {client_type}")

    st.plotly_chart(fig, use_container_width=True)

    st.write("This curve is artificially generated based on the selected client type.")

# Placeholder for classification results
st.header("Classification")
st.write("Prediction RS / RP (to be connected with the model)")

# Placeholder for forecasting results
st.header("Forecasting")
st.write("Future consumption prediction (to be connected)")
