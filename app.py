import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ========================
# TITLE
# ========================
st.title("Dashboard - Electricity Consumption")

st.write("AI Project: RS (Secondary Residence) vs RP (Primary Residence)")

# ========================
# GENERATION PART
# ========================

# Dropdown to choose client type
client_type = st.selectbox("Select client type", ["RP", "RS"])

# Function to generate curves
def generate_curve(client_type="RP", n=48):
    x = np.arange(n)

    if client_type == "RP":
        curve = 1.5 + 0.8*np.sin(2*np.pi*(x-10)/24) + 0.4*np.sin(2*np.pi*x/12)
    else:
        curve = 0.8 + 0.3*np.sin(2*np.pi*(x-12)/24)

    noise = np.random.normal(0, 0.1, n)
    curve = np.maximum(curve + noise, 0)

    return pd.DataFrame({
        "time": x,
        "consumption": curve
    })

# Button
if st.button("Generate curve"):
    df = generate_curve(client_type)

    fig = px.line(df, x="time", y="consumption",
                  title=f"Generated curve - {client_type}")

    st.plotly_chart(fig, width="stretch")

    st.write("This curve is artificially generated based on the selected client type.")

# ========================
# DATA PREPARATION (PART 1)
# ========================
st.header("Data Preparation")

try:
    df_features = pd.read_csv("features_clients_pour_clustering.csv")

    st.write("Preview of prepared data:")
    st.dataframe(df_features.head())

except:
    st.warning("Data preparation file not found.")

# ========================
# CLUSTERING (PART 2)
# ========================
st.header("Clustering Results")

try:
    df_clusters = pd.read_csv("features_clients_avec_labels.csv")

    st.write("Preview of clustering results (RP / RS):")
    st.dataframe(df_clusters.head())

except:
    st.warning("Clustering results file not found.")

# ========================
# CLASSIFICATION (TO DO)
# ========================
st.header("Classification")
st.write("Prediction RS / RP (to be connected with the model)")

# ========================
# FORECASTING (TO DO)
# ========================
st.header("Forecasting")
st.write("Future consumption prediction (to be connected)")
