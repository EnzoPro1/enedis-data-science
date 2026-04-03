import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ========================
# PAGE CONFIG
# ========================
st.set_page_config(page_title="Electricity Consumption Dashboard", layout="wide")

# ========================
# TITLE
# ========================
st.title("Dashboard - Electricity Consumption")
st.write("AI Project: RS (Secondary Residence) vs RP (Primary Residence)")

# ========================
# GENERATION PART
# ========================
st.header("Curve Generation")

client_type = st.selectbox("Select client type", ["RP", "RS"])

def generate_curve(client_type="RP", n=48):
    x = np.arange(n)

    if client_type == "RP":
        curve = 1.5 + 0.8 * np.sin(2 * np.pi * (x - 10) / 24) + 0.4 * np.sin(2 * np.pi * x / 12)
    else:
        curve = 0.8 + 0.3 * np.sin(2 * np.pi * (x - 12) / 24)

    noise = np.random.normal(0, 0.1, n)
    curve = np.maximum(curve + noise, 0)

    return pd.DataFrame({
        "time": x,
        "consumption": curve
    })

if st.button("Generate curve"):
    df_generated = generate_curve(client_type)

    fig_generated = px.line(
        df_generated,
        x="time",
        y="consumption",
        title=f"Generated curve - {client_type}"
    )

    st.plotly_chart(fig_generated, use_container_width=True)
    st.write("This curve is artificially generated based on the selected client type.")

# ========================
# DATA PREPARATION
# ========================
st.header("Data Preparation")

try:
    df_prep = pd.read_csv("features_clients_pour_clustering.csv", sep=";")

    st.write("Preview of prepared data:")
    st.dataframe(df_prep.head())

    st.write("Number of columns:", len(df_prep.columns))
    st.write("Columns:", list(df_prep.columns))

except Exception:
    st.warning("Data preparation file not found.")

# ========================
# CLUSTERING
# ========================
st.header("Clustering Results")

try:
    df_clusters = pd.read_csv("features_clients_avec_labels.csv", sep=";")

    st.write("Preview of clustering results (RP / RS):")
    st.dataframe(df_clusters.head())

    st.write("Number of columns:", len(df_clusters.columns))
    st.write("Columns:", list(df_clusters.columns))

    if "Label" in df_clusters.columns:
        st.write("Cluster label distribution:")
        st.dataframe(df_clusters["Label"].value_counts().reset_index().rename(
            columns={"index": "Label", "Label": "Count"}
        ))

except Exception:
    st.warning("Clustering results file not found.")

# ========================
# CLASSIFICATION
# ========================
st.header("Classification")

try:
    df_classif = pd.read_csv("features_clients_avec_labels.csv", sep=";")

    st.write("Dataset used for RP / RS classification:")
    st.dataframe(df_classif.head())

    st.write("""
The classification model uses the engineered features and the cluster labels (RP / RS).
A neural network (MLPClassifier) is trained to predict whether a client is RP or RS.
""")

    st.write("Input features for classification:")
    classif_features = [col for col in df_classif.columns if col not in ["ID", "Label"]]
    st.write(classif_features)

    st.write("Target variable:")
    st.write("Label")

except Exception:
    st.warning("Classification file not found.")

# ========================
# FORECASTING
# ========================
st.header("Forecasting")

try:
    # 1. Load forecasting data
    df_forecast = pd.read_csv("serie_temporelle_journaliere_pour_forecasting.csv", sep=";")

    # 2. Load labels RP / RS
    df_labels = pd.read_csv("features_clients_avec_labels.csv", sep=";")

    # 3. Keep only useful columns for merge
    df_labels_small = df_labels[["ID", "Label"]]

    # 4. Merge forecasting data with RP/RS labels
    df_forecast = df_forecast.merge(df_labels_small, on="ID", how="left")

    st.write("Dataset used for consumption forecasting:")
    st.dataframe(df_forecast.head())

    st.write("""
Forecasting is performed on daily electricity consumption.
Here, the dashboard compares the average daily consumption depending on the client type: RP or RS.
""")

    st.write("Number of columns:", len(df_forecast.columns))
    st.write("Columns:", list(df_forecast.columns))

    # 5. Select RP or RS
    forecast_label = st.selectbox("Select client type for forecasting", ["RP", "RS"])

    # 6. Filter data by selected label
    df_filtered = df_forecast[df_forecast["Label"] == forecast_label].copy()

    # 7. Convert date column
    df_filtered["date_jour"] = pd.to_datetime(df_filtered["date_jour"])

    # 8. Compute average daily consumption for selected type
    df_avg = (
        df_filtered.groupby("date_jour")["conso_journaliere_kWh"]
        .mean()
        .reset_index()
    )

    # 9. Plot average curve
    fig_forecast = px.line(
        df_avg,
        x="date_jour",
        y="conso_journaliere_kWh",
        title=f"Average daily consumption - {forecast_label}"
    )

    st.plotly_chart(fig_forecast, use_container_width=True)

except Exception:
    st.warning("Forecasting file not found.")
