import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

forecasts = pd.read_csv("precomputed_forecasts.csv")
anomalies = pd.read_csv("anomalies_detected.csv")
feat_imp = pd.read_csv("feature_importance.csv")

from forecasting_ai import chatbot

st.title("AI Demand Forecasting Dashboard")

store = st.number_input("Store ID", min_value=1)
item = st.number_input("Item ID", min_value=1)

filtered = forecasts[(forecasts.store==store) & (forecasts.item==item)]

if len(filtered):
    st.subheader("Forecast Chart")
    plt.figure(figsize=(10,4))
    plt.plot(filtered['date'], filtered['rf_pred'], label='RF')
    plt.plot(filtered['date'], filtered['xgb_pred'], label='XGB')
    plt.legend()
    st.pyplot(plt)

    st.subheader("Anomalies")
    st.write(anomalies[(anomalies.store==store) & (anomalies.item==item)])

    st.subheader("Feature Importance")
    st.write(feat_imp.head(10))

st.subheader("Chatbot")
query = st.text_input("Ask something...")
if query:
    st.write(chatbot(query))
