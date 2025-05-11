import streamlit as st

import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go

# App title
st.set_page_config(page_title="Financial Time Series Forecast", layout="centered")
st.title("📈 Financial Time Series Forecasting App")

# Sidebar instructions
st.sidebar.header("📂 Upload Data")
st.sidebar.markdown("Upload a CSV file with **date** and **value** columns.")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# Forecast slider
forecast_period = st.sidebar.slider("Select Forecast Horizon (days)", 7, 365, 90)

# Proceed if a file is uploaded
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Basic validation
        if "date" not in df.columns or "value" not in df.columns:
            st.error("❌ CSV must contain 'date' and 'value' columns.")
        else:
            # Preprocessing
            df["date"] = pd.to_datetime(df["date"])
            df = df[["date", "value"]].rename(columns={"date": "ds", "value": "y"})

            st.success("✅ Data uploaded successfully!")
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())

            # Prophet model
            with st.spinner("Training forecasting model..."):
                model = Prophet()
                model.fit(df)

                # Create future dataframe
                future = model.make_future_dataframe(periods=forecast_period)
                forecast = model.predict(future)

            # Plot
            st.subheader("📊 Forecast Plot")
            fig = plot_plotly(model, forecast)
            st.plotly_chart(fig)

            # Display forecast data
            st.subheader("📋 Forecasted Values")
            st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_period))

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
else:
    st.info("Please upload a CSV file to begin.")

# Footer
st.markdown("""
---
*Built with [Facebook Prophet](https://facebook.github.io/prophet/) and Streamlit*
""")
