import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pmdarima import auto_arima
from datetime import timedelta

# Set up Streamlit page
st.set_page_config(page_title="📈 ARIMA Time Series Forecast", layout="centered")
st.title("📉 Financial Time Series Forecasting (ARIMA Model)")

# Sidebar - File upload and options
st.sidebar.header("📂 Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV with 'date' and 'value' columns", type=["csv"])

forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", min_value=7, max_value=365, value=90, step=1)

# Process file if uploaded
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if 'date' not in df.columns or 'value' not in df.columns:
            st.error("❌ CSV must contain 'date' and 'value' columns.")
        else:
            # Prepare data
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            ts = df.set_index('date')['value']

            st.success("✅ File successfully loaded!")
            st.write("Data preview:")
            st.dataframe(df.head())

            # Fit ARIMA model
            with st.spinner("Fitting ARIMA model..."):
                model = auto_arima(ts, seasonal=False, stepwise=True, suppress_warnings=True)
                forecast, conf_int = model.predict(n_periods=forecast_horizon, return_conf_int=True)

                # Forecast index
                last_date = ts.index[-1]
                forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_horizon)

                # Forecast DataFrame
                forecast_df = pd.DataFrame({
                    'date': forecast_dates,
                    'forecast': forecast,
                    'lower': conf_int[:, 0],
                    'upper': conf_int[:, 1]
                })

            # Plotting
            st.subheader("📊 Forecast Plot")
            fig = go.Figure()

            # Historical data
            fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines', name='Actual'))

            # Forecast
            fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['forecast'], mode='lines', name='Forecast'))
            fig.add_trace(go.Scatter(
                x=forecast_df['date'], y=forecast_df['upper'],
                mode='lines', name='Upper Bound', line=dict(width=0), showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df['date'], y=forecast_df['lower'],
                mode='lines', name='Lower Bound', fill='tonexty', line=dict(width=0), fillcolor='rgba(0,100,80,0.2)', showlegend=True
            ))

            fig.update_layout(title="Forecast with ARIMA",
                              xaxis_title="Date", yaxis_title="Value",
                              template="plotly_white")
            st.plotly_chart(fig)

            # Show forecast data
            st.subheader("📋 Forecasted Values")
            st.dataframe(forecast_df)

    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
else:
    st.info("Please upload a CSV file to begin.")

# Footer
st.markdown("---")
st.caption("Developed using ARIMA modeling with pmdarima. Suitable for stationary or differenced financial time series.")
