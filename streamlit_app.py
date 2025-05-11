import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import timedelta

# --- Streamlit Page Setup ---
st.set_page_config(page_title="📈 Financial Forecasting with Exponential Smoothing", layout="centered")
st.title("📉 Financial Time Series Forecasting (ETS Model)")
st.markdown("Upload a CSV with a **`date`** and **`value`** column to generate a forecast.")

# --- Sidebar ---
st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
forecast_days = st.sidebar.slider("Forecast Horizon (days)", min_value=7, max_value=365, value=60, step=1)

# --- Load and Validate Data ---
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        if 'date' not in df.columns or 'value' not in df.columns:
            st.error("❌ The CSV file must contain 'date' and 'value' columns.")
        else:
            # Parse date and sort
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            df = df.set_index('date')
            ts = df['value']

            st.success("✅ File loaded successfully!")
            st.write("Data Preview:")
            st.dataframe(df.head())

            # --- Fit ETS Model ---
            with st.spinner("Fitting Exponential Smoothing model..."):
                model = ExponentialSmoothing(ts, trend="add", seasonal=None, initialization_method="estimated")
                model_fit = model.fit()

                # Forecast future values
                forecast_index = pd.date_range(start=ts.index[-1] + timedelta(days=1), periods=forecast_days)
                forecast = model_fit.forecast(steps=forecast_days)

                # Confidence intervals (basic ±1.96*std_resid approximation)
                resid_std = np.std(model_fit.resid)
                lower = forecast - 1.96 * resid_std
                upper = forecast + 1.96 * resid_std

                forecast_df = pd.DataFrame({
                    'date': forecast_index,
                    'forecast': forecast,
                    'lower': lower,
                    'upper': upper
                })

            # --- Plot ---
            st.subheader("📊 Forecast Plot")
            fig = go.Figure()

            # Actual data
            fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines', name='Actual'))

            # Forecast
            fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['forecast'], mode='lines', name='Forecast'))
            fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['upper'], line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(
                x=forecast_df['date'], y=forecast_df['lower'], fill='tonexty',
                fillcolor='rgba(0,176,246,0.2)', line=dict(width=0), name='Confidence Interval'
            ))

            fig.update_layout(
                title="Exponential Smoothing Forecast",
                xaxis_title="Date", yaxis_title="Value",
                template="plotly_white"
            )
            st.plotly_chart(fig)

            # --- Forecast Data Output ---
            st.subheader("📋 Forecasted Values")
            st.dataframe(forecast_df.set_index('date').round(2))

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
else:
    st.info("Please upload a CSV file to begin.")

# --- Footer ---
st.markdown("---")
st.caption("Built using Holt-Winters Exponential Smoothing from `statsmodels`.")
