
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Forecasting Penjualan Supermarket",
    page_icon="📈",
    layout="wide"
)

# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv("sales_clean.csv")
forecast_df = pd.read_csv("forecast_result.csv")

# ==========================
# SIDEBAR
# ==========================
st.sidebar.title("Menu")

menu = st.sidebar.radio(
    "Pilih Halaman",
    [
        "🏠 Home",
        "📊 Dataset",
        "📈 Visualisasi",
        "🔮 Forecast",
        "📋 Evaluasi",
        "📝 Kesimpulan"
    ]
)

# ==========================
# HOME
# ==========================
if menu == "🏠 Home":

    st.title("📈 Forecasting Penjualan Supermarket")

    st.write("""
    Dashboard ini menampilkan hasil forecasting
    penjualan supermarket menggunakan metode ARIMA.
    """)

# ==========================
# DATASET
# ==========================
elif menu == "📊 Dataset":

    st.title("Dataset")

    st.dataframe(df)

# ==========================
# VISUALISASI
# ==========================
elif menu == "📈 Visualisasi":

    st.title("Visualisasi Penjualan")

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        pd.to_datetime(df["Date"]),
        df["Sales"]
    )

    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Sales")

    st.pyplot(fig)

# ==========================
# FORECAST
# ==========================
elif menu == "🔮 Forecast":

    st.title("Forecast")

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        pd.to_datetime(forecast_df["Date"]),
        forecast_df["Actual"],
        label="Actual"
    )

    ax.plot(
        pd.to_datetime(forecast_df["Date"]),
        forecast_df["Forecast"],
        label="Forecast"
    )

    ax.legend()

    st.pyplot(fig)

    st.dataframe(forecast_df)

# ==========================
# EVALUASI
# ==========================
elif menu == "📋 Evaluasi":

    st.title("Evaluasi Model")

    st.metric("RMSE", f"{rmse:.2f}")

    st.metric("MAPE", f"{mape*100:.2f}%")

# ==========================
# KESIMPULAN
# ==========================
elif menu == "📝 Kesimpulan":

    st.title("Kesimpulan")

    st.write("""
    Model ARIMA berhasil digunakan
    untuk melakukan forecasting penjualan supermarket.
    """)
