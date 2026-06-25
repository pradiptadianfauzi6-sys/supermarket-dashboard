
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

forecast_df = pd.read_csv(
    "forecast_result.csv",
    sep=";",
    decimal=","
)

evaluation = pd.read_csv("evaluation.csv")

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

    st.title("🔮 Forecast Penjualan")

    st.write("""
    Halaman ini menampilkan hasil forecasting penjualan menggunakan
    model ARIMA(3,0,3). Grafik berikut memperlihatkan perbandingan
    antara data aktual dan hasil prediksi pada data testing.
    """)

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        pd.to_datetime(forecast_df["Order Date"]),
        forecast_df["Actual"],
        label="Data Aktual"
    )

    ax.plot(
        pd.to_datetime(forecast_df["Order Date"]),
        forecast_df["Prediction"],
        label="Prediksi ARIMA(3,0,3)"
    )

    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Penjualan")
    ax.legend()

    st.pyplot(fig)

    st.subheader("Hasil Forecasting")

    st.dataframe(forecast_df)

# ==========================
# EVALUASI
# ==========================
elif menu == "📋 Evaluasi":

    st.title("📋 Evaluasi Model")

    rmse = evaluation.loc[
        evaluation["Metric"] == "RMSE",
        "Value"
    ].values[0]

    mape = evaluation.loc[
        evaluation["Metric"] == "MAPE",
        "Value"
    ].values[0]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "RMSE",
            f"{rmse:,.2f}"
        )

    with col2:
        st.metric(
            "MAPE",
            f"{mape:,.2f}%"
        )

    st.info(
        "Nilai RMSE dan MAPE diambil dari hasil evaluasi model yang telah disimpan pada file evaluation.csv."
    )

# ==========================
# KESIMPULAN
# ==========================
elif menu == "📝 Kesimpulan":

    st.title("Kesimpulan")

    st.write("""
    Model ARIMA berhasil digunakan
    untuk melakukan forecasting penjualan supermarket.
    """)
