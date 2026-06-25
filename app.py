
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

df = pd.read_csv("sales_clean.csv")

forecast_df = pd.read_csv("forecast_result.csv")

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
        "🤖 Model ARIMA",
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
# MODEL ARIMA
# ==========================
elif menu == "🤖 Model ARIMA":

    st.title("🤖 Model ARIMA")

    st.write("""
    Halaman ini menampilkan proses pembentukan model ARIMA
    yang digunakan untuk forecasting penjualan supermarket.
    Tahapan meliputi uji stasioneritas menggunakan ADF Test,
    identifikasi parameter melalui plot ACF dan PACF,
    serta penentuan model terbaik berdasarkan nilai AIC.
    """)
    st.subheader("Model Terbaik")

st.success("""
Model terbaik yang diperoleh berdasarkan nilai AIC adalah
ARIMA (3,0,3). Model ini kemudian digunakan untuk melakukan
forecasting penjualan supermarket pada data testing.
""")

st.subheader("Hasil Uji Stasioneritas (ADF Test)")

st.info("""
Berdasarkan hasil Augmented Dickey-Fuller (ADF) Test,
data telah memenuhi kondisi stasioner sehingga dapat
digunakan dalam proses pemodelan ARIMA.
""")
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
        evaluation["Metrik"] == "RMSE",
        "Nilai"
    ].values[0]

    mae = evaluation.loc[
        evaluation["Metrik"] == "MAE",
        "Nilai"
    ].values[0]

    smape = evaluation.loc[
        evaluation["Metrik"] == "SMAPE",
        "Nilai"
    ].values[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("RMSE", f"{rmse:.6f}")

    with col2:
        st.metric("MAE", f"{mae:.6f}")

    with col3:
        st.metric("SMAPE", f"{smape:.2f}%")

    st.info(
        "Nilai RMSE, MAE, dan SMAPE diambil dari hasil evaluasi model ARIMA(3,0,3) yang disimpan pada file evaluation.csv."
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
