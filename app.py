
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
    Dashboard ini menampilkan hasil analisis
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

    st.title("📈 Visualisasi Penjualan")

    st.write("""
    Halaman ini menampilkan hasil eksplorasi data (Exploratory Data Analysis)
    yang digunakan untuk memahami pola penjualan supermarket sebelum dilakukan
    proses pemodelan menggunakan model ARIMA.
    """)

    st.subheader("Analisis Agregat Penjualan Berdasarkan Bulan")

    st.image(
        "monthly_bar_chart.png",
        use_container_width=True
    )

    st.write("""
    Grafik menunjukkan total akumulasi penjualan setiap bulan selama periode
    2015–2018. Terlihat adanya fluktuasi penjualan pada setiap bulan dengan
    beberapa periode mengalami peningkatan penjualan yang cukup signifikan.
    """)

    st.subheader("Distribusi Frekuensi Nilai Penjualan Harian")

    st.image(
        "distribution_plot.png",
        use_container_width=True
    )

    st.write("""
    Distribusi data menunjukkan bahwa sebagian besar nilai penjualan berada
    pada rentang yang relatif rendah, sedangkan nilai penjualan yang tinggi
    hanya terjadi pada beberapa periode tertentu.
    """)
    # ===========================
# MODEL ARIMA
# ===========================

elif menu == "🤖 Model ARIMA":

    st.title("🤖 Model ARIMA")

    st.write("""
    Halaman ini menampilkan tahapan pembentukan model ARIMA yang digunakan
    untuk melakukan forecasting penjualan supermarket. Tahapan meliputi
    pengujian stasioneritas data, identifikasi parameter menggunakan plot
    ACF dan PACF, serta penentuan model terbaik berdasarkan nilai AIC.
    """)

    # ======================
    # Hasil Uji Stasioneritas
    # ======================

    st.subheader("📈 Plot Data Stasioner")

    st.image(
        "adf_plot.png",
        use_container_width=True
    )

    st.write("""
    Grafik di atas menunjukkan data penjualan yang telah memenuhi kondisi
    stasioner sehingga dapat langsung digunakan dalam proses pemodelan
    ARIMA tanpa dilakukan proses differencing (d = 0).
    """)

    # ======================
    # Plot ACF & PACF
    # ======================

    st.subheader("📊 Plot ACF dan PACF")

    st.image(
        "acf_pacf_plot.png",
        use_container_width=True
    )

    st.write("""
    Plot Autocorrelation Function (ACF) digunakan untuk membantu
    menentukan nilai parameter q, sedangkan Partial Autocorrelation
    Function (PACF) digunakan untuk menentukan nilai parameter p.
    Kedua grafik tersebut digunakan sebagai dasar dalam proses
    identifikasi parameter model ARIMA.
    """)

    # ======================
    # Model Terbaik
    # ======================

    st.subheader("🏆 Model Terbaik")

    st.success("""
Model terbaik yang diperoleh berdasarkan proses identifikasi parameter
dan pencarian nilai Akaike Information Criterion (AIC) terkecil adalah
**ARIMA(3,0,3)**.

Model tersebut kemudian digunakan untuk melakukan proses forecasting
terhadap data testing sehingga diperoleh hasil prediksi penjualan yang
selanjutnya dievaluasi menggunakan metrik RMSE, MAE, dan SMAPE.
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
