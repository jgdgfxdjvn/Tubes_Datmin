import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import streamlit as st # Diperlukan untuk st.error/warning

def train_regression_model(df, test_size=0.2, random_state=42):
    """
    Melatih model regresi (RandomForestRegressor) untuk memprediksi harga rumah.
    Mengembalikan model yang dilatih, data uji, prediksi, dan metrik evaluasi.
    """
    # Pastikan kolom target 'harga' ada
    if 'harga' not in df.columns:
        st.error("Kolom 'harga' (target) tidak ditemukan di DataFrame. Tidak bisa melatih model.")
        return None, None, None, None, None

    # Pisahkan fitur (X) dan target (y)
    # Gunakan semua kolom numerik yang sudah diproses sebagai fitur, kecuali 'harga'
    X = df.drop(columns=['harga'])
    y = df['harga']

    # Pastikan X tidak kosong setelah drop 'harga'
    if X.empty:
        st.error("Tidak ada fitur yang tersisa untuk melatih model setelah menghapus kolom target.")
        return None, None, None, None, None

    # Bagi data menjadi set pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Inisialisasi dan latih model Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1) # n_jobs=-1 untuk parallel processing
    model.fit(X_train, y_train)

    # Buat prediksi pada data uji
    y_pred = model.predict(X_test)

    # Hitung metrik evaluasi
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    metrics = {
        'r2': r2,
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }

    return model, X_test, y_test, y_pred, metrics

def make_regression_prediction(model, new_data_df_scaled):
    """Membuat prediksi harga rumah menggunakan model yang sudah dilatih.
       Diasumsikan new_data_df_scaled sudah diskala.
    """
    if model:
        try:
            prediction = model.predict(new_data_df_scaled)
            return prediction
        except Exception as e:
            st.error(f"Error saat membuat prediksi: {e}. Pastikan format dan skala input sesuai.")
            return None
    st.warning("Model belum dilatih untuk membuat prediksi.")
    return None