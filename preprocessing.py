# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st # Diperlukan untuk st.error/warning
import os # Diperlukan untuk debugging path

# Inisialisasi scaler di level global (sesuai implementasi Anda)
global_scaler = StandardScaler()

def load_data(filepath):
    """Memuat dataset dari file Excel."""
    # --- BAGIAN DEBUGGING PATH (bisa dihapus nanti jika sudah fix) ---
    current_dir = os.getcwd()
    st.error(f"DEBUG: Direktori kerja saat ini: {current_dir}")
    files_in_dir = os.listdir(current_dir)
    st.error(f"DEBUG: File di direktori ini: {', '.join(files_in_dir)}")
    # --- AKHIR BAGIAN DEBUGGING PATH ---

    try:
        df = pd.read_excel(filepath)
        return df
    except FileNotFoundError:
        st.error(f"Error: File '{filepath}' tidak ditemukan. Pastikan file Excel berada di direktori yang sama.")
        return None
    except Exception as e:
        st.error(f"Error saat memuat data dari '{filepath}': {e}")
        return None

def preprocess_data(df):
    """
    Melakukan pra-pemrosesan data: penanganan nama kolom, konversi tipe data,
    penanganan missing values, encoding kategorikal, dan scaling fitur numerik.
    Mengembalikan DataFrame yang sudah diproses dan objek StandardScaler yang sudah dilatih.
    """
    df_processed = df.copy()

    # Penanganan Nama Kolom: Ubah ke lowercase, ganti spasi dengan underscore
    df_processed.columns = df_processed.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.lower()
    # print("  - Nama kolom telah dibersihkan dan diubah ke huruf kecil.") # Di Streamlit ini tidak muncul, hanya untuk debugging

    # Penanganan Kolom 'harga': Konversi ke numerik jika formatnya masih teks
    if 'harga' in df_processed.columns:
        df_processed['harga'] = df_processed['harga'].astype(str).str.replace('rp.', '', regex=False).str.replace('.', '', regex=False).astype(float)
        # print("  - Kolom 'harga' telah dikonversi ke format numerik.")
    # else:
        # st.warning("Kolom 'harga' tidak ditemukan. Pastikan nama kolom target Anda benar.")

    # Penanganan Missing Values: Isi numerik dengan median, kategorikal dengan mode
    for col in df_processed.select_dtypes(include=np.number).columns:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            # st.info(f"Mengisi missing values di kolom numerik '{col}' dengan median.")

    # Mengisi missing values untuk kolom bertipe 'object' (string) dengan mode
    for col in df_processed.select_dtypes(include='object').columns:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
            # st.info(f"Mengisi missing values di kolom kategorikal '{col}' dengan mode.")

    # >>>>> BAGIAN PERBAIKAN: Encoding Semua Kolom Kategorikal (tipe 'object') <<<<<
    categorical_cols_to_encode = df_processed.select_dtypes(include='object').columns.tolist()

    # Pastikan kolom target 'harga' tidak ada di daftar ini jika ia masih string
    if 'harga' in categorical_cols_to_encode:
        categorical_cols_to_encode.remove('harga')

    if categorical_cols_to_encode:
        # print("  - Melakukan Label Encoding pada kolom-kolom kategorikal berikut:")
        for col in categorical_cols_to_encode:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str)) # Pastikan semua nilai adalah string
            # print(f"    - Kolom '{col}'")
    # else:
        # print("  - Tidak ditemukan kolom kategorikal (tipe object) untuk di-encode.")


    # Scaling Fitur Numerik (Kecuali target 'harga')
    numeric_features_to_scale = df_processed.select_dtypes(include=np.number).columns.tolist()
    if 'harga' in numeric_features_to_scale:
        numeric_features_to_scale.remove('harga') # Jangan scale kolom target

    scaler = StandardScaler() # Inisialisasi scaler
    if numeric_features_to_scale:
        df_processed[numeric_features_to_scale] = scaler.fit_transform(df_processed[numeric_features_to_scale])
        # st.info(f"Melakukan StandardScaler pada fitur: {', '.join(numeric_features_to_scale)}.")
    # else:
        # st.warning("Tidak ada fitur numerik yang tersisa untuk diskala setelah mengecualikan 'harga'.")

    return df_processed, scaler # Mengembalikan scaler yang sudah dilatih