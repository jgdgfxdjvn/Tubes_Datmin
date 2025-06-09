# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os # Tambahkan import os untuk debugging path

# Import modul-modul yang kita buat
from preprocessing import load_data, preprocess_data
from modeling import train_regression_model, make_regression_prediction
from clustering import categorize_price, plot_price_categories_distribution
from utilitas import plot_feature_importance, plot_residuals

def main():
    st.set_page_config(layout="wide", page_title="Prediksi Harga Rumah")
    st.title("🏡 Prediksi Harga Penjualan Rumah")
    st.write("Aplikasi ini menggunakan data penjualan rumah untuk memprediksi harga, melakukan kategorisasi harga, dan analisis regresi.")

    # --- Sidebar untuk Navigasi dan Konfigurasi ---
    st.sidebar.title("Navigasi")
    page = st.sidebar.radio("Pilih Halaman", ["Beranda", "Eksplorasi Data", "Modeling Regresi", "Kategorisasi Harga", "Tentang"])

    # Memuat data dari file Excel
    df = load_data('data_rumah.xlsx') # Pastikan nama file sesuai

    if df is not None:
        # --- Ambil informasi sebelum preprocessing untuk display cleaning ---
        initial_missing_values = df.isnull().sum()
        initial_dtypes = df.dtypes
        # --- Akhir ambil informasi sebelum preprocessing ---

        # Lakukan pra-pemrosesan data
        df_processed, scaler_fitted = preprocess_data(df.copy())
        st.session_state['data_scaler'] = scaler_fitted # Simpan scaler di session state

        # --- Ambil informasi setelah preprocessing untuk display cleaning ---
        processed_missing_values = df_processed.isnull().sum()
        processed_dtypes = df_processed.dtypes
        # --- Akhir ambil informasi setelah preprocessing ---


        if page == "Beranda":
            st.header("Selamat Datang!")
            st.write("Aplikasi ini dirancang untuk membantu Anda memahami faktor-faktor yang memengaruhi harga rumah. Silakan pilih menu di sidebar untuk memulai analisis.")
            st.subheader("Sekilas Data Mentah")
            st.write(f"Jumlah baris: {df.shape[0]}, Jumlah kolom: {df.shape[1]}")
            # Format harga untuk tampilan di df.head()
            st.dataframe(df.head().style.format({'HARGA': 'Rp {:,.0f}'}))

            st.write("---")
            st.subheader("Sekilas Data Setelah Diproses")
            # Format harga untuk tampilan di df_processed.head()
            st.dataframe(df_processed.head().style.format({'harga': 'Rp {:,.0f}'}))
            st.info("""
                **Catatan:** Data setelah diproses terlihat dalam bentuk angka desimal (bahkan untuk kolom seperti 'kondisi') karena telah melalui tahap *encoding* (mengubah teks menjadi angka) dan *scaling* (menyeragamkan rentang nilai). Ini adalah format yang wajib bagi model *machine learning* untuk dapat melakukan prediksi dan analisis.
                Kolom 'harga' tidak diskala agar tetap dalam skala aslinya untuk prediksi.
            """)


        elif page == "Eksplorasi Data":
            st.header("📊 Eksplorasi Data")

            # --- BAGIAN BARU: Ringkasan Data Cleaning ---
            st.subheader("🧹 Ringkasan Data Cleaning")
            st.markdown("Berikut adalah gambaran perubahan data setelah proses *cleaning* dan *preprocessing*:")

            st.markdown("#### Status Missing Values")
            col1, col2 = st.columns(2)
            with col1:
                st.write("##### Sebelum Cleaning")
                if initial_missing_values.sum() == 0:
                    st.success("Tidak ada missing values pada data mentah.")
                else:
                    st.dataframe(initial_missing_values[initial_missing_values > 0].reset_index().rename(columns={'index': 'Kolom', 0: 'Jumlah Missing Values'}))
            with col2:
                st.write("##### Setelah Cleaning")
                if processed_missing_values.sum() == 0:
                    st.success("Semua missing values telah ditangani.")
                else:
                    st.dataframe(processed_missing_values[processed_missing_values > 0].reset_index().rename(columns={'index': 'Kolom', 0: 'Jumlah Missing Values'}))

            st.markdown("#### Perubahan Tipe Data & Transformasi")
            st.write("Perbandingan tipe data sebelum dan setelah preprocessing:")
            col1, col2 = st.columns(2)
            with col1:
                st.write("##### Tipe Data Awal")
                st.dataframe(initial_dtypes.reset_index().rename(columns={'index': 'Kolom', 0: 'Tipe Data'}))
            with col2:
                st.write("##### Tipe Data Setelah Preprocessing")
                st.dataframe(processed_dtypes.reset_index().rename(columns={'index': 'Kolom', 0: 'Tipe Data'}))

            st.info("""
                **Langkah-langkah Cleaning & Preprocessing yang Dilakukan:**
                -   **Pembersihan Nama Kolom:** Mengubah nama kolom menjadi huruf kecil dan mengganti spasi dengan garis bawah.
                -   **Penanganan Missing Values:** Mengisi missing values pada kolom numerik dengan median, dan pada kolom kategorikal (teks) dengan nilai yang paling sering muncul (mode).
                -   **Encoding Kategorikal:** Mengubah kolom teks (seperti 'carport', 'kondisi') menjadi angka menggunakan Label Encoding.
                -   **Feature Scaling:** Menyeragamkan rentang nilai fitur numerik (kecuali kolom 'harga') menggunakan StandardScaler, sehingga rata-rata menjadi 0 dan standar deviasi 1.
            """)
            st.write("---") # Garis pemisah
            # --- AKHIR BAGIAN BARU ---

            st.subheader("Statistik Deskriptif")
            st.write(df_processed.describe())

            st.subheader("Distribusi Harga Rumah")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df_processed['harga'], kde=True, ax=ax)
            ax.set_title('Distribusi Harga Rumah')
            ax.set_xlabel('Harga (Rupiah)')
            ax.set_ylabel('Frekuensi')
            st.pyplot(fig)

            st.subheader("Korelasi Antar Fitur Numerik")
            fig, ax = plt.subplots(figsize=(12, 10))
            numeric_cols = df_processed.select_dtypes(include=np.number).columns
            sns.heatmap(df_processed[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            ax.set_title('Matriks Korelasi Fitur')
            st.pyplot(fig)

        elif page == "Modeling Regresi":
            st.header("📈 Modeling Regresi (Prediksi Harga)")
            st.write("Latih model Random Forest Regressor untuk memprediksi harga rumah berdasarkan fitur yang tersedia.")

            st.subheader("Parameter Model")
            test_size = st.slider("Ukuran Data Uji (%)", 10, 50, 20) / 100
            random_state = st.slider("Random State", 0, 100, 42)

            model = None
            X_test_global = None

            if st.button("Latih Model Regresi"):
                with st.spinner("Melatih model..."):
                    if 'harga' in df_processed.columns:
                        model, X_test_global, y_test, y_pred, metrics = train_regression_model(df_processed, test_size, random_state)
                        st.session_state['trained_model'] = model
                        st.session_state['X_test_columns'] = X_test_global.columns.tolist()
                        st.success("Model berhasil dilatih!")

                        st.subheader("Metrik Evaluasi Model")
                        st.write(f"R-squared: **{metrics['r2']:.4f}**")
                        st.write(f"MAE (Mean Absolute Error): **Rp {metrics['mae']:.2f}**")
                        st.write(f"MSE (Mean Squared Error): **Rp {metrics['mse']:.2f}**")
                        st.write(f"RMSE (Root Mean Squared Error): **Rp {metrics['rmse']:.2f}**")

                        st.subheader("Visualisasi Prediksi")
                        fig_res = plot_residuals(y_test, y_pred)
                        st.pyplot(fig_res)

                        if model is not None and hasattr(model, 'feature_importances_'):
                            fig_fi = plot_feature_importance(model, X_test_global.columns)
                            st.pyplot(fig_fi)
                        else:
                            st.info("Model yang digunakan tidak memiliki atribut feature_importances_.")
                    else:
                        st.error("Kolom 'harga' tidak ditemukan di data yang diproses. Pastikan data Anda memiliki kolom harga yang valid.")

            st.subheader("Buat Prediksi Harga Rumah Baru")
            st.write("Masukkan nilai fitur untuk memprediksi harga rumah.")
            st.info("Pastikan Anda sudah melatih model regresi terlebih dahulu di atas.")

            feature_cols_for_pred = st.session_state.get('X_test_columns', [
                'luas_tanah_m2', 'luas_bangunan_m2', 'jumlah_kamar_tidur',
                'jumlah_kamar_mandi', 'jumlah_lantai', 'carport', 'listrik_va'
            ])

            input_features_dict = {}
            st.write("---")
            st.markdown("### Masukkan Detail Rumah Baru:")
            col1, col2, col3 = st.columns(3)

            with col1:
                input_features_dict['luas_tanah_m2'] = st.number_input("Luas Tanah (m²)", min_value=10, max_value=2000, value=100)
                input_features_dict['jumlah_kamar_tidur'] = st.slider("Jumlah Kamar Tidur", 1, 6, 3)
                input_features_dict['jumlah_lantai'] = st.slider("Jumlah Lantai", 1, 5, 1)

            with col2:
                input_features_dict['luas_bangunan_m2'] = st.number_input("Luas Bangunan (m²)", min_value=10, max_value=1000, value=80)
                input_features_dict['jumlah_kamar_mandi'] = st.slider("Jumlah Kamar Mandi", 1, 4, 2)

            with col3:
                carport_str = st.selectbox("Carport", ['Ada', 'Tidak Ada'])
                input_features_dict['carport'] = 1 if carport_str == 'Ada' else 0
                input_features_dict['listrik_va'] = st.number_input("Daya Listrik (VA)", min_value=900, max_value=30000, value=2200, step=100)

            new_data_raw = pd.DataFrame([input_features_dict])

            new_data_processed = None
            if 'trained_model' in st.session_state and st.session_state['X_test_columns'] and 'data_scaler' in st.session_state:
                try:
                    new_data_aligned = new_data_raw.reindex(columns=st.session_state['X_test_columns'], fill_value=0)
                    new_data_processed = st.session_state['data_scaler'].transform(new_data_aligned)
                    new_data_processed = pd.DataFrame(new_data_processed, columns=st.session_state['X_test_columns'])

                except KeyError as e:
                    st.error(f"Error kolom input tidak cocok dengan model: {e}. Pastikan nama dan jumlah kolom input benar.")
                    new_data_processed = None
                except Exception as e:
                    st.error(f"Error saat pra-pemrosesan data input baru: {e}. Pastikan tipe data dan format sesuai.")
                    new_data_processed = None
            else:
                st.warning("Model belum dilatih, kolom fitur tidak terdefinisi, atau scaler tidak tersedia. Silakan latih model terlebih dahulu.")
                new_data_processed = None

            if st.button("Prediksi Harga") and new_data_processed is not None:
                if 'trained_model' in st.session_state:
                    model_loaded = st.session_state['trained_model']
                    predicted_price = make_regression_prediction(model_loaded, new_data_processed)
                    if predicted_price is not None:
                        st.success(f"Harga prediksi untuk rumah ini adalah: **Rp {predicted_price[0]:,.2f}**")
                    else:
                        st.error("Gagal melakukan prediksi. Pastikan model telah dilatih dan input data sudah benar (termasuk skala).")
                else:
                    st.warning("Silakan latih model regresi terlebih dahulu di bagian atas.")


        elif page == "Kategorisasi Harga":
            st.header("🔍 Kategorisasi Harga Rumah")
            st.write("Mengategorikan harga rumah menjadi 'Murah', 'Normal', dan 'Mahal' berdasarkan kuantil.")

            st.subheader("Metode Kategorisasi")
            categorization_method = st.selectbox("Pilih Metode", ["Kuantil (33% | 66%)"])

            if st.button("Lakukan Kategorisasi"):
                with st.spinner("Melakukan kategorisasi harga..."):
                    df_categorized = categorize_price(df_processed.copy(), price_column='harga', method='quantiles')
                    st.success("Kategorisasi harga berhasil!")

                    st.subheader("Hasil Kategorisasi")
                    st.write("DataFrame dengan kolom 'kategori_harga' baru (5 baris pertama):")
                    # Format kolom harga di tampilan df_categorized
                    st.dataframe(df_categorized.head().style.format({'harga': 'Rp {:,.0f}'}))


                    st.subheader("Distribusi Rumah per Kategori Harga")
                    category_counts = df_categorized['kategori_harga'].value_counts().reindex(['Murah', 'Normal', 'Mahal'])
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis', ax=ax1)
                    ax1.set_title('Jumlah Rumah per Kategori Harga')
                    ax1.set_xlabel('Kategori Harga')
                    ax1.set_ylabel('Jumlah Rumah')
                    st.pyplot(fig1)

                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    sns.boxplot(x='kategori_harga', y='harga', data=df_categorized, order=['Murah', 'Normal', 'Mahal'], palette='viridis', ax=ax2)
                    ax2.set_title('Distribusi Harga per Kategori Harga')
                    ax2.set_xlabel('Kategori Harga')
                    ax2.set_ylabel('Harga (Rupiah)')
                    st.pyplot(fig2)

                    st.subheader("Profil Rata-rata Fitur per Kategori Harga")
                    category_summary = df_categorized.groupby('kategori_harga').mean(numeric_only=True).drop(columns=['harga'], errors='ignore')
                    # Format kolom harga (jika ada di category_summary, walaupun kita sudah drop 'harga')
                    # Kita bisa format semua kolom numerik agar rapi
                    styled_summary = category_summary.style.format({col: '{:,.2f}' for col in category_summary.columns if col != 'harga'})
                    st.dataframe(styled_summary) # Tampilkan yang sudah diformat
                    st.write("Tabel ini menunjukkan rata-rata fitur (misalnya, luas tanah, jumlah kamar) untuk setiap kategori harga, membantu memahami karakteristiknya.")


        elif page == "Tentang":
            st.header("Tentang Aplikasi Ini")
            st.write("""
            Aplikasi ini dibangun menggunakan **Streamlit** untuk mendemonstrasikan analisis data terhadap dataset harga rumah dari Kaggle (disimpan sebagai `data_rumah.xlsx`).
            Tujuan aplikasi ini adalah memberikan wawasan tentang pasar properti melalui:
            -   **Eksplorasi Data**: Melihat statistik deskriptif dan visualisasi dasar data untuk memahami distribusi fitur.
            -   **Modeling Regresi**: Melatih model regresi (Random Forest Regressor) untuk memprediksi harga rumah berdasarkan karakteristiknya, serta mengevaluasi performa model.
            -   **Kategorisasi Harga**: Mengategorikan rumah menjadi segmen 'Murah', 'Normal', dan 'Mahal' berdasarkan harga.

            **Sumber Data:**
            Dataset ini berasal dari Kaggle, dengan nama file `data_rumah.xlsx` di proyek ini.

            **Dibuat oleh:** [Nama Anda/Organisasi Anda]
            """)

    else:
        st.error("Gagal memuat data. Pastikan file 'data_rumah.xlsx' ada di direktori yang sama dan tidak rusak.")

if __name__ == '__main__':
    if 'trained_model' not in st.session_state:
        st.session_state['trained_model'] = None
    if 'X_test_columns' not in st.session_state:
        st.session_state['X_test_columns'] = []
    if 'data_scaler' not in st.session_state:
        st.session_state['data_scaler'] = None

    main()