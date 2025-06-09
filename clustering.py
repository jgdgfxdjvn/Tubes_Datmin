import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.cluster import KMeans # Tidak lagi dibutuhkan untuk kategorisasi


def categorize_price(df_processed, price_column='harga', method='quantiles'):
    """
    Mengkategorikan harga rumah menjadi 'Murah', 'Normal', 'Mahal'.
    Mengembalikan DataFrame dengan kolom 'kategori_harga' baru.
    """
    df_categorized = df_processed.copy()

    if price_column not in df_categorized.columns:
        print(f"❌ Error: Kolom '{price_column}' tidak ditemukan untuk kategorisasi harga.")
        return df_categorized

    print(f"\n--- Tahap Kategorisasi Harga: {price_column} ---")

    if method == 'quantiles':
        # Hitung kuantil (33.3% dan 66.7%)
        q1 = df_categorized[price_column].quantile(0.33)
        q2 = df_categorized[price_column].quantile(0.66)

        # Definisikan bin dan label
        bins = [df_categorized[price_column].min() - 1, q1, q2, df_categorized[price_column].max() + 1]
        labels = ['Murah', 'Normal', 'Mahal']

        df_categorized['kategori_harga'] = pd.cut(df_categorized[price_column], bins=bins, labels=labels, include_lowest=True, right=True)
        print(f"  ✔️ Harga dikategorikan berdasarkan kuantil:\n    Murah: < Rp {q1:,.2f}\n    Normal: Rp {q1:,.2f} - Rp {q2:,.2f}\n    Mahal: > Rp {q2:,.2f}")
    else:
        print("  ❌ Metode kategorisasi tidak valid. Menggunakan metode default 'quantiles'.")
        return categorize_price(df_processed, price_column, 'quantiles') # Rekursif ke metode default

    print("--- Kategorisasi Harga Selesai ---")
    return df_categorized

def plot_price_categories_distribution(df_categorized, price_column='harga', category_column='kategori_harga'):
    """
    Membuat plot distribusi harga per kategori.
    """
    if category_column not in df_categorized.columns:
        print(f"❌ Error: Kolom '{category_column}' tidak ditemukan untuk visualisasi kategori.")
        return None

    print(f"\n--- Visualisasi Distribusi Kategori Harga: {category_column} ---")

    # Hitung jumlah rumah per kategori
    category_counts = df_categorized[category_column].value_counts().reindex(['Murah', 'Normal', 'Mahal'])

    # Plot bar chart jumlah rumah per kategori
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis', ax=ax1)
    ax1.set_title('Jumlah Rumah per Kategori Harga')
    ax1.set_xlabel('Kategori Harga')
    ax1.set_ylabel('Jumlah Rumah')
    plt.tight_layout()
    plt.show() # Tampilkan plot di Jupyter/Streamlit

    # Plot box plot atau violin plot untuk melihat distribusi harga dalam setiap kategori
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=category_column, y=price_column, data=df_categorized, order=['Murah', 'Normal', 'Mahal'], palette='viridis', ax=ax2)
    ax2.set_title(f'Distribusi {price_column.title()} per Kategori Harga')
    ax2.set_xlabel('Kategori Harga')
    ax2.set_ylabel(f'{price_column.title()} (Rupiah)')
    plt.tight_layout()
    plt.show() # Tampilkan plot di Jupyter/Streamlit