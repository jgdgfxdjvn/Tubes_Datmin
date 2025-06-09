import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import streamlit as st

def plot_feature_importance(model, feature_names):
    """Membuat bar plot untuk feature importance dari model berbasis tree."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax)
        ax.set_title('Feature Importance')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        plt.tight_layout()
        return fig
    else:
        st.warning("Model yang diberikan tidak memiliki atribut 'feature_importances_'. Visualisasi ini hanya untuk model berbasis tree.")
        return None

def plot_residuals(y_test, y_pred):
    """Membuat scatter plot untuk residual (aktual vs prediksi)."""
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, ax=ax, alpha=0.6)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_title('Residual Plot (Aktual vs Prediksi)')
    ax.set_xlabel('Nilai Prediksi')
    ax.set_ylabel('Residual (Aktual - Prediksi)')
    plt.tight_layout()
    return fig