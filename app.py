# Fichier: app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from glob import glob

st.set_page_config(layout="wide", page_title="Analyse de Portefeuille Dynamique")

@st.cache_data
def load_results(results_file):
    with open(results_file, 'rb') as f:
        return pickle.load(f)

def main():
    st.title("📈 Tableau de Bord d'Analyse de Portefeuille Dynamique")
    st.markdown("Analyse comparative des stratégies de gestion de portefeuille (BRVM).")

    results_files = glob("results/full_experiment_results_*.pkl")
    if not results_files:
        st.error("Aucun fichier de résultats trouvé. Veuillez lancer `python run_training.py`.")
        return

    latest_results_file = max(results_files, key=os.path.getctime)
    results = load_results(latest_results_file)
    st.success(f"Résultats chargés depuis : `{os.path.basename(latest_results_file)}`")

    st.sidebar.header("Filtres de Visualisation")
    freq_options = ['daily', 'weekly', 'monthly']
    selected_freq = st.sidebar.selectbox("Fréquence de rééquilibrage", freq_options, index=1)
    
    use_copula_options = st.sidebar.multiselect(
        "Modèles SAC à afficher",
        ['Avec Copule', 'Sans Copule'],
        default=['Avec Copule', 'Sans Copule']
    )

    metrics_data, plot_data = [], {}
    for name, data in results.items():
        is_sac_copula = 'SAC' in name and 'copula' in name and 'Avec Copule' in use_copula_options
        is_sac_nocopula = 'SAC' in name and 'nocopula' in name and 'Sans Copule' in use_copula_options
        is_benchmark = 'SAC' not in name

        if selected_freq in name and (is_sac_copula or is_sac_nocopula or is_benchmark):
            metrics_row = {'Stratégie': name}
            metrics_row.update(data['metrics'])
            metrics_data.append(metrics_row)
            plot_data[name] = data['portfolio_values']

    if not metrics_data:
        st.warning(f"Aucun résultat à afficher pour les filtres sélectionnés.")
        return

    metrics_df = pd.DataFrame(metrics_data).set_index('Stratégie').sort_values('Sharpe', ascending=False)

    st.header(f"Analyse pour la Fréquence `{selected_freq.upper()}`")

    fig, ax = plt.subplots(figsize=(14, 7))
    for name, df in plot_data.items():
        ax.plot(df['Date'], df['PortfolioValue'], label=name, lw=2.5 if 'SAC' in name else 1.5, alpha=0.9 if 'SAC' in name else 0.7)
    
    ax.set_title(f"Évolution de la Valeur du Portefeuille (Fréquence: {selected_freq})", fontsize=16)
    ax.set_ylabel("Valeur du Portefeuille (FCFA)", fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    st.pyplot(fig)

    st.subheader("Tableau Comparatif des Métriques")
    st.dataframe(metrics_df.style.format({
        'Rendement Ann.': '{:,.2%}', 'Volatilité Ann.': '{:,.2%}',
        'Sharpe': '{:,.2f}', 'Max Drawdown': '{:,.2%}'
    }).background_gradient(cmap='viridis', subset=['Sharpe']))

    st.header("🏆 Analyse Globale des Modèles")
    all_metrics = []
    for name, data in results.items():
        metrics_row = {'Modèle': name}
        metrics_row.update(data['metrics'])
        all_metrics.append(metrics_row)

    if all_metrics:
        all_metrics_df = pd.DataFrame(all_metrics).set_index('Modèle')
        best_model_name = all_metrics_df['Sharpe'].idxmax()
        best_freq = best_model_name.split('_')[1]

        col1, col2 = st.columns(2)
        col1.metric("Meilleur Modèle Global (Sharpe)", best_model_name)
        col2.metric("Meilleure Fréquence", best_freq.upper())
        
        st.dataframe(all_metrics_df.sort_values('Sharpe', ascending=False).style.format({
            'Rendement Ann.': '{:,.2%}', 'Volatilité Ann.': '{:,.2%}', 'Sharpe': '{:,.2f}', 'Max Drawdown': '{:,.2%}'
        }))

if __name__ == "__main__":
    main()