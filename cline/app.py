import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.trainer import PortfolioTrainer
from src.config import REBALANCING_FREQUENCIES, DATA_PERIODS, STOCK_PICKING_PARAMS, RL_ENV_PARAMS
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# --- Fonctions de chargement des données (à adapter à vos sources réelles) ---
@st.cache_data
def load_all_data():
    """Charge toutes les données de prix OHLCV."""
    # Exemple de chargement de données fictives pour le développement
    # Remplacez ceci par votre logique de chargement réelle (ex: depuis des fichiers CSV/Excel)
    dates = pd.to_datetime(pd.date_range(start='2018-01-01', periods=1000, freq='D'))
    tickers = [f'ACTIF_{chr(65+i)}' for i in range(STOCK_PICKING_PARAMS['K'])]
    
    data_dict = {}
    for ticker in tickers:
        data_dict[ticker] = pd.DataFrame({
            'Close': np.random.rand(1000) * 100 + 50,
            'High': np.random.rand(1000) * 10 + 150,
            'Low': np.random.rand(1000) * 10 + 40,
            'Volume': np.random.rand(1000) * 100000
        }, index=dates)
    all_data_multiindex = pd.concat(data_dict, axis=1, names=['Ticker', 'Feature'])
    return all_data_multiindex

@st.cache_data
def load_fundamentals_panel():
    """Charge les données fondamentales."""
    # Exemple de chargement de données fictives
    tickers = [f'ACTIF_{chr(65+i)}' for i in range(STOCK_PICKING_PARAMS['K'])]
    fundamentals_data = {
        'nb_actions': np.random.randint(100000, 1000000, len(tickers)),
        'Secteur': np.random.choice(['Finance', 'Technologie', 'Industrie'], len(tickers)),
        'Pays': np.random.choice(['BENIN', 'COTE D\'IVOIRE', 'SENEGAL'], len(tickers)),
        'Dividende': np.random.rand(len(tickers)) * 5
    }
    fundamentals_panel_df = pd.DataFrame(fundamentals_data, index=tickers)
    return fundamentals_panel_df

@st.cache_data
def load_dividendes():
    """Charge les données de dividendes."""
    # Exemple de chargement de données fictives
    tickers = [f'ACTIF_{chr(65+i)}' for i in range(STOCK_PICKING_PARAMS['K'])]
    dividendes_data = {
        'Year': [2018, 2019, 2020, 2021, 2022, 2023, 2024],
        **{ticker: np.random.rand(7) * 2 for ticker in tickers}
    }
    dividendes_df = pd.DataFrame(dividendes_data)
    return dividendes_df

# --- Interface Streamlit ---
st.set_page_config(layout="wide", page_title="Optimisation de Portefeuille Dynamique")

st.title("📈 Optimisation de Portefeuille Dynamique par RL et Copules R-Vine")
st.markdown("""
Cette application démontre une approche avancée de gestion de portefeuille d'actions,
combinant l'apprentissage par renforcement (Soft Actor-Critic), la modélisation des dépendances
par copules R-Vine et la minimisation du risque extrême (CVaR).
""")

# --- Chargement des données ---
with st.spinner("Chargement des données..."):
    all_data = load_all_data()
    fundamentals_panel = load_fundamentals_panel()
    dividendes = load_dividendes()
st.success("Données chargées avec succès!")

# --- Paramètres de l'utilisateur ---
st.sidebar.header("Paramètres de l'Expérience")

initial_capital = st.sidebar.number_input("Capital Initial", value=RL_ENV_PARAMS['INITIAL_CAPITAL'], min_value=1000.0, step=1000.0)
rebalancing_freq_name = st.sidebar.selectbox(
    "Fréquence de Rééquilibrage",
    options=list(REBALANCING_FREQUENCIES.keys()),
    index=1 # weekly par défaut
)
use_copula = st.sidebar.checkbox("Utiliser la Copule R-Vine", value=True)
run_experiment_button = st.sidebar.button("Lancer l'Expérience")

# --- Exécution de l'expérience ---
if run_experiment_button:
    st.subheader("Résultats de l'Expérience")
    
    trainer = PortfolioTrainer(all_data, fundamentals_panel, dividendes)
    
    with st.spinner(f"Exécution de l'expérience ({rebalancing_freq_name}, Copule={use_copula})... Cela peut prendre du temps."):
        trainer.run_experiment(rebalancing_freq_name, use_copula)
        results = trainer.get_results()
    
    st.success("Expérience terminée!")

    # --- Visualisation des performances ---
    st.subheader("Évolution de la Valeur du Portefeuille")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for strategy_name, data in results.items():
        portfolio_values_df = data['portfolio_values']
        ax.plot(portfolio_values_df['Date'], portfolio_values_df['PortfolioValue'], label=strategy_name)
    
    ax.set_title("Valeur du Portefeuille au Fil du Temps")
    ax.set_xlabel("Date")
    ax.set_ylabel("Valeur du Portefeuille")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # --- Tableau des métriques ---
    st.subheader("Tableau Comparatif des Métriques de Performance")
    
    metrics_data = []
    for strategy_name, data in results.items():
        metrics_row = {'Stratégie': strategy_name}
        metrics_row.update(data['metrics'])
        metrics_data.append(metrics_row)
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df.set_index('Stratégie').style.format("{:.2%}"))

    st.markdown("---")
    st.markdown("Développé par Naofal AKANHO")

else:
    st.info("Configurez les paramètres dans la barre latérale et cliquez sur 'Lancer l'Expérience'.")
