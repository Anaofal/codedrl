# Fichier: run_training.py
import os
import pandas as pd
from src.trainer import PortfolioTrainer

def main():
    print("--- Démarrage de l'orchestrateur d'entraînement ---")
    try:
        all_data = pd.read_pickle('processed_data/all_data.pkl')
        dividendes = pd.read_pickle('processed_data/dividendes.pkl')
        fundamentals_panel = pd.read_pickle('processed_data/fundamentals_panel.pkl')
        print("✅ Données pré-traitées chargées.")
    except FileNotFoundError:
        print("❌ ERREUR: Fichiers non trouvés. Exécutez 'notebooks/00_pretraitement.ipynb' d'abord.")
        return

    trainer = PortfolioTrainer(all_data, dividendes, fundamentals_panel)

    # --- OPTIMISATION : AJUSTEMENT STRATÉGIQUE DES ÉPISODES ---
    experiments = [
        {'freq_name': 'daily', 'freq_days': 1, 'use_copula': True, 'episodes': 10}, # Réduit pour la vitesse
        {'freq_name': 'daily', 'freq_days': 1, 'use_copula': False, 'episodes': 10}, # Réduit pour la vitesse
        {'freq_name': 'weekly', 'freq_days': 7, 'use_copula': True, 'episodes': 40},
        {'freq_name': 'weekly', 'freq_days': 7, 'use_copula': False, 'episodes': 40},
        {'freq_name': 'monthly', 'freq_days': 30, 'use_copula': True, 'episodes': 30},
        {'freq_name': 'monthly', 'freq_days': 30, 'use_copula': False, 'episodes': 30},
    ]

    for exp in experiments:
        trainer.run_full_experiment(
            freq_name=exp['freq_name'],
            freq_days=exp['freq_days'],
            use_copula=exp['use_copula'],
            num_episodes=exp['episodes']
        )
    
    trainer.save_results()
    print("\n🎉🎉🎉 TOUTES LES EXPÉRIENCES SONT TERMINÉES ! 🎉🎉🎉")

if __name__ == "__main__":
    main()