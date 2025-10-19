# Fichier: src/visualizer.py
# Ce module peut être étendu plus tard pour des visualisations plus complexes.
# Pour l'instant, la logique de visualisation est intégrée dans app.py pour plus de simplicité.
# Vous pouvez y déplacer les fonctions de plotting de l'app Streamlit si le projet grandit.

import matplotlib.pyplot as plt
import seaborn as sns

def plot_performance_comparison(results_dict, title):
    """
    Génère un graphique comparant l'évolution de la valeur des portefeuilles.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    for name, data in results_dict.items():
        df = data['portfolio_values']
        ax.plot(df['Date'], df['PortfolioValue'], label=name)
    
    ax.set_title(title, fontsize=16)
    ax.set_ylabel("Valeur du Portefeuille (FCFA)")
    ax.legend()
    ax.grid(True, linestyle='--')
    return fig