# =============================================================================
# MODULE DE VISUALISATION (Version Professionnelle)
# =============================================================================
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
from scipy import stats
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

class PortfolioVisualizer:
    """Génère des visualisations professionnelles pour l'analyse de portefeuille."""
    
    def __init__(self, base_dir="../plots"):
        """
        Initialise le visualiseur avec un style professionnel.
        
        Args:
            base_dir (str): Répertoire de base pour sauvegarder les graphiques
        """
        self.base_dir = base_dir
        # Création des sous-répertoires pour chaque type de graphique
        self.subdirs = {
            '1_days': ['episodes', 'evaluation', 'training'],
            '7_days': ['episodes', 'evaluation', 'training'],
            '30_days': ['episodes', 'evaluation', 'training'],
            'comparison': []
        }
        self._setup_directories()
        
        # Configuration du style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        
        # Palette de couleurs professionnelle
        self.colors = sns.color_palette("husl", 8)
        
    def _setup_directories(self):
        """Crée la structure de répertoires pour les graphiques."""
        for subdir, nested_dirs in self.subdirs.items():
            main_path = os.path.join(self.base_dir, subdir)
            os.makedirs(main_path, exist_ok=True)
            for nested in nested_dirs:
                os.makedirs(os.path.join(main_path, nested), exist_ok=True)

    def _format_currency(self, x, p):
        """Formate les valeurs monétaires en FCFA."""
        if abs(x) >= 1e9:
            return f'{x/1e9:.1f}B'
        elif abs(x) >= 1e6:
            return f'{x/1e6:.1f}M'
        elif abs(x) >= 1e3:
            return f'{x/1e3:.1f}K'
        return f'{x:.0f}'

    def _format_percentage(self, x, p):
        """Formate les pourcentages."""
        return f'{x:.1f}%'
        
    def _validate_episode_data(self, episode_data: Dict[str, Any]) -> bool:
        """
        Valide les données de l'épisode avant la visualisation.
        
        Args:
            episode_data: Dictionnaire contenant les données de l'épisode
            
        Returns:
            bool: True si les données sont valides, False sinon
        """
        required_fields = [
            'episode_number', 'dates', 'total_values', 
            'portfolio_values', 'cash_values', 'returns'
        ]
        
        try:
            # Vérification des champs requis
            for field in required_fields:
                if field not in episode_data:
                    print(f"❌ Champ manquant: {field}")
                    return False
            
            # Vérification de la cohérence des longueurs
            lengths = {
                'dates': len(episode_data['dates']),
                'total_values': len(episode_data['total_values']),
                'portfolio_values': len(episode_data['portfolio_values']),
                'cash_values': len(episode_data['cash_values']),
                'returns': len(episode_data['returns'])
            }
            
            if len(set(lengths.values())) > 1:
                print("❌ Incohérence dans les longueurs des séries:")
                for key, value in lengths.items():
                    print(f"   - {key}: {value}")
                return False
            
            # Vérification des types et valeurs
            if not all(isinstance(v, (int, float)) for v in episode_data['total_values']):
                print("❌ Valeurs totales invalides")
                return False
            
            if not all(isinstance(v, (int, float)) for v in episode_data['returns']):
                print("❌ Rendements invalides")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de la validation: {str(e)}")
            return False
            
    def _calculate_performance_metrics(self, values: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
        """
        Calcule les métriques de performance complètes.
        
        Args:
            values: Série temporelle des valeurs du portefeuille
            returns: Série temporelle des rendements
            
        Returns:
            Dict[str, float]: Dictionnaire des métriques calculées
        """
        try:
            returns = np.array(returns)
            values = np.array(values)
            
            # Rendement total et annualisé
            total_return = (values[-1] / values[0] - 1) if len(values) > 1 else 0
            n_years = len(returns) / 52  # Supposant des données hebdomadaires
            annual_return = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0
            
            # Métriques de risque
            volatility = np.std(returns) * np.sqrt(52) if len(returns) > 0 else 0
            downside_returns = returns[returns < 0]
            downside_vol = np.std(downside_returns) * np.sqrt(52) if len(downside_returns) > 0 else 0
            
            # Ratios de performance
            risk_free_rate = 0.02  # Taux sans risque annuel
            excess_returns = returns - risk_free_rate/52
            sharpe = (np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(52)) if np.std(excess_returns) > 0 else 0
            sortino = (np.mean(excess_returns) / downside_vol) if downside_vol > 0 else 0
            
            # Maximum drawdown et ratio de Calmar
            peak = values[0]
            max_dd = 0
            for val in values[1:]:
                if val > peak:
                    peak = val
                dd = (peak - val) / peak
                max_dd = max(max_dd, dd)
            calmar = annual_return / max_dd if max_dd > 0 else 0
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'max_drawdown': max_dd,
                'calmar_ratio': calmar,
                'win_rate': np.mean(returns > 0) if len(returns) > 0 else 0,
                'skewness': stats.skew(returns) if len(returns) > 2 else 0,
                'kurtosis': stats.kurtosis(returns) if len(returns) > 2 else 0
            }
            
        except Exception as e:
            print(f"❌ Erreur dans le calcul des métriques: {e}")
            return {}

    def _calculate_drawdown_series(self, values):
        """Calcule la série temporelle des drawdowns."""
        peak = values[0]
        drawdowns = []
        for val in values:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0
            drawdowns.append(dd)
        return np.array(drawdowns)

    def _plot_drawdown(self, values, ax):
        """Trace la courbe de drawdown."""
        drawdowns = self._calculate_drawdown_series(values)
        ax.fill_between(range(len(drawdowns)), 0, -drawdowns * 100, color='red', alpha=0.3)
        ax.plot(range(len(drawdowns)), -drawdowns * 100, color='red', label='Drawdown')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True)
        ax.legend()

    def plot_portfolio_evolution(self, episode_data: Dict[str, Any], freq_days: int) -> str:
        """
        Génère un graphique détaillé de l'évolution du portefeuille.
        
        Args:
            episode_data: Données de l'épisode
            freq_days: Fréquence de rebalancement en jours
            
        Returns:
            str: Chemin du fichier PDF généré
        """
        try:
            episode_num = episode_data['episode_number']
            dates = pd.to_datetime(episode_data['dates'])
            
            # Configuration de la figure
            fig = plt.figure(figsize=(11, 7))
            gs = GridSpec(2, 1, height_ratios=[3, 1])
            
            # Graphique principal
            ax1 = fig.add_subplot(gs[0])
            
            # Valeur totale
            ax1.plot(dates, episode_data['total_values'], 
                    label='Valeur Totale', color=self.colors[0], linewidth=2)
            
            # Valeur du portefeuille et cash
            ax1.plot(dates, episode_data['portfolio_values'],
                    label='Valeur Portefeuille', color=self.colors[1], linestyle='--')
            ax1.plot(dates, episode_data['cash_values'],
                    label='Cash', color=self.colors[2], linestyle=':')
            
            # Formatting
            ax1.yaxis.set_major_formatter(FuncFormatter(self._format_currency))
            ax1.set_title(f'Évolution du Portefeuille - Épisode {episode_num}')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Valeur (FCFA)')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Drawdown
            ax2 = fig.add_subplot(gs[1])
            drawdowns = self._calculate_drawdown_series(episode_data['total_values'])
            ax2.fill_between(dates, 0, -drawdowns * 100, 
                           color='red', alpha=0.3, label='Drawdown')
            ax2.set_ylabel('Drawdown (%)')
            ax2.yaxis.set_major_formatter(FuncFormatter(self._format_percentage))
            ax2.grid(True, alpha=0.3)
            
            # Mise en page
            plt.tight_layout()
            
            # Sauvegarde
            save_dir = os.path.join(self.base_dir, f'{freq_days}_days', 'episodes')
            filepath = os.path.join(save_dir, f'ep_{episode_num:03d}_portfolio.pdf')
            plt.savefig(filepath, format='pdf', bbox_inches='tight', dpi=300)
            plt.close()
            
            return filepath
            
        except Exception as e:
            print(f"❌ Erreur graphique portefeuille: {str(e)}")
            if 'fig' in locals():
                plt.close()
            return None
    def plot_transaction_costs(self, episode_data: Dict[str, Any], freq_days: int) -> str:
        """
        Génère un graphique détaillé des coûts de transaction.
        
        Args:
            episode_data: Données de l'épisode
            freq_days: Fréquence de rebalancement en jours
            
        Returns:
            str: Chemin du fichier PDF généré
        """
        try:
            episode_num = episode_data['episode_number']
            dates = pd.to_datetime(episode_data['dates'][1:])  # Skip first date
            costs = episode_data['transaction_costs']
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))
            
            # Coûts par période
            bars = ax1.bar(dates, costs, color=self.colors[3],
                         alpha=0.7, label='Coût par période')
            ax1.set_title(f'Coûts de Transaction - Épisode {episode_num}')
            ax1.set_ylabel('Coût (FCFA)')
            ax1.yaxis.set_major_formatter(FuncFormatter(self._format_currency))
            
            # Ajouter les valeurs sur les barres
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:,.0f}',
                        ha='center', va='bottom', rotation=90)
            
            # Coûts cumulés
            cumul_costs = np.cumsum(costs)
            ax2.plot(dates, cumul_costs, color=self.colors[4],
                    linewidth=2, label='Coûts cumulés')
            ax2.fill_between(dates, 0, cumul_costs, alpha=0.3,
                           color=self.colors[4])
            ax2.set_ylabel('Coûts cumulés (FCFA)')
            ax2.yaxis.set_major_formatter(FuncFormatter(self._format_currency))
            
            # Stats
            stats_text = (
                f'Coût total: {cumul_costs[-1]:,.0f} FCFA\n'
                f'Coût moyen: {np.mean(costs):,.0f} FCFA\n'
                f'Coût max: {np.max(costs):,.0f} FCFA'
            )
            ax2.text(0.02, 0.95, stats_text,
                    transform=ax2.transAxes,
                    verticalalignment='top',
                    bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
            
            for ax in [ax1, ax2]:
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            plt.tight_layout()
            
            # Sauvegarde
            save_dir = os.path.join(self.base_dir, f'{freq_days}_days', 'episodes')
            filepath = os.path.join(save_dir, f'ep_{episode_num:03d}_costs.pdf')
            plt.savefig(filepath, format='pdf', bbox_inches='tight', dpi=300)
            plt.close()
            
            return filepath
            
        except Exception as e:
            print(f"❌ Erreur graphique coûts: {str(e)}")
            if 'fig' in locals():
                plt.close()
            return None

    def plot_allocation(self, episode_data: Dict[str, Any], freq_days: int) -> str:
        """
        Génère un graphique détaillé de l'allocation du portefeuille.
        
        Args:
            episode_data: Données de l'épisode
            freq_days: Fréquence de rebalancement en jours
            
        Returns:
            str: Chemin du fichier PDF généré
        """
        try:
            episode_num = episode_data['episode_number']
            dates = pd.to_datetime(episode_data['dates'])
            
            # Création du DataFrame des poids
            weights_df = pd.DataFrame(episode_data['weights_history'],
                                    index=dates)
            
            # Filtrer les actifs utilisés (>1% à un moment donné)
            used_assets = weights_df.columns[(weights_df > 0.01).any()]
            weights_df = weights_df[used_assets]
            
            # Configuration de la figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))
            
            # Graphique en aires empilées
            weights_df.plot(kind='area', stacked=True,
                          ax=ax1, colormap='viridis')
            ax1.set_title(f'Allocation du Portefeuille - Épisode {episode_num}')
            ax1.set_ylabel('Allocation (%)')
            ax1.yaxis.set_major_formatter(FuncFormatter(self._format_percentage))
            
            # Heatmap des poids
            sns.heatmap(weights_df.T, ax=ax2, cmap='RdYlBu_r',
                       cbar_kws={'label': 'Poids (%)'})
            ax2.set_title('Heatmap des Allocations')
            
            # Stats
            n_assets = (weights_df > 0.01).sum(axis=1)
            avg_assets = n_assets.mean()
            stats_text = (
                f'Nombre moyen d\'actifs: {avg_assets:.1f}\n'
                f'Max actifs: {n_assets.max()}\n'
                f'Min actifs: {n_assets.min()}'
            )
            ax1.text(0.02, 0.95, stats_text,
                    transform=ax1.transAxes,
                    verticalalignment='top',
                    bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
            
            plt.tight_layout()
            
            # Sauvegarde
            save_dir = os.path.join(self.base_dir, f'{freq_days}_days', 'episodes')
            filepath = os.path.join(save_dir, f'ep_{episode_num:03d}_allocation.pdf')
            plt.savefig(filepath, format='pdf', bbox_inches='tight', dpi=300)
            plt.close()
            
            return filepath
            
        except Exception as e:
            print(f"❌ Erreur graphique allocation: {str(e)}")
            if 'fig' in locals():
                plt.close()
            return None
            
    def _calculate_drawdown_series(self, values):
        """Calcule la série temporelle des drawdowns."""
        peak = values[0]
        drawdowns = []
        for val in values:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0
            drawdowns.append(dd)
        return np.array(drawdowns)

    def _plot_drawdown(self, values, ax):
        """Trace la courbe de drawdown."""
        drawdowns = self._calculate_drawdown_series(values)
        ax.fill_between(range(len(drawdowns)), 0, -drawdowns * 100, 
                       color='red', alpha=0.3)
        ax.plot(range(len(drawdowns)), -drawdowns * 100, 
               color='red', label='Drawdown')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True)
        ax.legend()

class EpisodeTracker:
    """Collecte et analyse les données de plusieurs épisodes."""
    
    def __init__(self, base_dir="../plots"):
        """
        Initialise le tracker d'épisodes.
        
        Args:
            base_dir: Répertoire de base pour les visualisations
        """
        self.episodes_data = []
        self.visualizer = PortfolioVisualizer(base_dir)
        
    def record_episode(self, env, episode_number: int, freq_days: int) -> None:
        """
        Enregistre les données d'un épisode et génère les visualisations.
        
        Args:
            env: L'environnement de trading
            episode_number: Numéro de l'épisode
            freq_days: Fréquence de rebalancement en jours
        """
        try:
            # Collection des données
            num_steps = len(env.returns_history)
            dates = [env.topk_dates[i][0] for i in range(num_steps)]
            dates.insert(0, env.topk_dates[0][0])  # Ajoute la date initiale
            
            episode_data = {
                'episode_number': episode_number,
                'dates': dates,
                'total_values': env.total_value_history.copy(),
                'portfolio_values': env.portfolio_value_history.copy(),
                'cash_values': env.cash_history.copy(),
                'returns': env.returns_history.copy(),
                'transaction_costs': env.transaction_costs_history.copy(),
                'weights_history': env.weights_history.copy(),
                'initial_cash': env.initial_cash,
                'final_total_value': env.total_value
            }
            
            # Enregistrement des données
            self.episodes_data.append(episode_data)
            
            # Génération des visualisations
            self.visualizer.plot_episode_summary(episode_data, freq_days)
            
            # Calcul et affichage des métriques principales
            returns = pd.Series(episode_data['returns'])
            sharpe = (returns.mean() / returns.std() * np.sqrt(52)
                     if not returns.empty and returns.std() > 0 else 0)
            total_return = episode_data['final_total_value']/env.initial_cash - 1
            
            print(f"\n✅ Épisode {episode_number} enregistré et visualisé")
            print(f"   Sharpe: {sharpe:.2f}")
            print(f"   Rendement: {total_return:+.1%}")
            
        except Exception as e:
            print(f"❌ Erreur enregistrement épisode: {str(e)}")

    def save_results(self, filepath='../results/training_summary.pkl'):
        """Sauvegarde les résultats d'entraînement."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.episodes_data, f)
            print(f"✅ Résultats sauvegardés: {filepath}")
        except Exception as e:
            print(f"❌ Erreur sauvegarde résultats: {str(e)}")

    def get_best_episode(self, metric='sharpe_ratio'):
        """Retourne le meilleur épisode selon la métrique spécifiée."""
        if not self.episodes_data:
            return None
        
        metrics_map = {
            'sharpe_ratio': lambda x: float('-inf') if 'returns' not in x 
                else (pd.Series(x['returns']).mean() / pd.Series(x['returns']).std() * np.sqrt(52)),
            'total_return': lambda x: x['final_total_value'] / x['initial_cash'] - 1,
            'max_drawdown': lambda x: -self._calculate_drawdown_series(x['total_values']).min()
        }
        
        metric_func = metrics_map.get(metric, lambda x: x.get(metric, float('-inf')))
        return max(self.episodes_data, key=metric_func)
        
    def compare_episodes(self, episode_ids: List[int], freq_days: int) -> str:
        """
        Compare les performances de plusieurs épisodes.
        
        Args:
            episode_ids: Liste des IDs des épisodes à comparer
            freq_days: Fréquence de rebalancement en jours
            
        Returns:
            str: Chemin du fichier PDF généré
        """
        try:
            fig = plt.figure(figsize=(15, 10))
            gs = GridSpec(2, 2)
            
            # 1. Comparaison des valeurs totales
            ax1 = fig.add_subplot(gs[0, :])
            for ep_id in episode_ids:
                ep_data = next((ep for ep in self.episodes_data if ep['episode_number'] == ep_id), None)
                if ep_data:
                    dates = pd.to_datetime(ep_data['dates'])
                    values = np.array(ep_data['total_values'])
                    normalized_values = values / values[0]
                    ax1.plot(dates, normalized_values, 
                            label=f'Episode {ep_id}', linewidth=2)
            
            ax1.set_title('Évolution des Valeurs (Normalisées)')
            ax1.set_ylabel('Valeur Relative')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Comparaison des rendements cumulés
            ax2 = fig.add_subplot(gs[1, 0])
            for ep_id in episode_ids:
                ep_data = next((ep for ep in self.episodes_data if ep['episode_number'] == ep_id), None)
                if ep_data and ep_data['returns']:
                    returns = pd.Series(ep_data['returns'])
                    cumul_returns = (1 + returns).cumprod() - 1
                    ax2.plot(range(len(returns)), cumul_returns * 100,
                            label=f'Episode {ep_id}', linewidth=2)
            
            ax2.set_title('Rendements Cumulés')
            ax2.set_xlabel('Période')
            ax2.set_ylabel('Rendement Cumulé (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Tableau comparatif
            ax3 = fig.add_subplot(gs[1, 1])
            metrics_data = []
            for ep_id in episode_ids:
                ep_data = next((ep for ep in self.episodes_data if ep['episode_number'] == ep_id), None)
                if ep_data:
                    returns = pd.Series(ep_data['returns'])
                    metrics = {
                        'Episode': ep_id,
                        'Rdt Total': f"{(ep_data['final_total_value']/ep_data['initial_cash'] - 1):+.1%}",
                        'Sharpe': f"{(returns.mean()/returns.std() * np.sqrt(52)):.2f}",
                        'Vol': f"{(returns.std() * np.sqrt(52)):.1%}",
                        'Max DD': f"{-self._calculate_drawdown_series(ep_data['total_values']).min():.1%}"
                    }
                    metrics_data.append(metrics)
            
            # Création et style du tableau
            if metrics_data:
                ax3.axis('tight')
                ax3.axis('off')
                table = ax3.table(
                    cellText=[[d[k] for k in metrics_data[0].keys()] for d in metrics_data],
                    colLabels=metrics_data[0].keys(),
                    loc='center',
                    cellLoc='center'
                )
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.5)
            
            plt.tight_layout()
            
            # Sauvegarde
            comparison_dir = os.path.join(self.visualizer.base_dir, f'{freq_days}_days/comparison')
            os.makedirs(comparison_dir, exist_ok=True)
            filepath = os.path.join(comparison_dir, f'episodes_comparison_{"-".join(map(str, episode_ids))}.pdf')
            plt.savefig(filepath, format='pdf', bbox_inches='tight', dpi=300)
            plt.close()
            
            return filepath
            
        except Exception as e:
            print(f"❌ Erreur comparaison épisodes: {str(e)}")
            if 'fig' in locals():
                plt.close()
            return None
            
            # 1. Comparaison des valeurs totales
            ax1 = fig.add_subplot(gs[0, :2])
            for ep_id in episode_ids:
                ep_data = next((ep for ep in self.episodes_data if ep['episode_number'] == ep_id), None)
                if ep_data:
                    ax1.plot(ep_data['dates'], ep_data['total_values'],
                            label=f'Episode {ep_id}')
            ax1.set_title('Évolution de la Valeur Totale')
            ax1.legend()
            ax1.grid(True)
            
            # 2. Comparaison des rendements (boxplot)
            ax2 = fig.add_subplot(gs[0, 2])
            returns_data = []
            labels = []
            for ep_id in episode_ids:
                ep_data = next((ep for ep in self.episodes_data if ep['episode_number'] == ep_id), None)
                if ep_data and ep_data['returns']:
                    returns_data.append(ep_data['returns'])
                    labels.append(f'Ep {ep_id}')
            if returns_data:
                ax2.boxplot(returns_data, labels=labels)
                ax2.set_title('Distribution des Rendements')
                ax2.grid(True)
            
            # 3. Tableau comparatif des métriques
            ax3 = fig.add_subplot(gs[1, :])
            metrics_data = []
            for ep_id in episode_ids:
                ep_data = next((ep for ep in self.episodes_data if ep['episode_number'] == ep_id), None)
                if ep_data:
                    metrics = self.visualizer._calculate_performance_metrics(
                        ep_data['total_values'],
                        ep_data['returns']
                    )
                    metrics['Episode'] = ep_id
                    metrics_data.append(metrics)
            
            if metrics_data:
                df_metrics = pd.DataFrame(metrics_data).set_index('Episode')
                df_metrics = df_metrics[['total_return', 'annual_return', 'volatility', 
                                       'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'win_rate']]
                
                table = ax3.table(
                    cellText=df_metrics.applymap(lambda x: f"{x:+.2%}" if isinstance(x, float) else str(x)).values,
                    colLabels=df_metrics.columns,
                    rowLabels=df_metrics.index,
                    loc='center',
                    cellLoc='center'
                )
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1.2, 1.5)
                ax3.axis('off')
            
            plt.tight_layout()
            
            if save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = os.path.join(self.base_dir, f'episodes_comparison_{timestamp}.png')
                plt.savefig(filepath, bbox_inches='tight')
                plt.close()
                print(f"✅ Comparaison sauvegardée: {filepath}")
            else:
                plt.show()
                
        except Exception as e:
            print(f"❌ Erreur lors de la comparaison des épisodes: {e}")