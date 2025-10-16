import pandas as pd
import numpy as np
from typing import List, Dict
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from src.config import STOCK_PICKING_PARAMS, RL_ENV_PARAMS
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class BenchmarkStrategies:
    """
    Implémente les stratégies de benchmark pour la comparaison:
    - Buy & Hold (B&H)
    - Equal Weight (EW)
    - Markowitz (Mean-Variance Optimization - MVO)
    """
    def __init__(self, all_data: pd.DataFrame, initial_capital: float = RL_ENV_PARAMS['INITIAL_CAPITAL']):
        self.all_data = all_data
        self.initial_capital = initial_capital
        self.stock_picking_k = STOCK_PICKING_PARAMS['K']
        logging.info("📊 BenchmarkStrategies initialisé.")

    def _get_prices_for_tickers(self, date: pd.Timestamp, tickers: List[str]) -> pd.Series:
        """Récupère les prix de clôture pour les tickers à une date donnée."""
        return self.all_data.loc[date, pd.IndexSlice[tickers, 'Close']].droplevel(0, axis=1)

    def run_buy_and_hold(self, topk_dates_data: List[Tuple[pd.Timestamp, List[str]]]) -> pd.DataFrame:
        """
        Exécute la stratégie Buy & Hold.
        Achète les K actifs sélectionnés à la première date et les conserve.
        """
        logging.info("🚀 Exécution de la stratégie Buy & Hold...")
        if not topk_dates_data:
            return pd.DataFrame(columns=['Date', 'PortfolioValue'])

        start_date, initial_tickers = topk_dates_data[0]
        
        # Récupérer les prix de clôture pour tous les tickers sur toute la période de test
        all_dates = [d for d, _ in topk_dates_data]
        
        # Créer un DataFrame de prix pour les tickers initiaux sur toutes les dates
        prices_df = pd.DataFrame(index=all_dates)
        for ticker in initial_tickers:
            prices_df[ticker] = self.all_data.loc[all_dates, pd.IndexSlice[ticker, 'Close']].droplevel(0, axis=1)
        
        prices_df = prices_df.dropna(axis=1) # Supprimer les actifs sans données complètes
        
        if prices_df.empty:
            logging.warning("Aucun actif valide pour Buy & Hold après nettoyage des données.")
            return pd.DataFrame(columns=['Date', 'PortfolioValue'])

        # Répartir le capital initial également entre les actifs disponibles
        num_assets = prices_df.shape[1]
        if num_assets == 0:
            return pd.DataFrame(columns=['Date', 'PortfolioValue'])
            
        initial_investment_per_asset = self.initial_capital / num_assets
        
        # Calculer le nombre d'actions achetées pour chaque actif
        initial_prices = prices_df.iloc[0]
        shares_held = initial_investment_per_asset / initial_prices
        
        # Calculer la valeur du portefeuille pour chaque date
        portfolio_values = (prices_df * shares_held).sum(axis=1)
        
        results = pd.DataFrame({
            'Date': portfolio_values.index,
            'PortfolioValue': portfolio_values.values
        })
        logging.info("✅ Buy & Hold terminé.")
        return results

    def run_equal_weight(self, topk_dates_data: List[Tuple[pd.Timestamp, List[str]]]) -> pd.DataFrame:
        """
        Exécute la stratégie Equal Weight.
        Rééquilibre le portefeuille pour avoir des poids égaux à chaque dateJ'ai implémenté l'agent SAC dans `src/agent.py` et les modèles de référence (benchmarks) dans `src/benchmarks.py`.

La prochaine étape consiste à évaluer et comparer les performances des modèles. Cela implique de créer un script d'entraînement et d'évaluation qui utilisera l'environnement, l'agent SAC et les benchmarks. Je vais créer un nouveau fichier `src/trainer.py` pour gérer ce processus.
</thinking>
<write_to_file>
<path>src/trainer.py</path>
<content>
import gymnasium as gym
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

from src.environment import PortfolioEnv
from src.agent import SACAgent
from src.benchmarks import BenchmarkStrategies
from src.data_preprocessing import DataPreprocessor
from src.config import DATA_PERIODS, REBALANCING_FREQUENCIES, SAC_AGENT_PARAMS, STOCK_PICKING_PARAMS, INDICATOR_COLS, RL_ENV_PARAMS
import logging
import os

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class PortfolioTrainer:
    """
    Gère l'entraînement, la validation et le test de l'agent SAC
    et la comparaison avec les benchmarks.
    """
    def __init__(self, all_data: pd.DataFrame, fundamentals_panel: pd.DataFrame, dividendes: pd.DataFrame):
        self.all_data = all_data
        self.fundamentals_panel = fundamentals_panel
        self.dividendes = dividendes
        self.preprocessor = DataPreprocessor(all_data, fundamentals_panel, dividendes)
        self.processed_data = self.preprocessor.preprocess()
        self.log_returns = self.processed_data['log_returns']
        self.uniform_pseudo_observations = self.processed_data['uniform_pseudo_observations']
        
        self.benchmark_runner = BenchmarkStrategies(all_data=self.all_data)
        
        # Définir la taille de l'état et de l'action pour l'agent SAC
        # K (poids) + 1 (valeur portefeuille) + 1 (cash) + K*INDICATORS + K*FUNDAMENTALS
        num_fundamental_features = 4 # nb_actions, secteur, pays, dividende
        num_indicator_features = len(INDICATOR_COLS)
        self.state_size = STOCK_PICKING_PARAMS['K'] + 1 + 1 + \
                          (STOCK_PICKING_PARAMS['K'] * num_indicator_features) + \
                          (STOCK_PICKING_PARAMS['K'] * num_fundamental_features)
        self.action_size = STOCK_PICKING_PARAMS['K']
        
        self.agent = SACAgent(self.state_size, self.action_size)
        
        self.results = {} # Pour stocker les résultats de toutes les simulations

    def _train_sac_agent(self, env: PortfolioEnv, period_name: str, model_path: str):
        """Entraîne l'agent SAC sur l'environnement donné."""
        logging.info(f"Début de l'entraînement de l'agent SAC pour la période '{period_name}'...")
        
        scores = deque(maxlen=100)
        avg_scores = []

        for i_episode in tqdm(range(1, SAC_AGENT_PARAMS['N_EPISODES'] + 1), desc=f"Entraînement {period_name}"):
            state, info = env.reset()
            episode_reward = 0
            done = False
            truncated = False
            
            while not done and not truncated:
                action = self.agent.act(state)
                next_state, reward, done, truncated, info = env.step(action)
                self.agent.step(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
            
            scores.append(episode_reward)
            avg_score = np.mean(scores)
            avg_scores.append(avg_score)
            
            if i_episode % 10 == 0:
                logging.info(f"Épisode {i_episode}/{SAC_AGENT_PARAMS['N_EPISODES']} - Score moyen: {avg_score:.2f}")
        
        self.agent.save_model(model_path)
        logging.info(f"Entraînement SAC terminé pour '{period_name}'. Modèle sauvegardé à {model_path}")
        return avg_scores

    def _evaluate_strategy(self, env: PortfolioEnv, agent: SACAgent = None) -> pd.DataFrame:
        """Évalue une stratégie (SAC ou benchmark) sur l'environnement."""
        logging.info(f"Évaluation de la stratégie sur la période '{env.period_name}'...")
        
        portfolio_values = []
        dates = []
        
        state, info = env.reset()
        portfolio_values.append(env.initial_capital)
        dates.append(info['date'])
        
        done = False
        truncated = False
        
        while not done and not truncated:
            if agent:
                action = agent.act(state)
            else: # Pour les benchmarks, l'action est gérée par l'environnement ou une logique externe
                action = np.zeros(self.action_size) # Action factice pour les benchmarks
            
            state, reward, done, truncated, info = env.step(action)
            portfolio_values.append(info['portfolio_value'])
            dates.append(info['date'])
            
        results_df = pd.DataFrame({'Date': dates, 'PortfolioValue': portfolio_values})
        return results_df

    def _calculate_metrics(self, portfolio_values_df: pd.DataFrame, strategy_name: str) -> Dict[str, float]:
        """Calcule les métriques de performance pour une stratégie."""
        returns = portfolio_values_df['PortfolioValue'].pct_change().dropna()
        
        if returns.empty:
            logging.warning(f"Pas assez de rendements pour calculer les métriques pour {strategy_name}.")
            return {
                'Cumulative Return': 0.0, 'Annualized Return': 0.0, 'Annualized Volatility': 0.0,
                'Sharpe Ratio': 0.0, 'Sortino Ratio': 0.0, 'Calmar Ratio': 0.0,
                'Max Drawdown': 0.0, 'CVaR': 0.0, 'Turnover': 0.0
            }

        # Rendement cumulé
        cumulative_return = (portfolio_values_df['PortfolioValue'].iloc[-1] / portfolio_values_df['PortfolioValue'].iloc[0]) - 1

        # Rendement annualisé
        num_years = (portfolio_values_df['Date'].iloc[-1] - portfolio_values_df['Date'].iloc[0]).days / 365.25
        annualized_return = (1 + cumulative_return)**(1/num_years) - 1 if num_years > 0 else 0.0

        # Volatilité annualisée
        annualized_volatility = returns.std() * np.sqrt(252) # 252 jours de trading par an

        # Ratio de Sharpe (taux sans risque = 0 pour simplification)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0.0

        # Sortino Ratio (nécessite le calcul du downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0.0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0.0

        # Maximum Drawdown
        peak = portfolio_values_df['PortfolioValue'].expanding(min_periods=1).max()
        drawdown = (portfolio_values_df['PortfolioValue'] / peak) - 1
        max_drawdown = drawdown.min()

        # Calmar Ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
        
        # CVaR (utiliser l'historique des rendements de l'environnement si disponible, sinon estimer)
        # Pour une évaluation post-hoc, on peut estimer le CVaR sur les rendements de la stratégie
        losses = -returns
        cvar_alpha = RL_ENV_PARAMS['CVAR_ALPHA']
        var_idx = int(np.ceil(cvar_alpha * len(losses))) - 1
        if var_idx < 0: var_idx = 0
        sorted_losses = np.sort(losses)
        var = sorted_losses[var_idx]
        cvar_losses = losses[losses >= var]
        cvar = np.mean(cvar_losses) if len(cvar_losses) > 0 else var

        # Turnover (simplifié, nécessite une logique de trading plus fine pour un calcul précis)
        # Pour l'instant, on met 0.0 ou une valeur indicative
        turnover = 0.0 # Placeholder, à implémenter si les actions de trading sont loggées

        metrics = {
            'Cumulative Return': cumulative_return,
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Calmar Ratio': calmar_ratio,
            'Max Drawdown': max_drawdown,
            'CVaR': cvar,
            'Turnover': turnover
        }
        logging.info(f"Métriques pour {strategy_name}: {metrics}")
        return metrics

    def run_experiment(self, rebalancing_freq_name: str, use_copula: bool):
        """
        Exécute un cycle complet d'entraînement, validation et test pour une configuration donnée.
        """
        logging.info(f"\n--- Démarrage de l'expérience: Fréquence={rebalancing_freq_name}, Copule={use_copula} ---")
        
        rebalancing_freq_days = REBALANCING_FREQUENCIES[rebalancing_freq_name]
        
        # --- Entraînement ---
        train_env = PortfolioEnv(
            all_data=self.all_data,
            fundamentals_panel=self.fundamentals_panel,
            dividendes=self.dividendes,
            rebalancing_freq_days=rebalancing_freq_days,
            period_name='train',
            use_copula=use_copula
        )
        
        # Assurez-vous que l'agent est réinitialisé pour chaque entraînement
        self.agent = SACAgent(self.state_size, self.action_size)
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"sac_agent_{rebalancing_freq_name}_copula_{use_copula}.pth")
        
        self._train_sac_agent(train_env, 'train', model_path)
        
        # --- Validation (Charger le modèle entraîné et évaluer) ---
        logging.info(f"Validation de l'agent SAC pour la période 'validation'...")
        self.agent.load_model(model_path) # Charger le modèle entraîné
        
        val_env = PortfolioEnv(
            all_data=self.all_data,
            fundamentals_panel=self.fundamentals_panel,
            dividendes=self.dividendes,
            rebalancing_freq_days=rebalancing_freq_days,
            period_name='validation',
            use_copula=use_copula
        )
        sac_val_results = self._evaluate_strategy(val_env, self.agent)
        sac_val_metrics = self._calculate_metrics(sac_val_results, f"SAC_Validation_{rebalancing_freq_name}_Copula_{use_copula}")
        
        # --- Test (Charger le modèle entraîné et évaluer) ---
        logging.info(f"Test de l'agent SAC pour la période 'test'...")
        self.agent.load_model(model_path) # Charger le modèle entraîné
        
        test_env = PortfolioEnv(
            all_data=self.all_data,
            fundamentals_panel=self.fundamentals_panel,
            dividendes=self.dividendes,
            rebalancing_freq_days=rebalancing_freq_days,
            period_name='test',
            use_copula=use_copula
        )
        sac_test_results = self._evaluate_strategy(test_env, self.agent)
        sac_test_metrics = self._calculate_metrics(sac_test_results, f"SAC_Test_{rebalancing_freq_name}_Copula_{use_copula}")
        
        self.results[f"SAC_{rebalancing_freq_name}_Copula_{use_copula}"] = {
            'portfolio_values': sac_test_results,
            'metrics': sac_test_metrics
        }

        # --- Benchmarks ---
        logging.info("Exécution des benchmarks...")
        
        # Récupérer les top-K dates pour la période de test pour les benchmarks
        benchmark_topk_dates = test_env.data_manager.generate_topk_dates_for_period(
            'test',
            K=self.stock_picking_k,
            window_size=STOCK_PICKING_PARAMS['WINDOW_SIZE_WEEKS']
        )

        # Buy & Hold
        bh_results = self.benchmark_runner.run_buy_and_hold(benchmark_topk_dates)
        bh_metrics = self._calculate_metrics(bh_results, f"BuyHold_{rebalancing_freq_name}")
        self.results[f"BuyHold_{rebalancing_freq_name}"] = {'portfolio_values': bh_results, 'metrics': bh_metrics}

        # Equal Weight
        ew_results = self.benchmark_runner.run_equal_weight(benchmark_topk_dates)
        ew_metrics = self._calculate_metrics(ew_results, f"EqualWeight_{rebalancing_freq_name}")
        self.results[f"EqualWeight_{rebalancing_freq_name}"] = {'portfolio_values': ew_results, 'metrics': ew_metrics}

        # Markowitz
        mvo_results = self.benchmark_runner.run_markowitz(benchmark_topk_dates, rebalancing_freq_days)
        mvo_metrics = self._calculate_metrics(mvo_results, f"Markowitz_{rebalancing_freq_name}")
        self.results[f"Markowitz_{rebalancing_freq_name}"] = {'portfolio_values': mvo_results, 'metrics': mvo_metrics}
        
        logging.info(f"--- Expérience terminée: Fréquence={rebalancing_freq_name}, Copule={use_copula} ---")

    def get_results(self) -> Dict:
        """Retourne tous les résultats des expériences."""
        return self.results

# Exemple d'utilisation (à supprimer ou commenter en production)
if __name__ == "__main__":
    # Créer des données fictives pour tester
    dates = pd.to_datetime(pd.date_range(start='2018-01-01', periods=1000, freq='D'))
    tickers = [f'ACTIF_{chr(65+i)}' for i in range(STOCK_PICKING_PARAMS['K'])]
    
    # all_data
    data_dict = {}
    for ticker in tickers:
        data_dict[ticker] = pd.DataFrame({
            'Close': np.random.rand(1000) * 100 + 50,
            'High': np.random.rand(1000) * 10 + 150,
            'Low': np.random.rand(1000) * 10 + 40,
            'Volume': np.random.rand(1000) * 100000
        }, index=dates)
    all_data_multiindex = pd.concat(data_dict, axis=1, names=['Ticker', 'Feature'])

    # fundamentals_panel
    fundamentals_data = {
        'nb_actions': np.random.randint(100000, 1000000, len(tickers)),
        'Secteur': np.random.choice(['Finance', 'Technologie', 'Industrie'], len(tickers)),
        'Pays': np.random.choice(['BENIN', 'COTE D\'IVOIRE', 'SENEGAL'], len(tickers)),
        'Dividende': np.random.rand(len(tickers)) * 5
    }
    fundamentals_panel_df = pd.DataFrame(fundamentals_data, index=tickers)

    # dividendes
    dividendes_data = {
        'Year': [2018, 2019, 2020, 2021, 2022, 2023, 2024],
        **{ticker: np.random.rand(7) * 2 for ticker in tickers}
    }
    dividendes_df = pd.DataFrame(dividendes_data)

    trainer = PortfolioTrainer(all_data_multiindex, fundamentals_panel_df, dividendes_df)
    
    # Exécuter une expérience (par exemple, rééquilibrage hebdomadaire avec copule)
    trainer.run_experiment(rebalancing_freq_name='weekly', use_copula=True)
    
    # Afficher les résultats
    all_results = trainer.get_results()
    for strategy, data in all_results.items():
        print(f"\n--- Stratégie: {strategy} ---")
        print("Valeurs du portefeuille (5 premières):")
        print(data['portfolio_values'].head())
        print("Métriques:")
        for metric, value in data['metrics'].items():
            print(f"  {metric}: {value:.4f}")
