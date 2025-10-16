import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from collections import deque
from src.config import RL_ENV_PARAMS, INDICATOR_COLS, STOCK_PICKING_PARAMS, DATA_PERIODS
from src.data_manager import BRVMTrainingManager, calculate_indicators
from src.copula import DynamicRVineCopula
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class PortfolioEnv(gym.Env):
    """
    Environnement de trading de portefeuille pour l'apprentissage par renforcement.
    Simule la gestion dynamique d'un portefeuille d'actions avec des coûts de transaction
    et une récompense basée sur le rendement ajusté au risque (CVaR).
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self,
                 all_data: pd.DataFrame,
                 fundamentals_panel: pd.DataFrame,
                 dividendes: pd.DataFrame,
                 rebalancing_freq_days: int,
                 period_name: str,
                 use_copula: bool = False,
                 render_mode: str = None):
        super().__init__()

        self.all_data = all_data
        self.fundamentals_panel = fundamentals_panel
        self.dividendes = dividendes
        self.rebalancing_freq_days = rebalancing_freq_days
        self.period_name = period_name
        self.use_copula = use_copula
        self.render_mode = render_mode

        self.initial_capital = RL_ENV_PARAMS['INITIAL_CAPITAL']
        self.transaction_cost_pct = RL_ENV_PARAMS['TRANSACTION_COST_PCT']
        self.risk_aversion_lambda = RL_ENV_PARAMS['RISK_AVERSION_LAMBDA']
        self.cvar_alpha = RL_ENV_PARAMS['CVAR_ALPHA']
        self.window_size_cvar_days = RL_ENV_PARAMS['WINDOW_SIZE_CVAR_DAYS']
        self.stock_picking_k = STOCK_PICKING_PARAMS['K']
        self.stock_picking_window_size = STOCK_PICKING_PARAMS['WINDOW_SIZE_WEEKS'] * 7 # Convertir en jours

        self.data_manager = BRVMTrainingManager(
            all_data=self.all_data,
            dividendes=self.dividendes,
            fundamentals_panel=self.fundamentals_panel,
            rebalancing_freq_days=self.rebalancing_freq_days
        )
        self.rebalancing_dates = self.data_manager.generate_rebalancing_dates_for_period(self.period_name)
        self.topk_dates_data = self.data_manager.generate_topk_dates_for_period(
            self.period_name,
            K=self.stock_picking_k,
            window_size=self.stock_picking_window_size // 7 # window_size en semaines pour generate_topk_dates_list
        )
        self.indicators_data = self.data_manager.generate_indicators_for_period(
            self.period_name,
            self.topk_dates_data,
            window_size=self.stock_picking_window_size # window_size en jours pour calculate_indicators_for_topk_periods
        )

        if self.use_copula:
            self.copula_model = DynamicRVineCopula(all_data=self.all_data)
            logging.info("Copule R-Vine activée pour la génération de scénarios.")
        else:
            self.copula_model = None
            logging.info("Copule R-Vine désactivée.")

        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.cash_in_hand = self.initial_capital
        self.current_weights = np.zeros(self.stock_picking_k)
        self.asset_prices_history = deque(maxlen=self.window_size_cvar_days)
        self.portfolio_returns_history = deque(maxlen=self.window_size_cvar_days)

        # Définition de l'espace d'action (poids du portefeuille)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.stock_picking_k,), dtype=np.float32)

        # Définition de l'espace d'observation
        # Composantes de l'état:
        # 1. Poids actuels du portefeuille (K)
        # 2. Valeur totale du portefeuille (1)
        # 3. Cash disponible (1)
        # 4. Indicateurs techniques des K actifs (K * len(INDICATOR_COLS))
        # 5. Indicateurs fondamentaux des K actifs (K * 4 - ex: nb_actions, secteur, pays, dividende)
        # 6. Facteurs de risque extraits via la copule (si utilisée, K) - Simplifié pour l'instant
        
        # Calculer la taille de l'espace d'observation
        num_fundamental_features = 4 # nb_actions, secteur, pays, dividende
        num_indicator_features = len(INDICATOR_COLS)
        
        low_bound = np.array([0.0] * self.stock_picking_k + # Poids
                             [0.0] + # Valeur portefeuille
                             [0.0] + # Cash
                             [0.0] * (self.stock_picking_k * num_indicator_features) + # Indicateurs techniques
                             [0.0] * (self.stock_picking_k * num_fundamental_features) # Indicateurs fondamentaux
                            )
        high_bound = np.array([1.0] * self.stock_picking_k + # Poids
                              [np.inf] + # Valeur portefeuille
                              [np.inf] + # Cash
                              [np.inf] * (self.stock_picking_k * num_indicator_features) + # Indicateurs techniques
                              [np.inf] * (self.stock_picking_k * num_fundamental_features) # Indicateurs fondamentaux
                             )
        
        self.observation_space = spaces.Box(low=low_bound, high=high_bound, dtype=np.float32)

        logging.info(f"Environnement initialisé pour la période '{period_name}' avec {self.stock_picking_k} actifs.")
        logging.info(f"Nombre de dates de rééquilibrage: {len(self.rebalancing_dates)}")

    def _get_current_data(self, date: pd.Timestamp, tickers: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Récupère les données de prix, fondamentales et indicateurs pour la date et les tickers donnés."""
        current_prices = self.all_data.loc[date, pd.IndexSlice[tickers, 'Close']].droplevel(0, axis=1)
        
        # Récupérer les fondamentaux pour les tickers actuels
        current_fundamentals = self.fundamentals_panel.loc[pd.IndexSlice[:, tickers], :].droplevel(0).loc[tickers]
        
        # Récupérer les indicateurs techniques pour les tickers actuels
        current_indicators = self.indicators_data.loc[pd.IndexSlice[date, tickers], :].droplevel(0)
        
        return current_prices, current_fundamentals, current_indicators

    def _calculate_cvar(self, returns_history: deque) -> float:
        """Calcule le CVaR à partir de l'historique des rendements."""
        if len(returns_history) < self.window_size_cvar_days:
            return 0.0 # Pas assez de données pour calculer le CVaR
        
        returns = np.array(returns_history)
        losses = -returns # Convertir les rendements en pertes
        
        # Calcul de la VaR
        var_idx = int(np.ceil(self.cvar_alpha * len(losses))) - 1
        if var_idx < 0: var_idx = 0 # S'assurer que l'index est valide
        sorted_losses = np.sort(losses)
        var = sorted_losses[var_idx]
        
        # Calcul de la CVaR (moyenne des pertes dépassant la VaR)
        cvar_losses = losses[losses >= var]
        if len(cvar_losses) > 0:
            cvar = np.mean(cvar_losses)
        else:
            cvar = var # Si aucune perte ne dépasse la VaR, la CVaR est la VaR
            
        return cvar

    def _get_observation(self, date: pd.Timestamp, tickers: List[str]) -> np.ndarray:
        """Construit le vecteur d'observation pour l'agent."""
        current_prices, current_fundamentals, current_indicators = self._get_current_data(date, tickers)

        # Assurer l'ordre des tickers pour les indicateurs et fondamentaux
        current_indicators = current_indicators.reindex(tickers).fillna(0)
        current_fundamentals = current_fundamentals.reindex(tickers).fillna(0)

        # Flatten les indicateurs et fondamentaux
        flat_indicators = current_indicators[INDICATOR_COLS].values.flatten()
        flat_fundamentals = current_fundamentals[['nb_actions', 'Secteur', 'Pays', 'Dividende']].values.flatten() # Assurez-vous que ces colonnes existent

        observation = np.concatenate([
            self.current_weights,
            [self.portfolio_value],
            [self.cash_in_hand],
            flat_indicators,
            flat_fundamentals
        ])
        return observation

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.cash_in_hand = self.initial_capital
        self.current_weights = np.zeros(self.stock_picking_k)
        self.asset_prices_history = deque(maxlen=self.window_size_cvar_days)
        self.portfolio_returns_history = deque(maxlen=self.window_size_cvar_days)
        
        # Initialiser avec la première date de rééquilibrage
        self.current_date_idx = 0
        current_date, current_tickers = self.topk_dates_data[self.current_date_idx]
        
        # Remplir l'historique des prix pour le calcul initial du CVaR
        # Pour simplifier, on utilise les prix de clôture des jours précédents la première date de rééquilibrage
        # Une implémentation plus robuste pourrait simuler ou charger un historique réel
        for _ in range(self.window_size_cvar_days):
            self.asset_prices_history.append(np.ones(self.stock_picking_k) * current_date.timestamp()) # Placeholder
            self.portfolio_returns_history.append(0.0) # Placeholder

        observation = self._get_observation(current_date, current_tickers)
        info = {"date": current_date, "tickers": current_tickers}
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Normaliser l'action pour que la somme des poids soit 1
        action = np.clip(action, 0, 1)
        if np.sum(action) > 0:
            action = action / np.sum(action)
        else:
            action = np.zeros_like(action) # Si tous les poids sont nuls, ne rien faire

        previous_portfolio_value = self.portfolio_value
        current_date, current_tickers = self.topk_dates_data[self.current_date_idx]
        next_date_idx = self.current_date_idx + 1
        done = next_date_idx >= len(self.topk_dates_data)

        if not done:
            next_date, next_tickers = self.topk_dates_data[next_date_idx]
            
            # 1. Calcul des rendements des actifs sur la période
            # Utiliser les prix de clôture de `current_date` et `next_date`
            prices_current = self.all_data.loc[current_date, pd.IndexSlice[current_tickers, 'Close']].droplevel(0, axis=1)
            prices_next = self.all_data.loc[next_date, pd.IndexSlice[next_tickers, 'Close']].droplevel(0, axis=1)

            # Assurer que les prix sont alignés et gérer les tickers manquants
            aligned_prices = pd.DataFrame(index=current_tickers)
            aligned_prices['current'] = prices_current
            aligned_prices['next'] = prices_next.reindex(current_tickers) # Reindex pour aligner

            # Gérer les NaN (actifs non disponibles ou disparus)
            aligned_prices = aligned_prices.dropna()
            
            # Si aucun actif n'est disponible, le portefeuille ne change pas de valeur
            if aligned_prices.empty:
                daily_returns = np.zeros(len(current_tickers))
            else:
                daily_returns = (aligned_prices['next'].values / aligned_prices['current'].values) - 1
                
            # 2. Calcul des coûts de transaction
            # Les poids sont pour les actifs *actuellement* dans le portefeuille
            # Si un actif n'est plus dans le top-K, son poids devient 0
            
            # Poids des actifs du portefeuille précédent
            prev_weights_full = np.zeros(len(self.all_data.columns.get_level_values(0).unique()))
            for i, ticker in enumerate(current_tickers):
                if i < len(self.current_weights): # S'assurer que l'index est valide
                    prev_weights_full[self.all_data.columns.get_level_values(0).unique().get_loc(ticker)] = self.current_weights[i]

            # Nouveaux poids pour les actifs du top-K actuel
            new_weights_full = np.zeros(len(self.all_data.columns.get_level_values(0).unique()))
            for i, ticker in enumerate(next_tickers):
                if i < len(action): # S'assurer que l'index est valide
                    new_weights_full[self.all_data.columns.get_level_values(0).unique().get_loc(ticker)] = action[i]

            # Calcul des coûts de transaction
            transaction_costs = np.sum(np.abs(new_weights_full - prev_weights_full)) * self.transaction_cost_pct * self.portfolio_value
            self.cash_in_hand -= transaction_costs

            # 3. Mise à jour des poids et de la valeur du portefeuille
            # Appliquer les rendements aux poids actuels pour obtenir la nouvelle valeur
            # Puis réallouer selon les nouveaux poids (action)
            
            # Valeur du portefeuille avant rééquilibrage
            portfolio_return_period = np.sum(self.current_weights * daily_returns[:len(self.current_weights)]) # Utiliser les rendements des actifs réellement détenus
            self.portfolio_value *= (1 + portfolio_return_period)
            
            # Mettre à jour les poids avec l'action de l'agent
            self.current_weights = action
            
            # Mettre à jour la valeur du portefeuille après rééquilibrage
            # Le cash est réinvesti selon les nouveaux poids
            self.cash_in_hand = self.portfolio_value * self.current_weights[0] # Exemple, si le cash est géré comme un actif
            
            # 4. Calcul de la récompense
            # Rendement du portefeuille sur la période
            portfolio_return = (self.portfolio_value - previous_portfolio_value) / previous_portfolio_value if previous_portfolio_value > 0 else 0.0
            self.portfolio_returns_history.append(portfolio_return)

            # Calcul du CVaR
            cvar = self._calculate_cvar(self.portfolio_returns_history)

            # Récompense finale
            reward = portfolio_return - self.risk_aversion_lambda * cvar - transaction_costs / self.initial_capital # Normaliser les coûts
            
            # 5. Préparation de l'observation pour le prochain état
            self.current_date_idx = next_date_idx
            next_observation = self._get_observation(next_date, next_tickers)
            info = {"date": next_date, "tickers": next_tickers, "portfolio_value": self.portfolio_value, "reward": reward}

        else:
            # Fin de l'épisode
            reward = 0.0
            next_observation = self._get_observation(current_date, current_tickers) # Dernière observation
            info = {"date": current_date, "tickers": current_tickers, "portfolio_value": self.portfolio_value, "reward": reward}

        return next_observation, reward, done, False, info

    def render(self):
        if self.render_mode == "human":
            print(f"Date: {self.topk_dates_data[self.current_date_idx][0].date()}, "
                  f"Valeur Portefeuille: {self.portfolio_value:.2f}, "
                  f"Cash: {self.cash_in_hand:.2f}")

    def close(self):
        pass

# Exemple d'utilisation (à supprimer ou commenter en production)
if __name__ == "__main__":
    # Créer des données fictives pour tester
    dates = pd.to_datetime(pd.date_range(start='2018-01-01', periods=200, freq='D'))
    tickers = ['ACTIF_A', 'ACTIF_B', 'ACTIF_C', 'ACTIF_D', 'ACTIF_E', 'ACTIF_F', 'ACTIF_G', 'ACTIF_H', 'ACTIF_I', 'ACTIF_J']
    
    # all_data
    data_dict = {}
    for ticker in tickers:
        data_dict[ticker] = pd.DataFrame({
            'Close': np.random.rand(200) * 100 + 50,
            'High': np.random.rand(200) * 10 + 150,
            'Low': np.random.rand(200) * 10 + 40,
            'Volume': np.random.rand(200) * 100000
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
        'Year': [2018, 2019, 2020, 2021, 2022, 2023],
        **{ticker: np.random.rand(6) * 2 for ticker in tickers}
    }
    dividendes_df = pd.DataFrame(dividendes_data)

    # Initialiser l'environnement
    env = PortfolioEnv(
        all_data=all_data_multiindex,
        fundamentals_panel=fundamentals_panel_df,
        dividendes=dividendes_df,
        rebalancing_freq_days=REBALANCING_FREQUENCIES['weekly'],
        period_name='train',
        use_copula=True,
        render_mode='human'
    )

    obs, info = env.reset()
    print(f"Observation initiale shape: {obs.shape}")
    print(f"Info initiale: {info}")

    for _ in range(5): # Simuler 5 étapes
        action = env.action_space.sample() # Action aléatoire
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        print(f"Reward: {reward:.4f}, Done: {done}, Truncated: {truncated}, Value: {info['portfolio_value']:.2f}")
        if done:
            break
    env.close()
