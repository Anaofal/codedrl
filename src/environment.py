# Fichier: src/environment.py
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from collections import deque
from src.config import RL_ENV_PARAMS, INDICATOR_COLS, STOCK_PICKING_PARAMS
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class PortfolioEnvironment(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, all_data: pd.DataFrame, topk_dates: List[Tuple[pd.Timestamp, List[str]]], 
                 indicators_topk: pd.DataFrame, fundamentals_panel: pd.DataFrame,
                 use_copula: bool = False, copula_function: Optional[Callable] = None):
        super().__init__()
        
        self.all_data = all_data
        self.topk_dates_data = topk_dates
        self.indicators_data = indicators_topk
        self.fundamentals_panel = fundamentals_panel
        self.use_copula = use_copula
        self.copula_function = copula_function

        # Paramètres de l'environnement
        self.initial_cash = RL_ENV_PARAMS['INITIAL_CAPITAL']
        self.transaction_cost_pct = RL_ENV_PARAMS['TRANSACTION_COST_PCT']
        self.risk_aversion_lambda = RL_ENV_PARAMS['RISK_AVERSION_LAMBDA']
        self.cvar_alpha = RL_ENV_PARAMS['CVAR_ALPHA']
        self.window_size_cvar_days = RL_ENV_PARAMS['WINDOW_SIZE_CVAR_DAYS']
        self.k_assets = STOCK_PICKING_PARAMS['K']
        
        self.n_total_assets = len(all_data.columns.get_level_values(0).unique())
        self.n_indicators = len(INDICATOR_COLS)
        self.n_fundamentals = fundamentals_panel.shape[1]

        # Espaces d'action et d'observation
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.k_assets,), dtype=np.float32)
        
        state_dim = self.k_assets + 2 + (self.k_assets * self.n_indicators) + (self.k_assets * self.n_fundamentals)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)
        
        logging.info(f"Environnement initialisé avec {len(self.topk_dates_data)} steps.")

    def _get_observation(self, date, tickers):
        try:
            indicators = self.indicators_data.loc[date, tickers].values.flatten()
        except KeyError:
            indicators = np.zeros(self.k_assets * self.n_indicators)

        try:
            fundamentals = self.fundamentals_panel.loc[date, tickers].values.flatten()
        except KeyError:
            fundamentals = np.zeros(self.k_assets * self.n_fundamentals)

        obs = np.concatenate([
            self.current_weights,
            [self.total_value / self.initial_cash], # Normalisé
            [self.cash / self.total_value if self.total_value > 0 else 1.0], # Normalisé
            indicators,
            fundamentals
        ]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.total_value = self.initial_cash
        self.cash = self.initial_cash
        self.current_weights = np.zeros(self.k_assets)
        self.selected_assets = self.topk_dates_data[0][1]
        
        self.total_value_history = [self.initial_cash]
        self.dates_history = [self.topk_dates_data[0][0]]
        self.returns_history = []
        
        obs = self._get_observation(self.topk_dates_data[0][0], self.selected_assets)
        info = {'current_date': self.topk_dates_data[0][0], 'selected_assets': self.selected_assets}
        return obs, info

    def step(self, action):
        action = np.clip(action, 0, 1)
        if np.sum(action) > 1e-6:
            action /= np.sum(action)
        else:
            action = np.ones(self.k_assets) / self.k_assets

        start_date, start_tickers = self.topk_dates_data[self.current_step]
        
        # Calcul des coûts de transaction
        turnover = np.sum(np.abs(action - self.current_weights))
        costs = turnover * self.transaction_cost_pct * self.total_value
        self.total_value -= costs
        self.cash = self.total_value * (1 - np.sum(action)) # Le cash est ce qui n'est pas alloué
        
        self.current_weights = action
        self.selected_assets = start_tickers
        
        self.current_step += 1
        done = self.current_step >= len(self.topk_dates_data) - 1
        
        if done:
            reward = 0
            next_obs = self._get_observation(start_date, start_tickers)
            info = {'current_date': start_date, 'total_value': self.total_value}
            return next_obs, reward, True, False, info

        end_date, _ = self.topk_dates_data[self.current_step]
        
        # Calcul du rendement
        prices_start = self.all_data.loc[start_date, pd.IndexSlice[start_tickers, 'Close']].droplevel(1).reindex(start_tickers)
        prices_end = self.all_data.loc[end_date, pd.IndexSlice[start_tickers, 'Close']].droplevel(1).reindex(start_tickers)
        
        returns = (prices_end / prices_start - 1).fillna(0).values
        period_return = np.sum(self.current_weights * returns)
        
        previous_total_value = self.total_value
        self.total_value *= (1 + period_return)
        
        self.total_value_history.append(self.total_value)
        self.dates_history.append(end_date)
        self.returns_history.append((self.total_value - previous_total_value) / previous_total_value)
        
        # Calcul de la récompense
        cvar = 0
        if self.use_copula and self.copula_function:
            sim_returns, _, _ = self.copula_function(start_tickers, start_date, (end_date - start_date).days)
            portfolio_sim_returns = sim_returns @ self.current_weights
            losses = -portfolio_sim_returns
            cvar = np.mean(losses[losses >= np.quantile(losses, self.cvar_alpha)])
        
        reward = period_return - self.risk_aversion_lambda * cvar
        
        next_obs = self._get_observation(end_date, self.topk_dates_data[self.current_step][1])
        info = {'current_date': end_date, 'total_value': self.total_value}
        
        return next_obs, reward, done, False, info