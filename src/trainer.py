# Fichier: src/trainer.py
import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm.auto import tqdm
import logging

from src.data_manager import BRVMTrainingManager
from src.environment import PortfolioEnvironment
from src.copula import DynamicRVineCopula
from src.agent import PortfolioSACAgent
from src.benchmarks import BenchmarkStrategies
from src.config import *

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class PortfolioTrainer:
    def __init__(self, all_data, dividendes, fundamentals_panel):
        self.all_data = all_data
        self.dividendes = dividendes
        self.fundamentals_panel = fundamentals_panel
        self.results = {}
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        logging.info("🎓 PortfolioTrainer initialisé.")

    def _train_agent(self, env, agent, num_episodes, model_suffix):
        best_sharpe = -np.inf
        patience, early_stop_counter = 5, 0

        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            
            with tqdm(total=len(env.topk_dates_data) - 1, desc=f"Ep {episode+1}/{num_episodes} ({model_suffix})", leave=False) as pbar:
                while not done:
                    if len(agent.replay_buffer) < SAC_AGENT_PARAMS['LEARNING_STARTS']:
                        action = np.ones(env.action_space.shape[0]) / env.action_space.shape[0]
                    else:
                        action = agent.select_action(obs, deterministic=False)
                    
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    agent.store_transition(obs, action, reward, next_obs, done)
                    obs = next_obs

                    if len(agent.replay_buffer) > SAC_AGENT_PARAMS['BATCH_SIZE']:
                        agent.update(SAC_AGENT_PARAMS['BATCH_SIZE'])
                    pbar.update(1)

            returns = pd.Series(env.returns_history)
            sharpe = (returns.mean() / returns.std()) * np.sqrt(52) if len(returns) > 1 and returns.std() > 1e-6 else -np.inf

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                early_stop_counter = 0
                agent.save(f"models/sac_agent_{model_suffix}_best.pth")
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    logging.info(f"   ⏹ Early stopping après {episode + 1} épisodes.")
                    break
        agent.save(f"models/sac_agent_{model_suffix}_final.pth")

    def _evaluate_agent(self, env, agent_path):
        agent = PortfolioSACAgent(
            state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0],
            k_assets=STOCK_PICKING_PARAMS['K'], n_total_assets=env.n_total_assets,
            n_indicators=len(INDICATOR_COLS), n_fundamentals=env.n_fundamentals
        )
        agent.load(agent_path)
        
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        return pd.DataFrame({'Date': env.dates_history, 'PortfolioValue': env.total_value_history})

    def _calculate_metrics(self, portfolio_df):
        if portfolio_df.empty or len(portfolio_df) < 2:
            return {k: 0.0 for k in ['Rendement Ann.', 'Volatilité Ann.', 'Sharpe', 'Max Drawdown']}
        
        returns = portfolio_df['PortfolioValue'].pct_change().dropna()
        if returns.empty: return {k: 0.0 for k in ['Rendement Ann.', 'Volatilité Ann.', 'Sharpe', 'Max Drawdown']}

        cum_return = (portfolio_df['PortfolioValue'].iloc[-1] / portfolio_df['PortfolioValue'].iloc[0]) - 1
        num_years = (portfolio_df['Date'].iloc[-1] - portfolio_df['Date'].iloc[0]).days / 365.25
        ann_return = (1 + cum_return)**(1/num_years) - 1 if num_years > 0 else 0
        
        freq_multiplier = 52 if self.current_freq_days == 7 else (252 if self.current_freq_days == 1 else 12)
        ann_vol = returns.std() * np.sqrt(freq_multiplier)
        sharpe = ann_return / ann_vol if ann_vol > 1e-6 else 0

        peak = portfolio_df['PortfolioValue'].expanding(min_periods=1).max()
        drawdown = (portfolio_df['PortfolioValue'] / peak) - 1
        max_dd = drawdown.min()

        return {'Rendement Ann.': ann_return, 'Volatilité Ann.': ann_vol, 'Sharpe': sharpe, 'Max Drawdown': max_dd}

    def run_full_experiment(self, freq_name, freq_days, use_copula, num_episodes):
        self.current_freq_days = freq_days
        model_suffix = f"{freq_name}_{'copula' if use_copula else 'nocopula'}"
        logging.info(f"\n{'#'*60}\n### EXPÉRIENCE: {model_suffix.upper()} ###\n{'#'*60}")

        manager = BRVMTrainingManager(self.all_data, self.dividendes, self.fundamentals_panel, freq_days)
        
        # 1. Training
        train_topk = manager.generate_topk_dates_for_period('train')
        train_indicators = manager.generate_indicators_for_period('train', train_topk)
        copula_sim = DynamicRVineCopula(self.all_data, n_simulations=500) if use_copula else None
        env_train = PortfolioEnvironment(self.all_data, train_topk, train_indicators, self.fundamentals_panel, use_copula=use_copula, copula_function=copula_sim.simulate_period_returns if use_copula else None)
        agent = PortfolioSACAgent(
            state_dim=env_train.observation_space.shape[0], action_dim=env_train.action_space.shape[0],
            k_assets=STOCK_PICKING_PARAMS['K'], n_total_assets=env_train.n_total_assets,
            n_indicators=len(INDICATOR_COLS), n_fundamentals=self.fundamentals_panel.shape[1]
        )
        self._train_agent(env_train, agent, num_episodes, model_suffix)

        # 2. Testing
        test_topk = manager.generate_topk_dates_for_period('test')
        test_indicators = manager.generate_indicators_for_period('test', test_topk)
        env_test = PortfolioEnvironment(self.all_data, test_topk, test_indicators, self.fundamentals_panel, use_copula=False) # Pas de copule pour le test réel
        
        agent_path = f"models/sac_agent_{model_suffix}_best.pth"
        sac_results_df = self._evaluate_agent(env_test, agent_path)
        self.results[f"SAC_{model_suffix}"] = {
            'portfolio_values': sac_results_df,
            'metrics': self._calculate_metrics(sac_results_df)
        }

        # 3. Benchmarks (only once per frequency)
        if f"BuyHold_{freq_name}" not in self.results:
            bench = BenchmarkStrategies(self.all_data)
            bh_df = bench.run_buy_and_hold(test_topk)
            ew_df = bench.run_equal_weight(test_topk)
            mvo_df = bench.run_markowitz(test_topk)
            self.results[f"BuyHold_{freq_name}"] = {'portfolio_values': bh_df, 'metrics': self._calculate_metrics(bh_df)}
            self.results[f"EqualWeight_{freq_name}"] = {'portfolio_values': ew_df, 'metrics': self._calculate_metrics(ew_df)}
            self.results[f"Markowitz_{freq_name}"] = {'portfolio_values': mvo_df, 'metrics': self._calculate_metrics(mvo_df)}

    def save_results(self):
        filepath = f'results/full_experiment_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)
        logging.info(f"\n✅ Résultats complets sauvegardés dans : {filepath}")