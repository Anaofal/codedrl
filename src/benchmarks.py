# Fichier: src/benchmarks.py
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
from src.config import RL_ENV_PARAMS
import logging
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

class BenchmarkStrategies:
    def __init__(self, all_data: pd.DataFrame, initial_capital: float = RL_ENV_PARAMS['INITIAL_CAPITAL'], transaction_cost: float = RL_ENV_PARAMS['TRANSACTION_COST_PCT']):
        self.all_data = all_data
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        logging.info("📊 BenchmarkStrategies initialisé.")

    def _simulate_strategy(self, topk_dates_data, weight_calculator_func):
        if not topk_dates_data: return pd.DataFrame(columns=['Date', 'PortfolioValue'])

        portfolio_value = self.initial_capital
        portfolio_values = [self.initial_capital]
        dates = [topk_dates_data[0][0]]
        current_weights_dict = {ticker: 0.0 for ticker in topk_dates_data[0][1]}

        for i in tqdm(range(len(topk_dates_data) - 1), desc=f"Simulation {weight_calculator_func.__name__}"):
            start_date, start_tickers = topk_dates_data[i]
            end_date, _ = topk_dates_data[i+1]

            target_weights_dict = weight_calculator_func(start_date, start_tickers)
            
            all_involved_tickers = list(set(current_weights_dict.keys()) | set(target_weights_dict.keys()))
            turnover = sum(abs(target_weights_dict.get(t, 0) - current_weights_dict.get(t, 0)) for t in all_involved_tickers)
            costs = turnover * self.transaction_cost * portfolio_value
            portfolio_value -= costs
            
            current_weights_dict = target_weights_dict

            prices_start = self.all_data.loc[start_date, pd.IndexSlice[start_tickers, 'Close']].droplevel(1).reindex(start_tickers)
            prices_end = self.all_data.loc[end_date, pd.IndexSlice[start_tickers, 'Close']].droplevel(1).reindex(start_tickers)
            
            returns = (prices_end / prices_start - 1).fillna(0)
            period_return = sum(current_weights_dict.get(t, 0) * returns.get(t, 0) for t in start_tickers)
            portfolio_value *= (1 + period_return)

            portfolio_values.append(portfolio_value)
            dates.append(end_date)
        
        return pd.DataFrame({'Date': dates, 'PortfolioValue': portfolio_values})

    def run_buy_and_hold(self, topk_dates_data):
        logging.info("🚀 Exécution de la stratégie Buy & Hold...")
        if not topk_dates_data: return pd.DataFrame(columns=['Date', 'PortfolioValue'])
        
        start_date, initial_tickers = topk_dates_data[0]
        all_dates = sorted(list(set([d for d, _ in topk_dates_data])))
        
        prices_df = self.all_data.loc[all_dates, pd.IndexSlice[initial_tickers, 'Close']].droplevel(1, axis=1).ffill().bfill()
        if prices_df.empty: return pd.DataFrame(columns=['Date', 'PortfolioValue'])

        initial_investment_per_asset = self.initial_capital / len(initial_tickers)
        initial_prices = prices_df.iloc[0]
        shares_held = initial_investment_per_asset / initial_prices
        
        portfolio_values = (prices_df * shares_held).sum(axis=1)
        return pd.DataFrame({'Date': portfolio_values.index, 'PortfolioValue': portfolio_values.values})

    def run_equal_weight(self, topk_dates_data):
        logging.info("🚀 Exécution de la stratégie Equal Weight...")
        def ew_calculator(date, tickers):
            return {ticker: 1.0 / len(tickers) for ticker in tickers}
        ew_calculator.__name__ = "EqualWeight"
        return self._simulate_strategy(topk_dates_data, ew_calculator)

    def run_markowitz(self, topk_dates_data, window_days=252):
        logging.info("🚀 Exécution de la stratégie Markowitz (MVO)...")
        def mvo_calculator(date, tickers):
            start_hist = date - pd.Timedelta(days=window_days)
            hist_prices = self.all_data.loc[start_hist:date, pd.IndexSlice[tickers, 'Close']].droplevel(1, axis=1).ffill().bfill()
            
            if len(hist_prices) < 20 or hist_prices.shape[1] < 2:
                return {ticker: 1.0 / len(tickers) for ticker in tickers}
            
            try:
                mu = expected_returns.mean_historical_return(hist_prices)
                S = risk_models.sample_cov(hist_prices)
                ef = EfficientFrontier(mu, S)
                ef.max_sharpe()
                weights = ef.clean_weights()
                return weights
            except Exception:
                return {ticker: 1.0 / len(tickers) for ticker in tickers}
        mvo_calculator.__name__ = "Markowitz"
        return self._simulate_strategy(topk_dates_data, mvo_calculator)