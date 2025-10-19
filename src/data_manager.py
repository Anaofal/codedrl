# Manages data loading, preprocessing, and storage
# Fichier: src/data_manager.py
import os
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import List, Dict, Tuple
from tqdm.auto import tqdm
import ta
import warnings
from src.config import INDICATOR_COLS, STOCK_PICKING_PARAMS, DATA_PERIODS

warnings.filterwarnings('ignore', category=RuntimeWarning)

def _calculate_topk_for_date(date, prices_dict, dividendes_by_year, K, window_size, weights):
    week_start = date - timedelta(weeks=window_size)
    all_tickers = list(prices_dict.keys())
    features, valid_mask = [], []
    prev_year_dividends = dividendes_by_year.get(date.year - 1, pd.Series(dtype=float))

    for ticker in all_tickers:
        df = prices_dict[ticker]
        df_window = df[(df['Date'] >= week_start) & (df['Date'] < date)]
        if len(df_window) < 2:
            features.append([np.nan] * 4)
            valid_mask.append(0)
            continue
        
        valid_mask.append(1)
        mu = (df_window['Close'].iloc[-1] / df_window['Close'].iloc[0]) - 1 if df_window['Close'].iloc[0] > 0 else 0
        sigma = df_window['Close'].pct_change().std()
        L = df_window['Volume'].mean()
        last_price = df_window['Close'].iloc[-1]
        dividend_value = prev_year_dividends.get(ticker, 0)
        D = (dividend_value / last_price) if last_price > 0 and not pd.isna(dividend_value) else 0
        features.append([mu, sigma, L, D])

    features = np.nan_to_num(np.array(features))
    valid_mask = np.array(valid_mask, dtype=bool)
    ranks = np.full_like(features, 0.0)
    for j in range(features.shape[1]):
        col = features[:, j]
        valid_feature_mask = valid_mask & (col != 0)
        if valid_feature_mask.sum() > 0:
            ranks[valid_feature_mask, j] = pd.Series(col[valid_feature_mask]).rank(pct=True).values

    scores = (weights['mu'] * ranks[:, 0] - weights['sigma'] * ranks[:, 1] + weights['L'] * ranks[:, 2] + weights['D'] * ranks[:, 3])
    scores[~valid_mask] = -np.inf
    top_indices = np.argsort(scores)[::-1][:K]
    return [all_tickers[i] for i in top_indices]

def _generate_topk_dates_list(all_data, dividendes, rebalancing_dates, K, window_size, weights):
    prices_dict = {
        ticker: all_data[ticker][['Close', 'Volume']].reset_index()
        for ticker in all_data.columns.get_level_values(0).unique()
    }
    
    div_long = dividendes.melt(id_vars=['Year'], var_name='Ticker', value_name='Dividende').dropna(subset=['Year'])
    div_pivot = div_long.pivot_table(index='Ticker', columns='Year', values='Dividende')
    dividendes_by_year = {int(year): div_pivot[year] for year in div_pivot.columns}

    topk_dates = [
        (pd.to_datetime(date.date()), _calculate_topk_for_date(date, prices_dict, dividendes_by_year, K, window_size, weights))
        for date in tqdm(rebalancing_dates, desc="Génération de la liste Top-K")
    ]
    return topk_dates

def _calculate_indicators(stock_data):
    if stock_data is None or stock_data.empty or len(stock_data) < 20:
        return pd.DataFrame([{col: 0.0 for col in INDICATOR_COLS}])
    
    df_numeric = stock_data[['High', 'Low', 'Close', 'Volume']].apply(pd.to_numeric, errors='coerce').ffill().bfill()
    if len(df_numeric) < 20: return _calculate_indicators(None)

    high, low, close, volume = df_numeric['High'], df_numeric['Low'], df_numeric['Close'], df_numeric['Volume']
    data = {col: 0.0 for col in INDICATOR_COLS}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try: data["MACD"] = ta.trend.MACD(close, fillna=True).macd_diff().iloc[-1]
        except Exception: pass
        try: data["RSI"] = ta.momentum.RSIIndicator(close, fillna=True).rsi().iloc[-1]
        except Exception: pass
        try: data["CCI"] = ta.trend.CCIIndicator(high, low, close, fillna=True).cci().iloc[-1]
        except Exception: pass
        try: data["STOCH_K"] = ta.momentum.StochasticOscillator(high, low, close, fillna=True).stoch().iloc[-1]
        except Exception: pass
        try: data["WILLR"] = ta.momentum.WilliamsRIndicator(high, low, close, fillna=True).williams_r().iloc[-1]
        except Exception: pass
        try:
            bb = ta.volatility.BollingerBands(close, fillna=True)
            data["BB_Upper"], data["BB_Middle"], data["BB_Lower"] = bb.bollinger_hband().iloc[-1], bb.bollinger_mavg().iloc[-1], bb.bollinger_lband().iloc[-1]
        except Exception: pass
        try: data["EMA"] = ta.trend.EMAIndicator(close, fillna=True).ema_indicator().iloc[-1]
        except Exception: pass
        try: data["ATR"] = ta.volatility.AverageTrueRange(high, low, close, fillna=True).average_true_range().iloc[-1]
        except Exception: pass
        try: data["SAR"] = ta.trend.PSARIndicator(high, low, close, fillna=True).psar().iloc[-1]
        except Exception: pass
        try: 
            data["Close_Lag1"] = close.shift(1).iloc[-1]
            data["Close_Lag3"] = close.shift(3).iloc[-1]
            data["Close_Lag5"] = close.shift(5).iloc[-1]
        except IndexError: pass
        try: 
            data["SMA_50"] = close.rolling(50, min_periods=1).mean().iloc[-1]
            data["SMA_20"] = close.rolling(20, min_periods=1).mean().iloc[-1]
        except Exception: pass
        try: data["OBV"] = ta.volume.OnBalanceVolumeIndicator(close, volume, fillna=True).on_balance_volume().iloc[-1]
        except Exception: pass
        try: data["MFI"] = ta.volume.MFIIndicator(high, low, close, volume, fillna=True).money_flow_index().iloc[-1]
        except Exception: pass
        try: data["ADX"] = ta.trend.ADXIndicator(high, low, close, fillna=True).adx().iloc[-1]
        except Exception: pass
        try: data["Volume"] = volume.iloc[-1]
        except Exception: pass

    return pd.DataFrame([pd.Series(data).fillna(0).to_dict()])

def _calculate_indicators_for_topk_periods(all_data, topk_dates, window_size=70):
    results = []
    for date, tickers in tqdm(topk_dates, desc="Calcul des indicateurs"):
        for ticker in tickers:
            try:
                df_ticker = all_data[ticker].copy().reset_index()
                window_df = df_ticker[df_ticker['Date'] < date].tail(window_size)
                if window_df.empty: continue
                ind = _calculate_indicators(window_df).assign(Date=date, Ticker=ticker)
                results.append(ind)
            except Exception as e:
                print(f"⚠️ Erreur indicateurs pour {ticker} à {date}: {e}")
    return pd.concat(results, ignore_index=True).set_index(["Date", "Ticker"]) if results else pd.DataFrame()

class BRVMTrainingManager:
    def __init__(self, all_data, dividendes, fundamentals_panel, rebalancing_freq_days=7):
        self.all_data = all_data
        self.dividendes = dividendes
        self.fundamentals_panel = fundamentals_panel
        self.rebalancing_freq_days = rebalancing_freq_days
        self.periods = DATA_PERIODS
        self.available_dates = sorted(all_data.index.get_level_values(0).unique())
        print(f"🏛️ BRVMTrainingManager initialisé (fréquence: {self.rebalancing_freq_days} jours).")

    def _find_nearest_available_date(self, target_date):
        available_dates_array = np.array(self.available_dates)
        if target_date in available_dates_array: return pd.Timestamp(target_date)
        future_dates = available_dates_array[available_dates_array > target_date]
        return pd.Timestamp(future_dates[0]) if len(future_dates) > 0 else pd.Timestamp(available_dates_array[-1])

    def generate_rebalancing_dates_for_period(self, period_name):
        if period_name not in self.periods: raise ValueError(f"Période '{period_name}' invalide.")
        period_info = self.periods[period_name]
        current_date = self._find_nearest_available_date(period_info['start'])
        rebalancing_dates = []
        while current_date <= period_info['end']:
            rebalancing_dates.append(current_date)
            next_target_date = current_date + pd.Timedelta(days=self.rebalancing_freq_days)
            if next_target_date > period_info['end']: break
            current_date = self._find_nearest_available_date(next_target_date)
        return sorted(list(set(rebalancing_dates)))

    def generate_topk_dates_for_period(self, period_name, K=STOCK_PICKING_PARAMS['K'], window_size=STOCK_PICKING_PARAMS['WINDOW_SIZE_WEEKS']):
        rebalancing_dates = self.generate_rebalancing_dates_for_period(period_name)
        if not rebalancing_dates:
            print(f"⚠️ Aucune date de rééquilibrage pour '{period_name}'")
            return []
        return _generate_topk_dates_list(self.all_data, self.dividendes, rebalancing_dates, K, window_size, STOCK_PICKING_PARAMS['WEIGHTS'])

    def generate_indicators_for_period(self, period_name, topk_dates, window_size=70):
        if not topk_dates:
            print(f"⚠️ Aucune donnée Top-K pour '{period_name}'")
            return pd.DataFrame()
        return _calculate_indicators_for_topk_periods(self.all_data, topk_dates, window_size)