# Fichier : src/data_manager.py
# =============================================================================
# MODULE DE GESTION DE DONNÉES (Préparateur de Scénarios)
# =============================================================================

import os
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import List, Dict, Tuple
from tqdm.auto import tqdm
import ta # Importé pour le calcul des indicateurs techniques

# === CONSTANTES DU MODULE ===
WEIGHTS = {'mu': 0.25, 'sigma': 0.25, 'L': 0.25, 'D': 0.25}
INDICATOR_COLS = [
    "MACD", "RSI", "CCI", "STOCH_K", "WILLR", "BB_Upper", "BB_Middle", "BB_Lower",
    "EMA", "ATR", "SAR", "Close_Lag1", "Close_Lag3", "Close_Lag5", "SMA_50", "SMA_20",
    "OBV", "MFI", "ADX", "Volume"
]

# === FONCTIONS DE SUPPORT (Workers) ===

def _calculate_topk_for_date(date, prices_dict, dividendes_by_year, K, window_size, weights):
    """Calcule le Top-K pour une seule date."""
    week_start = date - timedelta(weeks=window_size)
    all_tickers = list(prices_dict.keys())
    features, valid_mask = [], []
    prev_year_dividends = dividendes_by_year.get(date.year - 1, pd.Series(dtype=float))

    for ticker in all_tickers:
        df = prices_dict[ticker]
        df_window = df[(df['Date'] >= week_start) & (df['Date'] < date)]
        if len(df_window) < 2: features.append([np.nan] * 4); valid_mask.append(0); continue
        valid_mask.append(1)
        mu = (df_window['Close'].iloc[-1] / df_window['Close'].iloc[0]) - 1 if df_window['Close'].iloc[0] > 0 else 0
        sigma = df_window['Close'].pct_change(fill_method=None).std()
        L = df_window['Volume'].mean()
        last_price = df_window['Close'].iloc[-1]
        dividend_value = prev_year_dividends.get(ticker, 0) if prev_year_dividends is not None and not prev_year_dividends.empty else 0
        D = (dividend_value / last_price) if last_price > 0 and not pd.isna(dividend_value) else 0
        features.append([mu, sigma, L, D])

    features = np.nan_to_num(np.array(features))
    valid_mask = np.array(valid_mask, dtype=bool)
    ranks = np.full_like(features, 0.0)
    for j in range(features.shape[1]):
        col = features[:, j]
        valid_feature_mask = valid_mask & (col != 0)
        if valid_feature_mask.sum() > 0: ranks[valid_feature_mask, j] = pd.Series(col[valid_feature_mask]).rank(pct=True)
            
    scores = (weights['mu'] * ranks[:, 0] - weights['sigma'] * ranks[:, 1] + weights['L'] * ranks[:, 2] + weights['D'] * ranks[:, 3])
    scores[~valid_mask] = -np.inf
    top_indices = np.argsort(scores)[::-1][:K]
    return [all_tickers[i] for i in top_indices]

def generate_topk_dates_list(all_data, dividendes, rebalancing_dates, K, window_size, weights):
    """Orchestre le calcul du Top-K pour une liste de dates."""
    print("Pré-traitement des données pour le Top-K...")
    prices_dict = {}
    for ticker in tqdm(all_data.columns.get_level_values(0).unique(), desc="Préparation des prix"):
        df = all_data[ticker].copy().reset_index()
        # Gestion des noms de colonnes selon la structure MultiIndex
        if 'Close' not in df.columns and 'Price' in df.columns: 
            df = df.rename(columns={'Price': 'Close'})
        if 'Volume' not in df.columns: df['Volume'] = 0.0
        # S'assurer que toutes les colonnes nécessaires existent
        required_cols = ['Date', 'Close', 'High', 'Low', 'Volume']
        for col in required_cols:
            if col not in df.columns and col != 'Date':
                df[col] = df['Close'] if col != 'Volume' else 0.0
        prices_dict[ticker] = df
    
    # --- LOGIQUE STABILISÉE DES DIVIDENDES (Remplacement de la logique précédente) ---
    # Convertir en format long et créer un pivot table (plus robuste)
    div_long = dividendes.melt(id_vars=['Year'], var_name='Ticker', value_name='Dividende').dropna(subset=['Year'])
    div_pivot = div_long.pivot_table(index='Ticker', columns='Year', values='Dividende')
    
    # Créer le dictionnaire {Année: Série de Tickers}
    dividendes_by_year = {
        int(year): div_pivot[year] for year in div_pivot.columns
    }
    # --- FIN LOGIQUE STABILISÉE ---
    
    topk_dates = []
    for date in tqdm(rebalancing_dates, desc="Génération de la liste Top-K"):
        top_k = _calculate_topk_for_date(date, prices_dict, dividendes_by_year, K, window_size, weights)
        topk_dates.append((pd.to_datetime(date.date()), top_k))
    return topk_dates

def calculate_indicators(stock_data):
    """Calcule les indicateurs techniques en gérant les erreurs individuellement."""
    if stock_data is None or stock_data.empty or len(stock_data) < 20:
        return pd.DataFrame([{col: 0.0 for col in INDICATOR_COLS}])

    df_numeric = stock_data[['High', 'Low', 'Close', 'Volume']].apply(pd.to_numeric, errors='coerce').dropna()
    if len(df_numeric) < 20: return calculate_indicators(None)
    
    high, low, close, volume = df_numeric['High'], df_numeric['Low'], df_numeric['Close'], df_numeric['Volume']
    data = {col: 0.0 for col in INDICATOR_COLS}

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
    try: data["Close_Lag1"], data["Close_Lag3"], data["Close_Lag5"] = close.shift(1).iloc[-1], close.shift(3).iloc[-1], close.shift(5).iloc[-1]
    except Exception: pass
    try: data["SMA_50"], data["SMA_20"] = close.rolling(50).mean().iloc[-1], close.rolling(20).mean().iloc[-1]
    except Exception: pass
    try: data["OBV"] = ta.volume.OnBalanceVolumeIndicator(close, volume, fillna=True).on_balance_volume().iloc[-1]
    except Exception: pass
    try: data["MFI"] = ta.volume.MFIIndicator(high, low, close, volume, fillna=True).money_flow_index().iloc[-1]
    except Exception: pass
    try: data["ADX"] = ta.trend.ADXIndicator(high, low, close, fillna=True).adx().iloc[-1]
    except Exception: pass
    try: data["Volume"] = volume.iloc[-1]
    except Exception: pass

    for key, value in data.items():
        if not isinstance(value, (int, float)) or not np.isfinite(value): data[key] = 0.0
            
    return pd.DataFrame([data])

def calculate_indicators_for_topk_periods(all_data, topk_dates, window_size=70):
    """Orchestre le calcul des indicateurs pour une liste de dates."""
    results = []
    for date, tickers in tqdm(topk_dates, desc="Calcul des indicateurs"):
        for ticker in tickers:
            try:
                df_ticker = all_data[ticker].copy().reset_index()
                window_df = df_ticker[df_ticker['Date'] < date].tail(window_size)
                if window_df.empty: continue
                ind = calculate_indicators(window_df).assign(Date=date, Ticker=ticker)
                results.append(ind)
            except Exception as e:
                print(f"⚠️ Erreur indicateurs pour {ticker} à {date}: {e}")
    return pd.concat(results, ignore_index=True).set_index(["Date", "Ticker"]) if results else pd.DataFrame()

# === CLASSE PRINCIPALE DU MODULE ===
class BRVMTrainingManager:
    """🏛️ Gestionnaire pour la préparation des données d'expérimentation."""
    def __init__(self, all_data, dividendes, fundamentals_panel, rebalancing_freq_days=7):
        self.all_data, self.dividendes, self.fundamentals_panel = all_data, dividendes, fundamentals_panel
        self.rebalancing_freq_days = rebalancing_freq_days
        self.periods = {'train': {'start': pd.Timestamp('2018-01-01'), 'end': pd.Timestamp('2022-12-31')},
                        'validation': {'start': pd.Timestamp('2023-01-01'), 'end': pd.Timestamp('2023-12-31')},
                        'test': {'start': pd.Timestamp('2024-01-01'), 'end': pd.Timestamp('2024-12-31')}}
        self.available_dates = sorted(all_data.index.get_level_values(0).unique())
        print(f"🏛️ BRVMTrainingManager initialisé (fréquence: {self.rebalancing_freq_days} jours).")

    def _find_nearest_available_date(self, target_date):
        """Trouve la date de trading disponible future la plus proche."""
        available_dates_array = np.array(self.available_dates)
        if target_date in available_dates_array: return pd.Timestamp(target_date)
        future_dates = available_dates_array[available_dates_array > target_date]
        return pd.Timestamp(future_dates[0]) if len(future_dates) > 0 else pd.Timestamp(available_dates_array[-1])

    def generate_rebalancing_dates_for_period(self, period_name):
        """Génère la liste des dates de rééquilibrage pour une période."""
        if period_name not in self.periods: raise ValueError(f"Période '{period_name}' invalide.")
        period_info = self.periods[period_name]
        print(f"\n📅 Génération des dates de rééquilibrage pour '{period_name.upper()}'...")
        current_date = self._find_nearest_available_date(period_info['start'])
        rebalancing_dates = []
        while current_date <= period_info['end']:
            rebalancing_dates.append(current_date)
            next_target_date = current_date + pd.Timedelta(days=self.rebalancing_freq_days)
            current_date = self._find_nearest_available_date(next_target_date)
            if rebalancing_dates and current_date == rebalancing_dates[-1]:
                 current_date = self._find_nearest_available_date(current_date + pd.Timedelta(days=1))
        print(f"   -> {len(rebalancing_dates)} dates générées.")
        return sorted(list(set(rebalancing_dates)))

    def generate_topk_dates_for_period(self, period_name, K=10, window_size=12):
        """Orchestre la génération de la liste Top-K pour une période."""
        rebalancing_dates = self.generate_rebalancing_dates_for_period(period_name)
        if not rebalancing_dates: 
            print(f"⚠️ Aucune date de rééquilibrage trouvée pour la période '{period_name}'")
            return []
        
        topk_result = generate_topk_dates_list(self.all_data, self.dividendes, rebalancing_dates, K, window_size, WEIGHTS)
        print(f"✅ Top-K généré : {len(topk_result)} dates avec {K} actifs sélectionnés")
        return topk_result

    def generate_indicators_for_period(self, period_name, topk_dates, window_size=70):
        """Orchestre la génération des indicateurs techniques pour une période."""
        if not topk_dates: 
            print(f"⚠️ Aucune donnée Top-K disponible pour la période '{period_name}'")
            return pd.DataFrame()
        
        indicators_result = calculate_indicators_for_topk_periods(self.all_data, topk_dates, window_size)
        print(f"✅ Indicateurs générés : {len(indicators_result)} observations pour la période '{period_name}'")
        return indicators_result
    
    def validate_data_integrity(self):
        """Valide l'intégrité des données chargées."""
        print("\n🔍 Validation de l'intégrité des données...")
        
        # Validation all_data
        n_dates = len(self.all_data.index.get_level_values(0).unique())
        n_assets = len(self.all_data.columns.get_level_values(0).unique())
        print(f"  -> all_data: {n_dates} dates × {n_assets} actifs")
        
        # Validation fundamentals_panel
        n_fundamental_dates = len(self.fundamentals_panel.index.get_level_values(0).unique())
        n_fundamental_assets = len(self.fundamentals_panel.index.get_level_values(1).unique())
        print(f"  -> fundamentals_panel: {n_fundamental_dates} dates × {n_fundamental_assets} actifs")
        
        # Validation dividendes
        print(f"  -> dividendes: {self.dividendes.shape}")
        
        # Vérification de cohérence
        if n_assets != n_fundamental_assets:
            print(f"⚠️ Incohérence: {n_assets} actifs dans all_data vs {n_fundamental_assets} dans fundamentals_panel")
        
        print("✅ Validation terminée")
        return True