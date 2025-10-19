# Fichier: src/copula.py (VERSION OPTIMISÉE - Parallélisation & Cache Hebdomadaire)
import pandas as pd
import numpy as np
import pyvinecopulib as pv
from arch import arch_model
from scipy import stats
import warnings
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

MIN_OBSERVATIONS = 60
MIN_UNIQUE_RETURNS = 25

def check_data_quality(returns_series):
    asset_name = returns_series.name if hasattr(returns_series, 'name') else "Actif"
    if len(returns_series) < MIN_OBSERVATIONS: return False, f"Insuffisant ({len(returns_series)} < {MIN_OBSERVATIONS})"
    n_unique = len(returns_series.unique())
    if n_unique < MIN_UNIQUE_RETURNS: return False, f"Série trop constante ({n_unique} < {MIN_UNIQUE_RETURNS})"
    vol_annualisee = returns_series.std() * np.sqrt(252)
    if vol_annualisee < 0.05: return False, f"Volatilité trop faible (σ={vol_annualisee:.2%})"
    pct_zeros = (returns_series == 0).sum() / len(returns_series)
    if pct_zeros > 0.60: return False, f"Illiquidité excessive ({pct_zeros*100:.1f}% de zéros)"
    return True, "OK"

def fit_garch_model(returns_series):
    asset_name = returns_series.name if hasattr(returns_series, 'name') else "Actif"
    is_valid, reason = check_data_quality(returns_series)
    if not is_valid:
        logging.warning(f"REJETÉ [{asset_name}]: {reason}")
        return None
    
    configs = [
        {'vol': 'GARCH', 'p': 1, 'q': 1, 'o': 1, 'dist': 't', 'name': 'GJR-GARCH(1,1)-t'},
        {'vol': 'GARCH', 'p': 1, 'q': 1, 'o': 0, 'dist': 't', 'name': 'GARCH(1,1)-t'},
        {'vol': 'GARCH', 'p': 1, 'q': 1, 'o': 0, 'dist': 'normal', 'name': 'GARCH(1,1)-N'},
    ]
    for config in configs:
        try:
            model = arch_model(returns_series * 100, **{k: v for k, v in config.items() if k != 'name'})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = model.fit(disp='off', show_warning=False)
            if res.convergence_flag:
                std_resid = (res.resid / res.conditional_volatility).dropna()
                if len(std_resid) < 20: continue
                dist_params = {'nu': res.params.get('nu', 8.0)} if config['dist'] == 't' else {}
                marginal_dist = {'cdf': getattr(stats, config['dist']).cdf, 'ppf': getattr(stats, config['dist']).ppf, 'params': dist_params}
                logging.info(f"✅ [{asset_name}]: {config['name']} ajusté.")
                return {'model': res, 'std_resid': std_resid, 'marginal_dist': marginal_dist}
        except Exception:
            continue
    logging.warning(f"REJETÉ [{asset_name}]: Toutes configs GARCH ont échoué.")
    return None

class DynamicRVineCopula:
    def __init__(self, all_data, n_simulations=1000):
        self.all_data = all_data
        self.n_simulations = n_simulations
        self.model_cache = {}

    def _fit_asset_model_task(self, asset, start_date, end_date):
        try:
            prices = self.all_data.loc(axis=1)[asset, 'Close'].loc[start_date:end_date].dropna()
            if len(prices) < MIN_OBSERVATIONS: return None
            returns = np.log(prices / prices.shift(1)).dropna()
            returns.name = asset
            garch_fit = fit_garch_model(returns)
            if garch_fit:
                uniform_values = garch_fit['marginal_dist']['cdf'](garch_fit['std_resid'].values, **garch_fit['marginal_dist']['params'])
                uniform_series = pd.Series(np.clip(uniform_values, 1e-6, 1 - 1e-6), index=garch_fit['std_resid'].index, name=asset)
                return (asset, garch_fit, uniform_series)
        except Exception as e:
            logging.error(f"Erreur GARCH pour {asset}: {e}")
        return None

    def _get_rolling_models(self, selected_assets, end_date, window_weeks):
        # --- OPTIMISATION : CACHE HEBDOMADAIRE ---
        # La clé de cache est basée sur le début de la semaine, pas sur la date exacte.
        cache_date = end_date - pd.to_timedelta(end_date.weekday(), unit='D')
        cache_key = (tuple(sorted(selected_assets)), cache_date, window_weeks)
        
        if cache_key in self.model_cache:
            logging.info(f"⚡️ Cache HIT pour la semaine du {cache_date.date()}")
            return self.model_cache[cache_key]

        start_date = end_date - pd.Timedelta(weeks=window_weeks)
        logging.info(f"🌀 AJUSTEMENT COPULE : {start_date.date()} -> {end_date.date()}")

        window_models, pit_panel = {}, pd.DataFrame()

        # --- OPTIMISATION : PARALLÉLISATION ---
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._fit_asset_model_task, asset, start_date, end_date): asset for asset in selected_assets}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    asset, garch_fit, uniform_series = result
                    pit_panel = pd.concat([pit_panel, uniform_series], axis=1)
                    window_models[asset] = {'garch_fit': garch_fit}

        if len(window_models) < 2:
            logging.error(f"❌ Moins de 2 actifs valides pour la copule.")
            self.model_cache[cache_key] = None # Cache l'échec pour la semaine
            return None

        pit_panel = pit_panel.dropna()
        if len(pit_panel) < 30:
            logging.error(f"❌ Observations communes insuffisantes ({len(pit_panel)}).")
            self.model_cache[cache_key] = None
            return None

        controls = pv.FitControlsVinecop(family_set=[pv.BicopFamily.gaussian, pv.BicopFamily.student, pv.BicopFamily.clayton, pv.BicopFamily.gumbel, pv.BicopFamily.frank], num_threads=4)
        try:
            copula = pv.Vinecop(data=pit_panel.values, controls=controls)
            result = {'copula': copula, 'models': window_models, 'valid_assets': list(pit_panel.columns)}
            self.model_cache[cache_key] = result
            return result
        except Exception as e:
            logging.error(f"❌ Échec de l'ajustement de la copule: {e}")
            self.model_cache[cache_key] = None
            return None

    def simulate_period_returns(self, selected_assets, current_date, period_days, window_weeks=52):
        models_pack = self._get_rolling_models(selected_assets, current_date, window_weeks)

        if models_pack is None:
            logging.warning(f"⚠️ Fallback historique pour {len(selected_assets)} actifs.")
            start_date = current_date - pd.Timedelta(weeks=window_weeks)
            fallback_returns = []
            for asset in selected_assets:
                try:
                    prices = self.all_data.loc(axis=1)[asset, 'Close'].loc[start_date:current_date].dropna()
                    returns = np.log(prices / prices.shift(1)).dropna()
                    mean_ret, std_ret = (returns.mean(), returns.std()) if len(returns) > 10 else (0.0, 0.01)
                    sim_ret = np.random.normal(mean_ret, std_ret, self.n_simulations)
                    fallback_returns.append(sim_ret)
                except:
                    fallback_returns.append(np.random.normal(0.0, 0.01, self.n_simulations))
            return np.column_stack(fallback_returns), None, selected_assets

        copula, asset_models, valid_assets = models_pack['copula'], models_pack['models'], models_pack['valid_assets']
        
        try:
            simulated_uniform = copula.simulate(n=self.n_simulations * period_days)
            daily_returns = np.zeros((self.n_simulations * period_days, len(valid_assets)))

            for i, asset in enumerate(valid_assets):
                garch_fit = asset_models[asset]['garch_fit']
                sim_std_resid = garch_fit['marginal_dist']['ppf'](simulated_uniform[:, i], **garch_fit['marginal_dist']['params'])
                forecasts = garch_fit['model'].forecast(horizon=period_days, reindex=False)
                sim_vol = np.sqrt(forecasts.variance.values.flatten()) / 100.0
                sim_mean = forecasts.mean.values.flatten() / 100.0
                sim_vol_tiled = np.tile(sim_vol, self.n_simulations)
                sim_mean_tiled = np.tile(sim_mean, self.n_simulations)
                daily_returns[:, i] = sim_mean_tiled + sim_std_resid[:len(sim_mean_tiled)] * sim_vol_tiled

            daily_reshaped = daily_returns.reshape(self.n_simulations, period_days, -1)
            period_returns = np.expm1(np.sum(np.log1p(daily_reshaped), axis=1))

            final_returns = np.zeros((self.n_simulations, len(selected_assets)))
            for i, asset in enumerate(selected_assets):
                if asset in valid_assets:
                    final_returns[:, i] = period_returns[:, valid_assets.index(asset)]
                else:
                    prices = self.all_data.loc(axis=1)[asset, 'Close'].loc[current_date - pd.Timedelta(weeks=window_weeks):current_date].dropna()
                    returns = np.log(prices / prices.shift(1)).dropna()
                    final_returns[:, i] = np.random.normal(returns.mean(), returns.std(), self.n_simulations)
            
            return np.nan_to_num(final_returns), daily_returns, valid_assets
        except Exception as e:
            logging.error(f"❌ Erreur de simulation, fallback total: {e}")
            return self.simulate_period_returns(selected_assets, current_date, period_days, window_weeks)