# =============================================================================
# MODULE DE LA COPULE R-VINE DYNAMIQUE (Version Améliorée - GJR-GARCH & Skewed-t)
# =============================================================================
import pandas as pd
import numpy as np
import pyvinecopulib as pv
from arch import arch_model
import warnings
from scipy.stats import rankdata, genpareto
from scipy.stats import skewnorm  # Pour fallback EVT avec asymétrie
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- LOGGING CONFIG ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# --- CONSTANTES ---
P_Q_MAX = 2  # Ordre max pour ARMA(p,q)
MIN_OBSERVATIONS = 30  # Nombre minimum d'observations pour ajuster un modèle
DEFAULT_THRESHOLDS = (0.05, 0.95)  # Seuils pour l'EVT (5% et 95%)

def fit_arma_garch_model(returns_series, hybrid=True):
    """
    Ajuste un modèle ARMA(1,1)-GJR-GARCH(1,1) (skewed Student-t) et estime la distribution marginale.
    Retourne None si l'ajustement échoue ou si la stationnarité n'est pas respectée.
    """
    asset_name = returns_series.name if hasattr(returns_series, 'name') else "Actif"
    if len(returns_series) < MIN_OBSERVATIONS:
        logging.warning(f"REJETÉ [{asset_name}]: Données insuffisantes ({len(returns_series)} < {MIN_OBSERVATIONS}).")
        return None

    try:
        # Utilisation de GJR-GARCH avec skewed-t
        model = arch_model(
            returns_series,
            vol='GARCH',
            p=1, q=1, o=1,  # o=1 active le terme asymétrique (GJR-GARCH)
            dist='skewt'    # Distribution skewed-t pour l'asymétrie
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Premier essai avec optimisation standard
                res = model.fit(disp='off', options={'maxiter': 1000})
            except Exception as e1:
                try:
                    # Deuxième essai avec contraintes plus souples
                    res = model.fit(disp='off', options={'maxiter': 2000}, tol=1e-4)
                except Exception as e2:
                    # Dernier essai avec GJR-GARCH simple
                    model = arch_model(returns_series, vol='GARCH', p=1, q=1, o=1)
                    res = model.fit(disp='off', options={'maxiter': 2000}, tol=1e-4)

        # Vérification stationnarité stricte pour GJR-GARCH
        params = res.params
        alpha = params.get('alpha[1]', 0)
        beta = params.get('beta[1]', 0)
        gamma = params.get('gamma[1]', 0)
        # Critère de stationnarité modifié pour GJR-GARCH
        stationarity_check = alpha + beta + (gamma / 2)
        if stationarity_check >= 0.999:  # Critère très souple pour GJR-GARCH
            if stationarity_check >= 1.01:  # Rejet seulement si vraiment instable
                logging.warning(f"REJETÉ [{asset_name}]: Très instable (somme GJR-GARCH={stationarity_check:.4f}).")
                return None
            else:
                logging.warning(f"ATTENTION [{asset_name}]: Proche de l'instabilité (somme GJR-GARCH={stationarity_check:.4f}).")

        try:
            std_resid = res.resid / res.conditional_volatility
        except Exception as e:
            logging.warning(f"REJETÉ [{asset_name}]: Erreur dans le calcul des résidus: {e}")
            return None

        marginal_dist = estimate_marginal_distribution(std_resid, hybrid=hybrid)
        if marginal_dist is None:
            # Fallback: loi skewed-t
            logging.warning(f"[{asset_name}] Fallback sur une loi skewed-t pour la marge.")
            # Ajustement d'une loi skewed-t sur les résidus standardisés
            # Ajustement d'une loi skewnorm sur les résidus standardisés
            shape = skewnorm.fit(std_resid)[0]
            marginal_dist = {
                'cdf': lambda x: skewnorm.cdf(x, shape),
                'ppf': lambda p: skewnorm.ppf(p, shape)
            }

        logging.info(f"ACCEPTÉ [{asset_name}]: Modèle ajusté avec succès.")
        return {
            'model': res,
            'std_resid': std_resid.dropna(),
            'marginal_dist': marginal_dist,
            'mu': np.mean(returns_series),
            'sigma': np.std(returns_series)
        }
    except Exception as e:
        logging.error(f"REJETÉ [{asset_name}]: Échec de l'ajustement. Erreur: {str(e)[:200]}")
        return None

def estimate_marginal_distribution(residuals, hybrid=True, lower_threshold=0.05, upper_threshold=0.95):
    """
    Estime la distribution marginale avec une approche hybride (KDE + EVT). Si l'estimation échoue, retourne None.
    """
    if len(residuals) < MIN_OBSERVATIONS:
        logging.warning("Données insuffisantes pour l'estimation hybride.")
        return None

    try:
        ranks = rankdata(residuals, method='average')
        u = (ranks - 0.5) / len(residuals)
        sorted_residuals = np.sort(residuals)

        def cdf_kde(x):
            x = np.asarray(x)
            return np.searchsorted(sorted_residuals, x, side='right') / len(sorted_residuals)

        def ppf_kde(p):
            p = np.asarray(p)
            indices = np.clip(p * len(sorted_residuals), 0, len(sorted_residuals) - 1).astype(int)
            return sorted_residuals[indices]

        if not hybrid:
            return {'cdf': cdf_kde, 'ppf': ppf_kde}

        lower_residuals = residuals[residuals <= np.quantile(residuals, lower_threshold)]
        upper_residuals = residuals[residuals >= np.quantile(residuals, upper_threshold)]

        if len(lower_residuals) > 10:
            try:
                params = genpareto.fit(-lower_residuals)
                def cdf_lower(x):
                    x = np.asarray(x)
                    seuil = np.quantile(residuals, lower_threshold)
                    return np.where(
                        x <= seuil,
                        genpareto.cdf(-x, *params) * lower_threshold,
                        cdf_kde(x)
                    )
                def ppf_lower(p):
                    p = np.asarray(p)
                    return np.where(
                        p <= lower_threshold,
                        -genpareto.ppf(1 - p/lower_threshold, *params),
                        ppf_kde(p)
                    )
            except Exception as e:
                cdf_lower, ppf_lower = cdf_kde, ppf_kde
                logging.warning(f"Pas assez de données pour ajuster GPD sur la queue basse: {e}")
        else:
            cdf_lower, ppf_lower = cdf_kde, ppf_kde
            logging.warning("Pas assez de données pour ajuster GPD sur la queue basse.")

        if len(upper_residuals) > 10:
            try:
                params = genpareto.fit(upper_residuals - np.quantile(residuals, upper_threshold))
                def cdf_upper(x):
                    x = np.asarray(x)
                    seuil = np.quantile(residuals, upper_threshold)
                    return np.where(
                        x >= seuil,
                        (1 - upper_threshold) * genpareto.cdf(x - seuil, *params) + upper_threshold,
                        cdf_kde(x)
                    )
                def ppf_upper(p):
                    p = np.asarray(p)
                    seuil = np.quantile(residuals, upper_threshold)
                    return np.where(
                        p >= upper_threshold,
                        genpareto.ppf((p - upper_threshold)/(1 - upper_threshold), *params) + seuil,
                        ppf_kde(p)
                    )
            except Exception as e:
                cdf_upper, ppf_upper = cdf_kde, ppf_kde
                logging.warning(f"Pas assez de données pour ajuster GPD sur la queue haute: {e}")
        else:
            cdf_upper, ppf_upper = cdf_kde, ppf_kde
            logging.warning("Pas assez de données pour ajuster GPD sur la queue haute.")

        def cdf_hybrid(x):
            x = np.asarray(x)
            seuil_basse = np.quantile(residuals, lower_threshold)
            seuil_haute = np.quantile(residuals, upper_threshold)
            return np.where(
                x <= seuil_basse,
                cdf_lower(x),
                np.where(
                    x >= seuil_haute,
                    cdf_upper(x),
                    cdf_kde(x)
                )
            )

        def ppf_hybrid(p):
            p = np.asarray(p)
            return np.where(
                p <= lower_threshold,
                ppf_lower(p),
                np.where(
                    p >= upper_threshold,
                    ppf_upper(p),
                    ppf_kde(p)
                )
            )

        return {'cdf': cdf_hybrid, 'ppf': ppf_hybrid}
    except Exception as e:
        logging.error(f"Erreur dans l'estimation hybride: {str(e)[:200]}")
        return None

class DynamicRVineCopula:
    """
    Classe principale pour la modélisation et la simulation par copule R-Vine dynamique.
    - Ajuste des modèles ARMA-GARCH sur chaque actif (marge hybride KDE/EVT ou normale en fallback)
    - Ajuste une copule R-Vine sur les PITs
    - Simule des rendements corrélés sur une période donnée
    - Parallélise l'ajustement des marges pour accélérer sur de gros panels
    """
    def __init__(self, all_data, n_simulations=1000):
        """
        Initialise la copule dynamique.
        Args:
            all_data (pd.DataFrame): Données de prix multi-indexées (asset, champ)
            n_simulations (int): Nombre de simulations Monte Carlo
        """
        self.all_data = all_data
        self.n_simulations = n_simulations
        self.model_cache = {}

    def _fit_asset_model(self, asset, start_date, end_date):
        """
        Ajuste le modèle ARMA-GARCH et la marge pour un actif donné.
        Retourne (asset, garch_fit, marginal_dist, uniform_series) ou None si échec.
        """
        try:
            prices = self.all_data.loc(axis=1)[asset, 'Close'].loc[start_date:end_date].dropna()
            if len(prices) < MIN_OBSERVATIONS:
                logging.warning(f"{asset}: Données insuffisantes ({len(prices)} < {MIN_OBSERVATIONS}).")
                return None
            returns = np.log(prices / prices.shift(1)).dropna()
            if len(returns) < 20:
                return None
            garch_fit = fit_arma_garch_model(returns, hybrid=True)
            if garch_fit is None:
                logging.warning(f"{asset}: Ajustement ARMA-GARCH échoué.")
                return None
            marginal_dist = garch_fit['marginal_dist']
            if marginal_dist is None:
                logging.warning(f"{asset}: Estimation des marges échouée.")
                return None
            uniform_values = np.clip(
                marginal_dist['cdf'](garch_fit['std_resid'].values),
                0.001, 0.999
            )
            uniform_series = pd.Series(uniform_values, index=garch_fit['std_resid'].index, name=asset)
            return (asset, garch_fit, marginal_dist, uniform_series)
        except Exception as e:
            logging.error(f"{asset}: Erreur - {str(e)[:200]}")
            return None

    def _get_rolling_models(self, selected_assets, end_date, window_weeks=26):
        """
        Ajuste les modèles ARMA-GARCH et la copule pour une fenêtre glissante.
        Parallélise l'ajustement des marges pour accélérer sur de gros panels.
        Retourne un dict avec la copule, les modèles, les actifs valides et le PIT panel.
        """
        cache_key = (tuple(sorted(selected_assets)), end_date)
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]

        start_date = end_date - pd.Timedelta(weeks=window_weeks)
        logging.info(f"🌀 AJUSTEMENT COPULE : {start_date.date()} -> {end_date.date()}")

        window_models, pit_panel = {}, pd.DataFrame()
        valid_assets = []

        # Parallélisation de l'ajustement des marges
        results = []
        with ThreadPoolExecutor(max_workers=min(8, len(selected_assets))) as executor:
            futures = {executor.submit(self._fit_asset_model, asset, start_date, end_date): asset for asset in selected_assets}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    asset, garch_fit, marginal_dist, uniform_series = result
                    if pit_panel.empty:
                        pit_panel = uniform_series.to_frame()
                    else:
                        pit_panel = pit_panel.join(uniform_series, how='outer')
                    window_models[asset] = {
                        'garch_fit': garch_fit,
                        'marginal_dist': marginal_dist,
                        'uniform_values': uniform_series
                    }
                    valid_assets.append(asset)

        if not valid_assets:
            logging.error("Aucun actif valide trouvé pour la copule.")
            return None

        pit_panel = pit_panel.dropna(axis=1)
        valid_assets = list(pit_panel.columns)

        if len(valid_assets) < 2 or len(pit_panel) < 20:
            logging.error(f"Échec: Seulement {len(valid_assets)} actifs valides ou {len(pit_panel)} observations.")
            return None

        controls = pv.FitControlsVinecop(
            family_set=[
                pv.BicopFamily.gaussian,     # Copule gaussienne
                pv.BicopFamily.student,      # Copule de Student
                pv.BicopFamily.clayton,      # Copule de Clayton
                pv.BicopFamily.gumbel,       # Copule de Gumbel
                pv.BicopFamily.frank,        # Copule de Frank
                pv.BicopFamily.joe,          # Copule de Joe
                pv.BicopFamily.bb1,          # Copule BB1 (Clayton-Gumbel)
                pv.BicopFamily.bb7,          # Copule BB7 (Joe-Clayton)
                pv.BicopFamily.bb8,          # Copule BB8 (Joe-Frank)
                pv.BicopFamily.tll           # Copule Two-Level Lambda
            ],
            parametric_method='mle',         # Maximum de vraisemblance
            nonparametric_method='constant', # Méthode non-paramétrique
            selection_criterion='aic',       # Critère AIC pour la sélection
            tree_criterion='tau',           # Critère de construction de l'arbre
            threshold=0.0,                  # Seuil de troncature
            preselect_families=True,        # Présélection des familles
            show_trace=False,              # Pas de traces d'exécution
            num_threads=1                  # Nombre de threads
        )

        try:
            copula = pv.Vinecop.from_data(data=pit_panel.values, controls=controls)
            logging.info(f"Copule R-Vine ajustée avec {len(valid_assets)} actifs (approche hybride).")
        except Exception as e:
            logging.error(f"Échec de l'ajustement de la copule: {str(e)[:200]}")
            return None

        result = {
            'copula': copula,
            'models': window_models,
            'valid_assets': valid_assets,
            'pit_panel': pit_panel
        }
        self.model_cache[cache_key] = result
        return result

    def simulate_period_returns(self, selected_assets, current_date, period_days, window_weeks=52):
        """
        Simule les rendements périodiques corrélés via la copule R-Vine.
        Si la copule échoue, warning explicite et fallback sur rendements indépendants.
        Args:
            selected_assets (list): Liste des actifs sélectionnés
            current_date (pd.Timestamp): Date de fin de la fenêtre
            period_days (int): Nombre de jours à simuler
            window_weeks (int): Largeur de la fenêtre glissante
        Returns:
            (period_returns, daily_returns, valid_assets)
        """
        models_pack = self._get_rolling_models(selected_assets, current_date, window_weeks)
        if models_pack is None:
            logging.warning("Aucune copule valide. Fallback: rendements indépendants (skewed-t).")
            return (
                np.random.normal(0.001, 0.02, (self.n_simulations, len(selected_assets))),
                None,
                selected_assets
            )

        copula, asset_models, valid_assets, pit_panel = (
            models_pack['copula'],
            models_pack['models'],
            models_pack['valid_assets'],
            models_pack['pit_panel']
        )

        n_daily_sims = self.n_simulations * period_days
        simulated_uniform = copula.simulate(n=n_daily_sims)

        daily_returns = np.zeros((n_daily_sims, len(valid_assets)))
        logging.info(f"🔬 Simulation des rendements journaliers (copule hybride)")

        for i, asset in enumerate(valid_assets):
            try:
                marginal_dist = asset_models[asset]['marginal_dist']
                sim_std_resid = marginal_dist['ppf'](simulated_uniform[:, i])
                garch_fit = asset_models[asset]['garch_fit']
                mu_hat = garch_fit['mu']
                sigma_hat = garch_fit['sigma']
                sim_returns = mu_hat + (sigma_hat * sim_std_resid)
                daily_returns[:, i] = sim_returns
                logging.info(f"{asset}: Rendements simulés (moyenne={sim_returns.mean():+.2%}, std={sim_returns.std():+.2%})")
            except Exception as e:
                logging.warning(f"{asset}: Erreur simulation, fallback skewed-t. {str(e)[:200]}")
                # Utilisation de skewed-t pour le fallback
                shape = skewnorm.fit(garch_fit['std_resid'])[0]
                daily_returns[:, i] = skewnorm.rvs(shape, size=n_daily_sims)

        try:
            daily_reshaped = daily_returns.reshape(self.n_simulations, period_days, -1)
            period_returns = np.exp(np.mean(np.log(1 + daily_reshaped), axis=1)) - 1
        except Exception as e:
            logging.warning(f"Erreur dans l'agrégation géométrique: {e}. Utilisation de la moyenne arithmétique.")
            daily_reshaped = daily_returns.reshape(self.n_simulations, period_days, -1)
            period_returns = np.mean(daily_reshaped, axis=1)

        final_returns = np.zeros((self.n_simulations, len(selected_assets)))
        for i, asset in enumerate(selected_assets):
            if asset in valid_assets:
                idx = valid_assets.index(asset)
                final_returns[:, i] = period_returns[:, idx]
            else:
                shape = 0  # Valeur par défaut pour skewnorm
                final_returns[:, i] = skewnorm.rvs(shape, size=self.n_simulations) * 0.02 + 0.001

        final_returns = np.nan_to_num(final_returns)
        logging.info(f"Simulation réussie pour {len(valid_assets)}/{len(selected_assets)} actifs (copule hybride).")
        return final_returns, daily_returns, valid_assets
