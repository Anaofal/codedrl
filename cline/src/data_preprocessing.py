import pandas as pd
import numpy as np
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm, t
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, List, Tuple
from tqdm.auto import tqdm

class DataPreprocessor:
    """
    Gère le prétraitement complet des données financières pour la modélisation
    des copules et l'apprentissage par renforcement.
    """
    def __init__(self, all_data: pd.DataFrame, fundamentals_panel: pd.DataFrame, dividendes: pd.DataFrame):
        self.all_data = all_data
        self.fundamentals_panel = fundamentals_panel
        self.dividendes = dividendes
        self.tickers = self.all_data.columns.get_level_values(0).unique().tolist()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        print("📊 DataPreprocessor initialisé.")

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoie les données (valeurs manquantes, infinis, etc.)."""
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill() # Remplir les NaN avec la dernière/première valeur valide
        return df

    def _calculate_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule les rendements logarithmiques."""
        # Assurez-vous que 'Close' est une colonne numérique
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        return df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Log_Return'])

    def _encode_categorical_features(self):
        """Encode les caractéristiques catégorielles (Secteur, Pays)."""
        for col in ['Secteur', 'Pays']:
            if col in self.fundamentals_panel.columns:
                le = LabelEncoder()
                self.fundamentals_panel[col] = le.fit_transform(self.fundamentals_panel[col])
                self.label_encoders[col] = le
        print("   -> Caractéristiques catégorielles encodées.")

    def _fit_arma_garch(self, series: pd.Series) -> Tuple[ARIMA, arch_model]:
        """
        Ajuste un modèle ARMA(1,1)-GARCH(1,1) à une série de rendements.
        Retourne les modèles ajustés.
        """
        # Ajustement ARMA(1,1)
        arma_model = ARIMA(series, order=(1, 0, 1))
        arma_results = arma_model.fit()
        
        # Résidus de l'ARMA
        residuals = arma_results.resid
        
        # Ajustement GARCH(1,1) sur les résidus
        garch_model = arch_model(residuals, vol='Garch', p=1, q=1)
        garch_results = garch_model.fit(disp='off') # disp='off' pour supprimer l'affichage
        
        return arma_results, garch_results

    def _transform_to_uniform_pseudo_observations(self, series: pd.Series, arma_results, garch_results) -> pd.Series:
        """
        Transforme les rendements en pseudo-observations uniformes via ARMA-GARCH
        et la fonction de répartition des résidus standardisés.
        """
        # Obtenir les résidus standardisés du modèle GARCH
        std_residuals = garch_results.resid / garch_results.conditional_volatility
        
        # Utiliser la fonction de répartition empirique (ECDF) pour obtenir des uniformes
        # Ou une distribution théorique si les résidus standardisés suivent une loi connue (ex: Student-t)
        # Pour l'instant, utilisons une approche non-paramétrique simple (rangs)
        uniform_pseudo_observations = pd.Series(std_residuals).rank(method='average').values / (len(std_residuals) + 1)
        
        return pd.Series(uniform_pseudo_observations, index=series.index)

    def preprocess(self) -> Dict[str, pd.DataFrame]:
        """
        Exécute toutes les étapes de prétraitement et retourne les données préparées.
        """
        print("🚀 Démarrage du prétraitement des données...")

        processed_data = {}
        log_returns_df = pd.DataFrame(index=self.all_data.index.get_level_values('Date').unique())

        # 1. Nettoyage et calcul des rendements logarithmiques pour chaque actif
        print("   -> Nettoyage et calcul des rendements logarithmiques...")
        for ticker in tqdm(self.tickers, desc="Actifs"):
            df_ticker = self.all_data[ticker].copy().reset_index().set_index('Date')
            df_ticker = self._clean_data(df_ticker)
            df_ticker = self._calculate_log_returns(df_ticker)
            log_returns_df[ticker] = df_ticker['Log_Return']
        
        log_returns_df = self._clean_data(log_returns_df) # Nettoyage final après fusion

        # 2. Encodage des caractéristiques catégorielles
        self._encode_categorical_features()
        processed_data['fundamentals'] = self.fundamentals_panel

        # 3. Modélisation univariée (ARMA-GARCH) et transformation en pseudo-observations uniformes
        print("   -> Modélisation ARMA-GARCH et transformation en pseudo-observations uniformes...")
        uniform_pseudo_observations_df = pd.DataFrame(index=log_returns_df.index)
        for ticker in tqdm(self.tickers, desc="Modélisation ARMA-GARCH"):
            if ticker in log_returns_df.columns and not log_returns_df[ticker].dropna().empty:
                try:
                    arma_results, garch_results = self._fit_arma_garch(log_returns_df[ticker].dropna())
                    uniform_pseudo_observations_df[ticker] = self._transform_to_uniform_pseudo_observations(
                        log_returns_df[ticker].dropna(), arma_results, garch_results
                    )
                except Exception as e:
                    print(f"⚠️ Erreur ARMA-GARCH pour {ticker}: {e}")
                    uniform_pseudo_observations_df[ticker] = np.nan
            else:
                uniform_pseudo_observations_df[ticker] = np.nan
        
        processed_data['log_returns'] = log_returns_df
        processed_data['uniform_pseudo_observations'] = self._clean_data(uniform_pseudo_observations_df)
        
        print("✅ Prétraitement des données terminé.")
        return processed_data

# Exemple d'utilisation (à supprimer ou commenter en production)
if __name__ == "__main__":
    # Créer des données fictives pour tester
    dates = pd.to_datetime(pd.date_range(start='2018-01-01', periods=100, freq='D'))
    tickers = ['ACTIF_A', 'ACTIF_B', 'ACTIF_C']
    
    # all_data
    data_dict = {}
    for ticker in tickers:
        data_dict[ticker] = pd.DataFrame({
            'Close': np.random.rand(100) * 100 + 50,
            'High': np.random.rand(100) * 10 + 150,
            'Low': np.random.rand(100) * 10 + 40,
            'Volume': np.random.rand(100) * 100000
        }, index=dates)
    all_data_multiindex = pd.concat(data_dict, axis=1, names=['Ticker', 'Feature'])

    # fundamentals_panel
    fundamentals_data = {
        'Secteur': ['Finance', 'Technologie', 'Industrie'],
        'Pays': ['BENIN', 'COTE D\'IVOIRE', 'SENEGAL']
    }
    fundamentals_panel_df = pd.DataFrame(fundamentals_data, index=tickers)

    # dividendes
    dividendes_data = {
        'Year': [2018, 2019, 2020],
        'ACTIF_A': [1.0, 1.2, 1.5],
        'ACTIF_B': [0.5, 0.6, 0.7],
        'ACTIF_C': [2.0, 2.1, 2.3]
    }
    dividendes_df = pd.DataFrame(dividendes_data)

    preprocessor = DataPreprocessor(all_data_multiindex, fundamentals_panel_df, dividendes_df)
    processed_data = preprocessor.preprocess()

    print("\nRendements logarithmiques:")
    print(processed_data['log_returns'].head())
    print("\nPseudo-observations uniformes:")
    print(processed_data['uniform_pseudo_observations'].head())
    print("\nDonnées fondamentales encodées:")
    print(processed_data['fundamentals'].head())
