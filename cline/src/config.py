import pandas as pd

# Paramètres de l'environnement de Reinforcement Learning
RL_ENV_PARAMS = {
    'INITIAL_CAPITAL': 100000.0,
    'TRANSACTION_COST_PCT': 0.001,  # 0.1% de frais de transaction
    'RISK_AVERSION_LAMBDA': 0.5,    # Coefficient d'aversion au risque pour le CVaR
    'CVAR_ALPHA': 0.05,             # Alpha pour le calcul du CVaR (5%)
    'WINDOW_SIZE_CVAR_DAYS': 60,    # Fenêtre glissante pour le calcul du CVaR (en jours)
}

# Paramètres du Stock Picking
STOCK_PICKING_PARAMS = {
    'K': 10,                        # Nombre d'actifs à sélectionner pour le portefeuille
    'WINDOW_SIZE_WEEKS': 12,        # Fenêtre glissante pour le calcul des indicateurs de stock picking (en semaines)
    'WEIGHTS': {'mu': 0.25, 'sigma': 0.25, 'L': 0.25, 'D': 0.25} # Poids des critères de sélection
}

# Fréquences de rééquilibrage en jours
REBALANCING_FREQUENCIES = {
    'daily': 1,
    'weekly': 7,
    'monthly': 30 # Approximation pour le mois
}

# Périodes de données
DATA_PERIODS = {
    'train': {'start': pd.Timestamp('2018-01-01'), 'end': pd.Timestamp('2022-12-31')},
    'validation': {'start': pd.Timestamp('2023-01-01'), 'end': pd.Timestamp('2023-12-31')},
    'test': {'start': pd.Timestamp('2024-01-01'), 'end': pd.Timestamp('2024-12-31')}
}

# Colonnes des indicateurs techniques (doit correspondre à celles utilisées dans data_manager)
INDICATOR_COLS = [
    "MACD", "RSI", "CCI", "STOCH_K", "WILLR", "BB_Upper", "BB_Middle", "BB_Lower",
    "EMA", "ATR", "SAR", "Close_Lag1", "Close_Lag3", "Close_Lag5", "SMA_50", "SMA_20",
    "OBV", "MFI", "ADX", "Volume"
]

# Paramètres de l'agent SAC
SAC_AGENT_PARAMS = {
    'BUFFER_SIZE': 1_000_000,
    'BATCH_SIZE': 256,
    'GAMMA': 0.99,
    'TAU': 0.005,
    'LR_ACTOR': 3e-4,
    'LR_CRITIC': 3e-4,
    'LR_ALPHA': 3e-4,
    'HIDDEN_SIZE': (256, 256),
    'AUTO_ENTROPY_TUNING': True,
    'TARGET_ENTROPY_FACTOR': -1.0, # -dim(action_space)
    'N_EPISODES': 100,
    'UPDATE_EVERY': 1, # Update networks every N steps
}
