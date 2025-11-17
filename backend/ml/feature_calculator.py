"""
Feature Calculator - Cálculo de features técnicas para ML

Usa pandas_ta para calcular todos os indicadores necessários
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    logger.warning("pandas_ta não disponível, usando cálculos manuais")


def calculate_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula 65 features técnicas para ML

    Args:
        df: DataFrame com candles (open, high, low, close, timestamp)

    Returns:
        DataFrame com apenas a última linha e 65 features
    """
    if len(df) < 200:
        logger.warning(f"Dados insuficientes: {len(df)} candles (mínimo: 200)")
        return None

    # Copiar para não modificar original
    df = df.copy().reset_index(drop=True)

    # Garantir colunas numéricas
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if HAS_PANDAS_TA:
        features_df = _calculate_with_pandas_ta(df)
    else:
        features_df = _calculate_manual(df)

    return features_df


def _calculate_with_pandas_ta(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features usando pandas_ta"""

    # 1. Trend indicators (SMAs)
    df['sma_20'] = ta.sma(df['close'], length=20)
    df['sma_50'] = ta.sma(df['close'], length=50)
    df['sma_100'] = ta.sma(df['close'], length=100)
    df['sma_200'] = ta.sma(df['close'], length=200)

    # 2. Trend indicators (EMAs)
    df['ema_9'] = ta.ema(df['close'], length=9)
    df['ema_21'] = ta.ema(df['close'], length=21)
    df['ema_50'] = ta.ema(df['close'], length=50)

    # 3. Bollinger Bands
    bbands = ta.bbands(df['close'], length=20, std=2)
    if bbands is not None:
        df['bb_upper'] = bbands['BBU_20_2.0']
        df['bb_middle'] = bbands['BBM_20_2.0']
        df['bb_lower'] = bbands['BBL_20_2.0']
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    else:
        df['bb_upper'] = df['bb_middle'] = df['bb_lower'] = df['bb_width'] = df['bb_position'] = 0

    # 4. Momentum indicators (RSI)
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    df['rsi_7'] = ta.rsi(df['close'], length=7)
    df['rsi_21'] = ta.rsi(df['close'], length=21)

    # 5. MACD
    macd = ta.macd(df['close'])
    if macd is not None:
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_histogram'] = macd['MACDh_12_26_9']
    else:
        df['macd'] = df['macd_signal'] = df['macd_histogram'] = 0

    # 6. Stochastic
    stoch = ta.stoch(df['high'], df['low'], df['close'])
    if stoch is not None:
        df['stoch_k'] = stoch['STOCHk_14_3_3']
        df['stoch_d'] = stoch['STOCHd_14_3_3']
    else:
        df['stoch_k'] = df['stoch_d'] = 0

    # 7. ATR
    df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)

    # 8. Price-based features
    df['price_change'] = df['close'].pct_change()
    df['price_change_5'] = df['close'].pct_change(5)
    df['price_change_10'] = df['close'].pct_change(10)
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['open_close_range'] = abs(df['open'] - df['close']) / df['close']

    # 9. Volume
    if 'volume' not in df.columns:
        df['volume'] = 0

    # 10. Time-based features
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    else:
        df['hour'] = df['day_of_week'] = df['day_of_month'] = df['is_weekend'] = 0

    # 11. Rolling statistics
    for window in [5, 10, 20]:
        df[f'close_rolling_mean_{window}'] = df['close'].rolling(window).mean()
        df[f'close_rolling_std_{window}'] = df['close'].rolling(window).std()
        df[f'volume_rolling_mean_{window}'] = df['volume'].rolling(window).mean()

    # 12. Lagged features
    for lag in [1, 2, 3, 5]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        df[f'rsi_lag_{lag}'] = df['rsi_14'].shift(lag)

    # Pegar última linha
    features_df = df.iloc[[-1]].copy()

    # Remover colunas que não são features
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'timeframe']
    features_df = features_df[[col for col in features_df.columns if col not in exclude_cols]]

    # Preencher NaN com 0
    features_df = features_df.fillna(0)

    return features_df


def _calculate_manual(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula features manualmente (fallback se pandas_ta não disponível)"""

    # 1. SMAs
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_100'] = df['close'].rolling(100).mean()
    df['sma_200'] = df['close'].rolling(200).mean()

    # 2. EMAs
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

    # 3. Bollinger Bands
    bb_middle = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = bb_middle + (bb_std * 2)
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_middle - (bb_std * 2)
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

    # 4. RSI manual
    for period in [7, 14, 21]:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # 5. MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    # 6. Stochastic
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14 + 1e-10))
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # 7. ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = true_range.rolling(14).mean()

    # 8. Price-based features
    df['price_change'] = df['close'].pct_change()
    df['price_change_5'] = df['close'].pct_change(5)
    df['price_change_10'] = df['close'].pct_change(10)
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['open_close_range'] = abs(df['open'] - df['close']) / df['close']

    # 9. Volume
    if 'volume' not in df.columns:
        df['volume'] = 0

    # 10. Time-based features
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    else:
        df['hour'] = df['day_of_week'] = df['day_of_month'] = df['is_weekend'] = 0

    # 11. Rolling statistics
    for window in [5, 10, 20]:
        df[f'close_rolling_mean_{window}'] = df['close'].rolling(window).mean()
        df[f'close_rolling_std_{window}'] = df['close'].rolling(window).std()
        df[f'volume_rolling_mean_{window}'] = df['volume'].rolling(window).mean()

    # 12. Lagged features
    for lag in [1, 2, 3, 5]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        df[f'rsi_lag_{lag}'] = df['rsi_14'].shift(lag)

    # Pegar última linha
    features_df = df.iloc[[-1]].copy()

    # Remover colunas que não são features
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'timeframe']
    features_df = features_df[[col for col in features_df.columns if col not in exclude_cols]]

    # Preencher NaN com 0
    features_df = features_df.fillna(0)

    return features_df
