"""
Feature Calculator - Cálculo de features técnicas para ML

Calcula TODAS as 65 features esperadas pelo modelo XGBoost
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    logger.warning("pandas_ta não disponível, usando cálculos manuais")


def calculate_ml_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Calcula TODAS as 65 features técnicas esperadas pelo modelo ML

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

    # Calcular todas as features
    df = _calculate_all_features(df)

    # Pegar última linha
    features_df = df.iloc[[-1]].copy()

    # Garantir que APENAS as 65 features esperadas estão presentes
    expected_features = [
        'returns_1', 'returns_5', 'returns_15',
        'candle_range', 'body_size', 'upper_shadow', 'lower_shadow',
        'is_bullish', 'is_bearish', 'is_doji',
        'sma_20', 'sma_50',
        'ema_9', 'ema_21',
        'rsi', 'rsi_oversold', 'rsi_overbought',
        'macd_line', 'macd_signal', 'macd_histogram', 'macd_bullish',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
        'atr',
        'stoch_k', 'stoch_d', 'stoch_oversold', 'stoch_overbought',
        'pattern_hammer', 'pattern_shooting_star', 'pattern_doji',
        'pattern_bullish_engulfing', 'pattern_bearish_engulfing',
        'pattern_piercing_pattern', 'pattern_dark_cloud_cover',
        'pattern_morning_star', 'pattern_evening_star',
        'pattern_three_white_soldiers', 'pattern_three_black_crows',
        'bullish_pattern_count', 'bearish_pattern_count',
        'ema_diff_9_21', 'sma_diff_20_50',
        'price_to_sma20', 'price_to_ema9',
        'momentum_5', 'momentum_15',
        'volatility_5', 'volatility_20',
        'volume_ratio',
        'ema9_slope', 'rsi_diff',
        'bb_squeeze',
        'hour', 'day_of_week', 'day_of_month',
        'is_asian_session', 'is_london_session', 'is_ny_session', 'is_weekend',
        'hour_sin', 'hour_cos'
    ]

    # Filtrar apenas features esperadas
    features_df = features_df[[col for col in expected_features if col in features_df.columns]]

    # Adicionar features faltando com valor 0
    for feat in expected_features:
        if feat not in features_df.columns:
            features_df[feat] = 0

    # Reordenar colunas
    features_df = features_df[expected_features]

    # Preencher NaN com 0
    features_df = features_df.fillna(0)

    # Validar que temos 65 features
    if len(features_df.columns) != 65:
        logger.error(f"ERRO: {len(features_df.columns)} features geradas, esperadas 65!")
        logger.error(f"Features faltando: {set(expected_features) - set(features_df.columns)}")
        return None

    return features_df


def _calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula todas as 65 features"""

    # ==========================
    # 1. PRICE-BASED FEATURES (7)
    # ==========================
    df['returns_1'] = df['close'].pct_change(1)
    df['returns_5'] = df['close'].pct_change(5)
    df['returns_15'] = df['close'].pct_change(15)
    df['candle_range'] = (df['high'] - df['low']) / df['close']
    df['body_size'] = abs(df['close'] - df['open']) / df['close']
    df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['close']
    df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['close']

    # ==========================
    # 2. CANDLESTICK FLAGS (3)
    # ==========================
    df['is_bullish'] = (df['close'] > df['open']).astype(int)
    df['is_bearish'] = (df['close'] < df['open']).astype(int)
    df['is_doji'] = (abs(df['close'] - df['open']) < (df['high'] - df['low']) * 0.1).astype(int)

    # ==========================
    # 3. MOVING AVERAGES (4)
    # ==========================
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()

    # ==========================
    # 4. RSI (3)
    # ==========================
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)

    # ==========================
    # 5. MACD (4)
    # ==========================
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_line'] = ema_12 - ema_26
    df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd_line'] - df['macd_signal']
    df['macd_bullish'] = (df['macd_line'] > df['macd_signal']).astype(int)

    # ==========================
    # 6. BOLLINGER BANDS (6)
    # ==========================
    bb_middle = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = bb_middle + (bb_std * 2)
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_middle - (bb_std * 2)
    df['bb_width'] = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_width'] + 1e-10)
    df['bb_squeeze'] = df['bb_width'] / df['bb_middle']

    # ==========================
    # 7. ATR (1)
    # ==========================
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()

    # ==========================
    # 8. STOCHASTIC (4)
    # ==========================
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14 + 1e-10))
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
    df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)

    # ==========================
    # 9. CANDLESTICK PATTERNS (12)
    # ==========================
    # Hammer: small body, long lower shadow
    df['pattern_hammer'] = (
        (df['body_size'] < 0.005) &
        (df['lower_shadow'] > df['body_size'] * 2) &
        (df['upper_shadow'] < df['body_size'] * 0.5)
    ).astype(int)

    # Shooting Star: small body, long upper shadow
    df['pattern_shooting_star'] = (
        (df['body_size'] < 0.005) &
        (df['upper_shadow'] > df['body_size'] * 2) &
        (df['lower_shadow'] < df['body_size'] * 0.5)
    ).astype(int)

    # Doji: já calculado
    df['pattern_doji'] = df['is_doji']

    # Bullish Engulfing: candle atual engolfa anterior (bullish)
    df['pattern_bullish_engulfing'] = (
        (df['is_bullish'] == 1) &
        (df['is_bearish'].shift(1) == 1) &
        (df['open'] < df['close'].shift(1)) &
        (df['close'] > df['open'].shift(1))
    ).astype(int)

    # Bearish Engulfing: candle atual engolfa anterior (bearish)
    df['pattern_bearish_engulfing'] = (
        (df['is_bearish'] == 1) &
        (df['is_bullish'].shift(1) == 1) &
        (df['open'] > df['close'].shift(1)) &
        (df['close'] < df['open'].shift(1))
    ).astype(int)

    # Piercing Pattern: 2 candles, primeiro bearish, segundo bullish
    df['pattern_piercing_pattern'] = (
        (df['is_bearish'].shift(1) == 1) &
        (df['is_bullish'] == 1) &
        (df['close'] > (df['open'].shift(1) + df['close'].shift(1)) / 2)
    ).astype(int)

    # Dark Cloud Cover: oposto do piercing
    df['pattern_dark_cloud_cover'] = (
        (df['is_bullish'].shift(1) == 1) &
        (df['is_bearish'] == 1) &
        (df['close'] < (df['open'].shift(1) + df['close'].shift(1)) / 2)
    ).astype(int)

    # Morning Star: 3 candles (bearish, doji, bullish)
    df['pattern_morning_star'] = (
        (df['is_bearish'].shift(2) == 1) &
        (df['is_doji'].shift(1) == 1) &
        (df['is_bullish'] == 1)
    ).astype(int)

    # Evening Star: oposto do morning
    df['pattern_evening_star'] = (
        (df['is_bullish'].shift(2) == 1) &
        (df['is_doji'].shift(1) == 1) &
        (df['is_bearish'] == 1)
    ).astype(int)

    # Three White Soldiers: 3 bullish consecutivos
    df['pattern_three_white_soldiers'] = (
        (df['is_bullish'] == 1) &
        (df['is_bullish'].shift(1) == 1) &
        (df['is_bullish'].shift(2) == 1)
    ).astype(int)

    # Three Black Crows: 3 bearish consecutivos
    df['pattern_three_black_crows'] = (
        (df['is_bearish'] == 1) &
        (df['is_bearish'].shift(1) == 1) &
        (df['is_bearish'].shift(2) == 1)
    ).astype(int)

    # Pattern counts
    bullish_patterns = [
        'pattern_hammer', 'pattern_bullish_engulfing', 'pattern_piercing_pattern',
        'pattern_morning_star', 'pattern_three_white_soldiers'
    ]
    bearish_patterns = [
        'pattern_shooting_star', 'pattern_bearish_engulfing', 'pattern_dark_cloud_cover',
        'pattern_evening_star', 'pattern_three_black_crows'
    ]

    df['bullish_pattern_count'] = df[bullish_patterns].sum(axis=1)
    df['bearish_pattern_count'] = df[bearish_patterns].sum(axis=1)

    # ==========================
    # 10. DERIVED FEATURES (6)
    # ==========================
    df['ema_diff_9_21'] = df['ema_9'] - df['ema_21']
    df['sma_diff_20_50'] = df['sma_20'] - df['sma_50']
    df['price_to_sma20'] = df['close'] / (df['sma_20'] + 1e-10)
    df['price_to_ema9'] = df['close'] / (df['ema_9'] + 1e-10)
    df['ema9_slope'] = df['ema_9'].diff(5)
    df['rsi_diff'] = df['rsi'].diff(5)

    # ==========================
    # 11. MOMENTUM & VOLATILITY (4)
    # ==========================
    df['momentum_5'] = df['close'] - df['close'].shift(5)
    df['momentum_15'] = df['close'] - df['close'].shift(15)
    df['volatility_5'] = df['returns_1'].rolling(5).std()
    df['volatility_20'] = df['returns_1'].rolling(20).std()

    # ==========================
    # 12. VOLUME (1)
    # ==========================
    if 'volume' in df.columns:
        volume_sma_20 = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (volume_sma_20 + 1e-10)
    else:
        df['volume_ratio'] = 1.0

    # ==========================
    # 13. TIME-BASED FEATURES (8)
    # ==========================
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Session flags (UTC)
        df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['is_london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['is_ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)

        # Circular encoding (hour)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    else:
        df['hour'] = 0
        df['day_of_week'] = 0
        df['day_of_month'] = 1
        df['is_weekend'] = 0
        df['is_asian_session'] = 0
        df['is_london_session'] = 0
        df['is_ny_session'] = 0
        df['hour_sin'] = 0
        df['hour_cos'] = 1

    return df
