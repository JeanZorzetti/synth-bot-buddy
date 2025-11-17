"""
Feature Engineering para Machine Learning
Prepara dados históricos com features para treinamento
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
import sys
from pathlib import Path

# Adicionar path do backend
sys.path.append(str(Path(__file__).parent.parent.parent))

from analysis.technical_analysis import TechnicalAnalysis
from analysis.patterns.candlestick_patterns import CandlestickPatterns
from analysis.patterns.chart_formations import ChartFormationDetector

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Cria features (variáveis) para Machine Learning a partir de dados de candles
    """

    def __init__(self):
        """Inicializa detectores de padrões"""
        self.candlestick_detector = CandlestickPatterns()
        self.chart_formations = ChartFormationDetector()

    def prepare_ml_dataset(
        self,
        df: pd.DataFrame,
        prediction_horizon: int = 15,  # Minutos para frente
        price_threshold: float = 0.5   # % mínimo de movimento
    ) -> pd.DataFrame:
        """
        Prepara dataset completo para ML

        Args:
            df: DataFrame com candles (OHLCV + timestamp)
            prediction_horizon: Quantos minutos à frente prever (ex: 15 = 15min)
            price_threshold: Movimento mínimo considerado (ex: 0.5 = 0.5%)

        Returns:
            DataFrame com todas as features + target variable
        """
        logger.info(f"Preparando dataset ML com {len(df)} candles")
        logger.info(f"Horizonte de previsão: {prediction_horizon} minutos")
        logger.info(f"Threshold: {price_threshold}%")

        # Criar cópia
        ml_df = df.copy()

        # 1. Features de Preço Básicas
        logger.info("Calculando features básicas de preço...")
        ml_df = self._add_price_features(ml_df)

        # 2. Indicadores Técnicos
        logger.info("Calculando indicadores técnicos...")
        ml_df = self._add_technical_indicators(ml_df)

        # 3. Padrões de Candlestick
        logger.info("Detectando padrões de candlestick...")
        ml_df = self._add_candlestick_patterns(ml_df)

        # 4. Features Derivadas
        logger.info("Calculando features derivadas...")
        ml_df = self._add_derived_features(ml_df)

        # 5. Features Temporais
        logger.info("Adicionando features temporais...")
        ml_df = self._add_temporal_features(ml_df)

        # 6. Target Variable (Label)
        logger.info("Criando target variable...")
        ml_df = self._create_target_variable(ml_df, prediction_horizon, price_threshold)

        # Remover NaN (devido a indicadores que precisam de histórico)
        initial_rows = len(ml_df)
        ml_df = ml_df.dropna()
        logger.info(f"Removidas {initial_rows - len(ml_df)} linhas com NaN")

        logger.info(f"Dataset final: {len(ml_df)} linhas, {len(ml_df.columns)} features")

        return ml_df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona features básicas de preço
        """
        # Retornos (variação percentual)
        df['returns_1'] = df['close'].pct_change(1) * 100
        df['returns_5'] = df['close'].pct_change(5) * 100
        df['returns_15'] = df['close'].pct_change(15) * 100

        # Range do candle
        df['candle_range'] = ((df['high'] - df['low']) / df['close']) * 100
        df['body_size'] = ((df['close'] - df['open']) / df['close']).abs() * 100

        # Upper/Lower shadows
        df['upper_shadow'] = df.apply(
            lambda x: ((x['high'] - max(x['open'], x['close'])) / x['close']) * 100,
            axis=1
        )
        df['lower_shadow'] = df.apply(
            lambda x: ((min(x['open'], x['close']) - x['low']) / x['close']) * 100,
            axis=1
        )

        # Tipo de candle
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        df['is_bearish'] = (df['close'] < df['open']).astype(int)
        df['is_doji'] = (df['body_size'] < 0.1).astype(int)

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona indicadores técnicos usando TechnicalAnalysis
        """
        # Criar instância de análise técnica
        analyzer = TechnicalAnalysis()

        # Calcular indicadores
        indicators = analyzer.calculate_all_indicators(df)

        # SMAs
        df['sma_20'] = indicators['sma_20']
        df['sma_50'] = indicators['sma_50']

        # EMAs
        df['ema_9'] = indicators['ema_9']
        df['ema_21'] = indicators['ema_21']

        # RSI
        df['rsi'] = indicators['rsi']
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)

        # MACD
        if 'macd' in indicators:
            df['macd_line'] = indicators['macd']['macd_line']
            df['macd_signal'] = indicators['macd']['signal_line']
            df['macd_histogram'] = indicators['macd']['histogram']
        else:
            # Fallback - usar colunas diretas
            df['macd_line'] = indicators.get('macd_line', 0)
            df['macd_signal'] = indicators.get('signal_line', 0)
            df['macd_histogram'] = indicators.get('histogram', 0)

        df['macd_bullish'] = (df['macd_histogram'] > 0).astype(int)

        # Bollinger Bands
        df['bb_upper'] = indicators.get('bb_upper', df['close'])
        df['bb_middle'] = indicators.get('bb_middle', df['close'])
        df['bb_lower'] = indicators.get('bb_lower', df['close'])
        df['bb_width'] = indicators.get('bb_width', 0)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 0.0001)

        # ATR
        df['atr'] = indicators['atr']

        # Stochastic
        if 'stochastic' in indicators:
            df['stoch_k'] = indicators['stochastic']['k']
            df['stoch_d'] = indicators['stochastic']['d']
        else:
            df['stoch_k'] = indicators.get('k', 50)
            df['stoch_d'] = indicators.get('d', 50)

        df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
        df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)

        return df

    def _add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona detecção de padrões de candlestick (one-hot encoding)
        """
        # Detectar padrões
        patterns = self.candlestick_detector.detect_all_patterns(df)

        # Inicializar colunas (0 = sem padrão)
        pattern_types = [
            'hammer', 'shooting_star', 'doji',
            'bullish_engulfing', 'bearish_engulfing',
            'piercing_pattern', 'dark_cloud_cover',
            'morning_star', 'evening_star',
            'three_white_soldiers', 'three_black_crows'
        ]

        for pattern_type in pattern_types:
            df[f'pattern_{pattern_type}'] = 0

        # Marcar onde padrões foram encontrados
        for pattern in patterns:
            idx = pattern.index
            pattern_name = pattern.name.lower().replace(' ', '_')

            # Mapear nomes variantes
            if 'doji' in pattern_name:
                pattern_name = 'doji'
            elif 'harami' in pattern_name:
                continue  # Não temos coluna separada para harami

            if f'pattern_{pattern_name}' in df.columns:
                df.at[idx, f'pattern_{pattern_name}'] = 1

        # Contar padrões bullish e bearish
        bullish_cols = ['pattern_hammer', 'pattern_bullish_engulfing',
                        'pattern_piercing_pattern', 'pattern_morning_star',
                        'pattern_three_white_soldiers']
        bearish_cols = ['pattern_shooting_star', 'pattern_bearish_engulfing',
                        'pattern_dark_cloud_cover', 'pattern_evening_star',
                        'pattern_three_black_crows']

        df['bullish_pattern_count'] = df[bullish_cols].sum(axis=1)
        df['bearish_pattern_count'] = df[bearish_cols].sum(axis=1)

        return df

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona features derivadas de combinações de indicadores
        """
        # Diferenças entre médias móveis
        df['ema_diff_9_21'] = df['ema_9'] - df['ema_21']
        df['sma_diff_20_50'] = df['sma_20'] - df['sma_50']

        # Distância do preço às médias móveis
        df['price_to_sma20'] = ((df['close'] - df['sma_20']) / df['sma_20']) * 100
        df['price_to_ema9'] = ((df['close'] - df['ema_9']) / df['ema_9']) * 100

        # Momentum
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['momentum_15'] = df['close'] - df['close'].shift(15)

        # Volatilidade rolante
        df['volatility_5'] = df['returns_1'].rolling(5).std()
        df['volatility_20'] = df['returns_1'].rolling(20).std()

        # Volume ratio (se disponível)
        if 'volume' in df.columns and df['volume'].sum() > 0:
            df['volume_ma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        else:
            df['volume_ratio'] = 1.0

        # Força da tendência (EMA slope)
        df['ema9_slope'] = df['ema_9'].diff(5) / df['ema_9'].shift(5) * 100

        # RSI momentum
        df['rsi_diff'] = df['rsi'].diff(5)

        # Bollinger squeeze (volatilidade baixa)
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean()).astype(int)

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona features baseadas no tempo
        """
        # Extrair componentes de data/hora
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_of_month'] = df['timestamp'].dt.day

        # Sessões de trading (ajustar conforme mercado)
        # Para synthetic indices 24/7, mas pode ter padrões de hora
        df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['is_london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['is_ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)

        # Fim de semana
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # One-hot encoding para hora (cíclico)
        # Converter hora em features cíclicas (sin/cos)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        return df

    def _create_target_variable(
        self,
        df: pd.DataFrame,
        horizon: int,
        threshold: float
    ) -> pd.DataFrame:
        """
        Cria a variável target (label) para classificação

        Target = 1 se preço SOBE >= threshold% em 'horizon' minutos
        Target = 0 caso contrário

        Args:
            horizon: Minutos à frente para verificar
            threshold: % mínimo de movimento
        """
        # Calcular preço futuro
        df['future_price'] = df['close'].shift(-horizon)

        # Calcular variação percentual
        df['future_return'] = ((df['future_price'] - df['close']) / df['close']) * 100

        # Criar target binário
        df['target'] = (df['future_return'] >= threshold).astype(int)

        # Também criar target contínuo (para análise)
        df['target_continuous'] = df['future_return']

        # Remover colunas auxiliares
        df = df.drop(['future_price', 'future_return'], axis=1)

        logger.info(f"Target distribution: {df['target'].value_counts().to_dict()}")

        return df

    def get_feature_names(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Retorna nomes das features organizados por categoria

        Returns:
            Dicionário com categorias de features
        """
        feature_categories = {
            'price_basic': [col for col in df.columns if col in [
                'returns_1', 'returns_5', 'returns_15', 'candle_range', 'body_size',
                'upper_shadow', 'lower_shadow', 'is_bullish', 'is_bearish', 'is_doji'
            ]],
            'technical': [col for col in df.columns if any(ind in col for ind in [
                'sma', 'ema', 'rsi', 'macd', 'bb_', 'atr', 'stoch'
            ])],
            'patterns': [col for col in df.columns if col.startswith('pattern_') or
                        'pattern_count' in col],
            'derived': [col for col in df.columns if any(der in col for der in [
                'diff', 'momentum', 'volatility', 'ratio', 'slope', 'squeeze'
            ])],
            'temporal': [col for col in df.columns if any(tmp in col for tmp in [
                'hour', 'day_', 'session', 'weekend'
            ])],
            'target': ['target', 'target_continuous']
        }

        # Contar features
        total_features = sum(len(v) for k, v in feature_categories.items() if k != 'target')
        logger.info(f"Total de features: {total_features}")

        for category, features in feature_categories.items():
            logger.info(f"  {category}: {len(features)} features")

        return feature_categories

    def split_features_target(
        self,
        df: pd.DataFrame,
        target_col: str = 'target'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separa features (X) e target (y)

        Args:
            df: DataFrame completo
            target_col: Nome da coluna target

        Returns:
            (X, y) - Features e target
        """
        # Colunas a remover (não são features)
        non_feature_cols = [
            'timestamp', 'symbol', 'timeframe',
            'open', 'high', 'low', 'close', 'volume',
            'target', 'target_continuous'
        ]

        # Features (X)
        feature_cols = [col for col in df.columns if col not in non_feature_cols]
        X = df[feature_cols].copy()

        # Target (y)
        y = df[target_col].copy()

        logger.info(f"X shape: {X.shape}")
        logger.info(f"y shape: {y.shape}")
        logger.info(f"Class distribution: {y.value_counts().to_dict()}")

        return X, y


def main():
    """
    Exemplo de uso
    """
    # Carregar dados coletados
    from pathlib import Path
    import pickle

    data_dir = Path(__file__).parent.parent / "data"
    pkl_file = list(data_dir.glob("R_100_1m_*.pkl"))[0]

    logger.info(f"Carregando dados de {pkl_file}")
    df = pd.read_pickle(pkl_file)

    logger.info(f"Dados carregados: {len(df)} candles")
    logger.info(f"Período: {df['timestamp'].min()} até {df['timestamp'].max()}")

    # Criar feature engineer
    engineer = FeatureEngineer()

    # Preparar dataset ML
    ml_df = engineer.prepare_ml_dataset(
        df,
        prediction_horizon=15,  # Prever 15 minutos à frente
        price_threshold=0.3     # Movimento de 0.3%
    )

    # Mostrar categorias de features
    feature_cats = engineer.get_feature_names(ml_df)

    # Separar X e y
    X, y = engineer.split_features_target(ml_df)

    # Salvar dataset processado
    output_file = data_dir / "ml_dataset_R100_1m.pkl"
    ml_df.to_pickle(output_file)
    logger.info(f"Dataset ML salvo em: {output_file}")

    # Mostrar amostra
    print("\n" + "="*60)
    print("DATASET ML - AMOSTRA")
    print("="*60)
    print(f"\nShape: {ml_df.shape}")
    print(f"\nPrimeiras linhas:")
    print(ml_df.head())
    print(f"\nFeatures:")
    print(X.head())
    print(f"\nTarget:")
    print(y.value_counts())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s: %(message)s'
    )

    main()
