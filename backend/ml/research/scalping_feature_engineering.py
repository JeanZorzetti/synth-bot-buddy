"""
Feature Engineering para Scalping ML Trading

Este script adiciona ~30 features técnicas aos dados de candles M5
para melhorar a taxa de sucesso de 50.3% (sem filtros) para 60-65% (com filtros ML).

Features incluídas:
- Indicadores técnicos: RSI, Bollinger Bands, Stochastic, MACD, EMA
- Candlestick patterns: Engulfing, Hammer, Shooting Star
- Price action: Higher highs/lows, Support/Resistance, Volatility squeeze

Autor: Claude Sonnet 4.5
Data: 18/12/2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ta  # pip install ta


class ScalpingFeatureEngineer:
    """
    Engenharia de features para scalping em timeframe M5
    """

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame com colunas [timestamp, open, high, low, close, volume]
        """
        self.df = df.copy()

        # Validar colunas necessárias (volume é opcional)
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Colunas faltando: {missing_cols}")

        # Adicionar coluna volume se não existir (synthetic indices não têm volume real)
        if 'volume' not in self.df.columns:
            self.df['volume'] = 0
            print("[INFO] Coluna 'volume' não encontrada, adicionada com zeros")

        print(f"[OK] Feature Engineer inicializado com {len(self.df)} candles")

    def add_all_features(self) -> pd.DataFrame:
        """
        Adiciona TODAS as features técnicas ao DataFrame

        Returns:
            DataFrame com ~30+ features adicionadas
        """
        print("\n[FEATURE] Adicionando features técnicas...")

        # Grupo 1: Indicadores clássicos
        self.df = self._add_rsi_features()
        self.df = self._add_bollinger_bands()
        self.df = self._add_stochastic()
        self.df = self._add_macd()
        self.df = self._add_ema_features()

        # Grupo 2: Candlestick patterns
        self.df = self._add_candlestick_patterns()

        # Grupo 3: Price action
        self.df = self._add_price_action()

        # Grupo 4: Volatilidade
        self.df = self._add_volatility_features()

        # Remover NaN (primeiras linhas terão NaN por causa de rolling windows)
        initial_len = len(self.df)
        self.df = self.df.dropna()
        dropped = initial_len - len(self.df)

        print(f"[OK] {len(self.df.columns)} features criadas")
        print(f"[INFO] {dropped} linhas removidas (NaN em rolling windows)")

        return self.df

    def _add_rsi_features(self) -> pd.DataFrame:
        """Adiciona RSI (Relative Strength Index)"""
        print("   - RSI (14, 7)")

        # RSI 14 (padrão)
        self.df['rsi_14'] = ta.momentum.RSIIndicator(
            self.df['close'], window=14
        ).rsi()

        # RSI 7 (mais rápido para scalping)
        self.df['rsi_7'] = ta.momentum.RSIIndicator(
            self.df['close'], window=7
        ).rsi()

        # Flags de oversold/overbought
        self.df['rsi_oversold'] = (self.df['rsi_14'] < 30).astype(int)
        self.df['rsi_overbought'] = (self.df['rsi_14'] > 70).astype(int)

        # RSI momentum (mudança em relação ao candle anterior)
        self.df['rsi_momentum'] = self.df['rsi_14'].diff()

        return self.df

    def _add_bollinger_bands(self) -> pd.DataFrame:
        """Adiciona Bollinger Bands"""
        print("   - Bollinger Bands (20, 2)")

        bb = ta.volatility.BollingerBands(
            self.df['close'], window=20, window_dev=2
        )

        self.df['bb_upper'] = bb.bollinger_hband()
        self.df['bb_lower'] = bb.bollinger_lband()
        self.df['bb_middle'] = bb.bollinger_mavg()

        # Posição relativa dentro das bandas (0 = banda inferior, 1 = banda superior)
        bb_range = self.df['bb_upper'] - self.df['bb_lower']
        bb_range = bb_range.replace(0, np.nan)  # Evitar divisão por zero
        self.df['bb_position'] = (self.df['close'] - self.df['bb_lower']) / bb_range

        # Largura das bandas (normalizada pelo preço médio)
        self.df['bb_width'] = bb_range / self.df['bb_middle']

        # Flags de toque nas bandas
        self.df['bb_touch_upper'] = (self.df['high'] >= self.df['bb_upper']).astype(int)
        self.df['bb_touch_lower'] = (self.df['low'] <= self.df['bb_lower']).astype(int)

        return self.df

    def _add_stochastic(self) -> pd.DataFrame:
        """Adiciona Stochastic Oscillator"""
        print("   - Stochastic Oscillator (14, 3)")

        stoch = ta.momentum.StochasticOscillator(
            self.df['high'], self.df['low'], self.df['close'],
            window=14, smooth_window=3
        )

        self.df['stoch_k'] = stoch.stoch()
        self.df['stoch_d'] = stoch.stoch_signal()

        # Flags de oversold/overbought
        self.df['stoch_oversold'] = (self.df['stoch_k'] < 20).astype(int)
        self.df['stoch_overbought'] = (self.df['stoch_k'] > 80).astype(int)

        # Cruzamento K e D
        self.df['stoch_cross_up'] = (
            (self.df['stoch_k'] > self.df['stoch_d']) &
            (self.df['stoch_k'].shift(1) <= self.df['stoch_d'].shift(1))
        ).astype(int)

        self.df['stoch_cross_down'] = (
            (self.df['stoch_k'] < self.df['stoch_d']) &
            (self.df['stoch_k'].shift(1) >= self.df['stoch_d'].shift(1))
        ).astype(int)

        return self.df

    def _add_macd(self) -> pd.DataFrame:
        """Adiciona MACD"""
        print("   - MACD (12, 26, 9)")

        macd_indicator = ta.trend.MACD(
            self.df['close'],
            window_slow=26, window_fast=12, window_sign=9
        )

        self.df['macd'] = macd_indicator.macd()
        self.df['macd_signal'] = macd_indicator.macd_signal()
        self.df['macd_diff'] = macd_indicator.macd_diff()

        # Flag de sinal bullish (MACD acima da linha de sinal)
        self.df['macd_bullish'] = (self.df['macd'] > self.df['macd_signal']).astype(int)

        # Cruzamento MACD e signal
        self.df['macd_cross_up'] = (
            (self.df['macd'] > self.df['macd_signal']) &
            (self.df['macd'].shift(1) <= self.df['macd_signal'].shift(1))
        ).astype(int)

        self.df['macd_cross_down'] = (
            (self.df['macd'] < self.df['macd_signal']) &
            (self.df['macd'].shift(1) >= self.df['macd_signal'].shift(1))
        ).astype(int)

        return self.df

    def _add_ema_features(self) -> pd.DataFrame:
        """Adiciona Exponential Moving Averages"""
        print("   - EMA (9, 21, 50)")

        # EMAs rápidas para scalping
        self.df['ema_9'] = ta.trend.EMAIndicator(
            self.df['close'], window=9
        ).ema_indicator()

        self.df['ema_21'] = ta.trend.EMAIndicator(
            self.df['close'], window=21
        ).ema_indicator()

        self.df['ema_50'] = ta.trend.EMAIndicator(
            self.df['close'], window=50
        ).ema_indicator()

        # Cruzamento EMA 9 e 21
        self.df['ema_cross_up'] = (
            (self.df['ema_9'] > self.df['ema_21']) &
            (self.df['ema_9'].shift(1) <= self.df['ema_21'].shift(1))
        ).astype(int)

        self.df['ema_cross_down'] = (
            (self.df['ema_9'] < self.df['ema_21']) &
            (self.df['ema_9'].shift(1) >= self.df['ema_21'].shift(1))
        ).astype(int)

        # Distância relativa do preço às EMAs
        self.df['dist_to_ema_9'] = (self.df['close'] - self.df['ema_9']) / self.df['close']
        self.df['dist_to_ema_21'] = (self.df['close'] - self.df['ema_21']) / self.df['close']

        return self.df

    def _add_candlestick_patterns(self) -> pd.DataFrame:
        """Adiciona detecção de candlestick patterns"""
        print("   - Candlestick Patterns (Engulfing, Hammer, Shooting Star)")

        # Corpo e sombras dos candles
        self.df['body'] = abs(self.df['close'] - self.df['open'])
        self.df['upper_shadow'] = self.df['high'] - self.df[['open', 'close']].max(axis=1)
        self.df['lower_shadow'] = self.df[['open', 'close']].min(axis=1) - self.df['low']

        # Bullish Engulfing
        self.df['bullish_engulfing'] = (
            (self.df['close'].shift(1) < self.df['open'].shift(1)) &  # Candle anterior bearish
            (self.df['close'] > self.df['open']) &  # Candle atual bullish
            (self.df['open'] < self.df['close'].shift(1)) &  # Abre abaixo do close anterior
            (self.df['close'] > self.df['open'].shift(1))  # Fecha acima do open anterior
        ).astype(int)

        # Bearish Engulfing
        self.df['bearish_engulfing'] = (
            (self.df['close'].shift(1) > self.df['open'].shift(1)) &  # Candle anterior bullish
            (self.df['close'] < self.df['open']) &  # Candle atual bearish
            (self.df['open'] > self.df['close'].shift(1)) &  # Abre acima do close anterior
            (self.df['close'] < self.df['open'].shift(1))  # Fecha abaixo do open anterior
        ).astype(int)

        # Hammer (sombra inferior > 2x corpo, sombra superior < 0.3x corpo)
        self.df['hammer'] = (
            (self.df['lower_shadow'] > self.df['body'] * 2) &
            (self.df['upper_shadow'] < self.df['body'] * 0.3)
        ).astype(int)

        # Shooting Star (sombra superior > 2x corpo, sombra inferior < 0.3x corpo)
        self.df['shooting_star'] = (
            (self.df['upper_shadow'] > self.df['body'] * 2) &
            (self.df['lower_shadow'] < self.df['body'] * 0.3)
        ).astype(int)

        # Doji (corpo muito pequeno)
        avg_body = self.df['body'].rolling(20).mean()
        self.df['doji'] = (self.df['body'] < avg_body * 0.1).astype(int)

        return self.df

    def _add_price_action(self) -> pd.DataFrame:
        """Adiciona features de price action"""
        print("   - Price Action (Higher highs/lows, Support/Resistance)")

        # Higher highs / Lower lows (detecção de tendência)
        self.df['higher_high'] = (self.df['high'] > self.df['high'].shift(1)).astype(int)
        self.df['lower_low'] = (self.df['low'] < self.df['low'].shift(1)).astype(int)

        # Sequência de higher highs/lower lows
        self.df['hh_streak'] = self.df['higher_high'].rolling(3).sum()
        self.df['ll_streak'] = self.df['lower_low'].rolling(3).sum()

        # Support e Resistance (mínimo/máximo local)
        self.df['support'] = self.df['low'].rolling(20).min()
        self.df['resistance'] = self.df['high'].rolling(20).max()

        # Distância relativa para support/resistance
        self.df['dist_to_support'] = (self.df['close'] - self.df['support']) / self.df['close']
        self.df['dist_to_resistance'] = (self.df['resistance'] - self.df['close']) / self.df['close']

        # Flag de toque em support/resistance
        self.df['touch_support'] = (
            abs(self.df['low'] - self.df['support']) < self.df['close'] * 0.001
        ).astype(int)

        self.df['touch_resistance'] = (
            abs(self.df['high'] - self.df['resistance']) < self.df['close'] * 0.001
        ).astype(int)

        return self.df

    def _add_volatility_features(self) -> pd.DataFrame:
        """Adiciona features de volatilidade"""
        print("   - Volatility (ATR, Intrabar range, Squeeze)")

        # ATR (Average True Range)
        atr = ta.volatility.AverageTrueRange(
            self.df['high'], self.df['low'], self.df['close'], window=14
        )
        self.df['atr'] = atr.average_true_range()
        self.df['atr_pct'] = (self.df['atr'] / self.df['close']) * 100

        # Volatilidade intrabar (high-low range)
        self.df['intrabar_range'] = self.df['high'] - self.df['low']
        self.df['intrabar_range_pct'] = (self.df['intrabar_range'] / self.df['close']) * 100

        # Bollinger Bands Squeeze (volatilidade baixa)
        self.df['volatility_squeeze'] = (
            self.df['bb_width'] < self.df['bb_width'].rolling(20).quantile(0.2)
        ).astype(int)

        # Expansão de volatilidade (ATR aumentando)
        self.df['atr_expansion'] = (
            self.df['atr'] > self.df['atr'].shift(1)
        ).astype(int)

        return self.df

    def get_feature_importance_ready_df(self) -> pd.DataFrame:
        """
        Retorna DataFrame pronto para treinar modelo ML

        Remove colunas auxiliares e retorna apenas features + OHLCV
        """
        # Colunas auxiliares para remover (body, shadows, etc são intermediárias)
        cols_to_drop = ['body', 'upper_shadow', 'lower_shadow']
        existing_cols = [col for col in cols_to_drop if col in self.df.columns]

        df_ready = self.df.drop(columns=existing_cols)

        print(f"\n[OK] DataFrame pronto para ML com {len(df_ready.columns)} colunas")

        return df_ready

    def save_to_csv(self, output_path: str):
        """Salva DataFrame com features em CSV"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self.df.to_csv(output_path, index=False)
        print(f"[SAVE] Features salvas em {output_path}")


def process_symbol_features(symbol: str, timeframe: str = '5min'):
    """
    Processa features para um símbolo específico

    Args:
        symbol: '1HZ75V' ou '1HZ100V'
        timeframe: '1min' ou '5min'
    """
    print(f"\n{'='*70}")
    print(f"PROCESSANDO FEATURES: {symbol} ({timeframe})")
    print(f"{'='*70}")

    # Carregar dados
    data_dir = Path(__file__).parent / "data"
    csv_path = data_dir / f"{symbol}_{timeframe}_180days.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")

    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"[OK] Dados carregados: {len(df)} candles")

    # Criar features
    engineer = ScalpingFeatureEngineer(df)
    df_with_features = engineer.add_all_features()

    # Salvar
    output_dir = Path(__file__).parent / "data"
    output_path = output_dir / f"{symbol}_{timeframe}_180days_features.csv"
    engineer.save_to_csv(output_path)

    return df_with_features


if __name__ == "__main__":
    """
    Executar feature engineering para V100 M5
    """

    # Processar V100 M5 (nosso foco principal)
    df_v100 = process_symbol_features('1HZ100V', '5min')

    print("\n" + "="*70)
    print("RESUMO DE FEATURES CRIADAS")
    print("="*70)
    print(f"Total de colunas: {len(df_v100.columns)}")
    print(f"Total de candles: {len(df_v100)}")
    print(f"\nPrimeiras features:")
    feature_cols = [col for col in df_v100.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    for i, col in enumerate(feature_cols[:15], 1):
        print(f"   {i:2d}. {col}")
    print(f"   ... e {len(feature_cols) - 15} mais features\n")
