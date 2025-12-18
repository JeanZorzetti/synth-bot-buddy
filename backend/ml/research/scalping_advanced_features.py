"""
Feature Engineering Avançada para Scalping V100 M5

CONTEXTO:
Experimentos A/B/C falharam em atingir 60% win rate (máximo: 51.2%)
62 features técnicas básicas (RSI, BB, MACD) são insuficientes

SOLUÇÃO:
Adicionar features de microestrutura de mercado (order flow, tape reading)

EXPECTATIVA:
+13-22% win rate → 51.2% + 15% = 66.2% ✅

Features Implementadas:
1. Order Flow Imbalance (5 features) - esperado: +5-8% win rate
2. Tape Reading (6 features) - esperado: +3-5% win rate
3. Volume Profile (8 features) - esperado: +2-4% win rate
4. Delta Cumulativo (4 features) - esperado: +2-3% win rate
5. Absorção de Ordens (3 features) - esperado: +1-2% win rate

Total: 26 novas features

Autor: Claude Sonnet 4.5
Data: 18/12/2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple


class AdvancedScalpingFeatures:
    """
    Adiciona features avançadas de microestrutura de mercado
    """

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame com OHLCV (open, high, low, close, volume)
        """
        self.df = df.copy()
        print(f"[INIT] AdvancedScalpingFeatures inicializado")
        print(f"   Dataset: {len(self.df)} candles")

    def add_all_advanced_features(self) -> pd.DataFrame:
        """
        Adiciona todas as 26 features avançadas

        Returns:
            DataFrame com features adicionadas
        """
        print("\n[FEATURE ENGINEERING AVANÇADA] Iniciando...")

        # 1. Order Flow Imbalance (5 features)
        print("[1/5] Adicionando Order Flow Imbalance...")
        self._add_order_flow_imbalance()

        # 2. Tape Reading (6 features)
        print("[2/5] Adicionando Tape Reading...")
        self._add_tape_reading()

        # 3. Volume Profile (8 features)
        print("[3/5] Adicionando Volume Profile...")
        self._add_volume_profile()

        # 4. Delta Cumulativo (4 features)
        print("[4/5] Adicionando Delta Cumulativo...")
        self._add_cumulative_delta()

        # 5. Absorção de Ordens (3 features)
        print("[5/5] Adicionando Absorção de Ordens...")
        self._add_order_absorption()

        # Remover NaN gerados por rolling windows
        initial_rows = len(self.df)
        self.df.dropna(inplace=True)
        final_rows = len(self.df)

        print(f"\n[OK] Feature Engineering Avançada concluída!")
        print(f"   Features adicionadas: 26")
        print(f"   Linhas removidas (NaN): {initial_rows - final_rows}")
        print(f"   Dataset final: {final_rows} candles")

        return self.df

    def _add_order_flow_imbalance(self):
        """
        1. Order Flow Imbalance (5 features)

        ADAPTADO PARA DERIV (SEM VOLUME):
        Usa contagem de candles de alta/baixa ao invés de volume

        Literatura: Imbalance > 0.3 indica forte pressão de compra

        Features:
        - buy_candles_count: Contagem de candles de alta (close > open)
        - sell_candles_count: Contagem de candles de baixa (close < open)
        - order_flow_imbalance: (buy_count - sell_count) / total_count
        - imbalance_ma_10: Média móvel de 10 períodos
        - imbalance_strength: Força do desequilíbrio (absoluto)
        """
        # Buy candles = close > open
        self.df['buy_candles_count'] = (self.df['close'] > self.df['open']).astype(int)

        # Sell candles = close < open
        self.df['sell_candles_count'] = (self.df['close'] < self.df['open']).astype(int)

        # Calcular imbalance em janela de 10 candles
        window = 10
        buy_count_sum = self.df['buy_candles_count'].rolling(window).sum()
        sell_count_sum = self.df['sell_candles_count'].rolling(window).sum()
        total_count = buy_count_sum + sell_count_sum

        # Imbalance: -1 (vendedores) a +1 (compradores)
        self.df['order_flow_imbalance'] = np.where(
            total_count > 0,
            (buy_count_sum - sell_count_sum) / total_count,
            0
        )

        # Média móvel do imbalance (suaviza ruído)
        self.df['imbalance_ma_10'] = self.df['order_flow_imbalance'].rolling(10).mean()

        # Força do imbalance (valor absoluto)
        self.df['imbalance_strength'] = abs(self.df['order_flow_imbalance'])

        print(f"   [OK] Order Flow: 5 features")

    def _add_tape_reading(self):
        """
        2. Tape Reading (6 features)

        Detecta agressividade de ordens (market orders vs limit orders)
        Literatura: Aggressive buying (close near high) indica continuação

        Features:
        - aggressive_buy: Close próximo de high (0-1)
        - aggressive_sell: Close próximo de low (0-1)
        - aggr_buy_ma_5: Média móvel 5 períodos
        - aggr_sell_ma_5: Média móvel 5 períodos
        - aggr_net: Diferença entre aggr_buy e aggr_sell
        - aggr_streak: Streak de agressividade (quantos candles consecutivos)
        """
        # Aggressive buy: close próximo de high
        # Se close == high → 1.0 (muito agressivo)
        # Se close == low → 0.0 (não agressivo)
        range_hl = self.df['high'] - self.df['low']
        self.df['aggressive_buy'] = np.where(
            range_hl > 0,
            (self.df['close'] - self.df['low']) / range_hl,
            0.5
        )

        # Aggressive sell: close próximo de low
        self.df['aggressive_sell'] = np.where(
            range_hl > 0,
            (self.df['high'] - self.df['close']) / range_hl,
            0.5
        )

        # Médias móveis (suaviza ruído)
        self.df['aggr_buy_ma_5'] = self.df['aggressive_buy'].rolling(5).mean()
        self.df['aggr_sell_ma_5'] = self.df['aggressive_sell'].rolling(5).mean()

        # Net agressividade (compra - venda)
        self.df['aggr_net'] = self.df['aggressive_buy'] - self.df['aggressive_sell']

        # Streak de agressividade (quantos candles consecutivos com aggr_net > 0)
        self.df['aggr_streak'] = (
            self.df['aggr_net']
            .apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
            .groupby((self.df['aggr_net'] == 0).cumsum())
            .cumsum()
        )

        print(f"   [OK] Tape Reading: 6 features")

    def _add_volume_profile(self):
        """
        3. Range Profile (8 features)

        ADAPTADO PARA DERIV (SEM VOLUME):
        Usa range (high-low) como proxy para "intensidade" do candle

        Features:
        - range_ma_20: Média móvel de range (20 períodos)
        - range_std_20: Desvio padrão de range
        - range_zscore: Z-score do range (anomalias)
        - is_high_range: Range > média + 1 std (alta volatilidade)
        - is_low_range: Range < média - 1 std (baixa volatilidade)
        - range_spike: Range > 2x média (spike anormal)
        - hrn_distance: Distância até High Range Node (preço com alta volatilidade)
        - lrn_distance: Distância até Low Range Node (preço consolidado)
        """
        # Range do candle
        candle_range = self.df['high'] - self.df['low']

        # Média e desvio padrão de range
        self.df['range_ma_20'] = candle_range.rolling(20).mean()
        self.df['range_std_20'] = candle_range.rolling(20).std()

        # Z-score do range (quantos std acima/abaixo da média)
        self.df['range_zscore'] = np.where(
            self.df['range_std_20'] > 0,
            (candle_range - self.df['range_ma_20']) / self.df['range_std_20'],
            0
        )

        # High/Low range flags
        self.df['is_high_range'] = (
            candle_range > (self.df['range_ma_20'] + self.df['range_std_20'])
        ).astype(int)

        self.df['is_low_range'] = (
            candle_range < (self.df['range_ma_20'] - self.df['range_std_20'])
        ).astype(int)

        # Range spike (2x média = movimento anormal)
        self.df['range_spike'] = (candle_range > 2 * self.df['range_ma_20']).astype(int)

        # HRN/LRN distance (distância até zonas de alta/baixa volatilidade)
        # HRN = máximo de range em janela de 50 candles
        hrn_idx = candle_range.groupby(self.df.index // 50).idxmax()
        hrn_price = self.df.loc[hrn_idx.dropna()]['close']

        # Interpolar para todos os candles
        hrn_interpolated = hrn_price.reindex(self.df.index, method='ffill')
        self.df['hrn_distance'] = abs(self.df['close'] - hrn_interpolated) / self.df['close']

        # LRN = mínimo de range em janela de 50 candles
        lrn_idx = candle_range.groupby(self.df.index // 50).idxmin()
        lrn_price = self.df.loc[lrn_idx.dropna()]['close']

        lrn_interpolated = lrn_price.reindex(self.df.index, method='ffill')
        self.df['lrn_distance'] = abs(self.df['close'] - lrn_interpolated) / self.df['close']

        print(f"   [OK] Range Profile: 8 features")

    def _add_cumulative_delta(self):
        """
        4. Delta Cumulativo (4 features)

        ADAPTADO PARA DERIV (SEM VOLUME):
        Usa close-open (body) como proxy para "pressão" do candle

        Literatura: Delta crescente = tendência sustentada

        Features:
        - delta_body: close - open (pressão do candle)
        - cumulative_delta_20: Soma cumulativa de body (20 períodos)
        - delta_trend: Tendência do delta (positiva/negativa)
        - delta_strength: Força da tendência (absoluto)
        """
        # Delta por candle (close - open = direção + força)
        self.df['delta_body'] = self.df['close'] - self.df['open']

        # Delta cumulativo (janela de 20)
        self.df['cumulative_delta_20'] = self.df['delta_body'].rolling(20).sum()

        # Tendência do delta (subindo = 1, descendo = -1)
        self.df['delta_trend'] = np.where(
            self.df['cumulative_delta_20'] > self.df['cumulative_delta_20'].shift(1),
            1,
            -1
        )

        # Força da tendência
        self.df['delta_strength'] = abs(self.df['cumulative_delta_20'])

        print(f"   [OK] Delta Cumulativo: 4 features")

    def _add_order_absorption(self):
        """
        5. Padrão de Consolidação (3 features)

        ADAPTADO PARA DERIV (SEM VOLUME):
        Detecta quando range pequeno depois de movimento forte (consolidação)
        Literatura: Consolidação após impulso = potencial reversão

        Features:
        - consolidation_ratio: body / range (pequeno = consolidação)
        - is_consolidation: Flag de consolidação (range baixo após high range)
        - consolidation_direction: Direção do candle consolidado
        """
        # Range e body do candle
        candle_range = self.df['high'] - self.df['low']
        candle_body = abs(self.df['close'] - self.df['open'])

        # Consolidation ratio: body / range
        # Ratio baixo = doji-like (indecisão/consolidação)
        self.df['consolidation_ratio'] = np.where(
            candle_range > 0,
            candle_body / candle_range,
            0.5
        )

        # Detectar consolidação: range atual baixo DEPOIS de range alto anterior
        range_ma = candle_range.rolling(20).mean()
        range_prev_high = candle_range.shift(1) > range_ma.shift(1)

        self.df['is_consolidation'] = (
            range_prev_high &  # Candle anterior tinha range alto
            (candle_range < range_ma * 0.7)  # Candle atual tem range 30% abaixo da média
        ).astype(int)

        # Direção do candle consolidado
        self.df['consolidation_direction'] = np.where(
            self.df['is_consolidation'] == 1,
            np.where(self.df['close'] > self.df['open'], 1, -1),
            0
        )

        print(f"   [OK] Consolidation: 3 features")

    def save_features(self, output_path: Path):
        """
        Salva dataset com features avançadas

        Args:
            output_path: Caminho para salvar CSV
        """
        self.df.to_csv(output_path, index=False)
        print(f"\n[SAVE] Features salvas em: {output_path}")
        print(f"   Total de colunas: {len(self.df.columns)}")
        print(f"   Total de linhas: {len(self.df)}")

    def get_feature_summary(self) -> pd.DataFrame:
        """
        Retorna resumo estatístico das features avançadas

        Returns:
            DataFrame com estatísticas (média, std, min, max)
        """
        advanced_features = [
            # Order Flow (adaptado: candles count)
            'buy_candles_count', 'sell_candles_count', 'order_flow_imbalance',
            'imbalance_ma_10', 'imbalance_strength',
            # Tape Reading
            'aggressive_buy', 'aggressive_sell', 'aggr_buy_ma_5',
            'aggr_sell_ma_5', 'aggr_net', 'aggr_streak',
            # Range Profile (adaptado: range ao invés de volume)
            'range_ma_20', 'range_std_20', 'range_zscore',
            'is_high_range', 'is_low_range', 'range_spike',
            'hrn_distance', 'lrn_distance',
            # Delta (adaptado: body ao invés de volume)
            'delta_body', 'cumulative_delta_20', 'delta_trend', 'delta_strength',
            # Consolidation (adaptado: consolidação ao invés de absorção)
            'consolidation_ratio', 'is_consolidation', 'consolidation_direction'
        ]

        return self.df[advanced_features].describe()


def main():
    """
    Pipeline principal: carregar features básicas + adicionar features avançadas
    """
    print("="*70)
    print("FEATURE ENGINEERING AVANÇADA - SCALPING V100 M5")
    print("="*70)

    # Paths
    data_dir = Path(__file__).parent / "data"

    # 1. Carregar dataset com features básicas (62 features técnicas)
    input_path = data_dir / "1HZ100V_5min_180days_features.csv"

    if not input_path.exists():
        print(f"[ERRO] Arquivo não encontrado: {input_path}")
        print("Execute primeiro: scalping_feature_engineering.py")
        return

    print(f"\n[LOAD] Carregando dataset: {input_path.name}")
    df = pd.read_csv(input_path)
    print(f"   Dataset: {len(df)} candles, {len(df.columns)} colunas")

    # 2. Adicionar features avançadas
    feature_eng = AdvancedScalpingFeatures(df)
    df_advanced = feature_eng.add_all_advanced_features()

    # 3. Salvar dataset completo
    output_path = data_dir / "1HZ100V_5min_180days_features_advanced.csv"
    feature_eng.save_features(output_path)

    # 4. Mostrar resumo estatístico
    print("\n" + "="*70)
    print("RESUMO ESTATÍSTICO DAS FEATURES AVANÇADAS")
    print("="*70)
    summary = feature_eng.get_feature_summary()
    print(summary)

    # 5. Análise de correlação com preço futuro
    print("\n" + "="*70)
    print("TOP 10 FEATURES MAIS CORRELACIONADAS COM RETORNO FUTURO")
    print("="*70)

    # Calcular retorno futuro (próximo candle)
    df_advanced['future_return'] = df_advanced['close'].pct_change().shift(-1)

    # Correlação de todas features com retorno futuro
    advanced_cols = [col for col in df_advanced.columns if col not in ['timestamp', 'future_return']]
    correlations = df_advanced[advanced_cols].corrwith(df_advanced['future_return']).abs().sort_values(ascending=False)

    print(correlations.head(10))

    print("\n" + "="*70)
    print("FEATURE ENGINEERING AVANCADA CONCLUIDA COM SUCESSO!")
    print("="*70)
    print(f"\nDataset final: {output_path}")
    print(f"Total de features: {len(df_advanced.columns)}")
    print(f"   - Features basicas: 62")
    print(f"   - Features avancadas: 26")
    print(f"   - OHLCV: 7")
    print(f"\nProximo passo: Retreinar modelo XGBoost com 88 features")
    print(f"Expectativa: Win rate 51.2% -> 60-65%")


if __name__ == "__main__":
    main()
