"""
Feature Engineering para Scalping V100 M5

Adiciona 20+ indicadores técnicos robustos:
- Momentum: RSI, MACD, Stochastic
- Volatilidade: Bollinger Bands, ATR
- Tendência: ADX, EMA distances
- Microestrutura: Log returns, lagged returns, HL range

IMPORTANTE: Usa Log Returns para estacionariedade (necessário para Deep Learning)
"""
import pandas as pd
import pandas_ta as ta
import numpy as np
from pathlib import Path

def add_features(df):
    """
    Adiciona indicadores técnicos ao dataframe

    Retorna dataframe com 20+ features normalizadas
    """
    print("Gerando indicadores técnicos...")

    # 1. MOMENTUM & OSCILADORES
    print("  - RSI (7, 14)...")
    df['rsi_7'] = df.ta.rsi(length=7) / 100.0  # Normalizado 0-1
    df['rsi_14'] = df.ta.rsi(length=14) / 100.0

    print("  - MACD...")
    macd = df.ta.macd(fast=12, slow=26, signal=9)
    df['macd'] = macd['MACD_12_26_9'] / df['close']  # Normalizado pelo preço
    df['macd_signal'] = macd['MACDs_12_26_9'] / df['close']
    df['macd_hist'] = macd['MACDh_12_26_9'] / df['close']

    print("  - Stochastic...")
    stoch = df.ta.stoch()
    df['stoch_k'] = stoch['STOCHk_14_3_3'] / 100.0
    df['stoch_d'] = stoch['STOCHd_14_3_3'] / 100.0

    # 2. VOLATILIDADE (Crucial para V100)
    print("  - Bollinger Bands...")
    bb = df.ta.bbands(length=20, std=2)
    if bb is not None:
        # Encontrar nomes das colunas (podem variar por versão)
        bb_cols = bb.columns.tolist()
        upper_col = [c for c in bb_cols if 'BBU' in c][0]
        lower_col = [c for c in bb_cols if 'BBL' in c][0]

        # Largura das bandas = volatilidade
        df['bb_width'] = (bb[upper_col] - bb[lower_col]) / df['close']
        # Posição dentro das bandas (0 = banda inf, 1 = banda sup)
        df['bb_pos'] = (df['close'] - bb[lower_col]) / (bb[upper_col] - bb[lower_col])
    else:
        df['bb_width'] = 0.0
        df['bb_pos'] = 0.5

    print("  - ATR...")
    df['atr'] = df.ta.atr(length=14) / df['close']  # Normalizado

    # 3. TENDÊNCIA
    print("  - ADX...")
    adx = df.ta.adx()
    df['adx'] = adx['ADX_14'] / 100.0  # Força da tendência

    print("  - EMAs (9, 20, 50)...")
    # Distância das médias móveis (log space para estacionariedade)
    df['ema_9_dist'] = np.log(df['close'] / df.ta.ema(length=9))
    df['ema_20_dist'] = np.log(df['close'] / df.ta.ema(length=20))
    df['ema_50_dist'] = np.log(df['close'] / df.ta.ema(length=50))

    # 4. MICROESTRUTURA & RETORNOS (O que Conv1D ama!)
    print("  - Log Returns...")
    # Log Returns são ESTACIONÁRIOS (lei em Deep Learning)
    df['log_ret'] = np.log(df['close']).diff()

    # Lagged Returns (histórico de 1, 2, 3 candles atrás)
    df['lag_ret_1'] = df['log_ret'].shift(1)
    df['lag_ret_2'] = df['log_ret'].shift(2)
    df['lag_ret_3'] = df['log_ret'].shift(3)

    # High-Low Range (volatilidade intra-candle)
    df['hl_range'] = np.log(df['high'] / df['low'])

    # 5. VOLUME (se disponível)
    if 'volume' in df.columns:
        print("  - Volume features...")
        df['log_ret_vol'] = np.log(df['volume'] + 1).diff()  # +1 para evitar log(0)

    # Limpeza (remover NaNs dos indicadores)
    print("  - Removendo NaNs...")
    initial_len = len(df)
    df.dropna(inplace=True)
    final_len = len(df)
    print(f"    Removidos {initial_len - final_len} candles com NaNs")

    # Contagem de features
    feature_cols = [c for c in df.columns if c not in ['timestamp', 'label', 'label_metadata']]
    print(f"\n[OK] Features geradas! Total: {len(feature_cols)} colunas")
    print(f"   Features: {', '.join(feature_cols[:10])}...")

    return df

def process_pipeline():
    """
    Pipeline completo:
    1. Carregar dados brutos
    2. Gerar features técnicas
    3. Gerar labels pessimistas
    4. Salvar dataset enriquecido
    """
    print("="*70)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*70)

    # Caminhos
    base_dir = Path(__file__).parent / "data"
    input_file = base_dir / "1HZ100V_5min_180days.csv"
    output_file = base_dir / "1HZ100V_5min_rich_features.csv"

    # 1. Carregar
    print(f"\n[1/4] Carregando {input_file}...")
    df = pd.read_csv(input_file)
    print(f"   Shape inicial: {df.shape}")

    # 2. Gerar Features
    print(f"\n[2/4] Gerando features técnicas...")
    df_rich = add_features(df)
    print(f"   Shape após features: {df_rich.shape}")

    # 3. Labeling (Reutilizando lógica pessimista)
    print(f"\n[3/4] Gerando labels pessimistas...")
    from scalping_labeling import ScalpingLabeler
    labeler = ScalpingLabeler(df_rich, tp_pct=0.2, sl_pct=0.1, max_candles=20)
    df_labeled = labeler.generate_labels()
    print(f"   Shape após labeling: {df_labeled.shape}")

    # 4. Salvar
    print(f"\n[4/4] Salvando dataset enriquecido...")
    df_labeled.to_csv(output_file, index=False)
    print(f"   [OK] Salvo em: {output_file}")

    # Estatísticas finais
    print(f"\n{'='*70}")
    print(f"DATASET FINAL")
    print(f"{'='*70}")
    print(f"   Candles: {len(df_labeled):,}")

    # Features geradas
    feature_cols = [c for c in df_labeled.columns if c not in ['timestamp', 'label', 'label_metadata']]
    print(f"   Features: {len(feature_cols)}")
    print(f"\n   Lista de features:")
    for i, col in enumerate(feature_cols, 1):
        print(f"      {i:2d}. {col}")

    # Distribuição de labels
    label_counts = df_labeled['label'].value_counts()
    total = len(df_labeled)
    print(f"\n   Distribuição de labels:")
    print(f"      LONG (1):      {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/total*100:.1f}%)")
    print(f"      SHORT (-1):    {label_counts.get(-1, 0):,} ({label_counts.get(-1, 0)/total*100:.1f}%)")
    print(f"      NO_TRADE (0):  {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/total*100:.1f}%)")

    print(f"\n{'='*70}")
    print("[PRONTO] Agora seu modelo tem 'OLHOS'!")
    print("="*70)
    print("\nPróximos passos:")
    print("1. Modificar ScalpingDataset para usar estas features")
    print("2. Retreinar LSTM com input_dim = número de features")
    print("3. Expectativa: Win rate 55-58% (vs 54.3% atual)")

if __name__ == "__main__":
    process_pipeline()
