"""
Gera labels TP Before SL para múltiplos ativos sintéticos

Processa BOOM500, CRASH500, CRASH1000 e calcula win rate natural de cada um
para determinar qual ativo é melhor para scalping ML.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

class UniversalTPBeforeSLLabeler:
    """
    Labeler universal para qualquer ativo sintético
    """
    def __init__(
        self,
        df,
        tp_pct=2.0,
        sl_pct=1.0,
        max_hold_candles=20,
        slippage_pct=0.1,
        latency_candles=1,
    ):
        self.df = df.copy()
        self.tp_pct = tp_pct / 100.0
        self.sl_pct = sl_pct / 100.0
        self.max_hold_candles = max_hold_candles
        self.slippage_pct = slippage_pct / 100.0
        self.latency_candles = latency_candles

    def label_single_trade(self, entry_idx):
        """
        Simula um trade LONG e retorna label binário

        Returns:
            1 = TP hit (WIN)
            0 = SL ou timeout hit (LOSS)
            None = não há candles suficientes
        """
        # Entry com latência + slippage
        entry_candle_idx = entry_idx + self.latency_candles

        if entry_candle_idx >= len(self.df):
            return None

        entry_price = self.df.iloc[entry_candle_idx]['close']
        entry_price_with_slippage = entry_price * (1 + self.slippage_pct)

        # Calcular SL e TP
        sl_price = entry_price_with_slippage * (1 - self.sl_pct)
        tp_price = entry_price_with_slippage * (1 + self.tp_pct)

        # Simular tick-by-tick
        max_exit_idx = min(
            entry_candle_idx + self.max_hold_candles,
            len(self.df) - 1
        )

        for j in range(entry_candle_idx + 1, max_exit_idx + 1):
            candle = self.df.iloc[j]

            # TP hit? (WIN)
            if candle['high'] >= tp_price:
                return 1

            # SL hit? (LOSS)
            if candle['low'] <= sl_price:
                return 0

        # Timeout: fechar na close do último candle
        exit_price = self.df.iloc[max_exit_idx]['close']

        if exit_price > entry_price_with_slippage:
            return 1  # Timeout com lucro
        else:
            return 0  # Timeout com prejuízo

    def generate_labels(self):
        """
        Gera labels para todos os candles
        """
        self.df['tp_before_sl'] = -1

        max_processable = len(self.df) - self.max_hold_candles - self.latency_candles - 1

        for idx in tqdm(range(max_processable), desc="  Labeling"):
            label = self.label_single_trade(idx)

            if label is not None:
                self.df.loc[idx, 'tp_before_sl'] = label

        # Remover linhas sem label
        self.df = self.df[self.df['tp_before_sl'] != -1].reset_index(drop=True)

        return self.df

    def add_features(self):
        """
        Adiciona features técnicas
        """
        # Retornos
        self.df['return'] = self.df['close'].pct_change()

        # Volatilidade realizada
        self.df['realized_vol'] = self.df['return'].rolling(window=20).std()

        # RSI
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        self.df['rsi'] = 100 - (100 / (1 + rs))

        # ATR
        high_low = self.df['high'] - self.df['low']
        high_close = abs(self.df['high'] - self.df['close'].shift())
        low_close = abs(self.df['low'] - self.df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df['atr'] = true_range.rolling(window=14).mean()

        # Remover NaNs
        self.df = self.df.dropna().reset_index(drop=True)

        return self.df


def process_asset(asset_name, input_file, tp_pct=2.0, sl_pct=1.0, max_hold=20):
    """
    Processa um ativo e retorna estatísticas
    """
    print("="*70)
    print(f"PROCESSANDO: {asset_name}")
    print("="*70)

    # Carregar dados
    print(f"\n[LOAD] {input_file.name}")
    df = pd.read_csv(input_file)
    print(f"  Candles: {len(df):,}")

    # Labeling
    print(f"\n[LABELING] TP {tp_pct:.1f}% | SL {sl_pct:.1f}% | Timeout {max_hold} candles")
    labeler = UniversalTPBeforeSLLabeler(
        df,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        max_hold_candles=max_hold,
        slippage_pct=0.1,
        latency_candles=1,
    )

    df_labeled = labeler.generate_labels()

    # Features
    print(f"\n[FEATURES] Adicionando features...")
    df_labeled = labeler.add_features()

    # Estatísticas
    wins = (df_labeled['tp_before_sl'] == 1).sum()
    losses = (df_labeled['tp_before_sl'] == 0).sum()
    total = wins + losses
    win_rate = wins / total * 100 if total > 0 else 0

    print(f"\n[RESULTADOS]")
    print(f"  Total labels: {total:,}")
    print(f"  WINS (TP hit): {wins:,}")
    print(f"  LOSSES (SL/timeout): {losses:,}")
    print(f"  WIN RATE NATURAL: {win_rate:.2f}%")

    # Salvar
    output_file = input_file.parent / f"{asset_name}_tp_before_sl_labeled.csv"
    df_labeled.to_csv(output_file, index=False)
    print(f"\n[SAVE] {output_file.name}")

    return {
        'asset': asset_name,
        'candles': len(df_labeled),
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'output_file': output_file,
    }


def main():
    data_dir = Path(__file__).parent / "data"

    # Lista de ativos a processar
    assets = [
        ('CRASH500_M5', data_dir / "CRASH500_5min_180days.csv"),
        ('CRASH1000_M5', data_dir / "CRASH1000_M5_180days.csv"),
        ('BOOM500_M5', data_dir / "BOOM500_M5_180days.csv"),
    ]

    results = []

    for asset_name, input_file in assets:
        if not input_file.exists():
            print(f"\n[SKIP] {asset_name} - Arquivo não encontrado: {input_file}")
            continue

        try:
            result = process_asset(asset_name, input_file)
            results.append(result)
        except Exception as e:
            print(f"\n[ERRO] {asset_name}: {e}")

        print("\n")

    # Relatório comparativo
    if results:
        print("="*70)
        print("RESUMO COMPARATIVO - WIN RATE NATURAL")
        print("="*70)

        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('win_rate', ascending=False)

        print(f"\n{df_results[['asset', 'win_rate', 'wins', 'losses']].to_string(index=False)}")

        best = df_results.iloc[0]
        print(f"\n\nMELHOR ATIVO PARA ML SCALPING:")
        print(f"  {best['asset']} - Win Rate: {best['win_rate']:.2f}%")

        if best['win_rate'] >= 45:
            print(f"\n  STATUS: APROVADO para treinamento de modelo ML")
            print(f"  Expectativa: Modelo pode aprender padrões com WR >= 45%")
        else:
            print(f"\n  STATUS: REPROVADO para ML")
            print(f"  Win rate < 45% indica que o mercado não favorece scalping")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
