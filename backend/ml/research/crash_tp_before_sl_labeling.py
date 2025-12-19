"""
TP Before SL Labeling para CRASH 500

OBJETIVO: Prever "TP será atingido antes de SL?"

TARGET CORRETO (alinhado com execução real):
- Label = 1: TP (2%) atingido antes de SL (1%) ou timeout (20 candles)
- Label = 0: SL ou timeout atingido antes de TP

DIFERENÇA vs Survival Analysis:
- Survival: Prever "candles até crash" (regressão, não relacionado com lucro)
- TP Before SL: Prever "lucro ou perda" (classificação, direto ao ponto)

ESTRATÉGIA:
- Se modelo prever >= 70% chance de TP: ENTRAR LONG
- Se modelo prever < 70% chance de TP: FICAR FORA
- Win rate esperado: 60-70% (se modelo aprende os padrões)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

class TPBeforeSLLabeler:
    """
    Cria labels binários: 1 = TP atingido antes de SL, 0 = caso contrário
    """
    def __init__(
        self,
        df,
        tp_pct=2.0,           # Take Profit em %
        sl_pct=1.0,           # Stop Loss em %
        max_hold_candles=20,  # Timeout em candles
        slippage_pct=0.1,     # Slippage de execução
        latency_candles=1,    # Delay de entrada
    ):
        """
        Args:
            df: DataFrame com OHLC do CRASH 500
            tp_pct: Take profit em % (default 2%)
            sl_pct: Stop loss em % (default 1%)
            max_hold_candles: Timeout em candles (default 20)
            slippage_pct: Slippage em % (default 0.1%)
            latency_candles: Candles de delay (default 1)
        """
        self.df = df.copy()
        self.tp_pct = tp_pct / 100.0
        self.sl_pct = sl_pct / 100.0
        self.max_hold_candles = max_hold_candles
        self.slippage_pct = slippage_pct / 100.0
        self.latency_candles = latency_candles

    def label_single_trade(self, entry_idx):
        """
        Simula um trade e retorna label binário

        Returns:
            1 = TP atingido antes de SL/timeout
            0 = SL ou timeout atingido antes de TP
        """
        # 1. Entry com latência + slippage
        entry_candle_idx = entry_idx + self.latency_candles

        # Se não há candles suficientes no futuro, retornar None
        if entry_candle_idx >= len(self.df):
            return None

        entry_price = self.df.iloc[entry_candle_idx]['close']
        entry_price_with_slippage = entry_price * (1 + self.slippage_pct)

        # 2. Calcular SL e TP
        sl_price = entry_price_with_slippage * (1 - self.sl_pct)
        tp_price = entry_price_with_slippage * (1 + self.tp_pct)

        # 3. Simular tick-by-tick até max_hold_candles
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

        # Timeout: verificar se fechou com lucro ou prejuízo
        exit_price = self.df.iloc[max_exit_idx]['close']

        if exit_price > entry_price_with_slippage:
            return 1  # Timeout com lucro = WIN
        else:
            return 0  # Timeout com prejuízo = LOSS

    def generate_labels(self):
        """
        Gera labels binários para todos os candles

        Returns:
            DataFrame com coluna 'tp_before_sl' (0 ou 1)
        """
        print(f"\n[LABELING] Gerando labels TP Before SL...")
        print(f"  TP: {self.tp_pct*100:.1f}% | SL: {self.sl_pct*100:.1f}% | Timeout: {self.max_hold_candles} candles")

        # Inicializar coluna
        self.df['tp_before_sl'] = -1

        # Total de candles a processar
        max_processable = len(self.df) - self.max_hold_candles - self.latency_candles - 1

        # Gerar labels com barra de progresso
        valid_labels = 0
        for idx in tqdm(range(max_processable), desc="  Labeling"):
            label = self.label_single_trade(idx)

            if label is not None:
                self.df.loc[idx, 'tp_before_sl'] = label
                valid_labels += 1

        # Estatísticas
        wins = (self.df['tp_before_sl'] == 1).sum()
        losses = (self.df['tp_before_sl'] == 0).sum()
        total_valid = wins + losses

        if total_valid > 0:
            win_rate = wins / total_valid * 100
        else:
            win_rate = 0

        print(f"\n  Labels gerados: {total_valid:,}")
        print(f"  Wins (TP hit): {wins:,} ({win_rate:.2f}%)")
        print(f"  Losses (SL/timeout): {losses:,} ({100-win_rate:.2f}%)")

        # Remover linhas sem label
        self.df = self.df[self.df['tp_before_sl'] != -1].reset_index(drop=True)

        return self.df

    def add_features(self):
        """
        Adiciona features relevantes para o modelo

        Features:
        - OHLC normalizado
        - Volatilidade realizada (rolling std)
        - RSI
        - ATR (Average True Range)
        - Volume ratio
        """
        print(f"\n[FEATURES] Adicionando features...")

        # 1. Retornos
        self.df['return'] = self.df['close'].pct_change()

        # 2. Volatilidade realizada (20 candles)
        self.df['realized_vol'] = self.df['return'].rolling(window=20).std()

        # 3. RSI (14 candles)
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        self.df['rsi'] = 100 - (100 / (1 + rs))

        # 4. ATR (Average True Range, 14 candles)
        high_low = self.df['high'] - self.df['low']
        high_close = abs(self.df['high'] - self.df['close'].shift())
        low_close = abs(self.df['low'] - self.df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df['atr'] = true_range.rolling(window=14).mean()

        # 5. Remover NaNs
        self.df = self.df.dropna().reset_index(drop=True)

        print(f"  Features adicionadas: realized_vol, rsi, atr")
        print(f"  Candles após remover NaNs: {len(self.df):,}")

        return self.df

    def visualize_distribution(self, save_path=None):
        """
        Visualiza distribuição de labels e features
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Label distribution
        ax1 = axes[0, 0]
        label_counts = self.df['tp_before_sl'].value_counts()
        ax1.bar(['LOSS (0)', 'WIN (1)'], [label_counts.get(0, 0), label_counts.get(1, 0)])
        ax1.set_title('Label Distribution (TP Before SL)')
        ax1.set_ylabel('Count')
        ax1.grid(alpha=0.3)

        # Plot 2: Price evolution
        ax2 = axes[0, 1]
        sample_size = min(1000, len(self.df))
        sample_df = self.df.iloc[:sample_size]
        win_candles = sample_df[sample_df['tp_before_sl'] == 1]
        loss_candles = sample_df[sample_df['tp_before_sl'] == 0]

        ax2.plot(sample_df.index, sample_df['close'], label='Price', linewidth=0.8, alpha=0.5)
        ax2.scatter(win_candles.index, win_candles['close'],
                   color='green', s=10, alpha=0.5, label=f'WIN ({len(win_candles)})')
        ax2.scatter(loss_candles.index, loss_candles['close'],
                   color='red', s=10, alpha=0.5, label=f'LOSS ({len(loss_candles)})')
        ax2.set_title('Price + Labels (First 1000 candles)')
        ax2.set_xlabel('Candle Index')
        ax2.set_ylabel('Price')
        ax2.legend()
        ax2.grid(alpha=0.3)

        # Plot 3: Volatility distribution
        ax3 = axes[1, 0]
        self.df.boxplot(column='realized_vol', by='tp_before_sl', ax=ax3)
        ax3.set_title('Realized Volatility by Label')
        ax3.set_xlabel('Label (0=LOSS, 1=WIN)')
        ax3.set_ylabel('Realized Vol')

        # Plot 4: RSI distribution
        ax4 = axes[1, 1]
        self.df.boxplot(column='rsi', by='tp_before_sl', ax=ax4)
        ax4.set_title('RSI by Label')
        ax4.set_xlabel('Label (0=LOSS, 1=WIN)')
        ax4.set_ylabel('RSI')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n[PLOT] Salvo em: {save_path}")

        return fig

def main():
    print("="*70)
    print("CRASH 500 - TP BEFORE SL LABELING (TARGET CORRETO)")
    print("="*70)

    # Carregar dados
    data_dir = Path(__file__).parent / "data"
    input_file = data_dir / "CRASH500_5min_180days.csv"

    print(f"\n[LOAD] Carregando {input_file}...")
    df = pd.read_csv(input_file)
    print(f"  Candles originais: {len(df):,}")

    # Criar labeler
    labeler = TPBeforeSLLabeler(
        df,
        tp_pct=2.0,           # TP 2%
        sl_pct=1.0,           # SL 1%
        max_hold_candles=20,  # Timeout 20 candles (100 min em M5)
        slippage_pct=0.1,     # Slippage 0.1%
        latency_candles=1,    # Delay 1 candle
    )

    # Gerar labels
    df_labeled = labeler.generate_labels()

    # Adicionar features
    df_labeled = labeler.add_features()

    # Visualizar
    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    plot_path = reports_dir / "crash500_tp_before_sl_distribution.png"
    labeler.visualize_distribution(save_path=plot_path)

    # Salvar dataset
    output_file = data_dir / "CRASH500_5min_tp_before_sl_labeled.csv"
    df_labeled.to_csv(output_file, index=False)

    print(f"\n[OK] Dataset salvo!")
    print(f"  Arquivo: {output_file}")
    print(f"  Colunas: {list(df_labeled.columns)}")
    print(f"  Total candles: {len(df_labeled):,}")

    # Análise final
    print(f"\n{'='*70}")
    print(f"ESTRATÉGIA DE TRADING")
    print(f"{'='*70}")

    wins = (df_labeled['tp_before_sl'] == 1).sum()
    losses = (df_labeled['tp_before_sl'] == 0).sum()
    win_rate = wins / (wins + losses) * 100

    print(f"\n  Win Rate Natural (SL/TP/Timeout): {win_rate:.2f}%")
    print(f"  Total Wins: {wins:,}")
    print(f"  Total Losses: {losses:,}")

    print(f"\n  OBJETIVO DO MODELO:")
    print(f"    - Prever labels com acurácia > 70%")
    print(f"    - Apenas entrar em trades com P(WIN) >= 70%")
    print(f"    - Win rate esperado em produção: 60-70%")

    print(f"\n  DIFERENÇA vs Survival Analysis:")
    print(f"    ❌ Survival: Prever 'candles até crash' (não garante lucro)")
    print(f"    ✅ TP Before SL: Prever 'lucro ou perda' (direto ao ponto)")

    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()
