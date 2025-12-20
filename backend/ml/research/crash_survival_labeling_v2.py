"""
CRASH1000 - Survival Analysis (Deteccao de Crash)

NOVA ABORDAGEM: Prever EVENTO (crash) em vez de PRECO (TP/SL)

MOTIVACAO:
- CRASH1000 e um processo de contagem (sobe 1 tick ate crashar)
- Features OHLC/RSI/ATR tem correlacao ~0.02 com TP/SL (quase zero)
- Crash e um EVENTO DISCRETO (mais facil de prever que tendencia suave)

ESTRATEGIA:
1. Prever: "Vai crashar nos proximos N candles?" (Classificacao Binaria)
2. Se P(Crash) < 20%, entrar LONG e segurar por N/2 candles
3. Se P(Crash) > 20%, ficar de fora (Flat)

FEATURES ESPECIALIZADAS:
- ticks_since_last_crash: Tempo desde ultimo crash
- crash_size_lag1: Tamanho do ultimo crash
- avg_tick_velocity: Velocidade da subida
- acceleration: Mudanca na velocidade
"""
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

class CrashSurvivalLabeler:
    """
    Labeler para Survival Analysis em CRASH1000

    Target: crashed_in_next_N (1 = crash nos proximos N candles, 0 = caso contrario)
    """
    def __init__(self, df, lookforward=10, crash_threshold_pct=5.0):
        """
        Args:
            df: DataFrame com OHLC do CRASH1000
            lookforward: Janela de predicao (default 10 candles = 50 min M5)
            crash_threshold_pct: % de queda para considerar "crash" (default 5%)
        """
        self.df = df.copy()
        self.lookforward = lookforward
        self.crash_threshold = crash_threshold_pct / 100.0

    def detect_crashes(self):
        """
        Detecta crashes (quedas abruptas > threshold%)

        METODOS:
        1. Close-to-close (retorno do candle)
        2. Wick (high-low range dentro do candle)
        3. Shadow (low muito abaixo do open/close)

        Returns:
            DataFrame com coluna 'is_crash' (1 = crash, 0 = normal)
        """
        print(f"\n[CRASH DETECTION] Detectando crashes...")
        print(f"  Threshold: {self.crash_threshold*100:.1f}%")

        # Calcular retornos (close-to-close)
        self.df['return'] = self.df['close'].pct_change()

        # METODO 1: Close-to-close
        self.df['is_crash_close'] = (self.df['return'] < -self.crash_threshold).astype(int)

        # METODO 2: Wick (high-low range) - Detecta crash DENTRO do candle
        self.df['wick_range'] = (self.df['high'] - self.df['low']) / self.df['high']
        self.df['is_crash_wick'] = (self.df['wick_range'] > self.crash_threshold).astype(int)

        # METODO 3: Shadow (low muito abaixo de open/close) - Detecta spike DOWN
        self.df['body_midpoint'] = (self.df['open'] + self.df['close']) / 2
        self.df['lower_shadow'] = (self.df['body_midpoint'] - self.df['low']) / self.df['body_midpoint']
        self.df['is_crash_shadow'] = (self.df['lower_shadow'] > self.crash_threshold/2).astype(int)

        # Crash = Qualquer um dos 3 metodos detectar
        self.df['is_crash'] = (
            (self.df['is_crash_close'] == 1) |
            (self.df['is_crash_wick'] == 1) |
            (self.df['is_crash_shadow'] == 1)
        ).astype(int)

        # Estatisticas
        n_crashes_close = self.df['is_crash_close'].sum()
        n_crashes_wick = self.df['is_crash_wick'].sum()
        n_crashes_shadow = self.df['is_crash_shadow'].sum()
        n_crashes_total = self.df['is_crash'].sum()

        print(f"\n  Crashes detectados por metodo:")
        print(f"    Close-to-close: {n_crashes_close:,} ({n_crashes_close/len(self.df)*100:.2f}%)")
        print(f"    Wick (high-low): {n_crashes_wick:,} ({n_crashes_wick/len(self.df)*100:.2f}%)")
        print(f"    Shadow (spike): {n_crashes_shadow:,} ({n_crashes_shadow/len(self.df)*100:.2f}%)")
        print(f"    TOTAL (union): {n_crashes_total:,} ({n_crashes_total/len(self.df)*100:.2f}%)")

        # Estatisticas dos crashes
        crash_candles = self.df[self.df['is_crash'] == 1]
        if len(crash_candles) > 0:
            avg_wick = crash_candles['wick_range'].mean() * 100
            max_wick = crash_candles['wick_range'].max() * 100
            print(f"\n  Tamanho medio do wick (crashes): {avg_wick:.2f}%")
            print(f"  Maior wick (crash): {max_wick:.2f}%")

        return self.df

    def label_crash_in_next_n(self):
        """
        Cria label: crashed_in_next_N (1 = crash nos proximos N candles)
        """
        print(f"\n[LABELING] Criando label 'crashed_in_next_{self.lookforward}'...")

        self.df[f'crashed_in_next_{self.lookforward}'] = 0

        # Para cada candle, verificar se ha crash nos proximos N
        for i in tqdm(range(len(self.df) - self.lookforward), desc="  Labeling"):
            # Janela futura
            future_window = self.df.iloc[i+1:i+1+self.lookforward]

            # Ha crash na janela?
            if future_window['is_crash'].sum() > 0:
                self.df.loc[i, f'crashed_in_next_{self.lookforward}'] = 1

        # Remover ultimos N candles (nao tem janela completa)
        self.df = self.df.iloc[:-self.lookforward].copy()

        # Estatisticas
        n_crash_labels = self.df[f'crashed_in_next_{self.lookforward}'].sum()
        n_safe_labels = len(self.df) - n_crash_labels
        crash_rate = n_crash_labels / len(self.df) * 100

        print(f"\n  Labels criados:")
        print(f"    CRASH (1): {n_crash_labels:,} ({crash_rate:.2f}%)")
        print(f"    SAFE (0): {n_safe_labels:,} ({100-crash_rate:.2f}%)")

        return self.df

    def add_crash_specific_features(self):
        """
        Adiciona features especializadas para deteccao de crash
        """
        print(f"\n[FEATURES] Adicionando features crash-specific...")

        # 1. Ticks since last crash
        self.df['ticks_since_crash'] = 0
        counter = 0
        for i in range(len(self.df)):
            if self.df.iloc[i]['is_crash'] == 1:
                counter = 0
            else:
                counter += 1
            self.df.loc[i, 'ticks_since_crash'] = counter

        # 2. Crash size lag (tamanho do ultimo crash)
        crash_sizes = []
        last_crash_size = 0
        for i in range(len(self.df)):
            if self.df.iloc[i]['is_crash'] == 1:
                last_crash_size = self.df.iloc[i]['return']
            crash_sizes.append(last_crash_size)
        self.df['crash_size_lag1'] = crash_sizes

        # 3. Tick velocity (velocidade da subida)
        self.df['tick_velocity'] = self.df['close'].diff().rolling(window=5).mean()

        # 4. Acceleration (mudanca na velocidade)
        self.df['acceleration'] = self.df['tick_velocity'].diff()

        # 5. Volatilidade rolling (20 candles)
        self.df['rolling_volatility'] = self.df['return'].rolling(window=20).std()

        # 6. Price deviation from MA (distancia da media movel)
        self.df['ma_20'] = self.df['close'].rolling(window=20).mean()
        self.df['price_deviation'] = (self.df['close'] - self.df['ma_20']) / self.df['ma_20']

        # 7. Momentum (taxa de mudanca)
        self.df['momentum'] = self.df['close'].pct_change(periods=5)

        # Remover NaNs
        self.df = self.df.dropna().reset_index(drop=True)

        print(f"  Features adicionadas:")
        print(f"    - ticks_since_crash (contador)")
        print(f"    - crash_size_lag1 (tamanho do ultimo crash)")
        print(f"    - tick_velocity (velocidade da subida)")
        print(f"    - acceleration (mudanca na velocidade)")
        print(f"    - rolling_volatility (volatilidade 20 candles)")
        print(f"    - price_deviation (distancia da MA)")
        print(f"    - momentum (taxa de mudanca 5 candles)")

        print(f"\n  Total candles apos remover NaNs: {len(self.df):,}")

        return self.df

    def visualize_distribution(self, save_path=None):
        """
        Visualiza distribuicao de labels e features
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Label distribution
        ax1 = axes[0, 0]
        label_counts = self.df[f'crashed_in_next_{self.lookforward}'].value_counts()
        ax1.bar(['SAFE (0)', 'CRASH (1)'], [label_counts.get(0, 0), label_counts.get(1, 0)])
        ax1.set_title(f'Label Distribution (crashed_in_next_{self.lookforward})')
        ax1.set_ylabel('Count')
        ax1.grid(alpha=0.3)

        # Plot 2: Ticks since crash distribution
        ax2 = axes[0, 1]
        safe_candles = self.df[self.df[f'crashed_in_next_{self.lookforward}'] == 0]
        crash_candles = self.df[self.df[f'crashed_in_next_{self.lookforward}'] == 1]

        ax2.hist(safe_candles['ticks_since_crash'], bins=50, alpha=0.5, label='SAFE', color='green')
        ax2.hist(crash_candles['ticks_since_crash'], bins=50, alpha=0.5, label='CRASH', color='red')
        ax2.set_title('Ticks Since Crash Distribution')
        ax2.set_xlabel('Ticks Since Last Crash')
        ax2.set_ylabel('Count')
        ax2.legend()
        ax2.grid(alpha=0.3)

        # Plot 3: Tick velocity boxplot
        ax3 = axes[1, 0]
        data_to_plot = [
            safe_candles['tick_velocity'].dropna(),
            crash_candles['tick_velocity'].dropna()
        ]
        bp = ax3.boxplot(data_to_plot, tick_labels=['SAFE', 'CRASH'], patch_artist=True)
        bp['boxes'][0].set_facecolor('green')
        bp['boxes'][0].set_alpha(0.5)
        bp['boxes'][1].set_facecolor('red')
        bp['boxes'][1].set_alpha(0.5)
        ax3.set_title('Tick Velocity (SAFE vs CRASH)')
        ax3.set_ylabel('Velocity')
        ax3.grid(alpha=0.3)

        # Plot 4: Price evolution with crashes
        ax4 = axes[1, 1]
        sample_size = min(1000, len(self.df))
        sample_df = self.df.iloc[:sample_size]
        crash_idx = sample_df[sample_df['is_crash'] == 1].index

        ax4.plot(sample_df.index, sample_df['close'], linewidth=0.8, alpha=0.7, label='Price')
        ax4.scatter(crash_idx, sample_df.loc[crash_idx, 'close'],
                   color='red', s=30, alpha=0.7, label=f'Crash ({len(crash_idx)})')
        ax4.set_title('Price + Crashes (First 1000 candles)')
        ax4.set_xlabel('Candle Index')
        ax4.set_ylabel('Price')
        ax4.legend()
        ax4.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n[PLOT] Salvo em: {save_path}")

        return fig


def main():
    import sys

    # Detectar qual ativo e timeframe usar
    if len(sys.argv) > 1:
        asset = sys.argv[1]  # Ex: "CRASH500" ou "CRASH1000"
    else:
        asset = "CRASH500"  # Default: CRASH500

    if len(sys.argv) > 2:
        timeframe = sys.argv[2]  # Ex: "M1" ou "M5"
    else:
        timeframe = "M1"  # Default: M1

    print("="*70)
    print(f"{asset} {timeframe} - SURVIVAL ANALYSIS (CRASH DETECTION)")
    print("="*70)

    # Carregar dados
    data_dir = Path(__file__).parent / "data"
    input_file = data_dir / f"{asset}_{timeframe}_180days.csv"

    print(f"\n[LOAD] Carregando {input_file}...")
    df = pd.read_csv(input_file)
    print(f"  Candles originais: {len(df):,}")

    # Criar labeler
    labeler = CrashSurvivalLabeler(
        df,
        lookforward=10,        # Prever crash nos proximos 10 candles (50 min M5)
        crash_threshold_pct=5.0  # Queda > 5% = crash
    )

    # Detectar crashes
    df_labeled = labeler.detect_crashes()

    # Criar labels
    df_labeled = labeler.label_crash_in_next_n()

    # Adicionar features
    df_labeled = labeler.add_crash_specific_features()

    # Visualizar
    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    plot_path = reports_dir / "crash1000_survival_analysis_distribution.png"
    labeler.visualize_distribution(save_path=plot_path)

    # Salvar dataset
    output_file = data_dir / f"{asset}_{timeframe}_survival_labeled.csv"
    df_labeled.to_csv(output_file, index=False)

    print(f"\n[OK] Dataset salvo!")
    print(f"  Arquivo: {output_file}")
    print(f"  Total candles: {len(df_labeled):,}")
    print(f"  Colunas: {list(df_labeled.columns)}")

    # Analise final
    print(f"\n{'='*70}")
    print(f"ESTRATEGIA DE TRADING")
    print(f"{'='*70}")

    crash_rate = (df_labeled[f'crashed_in_next_{labeler.lookforward}'] == 1).sum() / len(df_labeled) * 100

    print(f"\n  Crash Rate (proximos 10 candles): {crash_rate:.2f}%")
    print(f"\n  ESTRATEGIA:")
    print(f"    1. Modelo preve P(Crash nos proximos 10 candles)")
    print(f"    2. Se P(Crash) < 20%, ENTRAR LONG")
    print(f"    3. Segurar por 5 candles (50% da janela)")
    print(f"    4. Sair (lucro garantido se nao crashar)")
    print(f"\n  VANTAGEM:")
    print(f"    - Crash e evento DISCRETO (mais facil de prever)")
    print(f"    - Features especializadas (ticks_since_crash, velocity)")
    print(f"    - Matematica a favor: Apostando CONTRA evento raro")

    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()
