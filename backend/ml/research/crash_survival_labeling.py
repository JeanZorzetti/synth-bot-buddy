"""
Survival Analysis Labeling para CRASH 500

OBJETIVO: Prever "Quantos candles faltam até o próximo Crash?"

CRASH 500 Características:
- Sobe gradualmente (tick a tick)
- A cada ~500 ticks, dá um crash (queda súbita de 10-30%)
- Crash é detectável: queda > 5% em 1-2 candles

ESTRATÉGIA:
- Se modelo prever >= 20 candles até crash: ENTRAR LONG
- Se modelo prever < 20 candles: FICAR FORA (evitar crash)
- Win rate esperado: 90-95% (só perdemos se crash for imprevisível)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class CrashSurvivalLabeler:
    """
    Detecta crashes e cria labels de survival (tempo até próximo crash)
    """
    def __init__(self, df, crash_threshold_pct=5.0, min_candles_between=10):
        """
        Args:
            df: DataFrame com OHLC do CRASH 500
            crash_threshold_pct: Queda percentual para considerar crash (default 5%)
            min_candles_between: Mínimo de candles entre crashes (filtro de ruído)
        """
        self.df = df.copy()
        self.crash_threshold = crash_threshold_pct / 100.0
        self.min_candles_between = min_candles_between

    def detect_high_volatility_zones(self, window=20, vol_threshold_pct=95):
        """
        Detecta zonas de alta volatilidade (proxy para risco de crash)

        Se o período não tem crashes, usamos volatilidade como proxy de risco
        """
        print(f"\n[VOLATILITY DETECTION] Detectando zonas de alta volatilidade...")

        # Calcular retornos
        self.df['return'] = self.df['close'].pct_change()

        # Volatilidade realizada (rolling std de retornos)
        self.df['realized_vol'] = self.df['return'].rolling(window=window).std()

        # Threshold dinâmico (percentil 95)
        vol_threshold = self.df['realized_vol'].quantile(vol_threshold_pct / 100.0)

        # Marcar zonas de alta volatilidade
        self.df['high_vol_zone'] = (self.df['realized_vol'] > vol_threshold).astype(int)

        n_high_vol = self.df['high_vol_zone'].sum()
        print(f"  Threshold volatilidade: {vol_threshold*100:.4f}%")
        print(f"  Candles em zona de alta vol: {n_high_vol:,} ({n_high_vol/len(self.df)*100:.1f}%)")

        # Retornar índices de alta volatilidade
        high_vol_indices = self.df[self.df['high_vol_zone'] == 1].index.tolist()

        return high_vol_indices

    def detect_crashes(self):
        """
        Detecta eventos de crash (quedas súbitas)

        Se nenhum crash detectado, usa volatilidade como proxy
        """
        print(f"\n[CRASH DETECTION] Detectando crashes (threshold: {self.crash_threshold*100:.1f}%)...")

        # Calcular retornos
        if 'return' not in self.df.columns:
            self.df['return'] = self.df['close'].pct_change()

        # Detectar quedas súbitas
        self.df['is_crash'] = (self.df['return'] < -self.crash_threshold).astype(int)

        # Filtrar crashes muito próximos (ruído)
        crash_indices = self.df[self.df['is_crash'] == 1].index.tolist()
        filtered_crashes = []

        for i, crash_idx in enumerate(crash_indices):
            # Primeiro crash sempre válido
            if i == 0:
                filtered_crashes.append(crash_idx)
                continue

            # Checar distância do crash anterior
            if crash_idx - filtered_crashes[-1] >= self.min_candles_between:
                filtered_crashes.append(crash_idx)

        # Atualizar coluna
        self.df['is_crash'] = 0
        self.df.loc[filtered_crashes, 'is_crash'] = 1

        n_crashes = len(filtered_crashes)
        avg_interval = np.diff(filtered_crashes).mean() if n_crashes > 1 else 0

        print(f"  Total crashes detectados: {n_crashes}")

        # Se nenhum crash, usar volatilidade
        if n_crashes == 0:
            print(f"  [FALLBACK] Nenhum crash detectado. Usando volatilidade como proxy...")
            return self.detect_high_volatility_zones()

        print(f"  Intervalo médio: {avg_interval:.1f} candles")
        print(f"  Crashes por 1000 candles: {n_crashes / len(self.df) * 1000:.1f}")

        return filtered_crashes

    def generate_survival_labels(self):
        """
        Gera labels de survival: quantos candles até o próximo crash?

        Returns:
            DataFrame com coluna 'candles_to_crash'
        """
        print(f"\n[LABELING] Gerando labels de Survival Analysis...")

        # Detectar crashes
        crash_indices = self.detect_crashes()

        # Inicializar label
        self.df['candles_to_crash'] = -1

        # Para cada candle, calcular distância até próximo crash
        for idx in range(len(self.df)):
            # Encontrar próximo crash
            future_crashes = [c for c in crash_indices if c > idx]

            if future_crashes:
                next_crash = future_crashes[0]
                self.df.loc[idx, 'candles_to_crash'] = next_crash - idx
            else:
                # Sem crashes futuros (fim do dataset)
                self.df.loc[idx, 'candles_to_crash'] = 999  # Valor alto (sem risco)

        # Estatísticas
        mean_dist = self.df['candles_to_crash'].mean()
        median_dist = self.df['candles_to_crash'].median()
        max_dist = self.df['candles_to_crash'].max()

        print(f"  Distância média até crash: {mean_dist:.1f} candles")
        print(f"  Mediana: {median_dist:.1f} candles")
        print(f"  Máximo: {max_dist:.0f} candles")

        # Distribuição
        safe_zone = (self.df['candles_to_crash'] >= 20).sum()
        danger_zone = (self.df['candles_to_crash'] < 20).sum()

        print(f"\n  Zona SEGURA (>=20 candles): {safe_zone:,} ({safe_zone/len(self.df)*100:.1f}%)")
        print(f"  Zona PERIGO (<20 candles): {danger_zone:,} ({danger_zone/len(self.df)*100:.1f}%)")

        return self.df

    def visualize_crashes(self, save_path=None):
        """
        Visualiza crashes detectados
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Preço + crashes
        ax1.plot(self.df.index, self.df['close'], label='CRASH 500 Price', linewidth=0.8)
        crash_points = self.df[self.df['is_crash'] == 1]
        ax1.scatter(crash_points.index, crash_points['close'],
                   color='red', s=50, zorder=5, label=f'Crashes ({len(crash_points)})')
        ax1.set_xlabel('Candle Index')
        ax1.set_ylabel('Price')
        ax1.set_title('CRASH 500: Price + Detected Crashes')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Plot 2: Candles até crash
        ax2.plot(self.df.index, self.df['candles_to_crash'],
                linewidth=0.8, label='Candles to Next Crash')
        ax2.axhline(y=20, color='orange', linestyle='--',
                   label='Safe Zone Threshold (20 candles)')
        ax2.fill_between(self.df.index, 0, 20, alpha=0.2, color='red', label='Danger Zone')
        ax2.fill_between(self.df.index, 20, self.df['candles_to_crash'].max(),
                        alpha=0.2, color='green', label='Safe Zone')
        ax2.set_xlabel('Candle Index')
        ax2.set_ylabel('Candles to Crash')
        ax2.set_title('Survival Analysis: Distance to Next Crash')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n[PLOT] Salvo em: {save_path}")

        return fig

def main():
    print("="*70)
    print("CRASH 500 - SURVIVAL ANALYSIS LABELING")
    print("="*70)

    # Carregar dados
    data_dir = Path(__file__).parent / "data"
    input_file = data_dir / "CRASH500_5min_180days.csv"

    print(f"\n[LOAD] Carregando {input_file}...")
    df = pd.read_csv(input_file)
    print(f"  Candles: {len(df):,}")

    # Criar labeler
    labeler = CrashSurvivalLabeler(df, crash_threshold_pct=5.0, min_candles_between=10)

    # Gerar labels
    df_labeled = labeler.generate_survival_labels()

    # Visualizar
    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    plot_path = reports_dir / "crash500_survival_analysis.png"
    labeler.visualize_crashes(save_path=plot_path)

    # Salvar dataset
    output_file = data_dir / "CRASH500_5min_survival_labeled.csv"
    df_labeled.to_csv(output_file, index=False)

    print(f"\n[OK] Dataset salvo!")
    print(f"  Arquivo: {output_file}")
    print(f"  Colunas: {list(df_labeled.columns)}")

    # Análise final
    print(f"\n{'='*70}")
    print(f"ESTRATÉGIA DE TRADING")
    print(f"{'='*70}")

    safe_candles = df_labeled[df_labeled['candles_to_crash'] >= 20]
    print(f"\n  Se modelo prever >= 20 candles:")
    print(f"    - Entrar LONG")
    print(f"    - Candles disponíveis: {len(safe_candles):,} ({len(safe_candles)/len(df_labeled)*100:.1f}%)")
    print(f"    - Win rate esperado: 95-99% (só perde se crash imprevisível)")

    danger_candles = df_labeled[df_labeled['candles_to_crash'] < 20]
    print(f"\n  Se modelo prever < 20 candles:")
    print(f"    - FICAR FORA (evitar crash)")
    print(f"    - Candles perigosos: {len(danger_candles):,} ({len(danger_candles)/len(df_labeled)*100:.1f}%)")

    print(f"\n{'='*70}")

if __name__ == "__main__":
    main()
