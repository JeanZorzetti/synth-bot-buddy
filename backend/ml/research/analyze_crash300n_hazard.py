"""
CRASH300N - Engenharia Reversa do Algoritmo
Foco: Features Temporais (Hazard Rate)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_hazard_features():
    print("="*70)
    print("CRASH300N - ENGENHARIA REVERSA (HAZARD FEATURES)")
    print("="*70)

    # 1. Carregar dados brutos (OHLC)
    data_dir = Path(__file__).parent / "data"
    files = list(data_dir.glob("*CRASH300*.csv"))
    if not files:
        print("ERRO: Nenhum arquivo CRASH300N encontrado em backend/ml/research/data/")
        print("   Por favor, gere ou baixe os dados primeiro.")
        return

    input_file = files[0]
    print(f"\n[LOAD] Carregando: {input_file.name}")
    df = pd.read_csv(input_file)

    # Garantir ordenacao temporal
    if 'epoch' in df.columns:
        df = df.sort_values('epoch').reset_index(drop=True)
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

    # 2. Definir o que e um CRASH (Threshold 0.5% como descobrimos)
    df['log_ret'] = np.log(df['close'] / df['open'])  # Intra-candle return

    CRASH_THRESHOLD = -0.005  # -0.5%
    df['is_crash'] = (df['log_ret'] <= CRASH_THRESHOLD).astype(int)

    print(f"\n[DATA]")
    print(f"  Total Candles: {len(df):,}")
    print(f"  Total Crashes: {df['is_crash'].sum():,} ({df['is_crash'].mean()*100:.2f}%)")

    # 3. ENGENHARIA DE FEATURES (A Magica)
    print(f"\n[FEATURES] Criando Features de 'Memoria'...")

    # Feature A: Candles desde o ultimo crash (Contador)
    df['crash_group'] = df['is_crash'].cumsum()
    df['candles_since_crash'] = df.groupby('crash_group').cumcount()

    # Feature B: Tamanho do ultimo crash (Lagged)
    crash_sizes = df['log_ret'].where(df['is_crash'] == 1)
    df['last_crash_magnitude'] = crash_sizes.ffill().shift(1)

    # Feature C: Media movel de crashes (Densidade de perigo)
    df['crash_density_50'] = df['is_crash'].rolling(window=50).sum().shift(1)

    # Limpeza de NaNs
    df = df.dropna()

    print(f"  Features criadas:")
    print(f"    - candles_since_crash (contador temporal)")
    print(f"    - last_crash_magnitude (tamanho do ultimo crash)")
    print(f"    - crash_density_50 (densidade de crashes)")

    # 4. LABELING (Target)
    df['target_next_crash'] = df['is_crash'].shift(-1).fillna(0).astype(int)

    # 5. ANALISE DE CORRELACAO
    print(f"\n{'='*70}")
    print(f"TESTE DE HIPOTESE: O CRASH E PREVISIVEL PELO TEMPO?")
    print(f"{'='*70}")

    features_to_test = ['candles_since_crash', 'last_crash_magnitude', 'crash_density_50']

    corr_matrix = df[features_to_test + ['target_next_crash']].corr()
    target_corr = corr_matrix['target_next_crash'].drop('target_next_crash')

    print(f"\nCorrelacao com Proximo Crash (Target):")
    for feat, corr_val in target_corr.sort_values(ascending=False).items():
        print(f"  {feat:<30}: {corr_val:+.6f}")

    # 6. VISUALIZACAO DA "ZONA DE PERIGO"
    print(f"\n[HAZARD CURVE] Calculando curva de perigo...")

    # Agrupar por 'candles_since_crash' e ver a media de 'target_next_crash'
    hazard_curve = df.groupby('candles_since_crash')['target_next_crash'].agg(['mean', 'count'])

    # Filtrar apenas grupos com pelo menos 30 samples (estatisticamente significante)
    hazard_curve = hazard_curve[hazard_curve['count'] >= 30]

    # Plotar apenas os primeiros 200 candles (onde tem mais dados)
    hazard_data = hazard_curve.head(200)

    plt.figure(figsize=(14, 7))

    # Plot 1: Curva de Hazard
    plt.subplot(1, 2, 1)
    plt.plot(hazard_data.index, hazard_data['mean'] * 100, 'b-', linewidth=2, label='Prob Real de Crash')
    plt.axhline(y=df['target_next_crash'].mean() * 100, color='r', linestyle='--',
                linewidth=1.5, label=f'Media Global ({df["target_next_crash"].mean()*100:.2f}%)')
    plt.xlabel('Candles desde o ultimo Crash', fontsize=12)
    plt.ylabel('Probabilidade de Crash no Proximo Candle (%)', fontsize=12)
    plt.title('Curva de Perigo (Hazard Rate)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Plot 2: Contagem de samples por grupo
    plt.subplot(1, 2, 2)
    plt.bar(hazard_data.index, hazard_data['count'], alpha=0.6, color='green')
    plt.xlabel('Candles desde o ultimo Crash', fontsize=12)
    plt.ylabel('Numero de Samples', fontsize=12)
    plt.title('Distribuicao de Samples', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    plot_path = data_dir.parent / "reports" / "crash300n_hazard_analysis.png"
    plot_path.parent.mkdir(exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    print(f"  Grafico salvo em: {plot_path}")

    # 7. ESTATISTICAS DA CURVA
    print(f"\n{'='*70}")
    print(f"DIAGNOSTICO DA HAZARD CURVE")
    print(f"{'='*70}")

    mean_global = df['target_next_crash'].mean()
    peak_prob = hazard_data['mean'].max()
    min_prob = hazard_data['mean'].min()
    peak_idx = hazard_data['mean'].idxmax()

    print(f"\n[ESTATISTICAS]")
    print(f"  Probabilidade media global: {mean_global*100:.4f}%")
    print(f"  Probabilidade minima (curva): {min_prob*100:.4f}%")
    print(f"  Probabilidade maxima (curva): {peak_prob*100:.4f}%")
    print(f"  Pico em: {peak_idx} candles apos ultimo crash")
    print(f"  Range: {(peak_prob - min_prob)*100:.4f}%")
    print(f"  Variacao relativa: {((peak_prob - min_prob) / mean_global)*100:.2f}%")

    # 8. TESTE DE TENDENCIA (Linear Regression)
    from scipy import stats

    x = hazard_data.index.values
    y = hazard_data['mean'].values

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    print(f"\n[REGRESSAO LINEAR]")
    print(f"  Slope (tendencia): {slope:+.8f}")
    print(f"  R-squared: {r_value**2:.6f}")
    print(f"  P-value: {p_value:.6f}")

    if p_value < 0.05:
        print(f"  Resultado: Tendencia SIGNIFICATIVA (p < 0.05)")
    else:
        print(f"  Resultado: Tendencia NAO significativa (p >= 0.05)")

    # 9. CONCLUSAO AUTOMATICA
    print(f"\n{'='*70}")
    print(f"CONCLUSAO FINAL")
    print(f"{'='*70}\n")

    # Criterios de decisao
    corr_threshold = 0.02  # Correlacao minima para considerar "previsivel"
    variance_threshold = 0.20  # Variacao relativa minima (20%)

    max_corr = abs(target_corr.max())
    variance_rel = ((peak_prob - min_prob) / mean_global)

    print(f"Criterios de Decisao:")
    print(f"  1. Correlacao maxima: {max_corr:.6f} (threshold: {corr_threshold})")
    print(f"  2. Variacao relativa: {variance_rel*100:.2f}% (threshold: {variance_threshold*100}%)")
    print(f"  3. P-value tendencia: {p_value:.6f} (threshold: 0.05)")

    print(f"\n{'='*70}")

    if max_corr > corr_threshold and variance_rel > variance_threshold and p_value < 0.05:
        print(f"VEREDICTO: ESPERANCA (Distribuicao de Weibull)")
        print(f"{'='*70}")
        print(f"\n  EXISTE UM PADRAO TEMPORAL!")
        print(f"  - A probabilidade de crash MUDA com o tempo")
        print(f"  - Efeito memoria detectado")
        print(f"  - O modelo PODE aprender com 'candles_since_crash'")
        print(f"\n  PROXIMA ACAO:")
        print(f"  >> Retreinar LSTM com features temporais (candles_since_crash)")
        print(f"  >> Expectativa: Modelo vai aprender a evitar 'Zona de Perigo'")
        print(f"  >> Win rate esperado: {(1 - peak_prob)*100:.1f}% (evitando pico)")

    elif max_corr > corr_threshold/2 or variance_rel > variance_threshold/2:
        print(f"VEREDICTO: INCERTO (Padrao Fraco)")
        print(f"{'='*70}")
        print(f"\n  Existe um padrao FRACO detectado")
        print(f"  - Correlacao baixa mas nao zero")
        print(f"  - Variacao existe mas e pequena")
        print(f"\n  PROXIMA ACAO:")
        print(f"  >> Tentar retreinar com features temporais")
        print(f"  >> Expectativa moderada de melhoria")

    else:
        print(f"VEREDICTO: PESADELO (Processo de Poisson / Memoryless)")
        print(f"{'='*70}")
        print(f"\n  NAO EXISTE PADRAO TEMPORAL!")
        print(f"  - Probabilidade de crash e constante")
        print(f"  - Processo 'Sem Memoria' (Falacia do Apostador)")
        print(f"  - Algoritmo da Deriv e 'perfeito' (RNG verdadeiro)")
        print(f"\n  CONCLUSAO:")
        print(f"  >> ML scalping e MATEMATICAMENTE INVIAVEL")
        print(f"  >> Nao existe previsao possivel com features temporais")
        print(f"  >> Recomendacao: Migrar para Forex/Indices reais")

    print(f"\n{'='*70}\n")

    # Salvar resultados em CSV para analise posterior
    results_file = data_dir.parent / "reports" / "crash300n_hazard_results.csv"
    hazard_data.to_csv(results_file)
    print(f"[SAVE] Resultados salvos em: {results_file}")


if __name__ == "__main__":
    analyze_hazard_features()
