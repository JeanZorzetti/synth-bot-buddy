"""
CRASH300N - FFT (Fast Fourier Transform) - O TESTE DEFINITIVO
Analise de Frequencia para Detectar Ciclos Ocultos no PRNG

Se houver QUALQUER periodicidade (mesmo mascarada por ruido), FFT encontra.
Se FFT = flat (ruido branco) -> Game Over.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq


def analyze_fft_intervals():
    print("="*70)
    print("CRASH300N - FFT ANALYSIS (O TESTE DEFINITIVO)")
    print("="*70)
    print("\nObjetivo: Detectar ciclos ocultos via analise de frequencia")
    print("Se houver periodicidade (300, 500, etc.), FFT mostra pico")
    print("Se FFT = flat -> Ruido Branco (Poisson) confirmado\n")

    # 1. Load Data
    data_dir = Path(__file__).parent / "data"
    files = list(data_dir.glob("*CRASH300*.csv"))
    if not files:
        print("\nERRO: Nenhum arquivo CRASH300N encontrado")
        return

    print(f"[LOAD] {files[0].name}")
    df = pd.read_csv(files[0])

    # Sort temporal
    if 'epoch' in df.columns:
        df = df.sort_values('epoch').reset_index(drop=True)
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"  Total candles: {len(df):,}")

    # 2. Extract Crash Intervals
    print(f"\n[EXTRACT] Detectando crashes e calculando intervalos...")

    df['log_ret'] = np.log(df['close'] / df['open'])
    df['is_crash'] = (df['log_ret'] <= -0.005).astype(int)

    crash_indices = df[df['is_crash'] == 1].index.values
    intervals = np.diff(crash_indices)

    print(f"  Total crashes: {len(crash_indices):,}")
    print(f"  Total intervalos: {len(intervals):,}")
    print(f"  Media: {intervals.mean():.2f} candles")
    print(f"  Std: {intervals.std():.2f} candles")

    # 3. FFT Analysis
    print(f"\n{'='*70}")
    print("FFT - ANALISE DE FREQUENCIA")
    print(f"{'='*70}")

    # Remover media (detrend) para evitar DC spike
    intervals_detrend = intervals - intervals.mean()

    # Compute FFT
    N = len(intervals_detrend)
    yf = fft(intervals_detrend)
    xf = fftfreq(N, 1.0)  # Frequencia em "1/candle"

    # Power Spectrum (magnitude ao quadrado)
    power = np.abs(yf)**2

    # Pegar apenas frequencias positivas
    positive_freq_mask = xf > 0
    xf_positive = xf[positive_freq_mask]
    power_positive = power[positive_freq_mask]

    # Normalizar power
    power_norm = power_positive / power_positive.max()

    print(f"\n[FFT] Computed")
    print(f"  Samples: {N}")
    print(f"  Frequency range: 0 to {xf_positive.max():.6f} cycles/candle")

    # 4. Find Dominant Frequencies
    # Top 5 picos
    top_indices = np.argsort(power_positive)[::-1][:10]
    top_freqs = xf_positive[top_indices]
    top_powers = power_positive[top_indices]
    top_periods = 1.0 / top_freqs  # Periodo em candles

    print(f"\n[DOMINANT FREQUENCIES] Top 10:")
    print(f"{'Rank':<6} | {'Freq (1/candle)':<18} | {'Period (candles)':<20} | {'Power':<15} | {'Power %':<10}")
    print(f"{'-'*6}-+-{'-'*18}-+-{'-'*20}-+-{'-'*15}-+-{'-'*10}")

    for i, (freq, period, pwr) in enumerate(zip(top_freqs, top_periods, top_powers)):
        pwr_pct = (pwr / power_positive.max()) * 100
        print(f"{i+1:<6} | {freq:<18.6f} | {period:<20.2f} | {pwr:<15.2e} | {pwr_pct:<10.2f}%")

    # 5. Test for White Noise (Flat Spectrum)
    # Se for ruido branco, todas as frequencias tem poder similar
    # Vamos calcular o "Spectral Flatness" (medida de quao flat e o espectro)

    # Spectral Flatness = (Geometric Mean) / (Arithmetic Mean)
    # = 1.0 -> Perfeitamente flat (ruido branco)
    # ~ 0.0 -> Tem picos (periodicidade)

    from scipy.stats import gmean

    geometric_mean = gmean(power_positive + 1e-10)  # +epsilon para evitar log(0)
    arithmetic_mean = np.mean(power_positive)

    spectral_flatness = geometric_mean / arithmetic_mean

    print(f"\n[SPECTRAL FLATNESS]")
    print(f"  Valor: {spectral_flatness:.6f}")
    print(f"  Interpretacao:")
    print(f"    1.0 = Ruido Branco (sem periodicidade)")
    print(f"    0.0 = Tons puros (forte periodicidade)")

    # 6. Statistical Test: Periodogram Test
    # Compara power vs distribuicao esperada de ruido branco
    from scipy.stats import chi2

    # Teste: Se for ruido branco, power segue distribuicao exponencial
    # Usamos Kolmogorov-Smirnov test
    from scipy.stats import kstest, expon

    # Normalizar power para ter media 1
    power_normalized = power_positive / power_positive.mean()

    # KS test vs exponencial(lambda=1)
    ks_stat, ks_pvalue = kstest(power_normalized, 'expon')

    print(f"\n[KOLMOGOROV-SMIRNOV TEST]")
    print(f"  H0: Power spectrum segue distribuicao de ruido branco")
    print(f"  KS Statistic: {ks_stat:.6f}")
    print(f"  P-value: {ks_pvalue:.6f}")

    if ks_pvalue > 0.05:
        print(f"  Resultado: NAO rejeitamos H0 (p > 0.05)")
        print(f"           -> Consistente com RUIDO BRANCO")
    else:
        print(f"  Resultado: Rejeitamos H0 (p < 0.05)")
        print(f"           -> Existe ESTRUTURA no espectro")

    # 7. Check for Expected Period (300 candles)
    # Se algoritmo tem ciclo de ~300, deveria ter pico em 1/300 = 0.00333
    expected_period = 300.0
    expected_freq = 1.0 / expected_period

    # Find closest frequency to expected
    closest_idx = np.argmin(np.abs(xf_positive - expected_freq))
    closest_freq = xf_positive[closest_idx]
    closest_period = 1.0 / closest_freq
    closest_power = power_positive[closest_idx]
    closest_power_pct = (closest_power / power_positive.max()) * 100

    print(f"\n[EXPECTED PERIOD TEST]")
    print(f"  Periodo esperado: {expected_period:.1f} candles (CRASH300N)")
    print(f"  Frequencia esperada: {expected_freq:.6f} cycles/candle")
    print(f"\n  Frequencia mais proxima encontrada:")
    print(f"    Freq: {closest_freq:.6f}")
    print(f"    Periodo: {closest_period:.2f} candles")
    print(f"    Power: {closest_power:.2e} ({closest_power_pct:.2f}% do maximo)")

    # Se power nessa frequencia for significativo (>10% do max), ha evidencia
    if closest_power_pct > 10.0:
        print(f"\n  -> EVIDENCIA de ciclo em ~{closest_period:.0f} candles!")
    else:
        print(f"\n  -> SEM evidencia de ciclo em ~{expected_period:.0f} candles")

    # 8. Plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot 1: Intervalos (Time Domain)
    ax1 = axes[0, 0]
    ax1.plot(intervals[:500], 'b-', alpha=0.7, linewidth=1)
    ax1.axhline(y=intervals.mean(), color='r', linestyle='--', linewidth=1, label=f'Media={intervals.mean():.1f}')
    ax1.set_title('Intervalos entre Crashes (Time Domain)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Interval (candles)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Power Spectrum (Full Range)
    ax2 = axes[0, 1]
    ax2.plot(xf_positive, power_norm, 'b-', alpha=0.7, linewidth=1)
    ax2.set_title('Power Spectrum (Normalized)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Frequency (cycles/candle)')
    ax2.set_ylabel('Normalized Power')
    ax2.set_xlim(0, 0.1)  # Zoom em baixas frequencias
    ax2.grid(True, alpha=0.3)

    # Highlight expected frequency
    ax2.axvline(x=expected_freq, color='r', linestyle='--', linewidth=2,
                label=f'Expected (1/{expected_period:.0f})')
    ax2.legend()

    # Plot 3: Power Spectrum (Log Scale)
    ax3 = axes[1, 0]
    ax3.semilogy(xf_positive, power_positive, 'g-', alpha=0.7, linewidth=1)
    ax3.set_title('Power Spectrum (Log Scale)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Frequency (cycles/candle)')
    ax3.set_ylabel('Power (log)')
    ax3.set_xlim(0, 0.1)
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=expected_freq, color='r', linestyle='--', linewidth=2)

    # Plot 4: Histogram of Power (Test for Exponential)
    ax4 = axes[1, 1]
    ax4.hist(power_normalized, bins=50, alpha=0.7, color='purple',
             density=True, label='Observed Power')

    # Overlay theoretical exponential
    x_theory = np.linspace(0, power_normalized.max(), 100)
    y_theory = expon.pdf(x_theory, scale=1.0)
    ax4.plot(x_theory, y_theory, 'r-', linewidth=2, label='Exponential (White Noise)')

    ax4.set_title('Power Distribution vs White Noise', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Normalized Power')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = data_dir.parent / "reports" / "crash300n_fft_analysis.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\n[PLOT] Graficos salvos em: {plot_path}")

    # 9. VEREDICTO FINAL
    print(f"\n{'='*70}")
    print("VEREDICTO FINAL - FFT ANALYSIS")
    print(f"{'='*70}\n")

    # Criterios de decisao
    # 1. Spectral Flatness > 0.8 -> Ruido branco
    # 2. KS p-value > 0.05 -> Consistente com ruido branco
    # 3. Power no periodo esperado < 10% -> Sem ciclo detectavel

    criteria_white_noise = 0

    print(f"[CRITERIOS]")

    # Criterio 1
    if spectral_flatness > 0.8:
        print(f"  1. Spectral Flatness: {spectral_flatness:.4f} > 0.8 -> RUIDO BRANCO")
        criteria_white_noise += 1
    else:
        print(f"  1. Spectral Flatness: {spectral_flatness:.4f} <= 0.8 -> ESTRUTURA")

    # Criterio 2
    if ks_pvalue > 0.05:
        print(f"  2. KS p-value: {ks_pvalue:.4f} > 0.05 -> RUIDO BRANCO")
        criteria_white_noise += 1
    else:
        print(f"  2. KS p-value: {ks_pvalue:.4f} <= 0.05 -> ESTRUTURA")

    # Criterio 3
    if closest_power_pct < 10.0:
        print(f"  3. Power em ~{expected_period:.0f}: {closest_power_pct:.2f}% < 10% -> SEM CICLO")
        criteria_white_noise += 1
    else:
        print(f"  3. Power em ~{expected_period:.0f}: {closest_power_pct:.2f}% >= 10% -> CICLO DETECTADO")

    print(f"\n[SCORE] {criteria_white_noise}/3 criterios de ruido branco")

    print(f"\n{'='*70}")

    if criteria_white_noise == 3:
        print(f"VEREDICTO: RUIDO BRANCO (Poisson Puro)")
        print(f"{'='*70}")
        print(f"\n  TODAS as 3 metricas convergem: RUIDO BRANCO")
        print(f"  - Spectral Flatness: {spectral_flatness:.4f} (proximo de 1.0)")
        print(f"  - KS Test: p={ks_pvalue:.4f} (nao rejeita H0)")
        print(f"  - Periodo {expected_period:.0f}: Power {closest_power_pct:.2f}% (insignificante)")
        print(f"\n  INTERPRETACAO:")
        print(f"  >> Intervalos sao PURAMENTE ESTOCASTICOS")
        print(f"  >> NAO existe periodicidade oculta")
        print(f"  >> Game Engine e Poisson puro: P(crash) = constante")
        print(f"  >> CSPRNG confirmado (Mersenne Twister ou melhor)")
        print(f"\n  EVIDENCIA COMPLETA (6 Testes):")
        print(f"  1. Hazard Curve: P-value=0.8448 (sem memoria)")
        print(f"  2. LSTM: Correlacao ~0.02 (sem correlacao)")
        print(f"  3. XGBoost: AUC=0.5012 (sem particoes)")
        print(f"  4. KAN B-Splines: -0.25% (sem funcao smooth)")
        print(f"  5. KAN Senoidais: -4.71% (sem periodicidade)")
        print(f"  6. FFT: Spectral Flatness=0.{int(spectral_flatness*10000)} (RUIDO BRANCO)")
        print(f"\n  >> 6/6 TESTES CONVERGEM: IMPOSSIVEL")
        print(f"\n  GAME OVER.")

    elif criteria_white_noise == 2:
        print(f"VEREDICTO: PROVAVEL RUIDO BRANCO (Edge Fraco)")
        print(f"{'='*70}")
        print(f"\n  2/3 criterios apontam para ruido branco")
        print(f"  Existe possibilidade de estrutura fraca, mas muito pequena")
        print(f"\n  POSSIBILIDADE:")
        print(f"  >> Periodo muito longo (> {len(intervals)} intervalos)")
        print(f"  >> Ou periodicidade mascarada por ruido forte")
        print(f"\n  RECOMENDACAO:")
        print(f"  >> Necessario mais dados (anos de historico)")
        print(f"  >> Ou migrar para Forex/Indices reais")

    else:
        print(f"SUCESSO! PERIODICIDADE DETECTADA!")
        print(f"{'='*70}")
        print(f"\n  FFT encontrou ESTRUTURA no espectro!")
        print(f"  - Spectral Flatness: {spectral_flatness:.4f} (distante de 1.0)")
        print(f"  - KS Test: p={ks_pvalue:.4f} (rejeita ruido branco)")
        print(f"\n  PERIODOS DOMINANTES:")
        for i, (freq, period, pwr) in enumerate(zip(top_freqs[:3], top_periods[:3], top_powers[:3])):
            pwr_pct = (pwr / power_positive.max()) * 100
            print(f"    {i+1}. Periodo: {period:.1f} candles (Power: {pwr_pct:.1f}%)")
        print(f"\n  INTERPRETACAO:")
        print(f"  >> Game Engine TEM periodicidade!")
        print(f"  >> PRNG tem ciclo detectavel")
        print(f"  >> Sistema e EXPLORAVEL com estrategia baseada em timing")
        print(f"\n  PROXIMA ACAO:")
        print(f"  >> Retreinar KAN focando nos periodos dominantes")
        print(f"  >> Criar features baseadas em modulo desses periodos")
        print(f"  >> Implementar estrategia de timing (entrar LONG apos periodos especificos)")

    print(f"\n{'='*70}\n")

    # Save results
    results = {
        'spectral_flatness': spectral_flatness,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pvalue,
        'expected_period': expected_period,
        'closest_period': closest_period,
        'closest_power_pct': closest_power_pct,
        'top_periods': top_periods.tolist(),
        'top_powers': top_powers.tolist()
    }

    import json
    results_file = data_dir.parent / "reports" / "crash300n_fft_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[SAVE] Resultados salvos em: {results_file}")


if __name__ == "__main__":
    analyze_fft_intervals()
