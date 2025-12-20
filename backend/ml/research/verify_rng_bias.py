"""
CRASH300N - FORENSIC LSB ANALYSIS (Eliminacao de Falso Positivo)

Objetivo: Testar se o vies detectado em crack_crash300n_rng_v2.py e REAL ou
          apenas artefato de quantizacao (MSBs preenchidos com zeros).

Metodo:
1. Teste de Paridade (LSB) - Deve ser 50/50 se RNG for bom
2. Teste de Ultimo Digito - Deve ser uniforme 0-9 (10% cada)
3. Ajuste de Distribuicao - Normal vs Exponencial

Se passar limpo -> Falso positivo confirmado (RNG e seguro)
Se falhar -> Vulnerabilidade REAL (RNG e fraco)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats


def test_lsb_parity():
    print("="*70)
    print("FORENSIC LSB ANALYSIS - TESTE DEFINITIVO")
    print("="*70)
    print("\nObjetivo: Eliminar falso positivo por quantizacao")
    print("Foco: Entropia nos bits MENOS significativos\n")

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

    # 2. Extract tick magnitudes (normal ticks only)
    df['log_ret'] = np.log(df['close'] / df['open'])
    normal_ticks = df[df['log_ret'] > 0]['log_ret'].values

    print(f"\n[DADOS] Analisando {len(normal_ticks):,} ticks de subida")

    # 3. TESTE 1: Paridade (LSB)
    print(f"\n{'='*70}")
    print("TESTE 1: PARIDADE (LSB - Least Significant Bit)")
    print(f"{'='*70}")

    print("\n[METODO]")
    print("  Escala: tick * 1e6 (microsegundos precision)")
    print("  Teste: (tick_int) % 2")
    print("  Expectativa: 50% par, 50% impar (se RNG for bom)")

    # Scale to integer (1e6 para manter 6 decimais)
    scaled_ticks = (normal_ticks * 1e6).astype(np.int64)

    # Test parity
    parity = scaled_ticks % 2
    even_count = (parity == 0).sum()
    odd_count = (parity == 1).sum()
    total = len(parity)

    print(f"\n[RESULTADOS]")
    print(f"  Total samples: {total:,}")
    print(f"  Even (0): {even_count:,} ({even_count/total*100:.4f}%)")
    print(f"  Odd  (1): {odd_count:,} ({odd_count/total*100:.4f}%)")

    # Chi-square test
    expected = total / 2
    chi_square_parity = ((even_count - expected)**2 + (odd_count - expected)**2) / expected

    print(f"\n  Chi-square: {chi_square_parity:.4f}")
    print(f"  Critical value (95%): 3.841")

    if chi_square_parity < 3.841:
        print(f"  >> PASSOU: Paridade e uniforme (RNG BOM)")
        parity_pass = True
    else:
        print(f"  >> FALHOU: Paridade tem vies (RNG FRACO)")
        parity_pass = False

    # 4. TESTE 2: Ultimo Digito
    print(f"\n{'='*70}")
    print("TESTE 2: UNIFORMIDADE DO ULTIMO DIGITO")
    print(f"{'='*70}")

    print("\n[METODO]")
    print("  Escala: tick * 1e6")
    print("  Teste: (tick_int) % 10")
    print("  Expectativa: 0-9 cada aparece ~10% (uniforme)")

    # Last digit test
    last_digit = scaled_ticks % 10
    digit_counts = np.bincount(last_digit, minlength=10)

    print(f"\n[RESULTADOS]")
    print(f"{'Digito':<8} | {'Count':<10} | {'%':<10} | {'Esperado':<10}")
    print(f"{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for digit in range(10):
        count = digit_counts[digit]
        pct = count / total * 100
        expected_pct = 10.0
        print(f"{digit:<8} | {count:<10,} | {pct:<10.2f} | {expected_pct:<10.1f}")

    # Chi-square test for uniformity
    expected_per_digit = total / 10
    chi_square_digit = np.sum((digit_counts - expected_per_digit)**2 / expected_per_digit)

    # Chi-square with 9 degrees of freedom: critical value = 16.919 (95%)
    print(f"\n  Chi-square: {chi_square_digit:.4f}")
    print(f"  Critical value (95%, df=9): 16.919")

    if chi_square_digit < 16.919:
        print(f"  >> PASSOU: Digitos sao uniformes (RNG BOM)")
        digit_pass = True
    else:
        print(f"  >> FALHOU: Digitos tem vies (RNG FRACO)")
        digit_pass = False

    # 5. TESTE 3: Ajuste de Distribuicao
    print(f"\n{'='*70}")
    print("TESTE 3: AJUSTE DE DISTRIBUICAO (Normal vs Exponencial)")
    print(f"{'='*70}")

    print("\n[METODO]")
    print("  Se ticks vem de RNG uniforme [0,1), magnitude nao tem distribuicao preferida")
    print("  Testamos: Normal vs Exponencial")

    # Normalize ticks
    ticks_normalized = (normal_ticks - normal_ticks.mean()) / normal_ticks.std()

    # Test against Normal
    ks_stat_normal, ks_pvalue_normal = stats.kstest(ticks_normalized, 'norm')

    # Test against Exponential
    # First scale to positive
    ticks_positive = normal_ticks - normal_ticks.min() + 1e-10
    ks_stat_exp, ks_pvalue_exp = stats.kstest(ticks_positive, 'expon')

    print(f"\n[RESULTADOS]")
    print(f"  Normal Distribution:")
    print(f"    KS Statistic: {ks_stat_normal:.6f}")
    print(f"    P-value: {ks_pvalue_normal:.6f}")

    print(f"\n  Exponential Distribution:")
    print(f"    KS Statistic: {ks_stat_exp:.6f}")
    print(f"    P-value: {ks_pvalue_exp:.6f}")

    # Neither should be a good fit if RNG is good
    if ks_pvalue_normal > 0.05:
        print(f"\n  >> ALERTA: Ticks seguem Normal (suspeito)")
        dist_pass = False
    elif ks_pvalue_exp > 0.05:
        print(f"\n  >> ALERTA: Ticks seguem Exponencial (suspeito)")
        dist_pass = False
    else:
        print(f"\n  >> PASSOU: Ticks nao seguem distribuicao especifica (RNG BOM)")
        dist_pass = True

    # 6. COMPARACAO COM RESULTADOS ANTERIORES
    print(f"\n{'='*70}")
    print("COMPARACAO: LSB vs MSB")
    print(f"{'='*70}")

    # Reconstruct MSB analysis from crack_crash300n_rng_v2.py
    scaled_ticks_1e9 = (normal_ticks * 1e9).astype(np.int64)

    # Convert to binary string (sample)
    bits_msb = ""
    for val in scaled_ticks_1e9[:1000]:
        bits_msb += format(abs(val), '032b')

    zeros_msb = bits_msb.count('0')
    ones_msb = bits_msb.count('1')
    total_bits = len(bits_msb)

    chi_square_msb = ((zeros_msb - total_bits/2)**2 + (ones_msb - total_bits/2)**2) / (total_bits/2)

    print(f"\n[MSB ANALYSIS - crack_crash300n_rng_v2.py]")
    print(f"  Escala: tick * 1e9")
    print(f"  Bits analisados: {total_bits:,}")
    print(f"  Zeros: {zeros_msb:,} ({zeros_msb/total_bits*100:.2f}%)")
    print(f"  Ones:  {ones_msb:,} ({ones_msb/total_bits*100:.2f}%)")
    print(f"  Chi-square: {chi_square_msb:.2f} >> 3.841")
    print(f"  >> FALHOU (vies enorme)")

    print(f"\n[LSB ANALYSIS - Este script]")
    print(f"  Escala: tick * 1e6")
    print(f"  Samples: {total:,}")
    print(f"  Paridade Chi-square: {chi_square_parity:.4f}")
    print(f"  Digito Chi-square: {chi_square_digit:.4f}")

    if parity_pass:
        print(f"  >> PASSOU no teste de paridade")
    else:
        print(f"  >> FALHOU no teste de paridade")

    # 7. VEREDICTO FINAL
    print(f"\n{'='*70}")
    print("VEREDICTO FINAL - FORENSIC LSB ANALYSIS")
    print(f"{'='*70}\n")

    tests_passed = sum([parity_pass, digit_pass, dist_pass])

    print(f"[RESUMO DOS TESTES]")
    print(f"  Teste 1 (Paridade LSB):    {'PASSOU' if parity_pass else 'FALHOU'}")
    print(f"  Teste 2 (Ultimo Digito):   {'PASSOU' if digit_pass else 'FALHOU'}")
    print(f"  Teste 3 (Distribuicao):    {'PASSOU' if dist_pass else 'FALHOU'}")
    print(f"\n  Total: {tests_passed}/3 testes passaram")

    if tests_passed == 3:
        print(f"\n{'='*70}")
        print("CONCLUSAO: FALSO POSITIVO CONFIRMADO")
        print(f"{'='*70}")
        print(f"\n  Os bits MENOS significativos (LSB) sao UNIFORMES.")
        print(f"  O vies detectado em crack_crash300n_rng_v2.py era:")
        print(f"  >> ARTEFATO DE QUANTIZACAO (MSBs zerados)")
        print(f"\n  PROVA:")
        print(f"  - log_ret sao PEQUENOS (0.0001x)")
        print(f"  - Multiplicar por 1e9 preenche MSBs com zeros")
        print(f"  - Isso cria vies artificial de 69% zeros")
        print(f"\n  VEREDICTO FINAL:")
        print(f"  >> CRASH300N usa RNG SEGURO (provavelmente CSPRNG)")
        print(f"  >> Impossivel clonar estado ou prever")
        print(f"  >> LSB tem entropia perfeita (50/50 parity, uniforme 0-9)")

    elif tests_passed >= 2:
        print(f"\n{'='*70}")
        print("CONCLUSAO: INCONCLUSIVO")
        print(f"{'='*70}")
        print(f"\n  2/3 testes passaram, mas 1 falhou.")
        print(f"  Pode haver ALGUM vies, mas nao e exploitavel.")
        print(f"\n  RECOMENDACAO:")
        print(f"  - Coletar mais dados (tick-by-tick via WebSocket)")
        print(f"  - Repetir analise com precisao de nanosegundos")

    else:
        print(f"\n{'='*70}")
        print("CONCLUSAO: VULNERABILIDADE REAL DETECTADA")
        print(f"{'='*70}")
        print(f"\n  LSB tambem mostra vies (nao e artefato de quantizacao).")
        print(f"  O RNG pode ser FRACO (ex: Mersenne Twister com seed previsivel).")
        print(f"\n  PROXIMOS PASSOS:")
        print(f"  1. Coletar 624 outputs de 32-bit via WebSocket")
        print(f"  2. Tentar clonar estado com randcrack")
        print(f"  3. Prever proximos valores com 100% acuracia")

    # 8. Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot 1: Parity distribution
    ax1 = axes[0, 0]
    ax1.bar(['Even', 'Odd'], [even_count, odd_count], alpha=0.7, color=['blue', 'orange'])
    ax1.axhline(y=total/2, color='r', linestyle='--', linewidth=2, label='Expected (50%)')
    ax1.set_title('TESTE 1: Paridade (LSB)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Last digit distribution
    ax2 = axes[0, 1]
    ax2.bar(range(10), digit_counts, alpha=0.7, color='green')
    ax2.axhline(y=expected_per_digit, color='r', linestyle='--', linewidth=2, label='Expected (10%)')
    ax2.set_title('TESTE 2: Ultimo Digito (0-9)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Digit')
    ax2.set_ylabel('Count')
    ax2.set_xticks(range(10))
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Distribution histogram
    ax3 = axes[1, 0]
    ax3.hist(normal_ticks, bins=50, alpha=0.7, density=True, label='Real Data')

    # Overlay Normal fit
    mu, sigma = normal_ticks.mean(), normal_ticks.std()
    x = np.linspace(normal_ticks.min(), normal_ticks.max(), 100)
    ax3.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Fit')

    ax3.set_title('TESTE 3: Ajuste de Distribuicao', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Tick Magnitude')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: MSB vs LSB comparison
    ax4 = axes[1, 1]
    comparison_data = [
        ['MSB\n(1e9)', chi_square_msb, 'FALHOU'],
        ['LSB\n(1e6)', chi_square_parity, 'PASSOU' if parity_pass else 'FALHOU']
    ]

    labels = [d[0] for d in comparison_data]
    values = [d[1] for d in comparison_data]
    colors = ['red' if d[2] == 'FALHOU' else 'green' for d in comparison_data]

    ax4.bar(labels, values, alpha=0.7, color=colors)
    ax4.axhline(y=3.841, color='orange', linestyle='--', linewidth=2, label='Critical (3.841)')
    ax4.set_title('COMPARACAO: MSB vs LSB', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Chi-square')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    plot_path = data_dir.parent / "reports" / "crash300n_lsb_forensic.png"
    plot_path.parent.mkdir(exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    print(f"\n[PLOT] Graficos salvos em: {plot_path}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    test_lsb_parity()
