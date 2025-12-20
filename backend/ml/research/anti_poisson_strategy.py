"""
CRASH300N - ANTI-POISSON STRATEGY
Gestao de Ruina baseada em Estatistica de Extremos

Em vez de prever QUANDO (impossivel), calculamos QUANTO aguentamos.
Objetivo: Sobreviver a 99.9% dos intervalos historicos.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats


def calculate_anti_poisson_strategy():
    print("="*70)
    print("CRASH300N - ESTRATEGIA ANTI-POISSON")
    print("="*70)
    print("\nPremissa: Crashes sao Poisson (nao previsiveis)")
    print("Objetivo: Dimensionar position size para sobreviver desertos\n")

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

    # 3. Estatistica de Extremos
    print(f"\n{'='*70}")
    print("ESTATISTICA DE EXTREMOS (Tail Risk)")
    print(f"{'='*70}")

    # Percentis
    percentiles = [50, 75, 90, 95, 99, 99.5, 99.9, 99.99]
    values = np.percentile(intervals, percentiles)

    print(f"\n[DISTRIBUICAO DOS INTERVALOS]")
    print(f"{'Percentil':<12} | {'Intervalo (candles)':<22} | {'Interpretacao':<30}")
    print(f"{'-'*12}-+-{'-'*22}-+-{'-'*30}")

    interpretations = [
        "Mediana (50% dos casos)",
        "3/4 dos casos",
        "90% dos casos",
        "95% dos casos",
        "99% dos casos (1 em 100)",
        "99.5% dos casos (1 em 200)",
        "99.9% dos casos (1 em 1000)",
        "99.99% dos casos (1 em 10k)"
    ]

    for p, v, interp in zip(percentiles, values, interpretations):
        print(f"{p:<12.2f}% | {v:<22.1f} | {interp:<30}")

    # 4. Teste de Distribuicao (Exponencial vs Real)
    print(f"\n[TESTE: Intervalo ~ Exponencial?]")

    # Se Poisson, intervalos seguem Exponencial com lambda = 1/mean
    lambda_param = 1.0 / intervals.mean()

    # Generate theoretical exponential
    theoretical_intervals = np.random.exponential(1.0/lambda_param, len(intervals))

    # KS Test
    ks_stat, ks_pvalue = stats.ks_2samp(intervals, theoretical_intervals)

    print(f"  H0: Intervalos seguem distribuicao Exponencial")
    print(f"  KS Statistic: {ks_stat:.6f}")
    print(f"  P-value: {ks_pvalue:.6f}")

    if ks_pvalue > 0.05:
        print(f"  Resultado: NAO rejeitamos H0 (p > 0.05)")
        print(f"           -> Consistente com POISSON")
    else:
        print(f"  Resultado: Rejeitamos H0 (p < 0.05)")
        print(f"           -> NAO e Poisson puro")

    # 5. Calculo de Position Size Seguro
    print(f"\n{'='*70}")
    print("CALCULO DE POSITION SIZE (Gestao de Ruina)")
    print(f"{'='*70}")

    # Parametros
    account_balance = 1000.0  # USD
    risk_percentiles = [95, 99, 99.9]

    print(f"\n[PREMISSAS]")
    print(f"  Saldo: ${account_balance:.2f}")
    print(f"  Asset: CRASH300N (sobe entre crashes)")
    print(f"  Risco: Crash de ~50-100 points (1.5% do indice)")
    print(f"  Estrategia: LONG continuo (Carry Trade)")

    # Estimar drawdown maximo
    # CRASH300N vale ~6000-7000 points
    # Crash tipico: 50-100 points = 1.5%
    crash_magnitude_pct = 0.015  # 1.5%

    print(f"\n[ANALISE DE DRAWDOWN]")
    print(f"  Crash magnitude (tipico): {crash_magnitude_pct*100:.1f}%")

    # Para cada nivel de confianca, calcular max position size
    print(f"\n[POSITION SIZE SEGURO]")
    print(f"{'Confianca':<12} | {'Max Intervalo':<15} | {'Max Drawdown':<15} | {'Position Size':<15} | {'Leverage':<10}")
    print(f"{'-'*12}-+-{'-'*15}-+-{'-'*15}-+-{'-'*15}-+-{'-'*10}")

    for risk_pct in risk_percentiles:
        max_interval = np.percentile(intervals, risk_pct)

        # Drawdown = crash_magnitude_pct
        max_drawdown_pct = crash_magnitude_pct

        # Position size que nao quebra conta
        # Se queremos perder no maximo X% da conta
        max_loss_pct = 0.02  # 2% max loss per trade

        # Position size = (max_loss_pct * account) / (max_drawdown_pct * price)
        # Simplificado: position_size = (max_loss / max_drawdown)
        position_size = (max_loss_pct * account_balance) / max_drawdown_pct

        # Leverage
        leverage = position_size / account_balance

        print(f"{risk_pct:<12.1f}% | {max_interval:<15.1f} | {max_drawdown_pct*100:<15.1f}% | ${position_size:<14.2f} | {leverage:<10.1f}x")

    # 6. Expectativa de Lucro (Carry Trade)
    print(f"\n{'='*70}")
    print("EXPECTATIVA DE LUCRO (Carry Trade)")
    print(f"{'='*70}")

    # CRASH300N sempre sobe entre crashes
    # Estimar rise medio
    df['rise'] = df['close'].pct_change()
    df_no_crash = df[df['is_crash'] == 0]
    avg_rise_per_candle = df_no_crash['rise'].mean()

    print(f"\n[CARRY TRADE STATS]")
    print(f"  Rise medio por candle (sem crash): {avg_rise_per_candle*100:.4f}%")
    print(f"  Rise acumulado em 50 candles: {(1 + avg_rise_per_candle)**50 - 1:.2%}")

    # Expectativa de lucro
    avg_interval = intervals.mean()
    expected_rise_per_cycle = (1 + avg_rise_per_candle)**avg_interval - 1
    crash_loss = crash_magnitude_pct

    net_expected_return = expected_rise_per_cycle - crash_loss

    print(f"\n[EXPECTATIVA POR CICLO]")
    print(f"  Intervalo medio: {avg_interval:.1f} candles")
    print(f"  Rise esperado: {expected_rise_per_cycle*100:.2f}%")
    print(f"  Crash loss: {crash_loss*100:.2f}%")
    print(f"  Net Return: {net_expected_return*100:.2f}%")

    if net_expected_return > 0:
        print(f"\n  -> EXPECTATIVA POSITIVA (estrategia viavel)")
    else:
        print(f"\n  -> EXPECTATIVA NEGATIVA (estrategia nao viavel)")

    # 7. Simulacao de Monte Carlo
    print(f"\n{'='*70}")
    print("SIMULACAO DE MONTE CARLO (10,000 trades)")
    print(f"{'='*70}")

    n_simulations = 10000
    n_trades = 100

    # Parametros
    position_size = (0.02 * account_balance) / crash_magnitude_pct  # 99% safe

    balances = []

    for sim in range(n_simulations):
        balance = account_balance
        for trade in range(n_trades):
            # Sample interval from distribution
            interval = np.random.choice(intervals)

            # Simulate EACH candle in the interval
            crash_occurred_in_interval = False
            cumulative_rise = 0.0

            for candle in range(int(interval)):
                # Check if crash happens in this candle
                if np.random.random() < 0.019:  # 1.9% chance per candle
                    crash_occurred_in_interval = True
                    # Crash happens, stop accumulating rise
                    break
                else:
                    # Normal rise
                    cumulative_rise += avg_rise_per_candle

            # Calculate profit/loss
            if crash_occurred_in_interval:
                # Lost from crash
                loss = position_size * crash_magnitude_pct
                # But gained from rise BEFORE crash
                profit = position_size * cumulative_rise
                balance += (profit - loss)
            else:
                # Full cycle without crash
                profit = position_size * cumulative_rise
                balance += profit

            if balance <= 0:
                break

        balances.append(balance)

    balances = np.array(balances)

    print(f"\n[RESULTADOS]")
    print(f"  Balance final medio: ${balances.mean():.2f}")
    print(f"  Balance final mediano: ${np.median(balances):.2f}")
    print(f"  Min: ${balances.min():.2f}")
    print(f"  Max: ${balances.max():.2f}")
    print(f"  Std: ${balances.std():.2f}")

    # Probability of ruin
    prob_ruin = (balances <= 0).sum() / len(balances)
    print(f"\n  Probabilidade de Ruina: {prob_ruin*100:.2f}%")

    # 8. Plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot 1: Distribuicao dos Intervalos
    ax1 = axes[0, 0]
    ax1.hist(intervals, bins=50, alpha=0.7, density=True, label='Real')

    # Overlay theoretical exponential
    x_theory = np.linspace(0, intervals.max(), 100)
    y_theory = lambda_param * np.exp(-lambda_param * x_theory)
    ax1.plot(x_theory, y_theory, 'r-', linewidth=2, label='Exponencial (Poisson)')

    ax1.set_title('Distribuicao dos Intervalos', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Intervalo (candles)')
    ax1.set_ylabel('Densidade')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Percentis
    ax2 = axes[0, 1]
    ax2.bar(range(len(percentiles)), values, alpha=0.7, color='blue')
    ax2.set_xticks(range(len(percentiles)))
    ax2.set_xticklabels([f"{p}%" for p in percentiles], rotation=45)
    ax2.set_title('Percentis dos Intervalos', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Intervalo (candles)')
    ax2.grid(True, alpha=0.3, axis='y')

    # Highlight 99.9%
    ax2.bar(percentiles.index(99.9), values[percentiles.index(99.9)],
            alpha=0.7, color='red', label='99.9% (Safe)')
    ax2.legend()

    # Plot 3: Monte Carlo Results
    ax3 = axes[1, 0]
    ax3.hist(balances, bins=50, alpha=0.7, color='green')
    ax3.axvline(x=account_balance, color='r', linestyle='--', linewidth=2,
                label=f'Initial (${account_balance:.0f})')
    ax3.set_title('Monte Carlo: Balance Final (10k simulations)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Balance ($)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Cumulative Probability
    ax4 = axes[1, 1]
    sorted_intervals = np.sort(intervals)
    cumulative_prob = np.arange(1, len(sorted_intervals) + 1) / len(sorted_intervals)
    ax4.plot(sorted_intervals, cumulative_prob, 'b-', linewidth=2)
    ax4.axhline(y=0.99, color='r', linestyle='--', linewidth=1, label='99% Confidence')
    ax4.axhline(y=0.999, color='orange', linestyle='--', linewidth=1, label='99.9% Confidence')
    ax4.set_title('Funcao Distribuicao Acumulada', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Intervalo (candles)')
    ax4.set_ylabel('Probabilidade Acumulada')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = data_dir.parent / "reports" / "crash300n_anti_poisson_strategy.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\n[PLOT] Graficos salvos em: {plot_path}")

    # 9. VEREDICTO FINAL
    print(f"\n{'='*70}")
    print("VEREDICTO FINAL - ESTRATEGIA ANTI-POISSON")
    print(f"{'='*70}\n")

    if net_expected_return > 0 and prob_ruin < 0.01:
        print(f"VIAVEL - Estrategia tem expectativa positiva")
        print(f"\n  Expectativa por ciclo: {net_expected_return*100:.2f}%")
        print(f"  Probabilidade de ruina: {prob_ruin*100:.2f}%")
        print(f"\n  PROXIMOS PASSOS:")
        print(f"  1. Implementar bot com position sizing dinamico")
        print(f"  2. Backtest em dados out-of-sample")
        print(f"  3. Paper trading por 1 mes")
        print(f"  4. Live trading com capital pequeno ($100)")
    else:
        print(f"NAO VIAVEL - Expectativa negativa ou risco de ruina alto")
        print(f"\n  Expectativa por ciclo: {net_expected_return*100:.2f}%")
        print(f"  Probabilidade de ruina: {prob_ruin*100:.2f}%")
        print(f"\n  RAZAO:")
        if net_expected_return <= 0:
            print(f"  >> Carry trade nao compensa o risco de crash")
        if prob_ruin >= 0.01:
            print(f"  >> Risco de ruina muito alto (>1%)")
        print(f"\n  RECOMENDACAO:")
        print(f"  >> Migrar para Forex/Indices reais")
        print(f"  >> Ou aceitar que CRASH/BOOM e casa de apostas (nao trading)")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    calculate_anti_poisson_strategy()
