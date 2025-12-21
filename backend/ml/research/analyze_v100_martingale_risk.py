"""
V100/V75 - ANALISE DE RISCO MARTINGALE
Objetivo: Encontrar a 'Sequencia da Morte' (Max Consecutive Candles)

Este script responde:
1. Qual a PIOR sequencia de velas da mesma cor ja vista?
2. Quanto de banca preciso para sobreviver a ela?
3. Qual o nivel maximo seguro para o Martingale?
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def analyze_streaks():
    print("="*70)
    print("V100/V75 - ANALISE DE RISCO MARTINGALE")
    print("Objetivo: Calcular a banca necessaria para sobreviver a pior tendencia")
    print("="*70)

    # 1. Carregar Dados (Tentar V100/V75)
    data_dir = Path(__file__).parent / "data"

    # Procurar arquivos V100 ou V75
    v100_files = list(data_dir.glob("*1HZ100V*.csv"))
    v75_files = list(data_dir.glob("*1HZ75V*.csv"))

    files = v100_files + v75_files

    if not files:
        print("\n[ERROR] Nenhum arquivo V100/V75 encontrado em:")
        print(f"  {data_dir}")
        print("\nArquivos disponiveis:")
        for f in data_dir.glob("*.csv"):
            print(f"  - {f.name}")
        return

    input_file = files[0]
    print(f"\n[LOAD] Carregando: {input_file.name}")
    df = pd.read_csv(input_file)

    # Sort temporal
    if 'epoch' in df.columns:
        df = df.sort_values('epoch').reset_index(drop=True)
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"  Total candles: {len(df):,}")

    # 2. Identificar Cor da Vela
    # 1 = Verde (Alta), -1 = Vermelha (Baixa), 0 = Doji
    df['color'] = np.sign(df['close'] - df['open'])

    total_candles = len(df)
    green_count = (df['color'] == 1).sum()
    red_count = (df['color'] == -1).sum()
    doji_count = (df['color'] == 0).sum()

    print(f"\n[DISTRIBUICAO DE CORES]")
    print(f"  Verde (Alta):    {green_count:,} ({green_count/total_candles*100:.2f}%)")
    print(f"  Vermelha (Baixa): {red_count:,} ({red_count/total_candles*100:.2f}%)")
    print(f"  Doji (Neutro):   {doji_count:,} ({doji_count/total_candles*100:.2f}%)")

    # Remover Dojis (geralmente repetem a aposta ou devolvem)
    df_no_doji = df[df['color'] != 0].copy()
    print(f"\n[FILTRO] Removendo Dojis: {len(df_no_doji):,} velas restantes")

    # 3. Contar Sequencias (Streaks)
    # Logica: Agrupar sequencias consecutivas de numeros iguais
    df_no_doji['block'] = (df_no_doji['color'] != df_no_doji['color'].shift(1)).cumsum()
    streaks = df_no_doji.groupby('block')['color'].agg(['mean', 'count'])
    streaks.columns = ['direction', 'length']

    # 4. Estatisticas de "Quebra de Banca"
    max_streak = streaks['length'].max()
    avg_streak = streaks['length'].mean()
    median_streak = streaks['length'].median()
    std_streak = streaks['length'].std()

    print(f"\n{'='*70}")
    print("ESTATISTICAS DE SEQUENCIA")
    print(f"{'='*70}")
    print(f"\n[GERAL]")
    print(f"  Total de Sequencias: {len(streaks):,}")
    print(f"  Sequencia Media:     {avg_streak:.2f} velas")
    print(f"  Sequencia Mediana:   {median_streak:.0f} velas")
    print(f"  Desvio Padrao:       {std_streak:.2f}")

    # Detalhes da pior sequencia
    worst_streak_idx = streaks['length'].idxmax()
    worst_direction = "ALTA (Verde)" if streaks.loc[worst_streak_idx, 'direction'] > 0 else "BAIXA (Vermelha)"

    print(f"\n[PIOR SEQUENCIA - 'A SEQUENCIA DA MORTE']")
    print(f"  >> MAXIMO: {max_streak} velas seguidas da mesma cor!")
    print(f"  Direcao: {worst_direction}")

    # Encontrar quando aconteceu
    worst_block = df_no_doji[df_no_doji['block'] == worst_streak_idx]
    if 'timestamp' in worst_block.columns:
        print(f"  Ocorrida em: {worst_block.iloc[0]['timestamp']}")

    # Percentis
    percentiles = [50, 75, 90, 95, 99, 99.5, 99.9]
    values = np.percentile(streaks['length'], percentiles)

    print(f"\n[DISTRIBUICAO DOS TAMANHOS]")
    print(f"{'Percentil':<12} | {'Tamanho':<10} | {'Interpretacao':<30}")
    print(f"{'-'*12}-+-{'-'*10}-+-{'-'*30}")

    interpretations = [
        "Metade dos casos",
        "75% dos casos",
        "90% dos casos",
        "95% dos casos",
        "99% dos casos (1 em 100)",
        "99.5% dos casos (1 em 200)",
        "99.9% dos casos (1 em 1000)"
    ]

    for p, v, interp in zip(percentiles, values, interpretations):
        print(f"{p:<12.1f}% | {v:<10.0f} | {interp:<30}")

    # 5. Simulacao de Custo Martingale
    print(f"\n{'='*70}")
    print("SIMULACAO DE CUSTO MARTINGALE (Entrada Inicial $1.00)")
    print(f"{'='*70}")

    print(f"\n{'Nivel':<6} | {'Aposta ($)':<12} | {'Perda Acum ($)':<16} | {'Banca Necessaria ($)':<20} | {'Notas':<30}")
    print(f"{'-'*6}-+-{'-'*12}-+-{'-'*16}-+-{'-'*20}-+-{'-'*30}")

    initial_stake = 1.0
    multiplier = 2.0  # Martingale Classico
    total_loss = 0
    current_stake = initial_stake

    # Simular ate a pior sequencia + 3 (margem de seguranca)
    safety_margin = 3
    max_levels = max_streak + safety_margin

    for i in range(1, max_levels + 1):
        total_loss += current_stake
        next_stake = current_stake * multiplier
        required_bank = total_loss + next_stake  # Precisa ter a proxima aposta

        marker = ""
        if i == int(values[percentiles.index(99)]):
            marker = "99% seguro (1 em 100 falha)"
        elif i == int(values[percentiles.index(99.9)]):
            marker = "99.9% seguro (1 em 1000 falha)"
        elif i == max_streak:
            marker = "MAXIMO HISTORICO"
        elif i > max_streak:
            marker = "ZONA DA MORTE (nunca visto)"

        print(f"{i:<6} | ${current_stake:<11.2f} | ${total_loss:<15.2f} | ${required_bank:<19.2f} | {marker:<30}")

        current_stake = next_stake

    # 6. Recomendacoes de "Smart Martingale"
    print(f"\n{'='*70}")
    print("RECOMENDACOES - SMART MARTINGALE")
    print(f"{'='*70}")

    # Encontrar nivel seguro (99.5% de confianca)
    safe_level = int(values[percentiles.index(99.5)])
    safe_bank = sum([initial_stake * (multiplier ** i) for i in range(safe_level)])
    safe_bank_with_buffer = safe_bank * 2  # 2x para margin

    print(f"\n[CONFIGURACAO CONSERVADORA - 99.5% Segura]")
    print(f"  Nivel Maximo: {safe_level}")
    print(f"  Banca Minima: ${safe_bank:.2f}")
    print(f"  Banca Recomendada (2x): ${safe_bank_with_buffer:.2f}")
    print(f"  Estrategia: Faz Martingale ate o Nivel {safe_level}, depois RESETA")
    print(f"  Risco: 1 em 200 sequencias vai ultrapassar esse limite")

    # Encontrar nivel agressivo (99%)
    aggressive_level = int(values[percentiles.index(99)])
    aggressive_bank = sum([initial_stake * (multiplier ** i) for i in range(aggressive_level)])
    aggressive_bank_with_buffer = aggressive_bank * 2

    print(f"\n[CONFIGURACAO AGRESSIVA - 99% Segura]")
    print(f"  Nivel Maximo: {aggressive_level}")
    print(f"  Banca Minima: ${aggressive_bank:.2f}")
    print(f"  Banca Recomendada (2x): ${aggressive_bank_with_buffer:.2f}")
    print(f"  Estrategia: Faz Martingale ate o Nivel {aggressive_level}, depois RESETA")
    print(f"  Risco: 1 em 100 sequencias vai ultrapassar esse limite")

    # 7. Graficos
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot 1: Distribuicao de tamanhos de sequencias (Log Scale)
    ax1 = axes[0, 0]
    streak_counts = streaks['length'].value_counts().sort_index()
    ax1.bar(streak_counts.index, streak_counts.values, alpha=0.7, color='blue')
    ax1.axvline(x=safe_level, color='green', linestyle='--', linewidth=2, label=f'Safe Level ({safe_level})')
    ax1.axvline(x=max_streak, color='red', linestyle='--', linewidth=2, label=f'Max Historic ({max_streak})')
    ax1.set_yscale('log')
    ax1.set_title('Frequencia de Sequencias (Log Scale)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Tamanho da Sequencia (velas)')
    ax1.set_ylabel('Frequencia (Log Scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Custo acumulado vs Nivel
    ax2 = axes[0, 1]
    levels = range(1, 16)
    cumulative_costs = [sum([initial_stake * (multiplier ** i) for i in range(lvl)]) for lvl in levels]
    ax2.plot(levels, cumulative_costs, 'o-', linewidth=2, markersize=8, color='red')
    ax2.axvline(x=safe_level, color='green', linestyle='--', linewidth=2, label=f'Safe ({safe_level})')
    ax2.axvline(x=aggressive_level, color='orange', linestyle='--', linewidth=2, label=f'Aggressive ({aggressive_level})')
    ax2.set_title('Custo Acumulado por Nivel (Martingale 2x)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Nivel')
    ax2.set_ylabel('Custo Acumulado ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # Plot 3: Distribuicao Cumulativa
    ax3 = axes[1, 0]
    sorted_lengths = np.sort(streaks['length'])
    cumulative_prob = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
    ax3.plot(sorted_lengths, cumulative_prob * 100, linewidth=2, color='blue')
    ax3.axhline(y=99, color='orange', linestyle='--', linewidth=1, label='99% (Aggressive)')
    ax3.axhline(y=99.5, color='green', linestyle='--', linewidth=1, label='99.5% (Safe)')
    ax3.axvline(x=max_streak, color='red', linestyle='--', linewidth=2, label=f'Max ({max_streak})')
    ax3.set_title('Funcao Distribuicao Acumulada', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Tamanho da Sequencia')
    ax3.set_ylabel('Probabilidade Acumulada (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Histograma de sequencias (zoom em 1-20)
    ax4 = axes[1, 1]
    streak_counts_zoom = streaks[streaks['length'] <= 20]['length'].value_counts().sort_index()
    ax4.bar(streak_counts_zoom.index, streak_counts_zoom.values, alpha=0.7, color='green')
    ax4.axvline(x=safe_level, color='darkgreen', linestyle='--', linewidth=2, label=f'Safe ({safe_level})')
    ax4.set_title('Distribuicao de Sequencias (Zoom 1-20)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Tamanho da Sequencia')
    ax4.set_ylabel('Frequencia')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = data_dir.parent / "reports" / "v100_martingale_risk.png"
    plot_path.parent.mkdir(exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    print(f"\n[PLOT] Graficos salvos em: {plot_path}")

    # 8. VEREDICTO FINAL
    print(f"\n{'='*70}")
    print("VEREDICTO FINAL - MARTINGALE EM V100/V75")
    print(f"{'='*70}\n")

    if max_streak > 12:
        print(f"[PERIGO EXTREMO]")
        print(f"  >> A pior sequencia historica foi {max_streak} velas!")
        print(f"  >> Martingale puro ate {max_streak} precisa de ${sum([initial_stake * (multiplier ** i) for i in range(max_streak)]):.2f}")
        print(f"  >> Isso VAI quebrar a banca eventualmente")
        print(f"\n[SOLUCAO: SMART MARTINGALE]")
        print(f"  1. Defina um LIMITE de nivel (ex: Nivel {safe_level})")
        print(f"  2. Se chegar no limite, ACEITE a perda e RESETE")
        print(f"  3. Matematica: Perder ${safe_bank:.2f} de vez em quando")
        print(f"     e mais barato que arriscar ${cumulative_costs[-1]:.2f} e perder tudo")
        print(f"\n[CONFIGURACAO RECOMENDADA]")
        print(f"  - Banca: ${safe_bank_with_buffer:.2f}")
        print(f"  - Nivel Max: {safe_level}")
        print(f"  - Win Rate esperado: 99.5% (1 em 200 perde)")

    else:
        print(f"[RISCO MODERADO]")
        print(f"  >> A pior sequencia foi {max_streak} velas (gerenciavel)")
        print(f"  >> Martingale e viavel com banca robusta")
        print(f"\n[CONFIGURACAO RECOMENDADA]")
        print(f"  - Banca: ${aggressive_bank_with_buffer:.2f}")
        print(f"  - Nivel Max: {aggressive_level}")
        print(f"  - Win Rate esperado: 99% (1 em 100 perde)")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    analyze_streaks()
