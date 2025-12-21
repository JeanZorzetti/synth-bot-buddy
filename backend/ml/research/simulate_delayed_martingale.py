"""
V100/V75 - DELAYED MARTINGALE SIMULATION ("THE VULTURE STRATEGY")

Objetivo: Simular entradas apenas APOS sequencias longas para economizar "municao" da banca.

A Logica do Abutre:
- Banca de $2000 aguenta 10 niveis (ate $1023 acumulado)
- Max historico foi 18 velas
- DELAY = 18 - 10 = 8 velas
- Esperamos 8 velas da mesma cor ANTES de entrar
- Resultado: Eficacia de banca de $262k com apenas $2k

Matematica:
- Se mercado faz 18 velas verdes:
  * Esperamos 8 primeiras passarem (custo $0)
  * Entramos na vela 9 (nosso Nivel 1)
  * Perdemos ate vela 18 (nosso Nivel 10)
  * Vela 19: Reversao, WIN!
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def simulate_delayed_martingale():
    print("="*70)
    print("V100/V75 - SIMULACAO DELAYED MARTINGALE (ABUTRE)")
    print("="*70)

    # --- CONFIGURACOES ---
    BANKROLL = 2000.0       # Sua banca
    INITIAL_STAKE = 1.0     # Aposta inicial
    MULTIPLIER = 2.0        # Martingale classico

    # O PULO DO GATO:
    # Banca de $2000 aguenta ate o nivel 10 ($1023 acumulado).
    # Max Historico foi 18.
    # Delay ideal = 18 (Pior Caso) - 10 (Sua Capacidade) = 8.
    DELAY_THRESHOLD = 8     # Esperar 8 velas da mesma cor antes de entrar

    MAX_LEVEL = 10          # Limite de niveis (safety)

    # 1. Carregar Dados
    data_dir = Path(__file__).parent / "data"
    files = list(data_dir.glob("*1HZ100V*.csv")) + list(data_dir.glob("*1HZ75V*.csv"))

    if not files:
        print("\n[ERROR] Nenhum arquivo V100/V75 encontrado.")
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

    # 2. Identificar Cores e Sequencias
    df['color'] = np.sign(df['close'] - df['open'])

    # Remover Dojis (assumir que nao quebram streak)
    df_no_doji = df[df['color'] != 0].copy().reset_index(drop=True)
    print(f"  Candles sem Dojis: {len(df_no_doji):,}")

    # Calcular tamanho das streaks
    df_no_doji['streak_id'] = (df_no_doji['color'] != df_no_doji['color'].shift(1)).cumsum()
    df_no_doji['streak_count'] = df_no_doji.groupby('streak_id').cumcount() + 1

    # 3. Configuracao
    print(f"\n{'='*70}")
    print("CONFIGURACAO DO ABUTRE")
    print(f"{'='*70}")
    print(f"\n[PARAMETROS]")
    print(f"  Banca: ${BANKROLL:.2f}")
    print(f"  Aposta inicial: ${INITIAL_STAKE:.2f}")
    print(f"  Multiplicador: {MULTIPLIER}x")
    print(f"  Delay (gatilho): {DELAY_THRESHOLD} velas consecutivas")
    print(f"  Max nivel permitido: {MAX_LEVEL}")

    # Calcular capacidade teorica
    total_capacity = sum([INITIAL_STAKE * (MULTIPLIER ** i) for i in range(MAX_LEVEL)])
    print(f"  Capacidade da banca: {MAX_LEVEL} niveis (${total_capacity:.2f} acumulado)")

    # 4. Simulacao
    print(f"\n{'='*70}")
    print("SIMULACAO EM ANDAMENTO...")
    print(f"{'='*70}\n")

    balance = BANKROLL
    peak_balance = BANKROLL
    max_drawdown = 0.0

    total_trades = 0
    wins = 0
    losses = 0
    busts = 0

    trades_log = []
    balance_history = [BANKROLL]

    # Variaveis de estado do Martingale
    in_position = False
    current_stake = 0
    current_level = 0
    bet_direction = 0  # 1 = Buy (verde), -1 = Sell (vermelha)
    entry_streak_size = 0
    entry_candle_idx = 0
    total_loss_in_sequence = 0

    # Iterar candle a candle
    for i in range(1, len(df_no_doji)):
        # Dados do candle anterior (que acabamos de ver fechar)
        prev_streak_count = df_no_doji.loc[i-1, 'streak_count']
        prev_color = df_no_doji.loc[i-1, 'color']

        # Dados do candle atual (resultado da aposta)
        current_color = df_no_doji.loc[i, 'color']

        # LOGICA DE ENTRADA (Se nao estamos posicionados)
        if not in_position:
            # Se a sequencia anterior atingiu o Delay, entramos CONTRA ela
            if prev_streak_count >= DELAY_THRESHOLD:
                in_position = True
                current_stake = INITIAL_STAKE
                current_level = 1
                # Apostar contra a tendencia (Reversao)
                bet_direction = -1 * prev_color
                entry_streak_size = prev_streak_count
                entry_candle_idx = i
                total_loss_in_sequence = 0

                # Debug para primeiros trades
                if total_trades < 5:
                    direction_str = "BUY (verde)" if bet_direction > 0 else "SELL (vermelha)"
                    print(f"  Gatilho #{total_trades+1}: Streak de {prev_streak_count} velas")
                    print(f"    -> Entrando {direction_str} no candle {i}")
                    print(f"    -> Nivel 1: ${current_stake:.2f}")

        # LOGICA DE EXECUCAO (Se estamos posicionados)
        else:
            # Verificar resultado
            # Se a cor do candle atual for igual a nossa aposta -> WIN
            if current_color == bet_direction:
                # WIN - Reversao aconteceu!
                profit = current_stake * 0.95  # 5% spread/comissao
                balance += profit
                wins += 1
                total_trades += 1

                # Log do trade
                trades_log.append({
                    'trade_num': total_trades,
                    'entry_candle': entry_candle_idx,
                    'exit_candle': i,
                    'entry_streak': entry_streak_size,
                    'levels_used': current_level,
                    'profit': profit,
                    'balance': balance,
                    'result': 'WIN'
                })

                # Reset
                in_position = False

                if total_trades <= 5:
                    print(f"    -> WIN no Nivel {current_level}!")
                    print(f"       Lucro: +${profit:.2f}")
                    print(f"       Banca: ${balance:.2f}")
                    print()

            else:
                # LOSS (Mercado continuou a tendencia)
                balance -= current_stake
                total_loss_in_sequence += current_stake

                # Verificar Quebra de Banca
                if balance <= 0:
                    print(f"\n[BUST] Banca quebrada no candle {i}!")
                    print(f"  Streak inicial: {entry_streak_size} velas")
                    print(f"  Nivel atingido: {current_level}")
                    print(f"  Perda total: ${total_loss_in_sequence:.2f}")

                    balance = 0
                    busts += 1
                    total_trades += 1
                    losses += 1

                    trades_log.append({
                        'trade_num': total_trades,
                        'entry_candle': entry_candle_idx,
                        'exit_candle': i,
                        'entry_streak': entry_streak_size,
                        'levels_used': current_level,
                        'profit': -total_loss_in_sequence,
                        'balance': 0,
                        'result': 'BUST'
                    })

                    break

                # Martingale: Dobrar aposta
                current_level += 1
                current_stake *= MULTIPLIER

                # Verificar se atingiu limite de niveis
                if current_level > MAX_LEVEL:
                    print(f"\n[STOP LOSS] Nivel maximo ({MAX_LEVEL}) atingido!")
                    print(f"  Perda total da sequencia: ${total_loss_in_sequence:.2f}")

                    losses += 1
                    total_trades += 1

                    trades_log.append({
                        'trade_num': total_trades,
                        'entry_candle': entry_candle_idx,
                        'exit_candle': i,
                        'entry_streak': entry_streak_size,
                        'levels_used': current_level - 1,
                        'profit': -total_loss_in_sequence,
                        'balance': balance,
                        'result': 'STOP_LOSS'
                    })

                    in_position = False

                # Verificar se saldo aguenta proxima aposta
                elif current_stake > balance:
                    print(f"\n[FORCED STOP] Saldo insuficiente para Nivel {current_level} (${current_stake:.2f})")
                    print(f"  Perda total da sequencia: ${total_loss_in_sequence:.2f}")

                    losses += 1
                    total_trades += 1
                    busts += 1

                    trades_log.append({
                        'trade_num': total_trades,
                        'entry_candle': entry_candle_idx,
                        'exit_candle': i,
                        'entry_streak': entry_streak_size,
                        'levels_used': current_level - 1,
                        'profit': -total_loss_in_sequence,
                        'balance': balance,
                        'result': 'FORCED_STOP'
                    })

                    in_position = False
                else:
                    # Continua para o proximo candle
                    if total_trades < 5 and current_level <= 3:
                        print(f"    -> LOSS. Subindo para Nivel {current_level}: ${current_stake:.2f}")

        # Tracking
        balance_history.append(balance)
        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
        if dd > max_drawdown:
            max_drawdown = dd

    # 5. Resultados Finais
    print(f"\n{'='*70}")
    print("RESULTADO FINAL - ESTRATEGIA DO ABUTRE")
    print(f"{'='*70}\n")

    roi = ((balance - BANKROLL) / BANKROLL) * 100
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    print(f"[PERFORMANCE]")
    print(f"  Banca Inicial:   ${BANKROLL:.2f}")
    print(f"  Banca Final:     ${balance:.2f}")
    print(f"  Lucro Liquido:   ${balance - BANKROLL:.2f} ({roi:+.2f}%)")
    print(f"  Max Drawdown:    {max_drawdown*100:.2f}%")
    print(f"  Peak Balance:    ${peak_balance:.2f}")

    print(f"\n[ESTATISTICAS DE TRADES]")
    print(f"  Total Trades:        {total_trades}")
    print(f"  Wins:                {wins} ({win_rate:.1f}%)")
    print(f"  Losses:              {losses}")
    print(f"  Busts (quebras):     {busts}")

    if total_trades > 0:
        avg_profit_per_trade = (balance - BANKROLL) / total_trades
        print(f"  Lucro medio/trade:   ${avg_profit_per_trade:.2f}")

    # Analise de trades
    if trades_log:
        trades_df = pd.DataFrame(trades_log)

        print(f"\n[ANALISE DETALHADA]")

        if wins > 0:
            win_trades = trades_df[trades_df['result'] == 'WIN']
            print(f"\n  TRADES VENCEDORES ({wins}):")
            print(f"    Streak medio de entrada: {win_trades['entry_streak'].mean():.1f} velas")
            print(f"    Niveis medios usados: {win_trades['levels_used'].mean():.1f}")
            print(f"    Lucro medio: ${win_trades['profit'].mean():.2f}")

        if losses > 0:
            loss_trades = trades_df[trades_df['result'] != 'WIN']
            print(f"\n  TRADES PERDEDORES ({losses}):")
            print(f"    Streak medio de entrada: {loss_trades['entry_streak'].mean():.1f} velas")
            print(f"    Niveis medios usados: {loss_trades['levels_used'].mean():.1f}")
            print(f"    Perda media: ${abs(loss_trades['profit'].mean()):.2f}")

    # 6. Graficos
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Plot 1: Equity Curve
    ax1 = axes[0, 0]
    ax1.plot(balance_history, linewidth=2, color='blue')
    ax1.axhline(y=BANKROLL, color='red', linestyle='--', linewidth=2, label=f'Inicial (${BANKROLL:.0f})')
    ax1.set_title(f'Equity Curve - Abutre Strategy (Delay {DELAY_THRESHOLD})', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Candles')
    ax1.set_ylabel('Banca ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Drawdown
    ax2 = axes[0, 1]
    peak_series = pd.Series(balance_history).expanding().max()
    drawdown_series = (peak_series - pd.Series(balance_history)) / peak_series * 100
    ax2.fill_between(range(len(drawdown_series)), drawdown_series, alpha=0.5, color='red')
    ax2.set_title('Drawdown (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Candles')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Trade Results
    if trades_log:
        ax3 = axes[1, 0]
        profits = [t['profit'] for t in trades_log]
        colors = ['green' if p > 0 else 'red' for p in profits]
        ax3.bar(range(len(profits)), profits, color=colors, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.set_title('Lucro por Trade', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Trade #')
        ax3.set_ylabel('Lucro ($)')
        ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Levels Distribution
    if trades_log:
        ax4 = axes[1, 1]
        levels = [t['levels_used'] for t in trades_log]
        ax4.hist(levels, bins=range(1, max(levels)+2), alpha=0.7, color='purple', edgecolor='black')
        ax4.set_title('Distribuicao de Niveis Usados', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Nivel Maximo Atingido')
        ax4.set_ylabel('Frequencia')
        ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    plot_path = data_dir.parent / "reports" / "v100_delayed_martingale.png"
    plot_path.parent.mkdir(exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    print(f"\n[PLOT] Graficos salvos em: {plot_path}")

    # 7. VEREDICTO FINAL
    print(f"\n{'='*70}")
    print("VEREDICTO FINAL - ABUTRE vs TRADICIONAL")
    print(f"{'='*70}\n")

    if busts == 0 and roi > 0:
        print(f"[SUCESSO ABSOLUTO]")
        print(f"  >> A estrategia sobreviveu aos 180 dias E lucrou!")
        print(f"  >> ROI: {roi:+.2f}%")
        print(f"  >> O 'filtro de entrada' (Delay {DELAY_THRESHOLD}) funcionou")
        print(f"  >> Voce transformou banca de $2k em eficacia de $262k")
        print(f"\n  COMPARACAO:")
        print(f"  - Martingale Tradicional: -$0.28/trade (expectativa negativa)")
        print(f"  - Delayed Martingale: ${avg_profit_per_trade:.2f}/trade")
        print(f"\n  PROXIMOS PASSOS:")
        print(f"  1. Forward test em dados out-of-sample (proximos 6 meses)")
        print(f"  2. Paper trading por 30 dias")
        print(f"  3. Live trading com banca de ${BANKROLL:.0f}")

    elif busts > 0:
        print(f"[FALHA - BUST DETECTADO]")
        print(f"  >> Mesmo com Delay {DELAY_THRESHOLD}, houve {busts} quebra(s)")
        print(f"  >> O mercado fez sequencia > {DELAY_THRESHOLD + MAX_LEVEL} velas")
        print(f"  >> Opcoes:")
        print(f"     1. Aumentar Delay para {DELAY_THRESHOLD + 2}")
        print(f"     2. Aumentar banca para ${BANKROLL * 2:.0f}")
        print(f"     3. Reduzir Max Level para {MAX_LEVEL - 2}")

    elif roi <= 0:
        print(f"[SOBREVIVEU MAS NAO LUCROU]")
        print(f"  >> Sem busts, mas ROI {roi:.2f}%")
        print(f"  >> Spread/comissoes comeram o lucro")
        print(f"  >> Poucos trades: {total_trades} em 180 dias")
        print(f"  >> Estrategia e SEGURA mas nao LUCRATIVA")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    simulate_delayed_martingale()
