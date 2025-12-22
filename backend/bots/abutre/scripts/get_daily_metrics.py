"""
ABUTRE BOT - Daily Metrics Extractor

Extrai métricas diárias do banco de dados SQLite para o FORWARD_TEST_LOG.md

Uso:
    python scripts/get_daily_metrics.py

Output:
    - Saldo atual
    - Total de trades (hoje e acumulado)
    - Win rate
    - Max drawdown
    - ROI
    - Últimos 5 trades
"""
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Adicionar backend ao path
backend_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(backend_dir))


def get_db_path():
    """Retorna o caminho do banco de dados"""
    return Path(__file__).parent.parent / "data" / "abutre.db"


def connect_db():
    """Conecta ao banco de dados"""
    db_path = get_db_path()
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        print("ℹ️  Bot precisa rodar pelo menos uma vez para criar o banco.")
        return None
    return sqlite3.connect(db_path)


def get_current_balance(cursor):
    """Retorna o saldo atual"""
    cursor.execute("""
        SELECT balance FROM balance_history
        ORDER BY timestamp DESC
        LIMIT 1
    """)
    result = cursor.fetchone()
    return result[0] if result else 2000.0  # Default inicial


def get_total_trades(cursor, today_only=False):
    """Retorna total de trades (opcionalmente apenas hoje)"""
    if today_only:
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute("""
            SELECT COUNT(*) FROM trades
            WHERE DATE(entry_time) = ?
        """, (today,))
    else:
        cursor.execute("SELECT COUNT(*) FROM trades")

    result = cursor.fetchone()
    return result[0] if result else 0


def get_win_rate(cursor):
    """Calcula win rate (%)"""
    cursor.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as wins
        FROM trades
        WHERE exit_time IS NOT NULL
    """)
    result = cursor.fetchone()

    if not result or result[0] == 0:
        return 0.0

    total, wins = result
    return (wins / total) * 100


def get_max_drawdown(cursor):
    """Calcula max drawdown (%)"""
    cursor.execute("""
        SELECT balance FROM balance_history
        ORDER BY timestamp ASC
    """)
    balances = [row[0] for row in cursor.fetchall()]

    if not balances:
        return 0.0

    peak = balances[0]
    max_dd = 0.0

    for balance in balances:
        if balance > peak:
            peak = balance
        dd = ((peak - balance) / peak) * 100
        if dd > max_dd:
            max_dd = dd

    return max_dd


def get_roi(cursor):
    """Calcula ROI (%)"""
    # Saldo inicial
    initial_balance = 2000.0

    # Saldo atual
    current_balance = get_current_balance(cursor)

    # ROI
    roi = ((current_balance - initial_balance) / initial_balance) * 100

    return roi, current_balance


def get_recent_trades(cursor, limit=5):
    """Retorna últimos N trades"""
    cursor.execute("""
        SELECT
            entry_time,
            direction,
            level,
            stake,
            profit,
            status
        FROM trades
        ORDER BY entry_time DESC
        LIMIT ?
    """, (limit,))

    return cursor.fetchall()


def get_today_events(cursor):
    """Retorna eventos importantes de hoje"""
    today = datetime.now().strftime('%Y-%m-%d')

    # Trades
    cursor.execute("""
        SELECT
            entry_time,
            direction,
            level,
            profit,
            status
        FROM trades
        WHERE DATE(entry_time) = ?
        ORDER BY entry_time ASC
    """, (today,))

    trades = cursor.fetchall()

    events = []
    for trade in trades:
        time = datetime.fromisoformat(trade[0]).strftime('%H:%M:%S')
        direction = trade[1]
        level = trade[2]
        profit = trade[3]
        status = trade[4]

        result = "WIN" if profit > 0 else "LOSS"
        events.append(f"[{time}] Trade {direction} L{level}: {result} (${profit:+.2f})")

    # Gatilhos detectados (se houver tabela de signals)
    try:
        cursor.execute("""
            SELECT
                timestamp,
                streak_count,
                direction
            FROM signals
            WHERE DATE(timestamp) = ?
            ORDER BY timestamp ASC
        """, (today,))

        signals = cursor.fetchall()
        for signal in signals:
            time = datetime.fromisoformat(signal[0]).strftime('%H:%M:%S')
            streak = signal[1]
            direction = signal[2]
            events.append(f"[{time}] Trigger detected: {streak} velas consecutivas → {direction}")
    except sqlite3.OperationalError:
        # Tabela signals não existe
        pass

    return events


def format_metrics():
    """Formata métricas do dia para o log"""
    conn = connect_db()
    if not conn:
        return None

    cursor = conn.cursor()

    # Coletar métricas
    roi, balance = get_roi(cursor)
    total_trades = get_total_trades(cursor)
    today_trades = get_total_trades(cursor, today_only=True)
    win_rate = get_win_rate(cursor)
    max_dd = get_max_drawdown(cursor)
    recent_trades = get_recent_trades(cursor, limit=5)
    today_events = get_today_events(cursor)

    conn.close()

    # Formatação
    today = datetime.now().strftime('%d/%m/%Y')

    print(f"\n{'='*60}")
    print(f"📊 MÉTRICAS DIÁRIAS - {today}")
    print(f"{'='*60}\n")

    print(f"💰 Saldo: ${balance:,.2f}")
    print(f"📈 ROI: {roi:+.2f}%")
    print(f"📊 Win Rate: {win_rate:.1f}%")
    print(f"📉 Max Drawdown: {max_dd:.2f}%")
    print(f"🔢 Total Trades: {total_trades} (hoje: {today_trades})")

    print(f"\n{'='*60}")
    print("📅 EVENTOS DE HOJE")
    print(f"{'='*60}\n")

    if today_events:
        for event in today_events:
            print(event)
    else:
        print("⏳ Nenhum evento registrado hoje")

    print(f"\n{'='*60}")
    print("📜 ÚLTIMOS 5 TRADES")
    print(f"{'='*60}\n")

    if recent_trades:
        print(f"{'Time':<20} | {'Dir':<5} | {'Lv':<3} | {'Stake':<8} | {'P&L':<10} | {'Status':<8}")
        print("-" * 70)
        for trade in recent_trades:
            time = datetime.fromisoformat(trade[0]).strftime('%d/%m %H:%M:%S')
            direction = trade[1]
            level = trade[2]
            stake = trade[3]
            profit = trade[4]
            status = trade[5]

            print(f"{time:<20} | {direction:<5} | L{level:<2} | ${stake:<7.2f} | ${profit:+<9.2f} | {status:<8}")
    else:
        print("⏳ Nenhum trade executado ainda")

    print(f"\n{'='*60}\n")

    # Retornar dados estruturados para atualizar o log
    return {
        'date': today,
        'balance': balance,
        'roi': roi,
        'win_rate': win_rate,
        'max_dd': max_dd,
        'total_trades': total_trades,
        'today_trades': today_trades,
        'events': today_events,
        'recent_trades': recent_trades
    }


def generate_log_entry():
    """Gera entrada formatada para copiar no FORWARD_TEST_LOG.md"""
    metrics = format_metrics()

    if not metrics:
        return

    print("\n📝 COPIE ESTA ENTRADA PARA O FORWARD_TEST_LOG.md:\n")
    print("-" * 60)

    # Descobrir qual dia é (calcular diferença de 22/12/2025)
    start_date = datetime(2025, 12, 22)
    current_date = datetime.now()
    day_number = (current_date - start_date).days + 1

    print(f"#### 📆 Dia {day_number} - {metrics['date']}\n")
    print(f"**Métricas:**")
    print(f"- Saldo: ${metrics['balance']:,.2f}")
    print(f"- Total Trades: {metrics['total_trades']} (hoje: {metrics['today_trades']})")
    print(f"- Win Rate: {metrics['win_rate']:.1f}%")
    print(f"- Max Drawdown: {metrics['max_dd']:.2f}%")
    print(f"- ROI: {metrics['roi']:+.2f}%\n")

    print(f"**Eventos:**")
    if metrics['events']:
        for event in metrics['events']:
            print(f"- {event}")
    else:
        print("- ⏳ Aguardando sinais do mercado")

    print(f"\n**Observações:**")

    # Análise automática
    if metrics['today_trades'] == 0:
        print("- ⏳ Nenhum trade executado hoje (aguardando gatilho)")
    elif metrics['win_rate'] == 100:
        print("- ✅ Win rate perfeito mantido")
    elif metrics['win_rate'] >= 90:
        print("- ✅ Win rate dentro do esperado (>90%)")
    else:
        print("- ⚠️ Win rate abaixo do esperado (<90%)")

    if metrics['max_dd'] > 30:
        print("- ❌ ATENÇÃO: Drawdown acima do limite (>30%)")
    elif metrics['max_dd'] > 20:
        print("- ⚠️ Drawdown elevado mas dentro do limite")
    else:
        print("- ✅ Drawdown sob controle")

    if metrics['roi'] > 0:
        print("- ✅ ROI positivo")
    else:
        print("- ⚠️ ROI negativo - monitorar")

    print("\n" + "-" * 60)


if __name__ == "__main__":
    print("\n🤖 ABUTRE BOT - Daily Metrics Extractor\n")
    generate_log_entry()
