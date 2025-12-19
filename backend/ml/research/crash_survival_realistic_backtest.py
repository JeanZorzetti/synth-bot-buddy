"""
CRASH 500 - BACKTEST REALISTA

Backtest que simula CORRETAMENTE a execução de trades com:
- Stop Loss e Take Profit em %
- Timeout de posição
- Slippage de execução
- Latência de entrada

Objetivo: Validar se o modelo LSTM Survival realmente funciona em condições reais
"""

import os
# Force CPU only (evita problemas com CUDA)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json

# Import do modelo
from crash_survival_model import LSTMSurvivalModel


class RealisticBacktester:
    """
    Backtest realista que simula trades com SL/TP
    """

    def __init__(
        self,
        model,
        device,
        initial_capital: float = 10000.0,
        position_size_pct: float = 2.0,  # 2% do capital por trade
        stop_loss_pct: float = 1.0,      # 1% SL
        take_profit_pct: float = 2.0,    # 2% TP
        max_hold_candles: int = 20,      # 20 candles = 20 min (M1) ou 100 min (M5)
        slippage_pct: float = 0.1,       # 0.1% slippage
        latency_candles: int = 1,        # 1 candle de latência
        safe_threshold: int = 20,        # Threshold de segurança
        lookback: int = 50,              # Lookback do modelo
    ):
        self.model = model
        self.device = device
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_hold_candles = max_hold_candles
        self.slippage_pct = slippage_pct
        self.latency_candles = latency_candles
        self.safe_threshold = safe_threshold
        self.lookback = lookback

        # Estado
        self.capital = initial_capital
        self.trades = []
        self.equity_curve = []

    def prepare_features(self, candles_df: pd.DataFrame) -> torch.Tensor:
        """
        Prepara features para predição (mesmo método do modelo)
        """
        if len(candles_df) < self.lookback:
            return None

        # Pegar últimos 70 candles (50 lookback + 20 buffer para rolling)
        required_candles = self.lookback + 20
        if len(candles_df) < required_candles:
            return None

        recent_candles = candles_df.iloc[-required_candles:].copy()

        # Calcular volatilidade realizada
        recent_candles['return'] = recent_candles['close'].pct_change()
        recent_candles['realized_vol'] = recent_candles['return'].rolling(window=20).std()

        # Remover NaNs
        recent_candles = recent_candles.dropna()

        if len(recent_candles) < self.lookback:
            return None

        # Pegar exatamente os últimos lookback candles
        recent_candles = recent_candles.iloc[-self.lookback:]

        # Features: OHLC + realized_vol
        features = recent_candles[['open', 'high', 'low', 'close', 'realized_vol']].values.astype(np.float32)

        # Normalização Min-Max por janela
        window_min = features.min(axis=0)
        window_max = features.max(axis=0)
        window_range = window_max - window_min + 1e-8
        features_norm = (features - window_min) / window_range

        # Converter para tensor [1, lookback, 5]
        tensor = torch.FloatTensor(features_norm).unsqueeze(0)

        return tensor

    def predict(self, candles_df: pd.DataFrame) -> dict:
        """
        Gera predição usando o modelo
        """
        features = self.prepare_features(candles_df)

        if features is None:
            return {'signal': 'WAIT', 'candles_to_risk': 0, 'is_safe': False}

        # Predição
        with torch.no_grad():
            features = features.to(self.device)
            candles_pred = self.model(features).cpu().item()

        # Decisão
        is_safe = candles_pred >= self.safe_threshold
        signal = 'LONG' if is_safe else 'WAIT'

        return {
            'signal': signal,
            'candles_to_risk': round(candles_pred, 1),
            'is_safe': is_safe
        }

    def simulate_trade(self, df: pd.DataFrame, entry_idx: int) -> dict:
        """
        Simula execução de um trade LONG

        Returns:
            Dict com resultado do trade
        """
        # 1. Preço de entrada (com latência + slippage)
        entry_candle_idx = entry_idx + self.latency_candles
        if entry_candle_idx >= len(df):
            return None

        entry_price = df.iloc[entry_candle_idx]['close']
        entry_price_with_slippage = entry_price * (1 + self.slippage_pct / 100)

        # 2. Calcular SL e TP
        sl_price = entry_price_with_slippage * (1 - self.stop_loss_pct / 100)
        tp_price = entry_price_with_slippage * (1 + self.take_profit_pct / 100)

        # 3. Tamanho da posição
        position_size = self.capital * (self.position_size_pct / 100)

        # 4. Simular evolução do trade
        exit_idx = None
        exit_reason = None
        exit_price = None

        for j in range(entry_candle_idx + 1, min(entry_candle_idx + 1 + self.max_hold_candles, len(df))):
            candle = df.iloc[j]

            # TP atingido?
            if candle['high'] >= tp_price:
                exit_idx = j
                exit_reason = 'take_profit'
                exit_price = tp_price
                break

            # SL atingido?
            if candle['low'] <= sl_price:
                exit_idx = j
                exit_reason = 'stop_loss'
                exit_price = sl_price
                break

        # Timeout?
        if exit_reason is None:
            exit_idx = min(entry_candle_idx + 1 + self.max_hold_candles, len(df) - 1)
            exit_reason = 'timeout'
            exit_price = df.iloc[exit_idx]['close']
            # Aplicar slippage na saída também
            exit_price = exit_price * (1 - self.slippage_pct / 100)

        # 5. Calcular P&L
        pnl = (exit_price - entry_price_with_slippage) / entry_price_with_slippage * position_size
        pnl_pct = (exit_price - entry_price_with_slippage) / entry_price_with_slippage * 100

        # 6. Atualizar capital
        self.capital += pnl

        # 7. Registro do trade
        trade = {
            'entry_idx': entry_idx,
            'entry_price': entry_price_with_slippage,
            'exit_idx': exit_idx,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'position_size': position_size,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'result': 'WIN' if pnl > 0 else 'LOSS',
            'hold_candles': exit_idx - entry_candle_idx,
            'timestamp_entry': df.iloc[entry_candle_idx]['timestamp'] if 'timestamp' in df.columns else entry_candle_idx,
            'timestamp_exit': df.iloc[exit_idx]['timestamp'] if 'timestamp' in df.columns else exit_idx,
        }

        return trade

    def run(self, df: pd.DataFrame) -> dict:
        """
        Executa backtest completo

        Args:
            df: DataFrame com OHLC + timestamp

        Returns:
            Dict com resultados
        """
        print(f"\n{'='*70}")
        print("BACKTEST REALISTA - CRASH 500 SURVIVAL")
        print(f"{'='*70}")

        print(f"\n[CONFIG]")
        print(f"   Capital inicial: ${self.initial_capital:,.2f}")
        print(f"   Stop Loss: {self.stop_loss_pct}%")
        print(f"   Take Profit: {self.take_profit_pct}%")
        print(f"   Max Hold: {self.max_hold_candles} candles")
        print(f"   Slippage: {self.slippage_pct}%")
        print(f"   Position Size: {self.position_size_pct}% por trade")
        print(f"   Safe Threshold: >= {self.safe_threshold} candles")

        # Reset estado
        self.capital = self.initial_capital
        self.trades = []
        self.equity_curve = []

        # Iterar sobre DataFrame
        print(f"\n[BACKTEST] Processando {len(df):,} candles...")

        i = self.lookback + 20  # Começar após lookback + buffer
        last_trade_idx = -100  # Evitar trades consecutivos

        while i < len(df) - self.max_hold_candles:
            # Progresso
            if i % 500 == 0:
                progress = (i / len(df)) * 100
                print(f"   Progresso: {progress:.1f}% | Trades: {len(self.trades)} | Capital: ${self.capital:,.2f}")

            # 1. Pegar janela de candles até o candle atual
            window = df.iloc[:i]

            # 2. Gerar predição
            prediction = self.predict(window)

            # 3. Decisão de entrada
            if prediction['signal'] == 'LONG':
                # Evitar trades muito próximos
                if i - last_trade_idx < 5:
                    i += 1
                    continue

                # Simular trade
                trade = self.simulate_trade(df, i)

                if trade:
                    self.trades.append(trade)
                    last_trade_idx = i

                    # Pular para depois do exit
                    i = trade['exit_idx'] + 1
                else:
                    i += 1
            else:
                i += 1

            # Registrar equity
            self.equity_curve.append({
                'idx': i,
                'capital': self.capital
            })

        # Calcular métricas
        print(f"\n[RESULTADOS]")
        metrics = self._calculate_metrics()

        return {
            'metrics': metrics,
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }

    def _calculate_metrics(self) -> dict:
        """
        Calcula métricas de performance
        """
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'total_pnl': 0.0,
                'total_pnl_pct': 0.0,
            }

        wins = [t for t in self.trades if t['result'] == 'WIN']
        losses = [t for t in self.trades if t['result'] == 'LOSS']

        # Breakdown por exit reason
        tp_trades = [t for t in self.trades if t['exit_reason'] == 'take_profit']
        sl_trades = [t for t in self.trades if t['exit_reason'] == 'stop_loss']
        timeout_trades = [t for t in self.trades if t['exit_reason'] == 'timeout']

        # Métricas básicas
        total_trades = len(self.trades)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0

        avg_pnl = np.mean([t['pnl'] for t in self.trades])
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0

        total_pnl = self.capital - self.initial_capital
        total_pnl_pct = (total_pnl / self.initial_capital) * 100

        # Profit factor
        gross_profit = sum([t['pnl'] for t in wins]) if wins else 0
        gross_loss = abs(sum([t['pnl'] for t in losses])) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Max drawdown
        equity = [self.initial_capital] + [self.initial_capital + sum([t['pnl'] for t in self.trades[:i+1]]) for i in range(len(self.trades))]
        peak = equity[0]
        max_dd = 0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / peak
            if dd > max_dd:
                max_dd = dd

        # Sharpe ratio (simplificado)
        returns = [t['pnl_pct'] for t in self.trades]
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if returns and np.std(returns) > 0 else 0

        # Average hold time
        avg_hold = np.mean([t['hold_candles'] for t in self.trades])

        metrics = {
            'total_trades': total_trades,
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': win_rate * 100,  # %

            'avg_pnl': avg_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,

            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,

            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_dd * 100,

            'avg_hold_candles': avg_hold,

            # Breakdown
            'tp_trades': len(tp_trades),
            'sl_trades': len(sl_trades),
            'timeout_trades': len(timeout_trades),

            'tp_rate': len(tp_trades) / total_trades * 100 if total_trades > 0 else 0,
            'sl_rate': len(sl_trades) / total_trades * 100 if total_trades > 0 else 0,
            'timeout_rate': len(timeout_trades) / total_trades * 100 if total_trades > 0 else 0,

            # Capital
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
        }

        # Print
        print(f"   Total Trades: {metrics['total_trades']}")
        print(f"   Wins: {metrics['winning_trades']} | Losses: {metrics['losing_trades']}")
        print(f"   Win Rate: {metrics['win_rate']:.2f}%")
        print(f"\n   P&L:")
        print(f"      Total: ${metrics['total_pnl']:,.2f} ({metrics['total_pnl_pct']:+.2f}%)")
        print(f"      Avg Win: ${metrics['avg_win']:,.2f}")
        print(f"      Avg Loss: ${metrics['avg_loss']:,.2f}")
        print(f"\n   Risk Metrics:")
        print(f"      Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"      Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"      Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print(f"\n   Exit Breakdown:")
        print(f"      Take Profit: {metrics['tp_trades']} ({metrics['tp_rate']:.1f}%)")
        print(f"      Stop Loss: {metrics['sl_trades']} ({metrics['sl_rate']:.1f}%)")
        print(f"      Timeout: {metrics['timeout_trades']} ({metrics['timeout_rate']:.1f}%)")
        print(f"\n   Avg Hold Time: {metrics['avg_hold_candles']:.1f} candles")

        return metrics


def main():
    print("="*70)
    print("CRASH 500 - BACKTEST REALISTA")
    print("="*70)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[DEVICE] {device}")

    # 1. Carregar modelo treinado
    print(f"\n[MODEL] Carregando modelo treinado...")
    model_path = Path(__file__).parent / "models" / "crash_survival_lstm.pth"

    model = LSTMSurvivalModel(input_dim=5, hidden_dim1=128, hidden_dim2=64)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print(f"   Modelo carregado: {model_path.name}")

    # 2. Carregar dados de teste
    print(f"\n[DATA] Carregando dados de teste...")
    data_dir = Path(__file__).parent / "data"
    df_full = pd.read_csv(data_dir / "CRASH500_5min_survival_labeled.csv")
    df_full = df_full.dropna()

    # Usar apenas test set (últimos 15%)
    test_start = int(0.85 * len(df_full))
    df_test = df_full.iloc[test_start:].reset_index(drop=True)

    print(f"   Total candles (full): {len(df_full):,}")
    print(f"   Test set: {len(df_test):,} candles")

    # 3. Criar backtester
    print(f"\n[BACKTEST] Inicializando backtester...")

    backtester = RealisticBacktester(
        model=model,
        device=device,
        initial_capital=10000.0,
        position_size_pct=2.0,
        stop_loss_pct=1.0,
        take_profit_pct=2.0,
        max_hold_candles=20,
        slippage_pct=0.1,
        latency_candles=1,
        safe_threshold=20,
        lookback=50,
    )

    # 4. Executar backtest
    results = backtester.run(df_test)

    # 5. Salvar resultados
    print(f"\n[SAVE] Salvando resultados...")

    output_dir = Path(__file__).parent / "reports"
    output_dir.mkdir(exist_ok=True)

    # Métricas
    metrics_file = output_dir / "crash500_realistic_backtest_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(results['metrics'], f, indent=2)

    # Trades
    trades_file = output_dir / "crash500_realistic_backtest_trades.json"
    with open(trades_file, 'w') as f:
        json.dump(results['trades'], f, indent=2)

    # Equity curve
    equity_file = output_dir / "crash500_realistic_backtest_equity.json"
    with open(equity_file, 'w') as f:
        json.dump(results['equity_curve'], f, indent=2)

    print(f"   Métricas: {metrics_file}")
    print(f"   Trades: {trades_file}")
    print(f"   Equity: {equity_file}")

    # 6. Comparação com backtest original
    print(f"\n{'='*70}")
    print("COMPARAÇÃO: BACKTEST ORIGINAL vs REALISTA")
    print(f"{'='*70}")

    print(f"\n{'Métrica':<30} | {'Original':<15} | {'Realista':<15}")
    print(f"{'-'*30}-+-{'-'*15}-+-{'-'*15}")
    print(f"{'Win Rate':<30} | {'91.81%':<15} | {results['metrics']['win_rate']:.2f}%")
    print(f"{'Total Trades':<30} | {'1,478':<15} | {results['metrics']['total_trades']:<15}")
    print(f"{'Profit Factor':<30} | {'N/A':<15} | {results['metrics']['profit_factor']:.2f}")
    print(f"{'Sharpe Ratio':<30} | {'N/A':<15} | {results['metrics']['sharpe_ratio']:.2f}")
    print(f"{'Max Drawdown':<30} | {'N/A':<15} | {results['metrics']['max_drawdown_pct']:.2f}%")

    print(f"\n{'='*70}")
    print("CONCLUSÃO")
    print(f"{'='*70}")

    if results['metrics']['win_rate'] >= 60:
        print("✅ MODELO APROVADO para produção!")
        print(f"   Win rate realista: {results['metrics']['win_rate']:.2f}% (>= 60%)")
    else:
        print("❌ MODELO REPROVADO")
        print(f"   Win rate realista: {results['metrics']['win_rate']:.2f}% (< 60%)")
        print("\n   OVERFITTING CONFIRMADO!")
        print("   O backtest original de 91.81% não refletiu a realidade.")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
