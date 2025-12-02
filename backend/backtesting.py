"""
Sistema de Backtesting para Análise Técnica
Testa indicadores e sinais em dados históricos
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class BacktestResult:
    """Resultado de um backtest"""

    def __init__(self):
        self.trades: List[Dict] = []
        self.initial_balance = 1000.0
        self.final_balance = 1000.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.win_rate = 0.0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.profit_factor = 0.0
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.largest_win = 0.0
        self.largest_loss = 0.0

    def calculate_metrics(self):
        """Calcula métricas de performance"""
        if not self.trades:
            return

        self.total_trades = len(self.trades)

        profits = [t['profit'] for t in self.trades]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]

        self.winning_trades = len(wins)
        self.losing_trades = len(losses)
        self.win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        self.total_profit = sum(wins) if wins else 0
        self.total_loss = abs(sum(losses)) if losses else 0
        self.profit_factor = (self.total_profit / self.total_loss) if self.total_loss > 0 else 0

        self.avg_win = np.mean(wins) if wins else 0
        self.avg_loss = abs(np.mean(losses)) if losses else 0
        self.largest_win = max(wins) if wins else 0
        self.largest_loss = abs(min(losses)) if losses else 0

        # Calcular drawdown máximo
        balance_curve = [self.initial_balance]
        for trade in self.trades:
            balance_curve.append(balance_curve[-1] + trade['profit'])

        peak = balance_curve[0]
        max_dd = 0
        for balance in balance_curve:
            if balance > peak:
                peak = balance
            dd = (peak - balance) / peak * 100
            if dd > max_dd:
                max_dd = dd

        self.max_drawdown = max_dd
        self.final_balance = balance_curve[-1]

        # Sharpe Ratio simplificado
        if profits:
            returns = np.array(profits) / self.initial_balance
            self.sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0

    def to_dict(self) -> Dict:
        """Converte resultado para dicionário"""
        return {
            'summary': {
                'initial_balance': round(self.initial_balance, 2),
                'final_balance': round(self.final_balance, 2),
                'total_profit': round(self.final_balance - self.initial_balance, 2),
                'total_profit_percent': round((self.final_balance / self.initial_balance - 1) * 100, 2),
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': round(self.win_rate, 2),
                'profit_factor': round(self.profit_factor, 2),
                'max_drawdown': round(self.max_drawdown, 2),
                'sharpe_ratio': round(self.sharpe_ratio, 2),
            },
            'trade_stats': {
                'avg_win': round(self.avg_win, 2),
                'avg_loss': round(self.avg_loss, 2),
                'largest_win': round(self.largest_win, 2),
                'largest_loss': round(self.largest_loss, 2),
                'avg_profit_per_trade': round((self.final_balance - self.initial_balance) / self.total_trades, 2) if self.total_trades > 0 else 0,
            },
            'trades': self.trades[:20]  # Últimas 20 trades para não sobrecarregar
        }


class Backtester:
    """Sistema de backtesting para indicadores técnicos"""

    def __init__(self, technical_analysis):
        self.ta = technical_analysis

    def run_backtest(
        self,
        df: pd.DataFrame,
        symbol: str,
        initial_balance: float = 1000.0,
        position_size_percent: float = 10.0,
        stop_loss_percent: float = 2.0,
        take_profit_percent: float = 4.0
    ) -> BacktestResult:
        """
        Executa backtest em dados históricos

        Args:
            df: DataFrame com OHLC data
            symbol: Símbolo do ativo
            initial_balance: Saldo inicial
            position_size_percent: % do saldo para cada trade
            stop_loss_percent: % de stop loss
            take_profit_percent: % de take profit

        Returns:
            BacktestResult com métricas de performance
        """
        result = BacktestResult()
        result.initial_balance = initial_balance

        current_balance = initial_balance
        open_position = None

        logger.info(f"Iniciando backtest para {symbol} com {len(df)} candles")

        # Iterar sobre os candles
        for i in range(100, len(df) - 1):  # Precisa de 100 candles para indicadores
            current_candle = df.iloc[i]
            historical_data = df.iloc[:i+1]

            # Calcular indicadores até este ponto
            indicators = self.ta.calculate_all_indicators(historical_data)

            if indicators is None:
                continue

            # Gerar sinal
            signal = self.ta.generate_signal(symbol, historical_data, indicators)

            # Se não há posição aberta, procurar entrada
            if open_position is None:
                if signal.signal_type in ['BUY', 'SELL'] and signal.confidence >= 60:
                    position_size = current_balance * (position_size_percent / 100)

                    open_position = {
                        'entry_candle': i,
                        'entry_price': signal.entry_price,
                        'signal_type': signal.signal_type,
                        'position_size': position_size,
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                        'entry_time': current_candle.name if hasattr(current_candle, 'name') else i,
                        'confidence': signal.confidence,
                        'reason': signal.reason
                    }

                    logger.debug(f"Entrada {signal.signal_type} @ {signal.entry_price} (confidence: {signal.confidence}%)")

            # Se há posição aberta, verificar saída
            else:
                next_candle = df.iloc[i + 1]
                close_position = False
                exit_price = None
                exit_reason = None

                # Verificar stop loss e take profit
                if open_position['signal_type'] == 'BUY':
                    if next_candle['low'] <= open_position['stop_loss']:
                        close_position = True
                        exit_price = open_position['stop_loss']
                        exit_reason = 'Stop Loss'
                    elif next_candle['high'] >= open_position['take_profit']:
                        close_position = True
                        exit_price = open_position['take_profit']
                        exit_reason = 'Take Profit'

                elif open_position['signal_type'] == 'SELL':
                    if next_candle['high'] >= open_position['stop_loss']:
                        close_position = True
                        exit_price = open_position['stop_loss']
                        exit_reason = 'Stop Loss'
                    elif next_candle['low'] <= open_position['take_profit']:
                        close_position = True
                        exit_price = open_position['take_profit']
                        exit_reason = 'Take Profit'

                # Fechar posição se condições atendidas
                if close_position:
                    if open_position['signal_type'] == 'BUY':
                        profit = (exit_price - open_position['entry_price']) / open_position['entry_price'] * open_position['position_size']
                    else:  # SELL
                        profit = (open_position['entry_price'] - exit_price) / open_position['entry_price'] * open_position['position_size']

                    current_balance += profit

                    trade_record = {
                        'entry_time': open_position['entry_time'],
                        'exit_time': next_candle.name if hasattr(next_candle, 'name') else i + 1,
                        'signal_type': open_position['signal_type'],
                        'entry_price': round(open_position['entry_price'], 5),
                        'exit_price': round(exit_price, 5),
                        'position_size': round(open_position['position_size'], 2),
                        'profit': round(profit, 2),
                        'profit_percent': round(profit / open_position['position_size'] * 100, 2),
                        'exit_reason': exit_reason,
                        'confidence': round(open_position['confidence'], 2),
                        'reason': open_position['reason'],
                        'balance_after': round(current_balance, 2)
                    }

                    result.trades.append(trade_record)

                    logger.debug(f"Saída @ {exit_price} - Profit: ${profit:.2f} ({exit_reason})")

                    open_position = None

        # Calcular métricas finais
        result.calculate_metrics()

        logger.info(f"Backtest completo: {result.total_trades} trades, Win Rate: {result.win_rate:.2f}%, Profit: ${result.final_balance - result.initial_balance:.2f}")

        return result
