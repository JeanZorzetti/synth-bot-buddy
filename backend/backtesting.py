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
        self.use_vectorized = True  # Flag para usar backtesting vetorizado por padrão

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
        for i in range(200, len(df) - 1):  # Precisa de 200 candles para indicadores
            current_candle = df.iloc[i]
            historical_data = df.iloc[:i+1]

            # Gerar sinal (já calcula indicadores internamente)
            signal = self.ta.generate_signal(historical_data, symbol)

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

    def run_vectorized_backtest(
        self,
        df: pd.DataFrame,
        strategy_signals: pd.Series,
        initial_balance: float = 1000.0,
        position_size_percent: float = 10.0,
        stop_loss_percent: float = 2.0,
        take_profit_percent: float = 4.0
    ) -> BacktestResult:
        """
        Backtesting vetorizado usando operações Pandas/NumPy
        10-100x mais rápido que backtesting iterativo

        Args:
            df: DataFrame com OHLC data
            strategy_signals: Series com sinais (-1: SELL, 0: NEUTRAL, 1: BUY)
            initial_balance: Saldo inicial
            position_size_percent: % do saldo para cada trade
            stop_loss_percent: % de stop loss
            take_profit_percent: % de take profit

        Returns:
            BacktestResult com métricas de performance
        """
        logger.info(f"Iniciando backtest vetorizado com {len(df)} candles")

        result = BacktestResult()
        result.initial_balance = initial_balance

        # Copiar DataFrame para não modificar original
        data = df.copy()
        data['signal'] = strategy_signals

        # Calcular retornos
        data['returns'] = data['close'].pct_change()

        # Aplicar sinais com lag (sinal em t afeta retorno em t+1)
        data['positions'] = data['signal'].shift(1)

        # Calcular retornos da estratégia
        data['strategy_returns'] = data['positions'] * data['returns']

        # Aplicar stop loss e take profit de forma vetorizada
        data['sl_triggered'] = False
        data['tp_triggered'] = False

        # Para posições LONG (1)
        long_mask = data['positions'] == 1
        data.loc[long_mask, 'sl_triggered'] = data.loc[long_mask, 'returns'] <= -stop_loss_percent / 100
        data.loc[long_mask, 'tp_triggered'] = data.loc[long_mask, 'returns'] >= take_profit_percent / 100

        # Para posições SHORT (-1)
        short_mask = data['positions'] == -1
        data.loc[short_mask, 'sl_triggered'] = data.loc[short_mask, 'returns'] >= stop_loss_percent / 100
        data.loc[short_mask, 'tp_triggered'] = data.loc[short_mask, 'returns'] <= -take_profit_percent / 100

        # Ajustar retornos quando SL/TP acionados
        data.loc[data['sl_triggered'], 'strategy_returns'] = -stop_loss_percent / 100 * data.loc[data['sl_triggered'], 'positions']
        data.loc[data['tp_triggered'], 'strategy_returns'] = take_profit_percent / 100 * data.loc[data['tp_triggered'], 'positions']

        # Aplicar position sizing
        position_multiplier = position_size_percent / 100
        data['strategy_returns'] = data['strategy_returns'] * position_multiplier

        # Calcular curva de capital (equity curve)
        data['cumulative_returns'] = (1 + data['strategy_returns']).cumprod()
        data['equity'] = initial_balance * data['cumulative_returns']

        # Calcular drawdown vetorizado
        data['running_max'] = data['equity'].expanding().max()
        data['drawdown'] = (data['equity'] - data['running_max']) / data['running_max'] * 100

        # Identificar trades individuais (mudanças de posição)
        data['position_change'] = data['positions'].diff()
        entries = data[data['position_change'] != 0].copy()

        # Criar lista de trades
        trades = []
        for i in range(len(entries) - 1):
            entry = entries.iloc[i]
            exit_entry = entries.iloc[i + 1]

            if entry['positions'] != 0:  # Ignora posições flat
                entry_idx = entry.name
                exit_idx = exit_entry.name

                # Calcular profit do trade
                trade_data = data.loc[entry_idx:exit_idx]
                trade_return = trade_data['strategy_returns'].sum()
                trade_profit = initial_balance * trade_return

                # Determinar motivo de saída
                if trade_data['sl_triggered'].any():
                    exit_reason = 'Stop Loss'
                elif trade_data['tp_triggered'].any():
                    exit_reason = 'Take Profit'
                else:
                    exit_reason = 'Signal Reversal'

                trade_record = {
                    'entry_time': entry_idx,
                    'exit_time': exit_idx,
                    'signal_type': 'BUY' if entry['positions'] == 1 else 'SELL',
                    'entry_price': entry['close'],
                    'exit_price': exit_entry['close'],
                    'position_size': initial_balance * position_multiplier,
                    'profit': trade_profit,
                    'profit_percent': trade_return * 100,
                    'exit_reason': exit_reason,
                    'confidence': 75.0,  # Placeholder (não temos dados de confiança em vetorizado)
                    'reason': 'Vectorized backtest',
                    'balance_after': data.loc[exit_idx, 'equity']
                }

                trades.append(trade_record)

        result.trades = trades
        result.final_balance = data['equity'].iloc[-1]

        # Calcular métricas agregadas
        result.calculate_metrics()

        # Calcular métricas adicionais de forma vetorizada
        if len(data['strategy_returns']) > 0:
            returns = data['strategy_returns'].dropna()

            # Sharpe Ratio
            if returns.std() > 0:
                result.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)

            # Max Drawdown
            result.max_drawdown = abs(data['drawdown'].min())

            # Win Rate
            winning_trades = returns[returns > 0]
            result.win_rate = (len(winning_trades) / len(returns) * 100) if len(returns) > 0 else 0

        logger.info(f"Backtest vetorizado completo: {len(trades)} trades, Win Rate: {result.win_rate:.2f}%, Profit: ${result.final_balance - result.initial_balance:.2f}")

        return result

    @staticmethod
    def calculate_max_drawdown_vectorized(returns: pd.Series) -> float:
        """
        Calcula max drawdown de forma vetorizada

        Args:
            returns: Series com retornos da estratégia

        Returns:
            Max drawdown (valor positivo em %)
        """
        # Calcular curva de capital
        cumulative = (1 + returns).cumprod()

        # Calcular running maximum
        running_max = cumulative.expanding().max()

        # Calcular drawdown
        drawdown = (cumulative - running_max) / running_max * 100

        return abs(drawdown.min())

    def compare_strategies(
        self,
        df: pd.DataFrame,
        strategies: Dict[str, pd.Series]
    ) -> Dict[str, BacktestResult]:
        """
        Compara múltiplas estratégias usando backtesting vetorizado

        Args:
            df: DataFrame com OHLC data
            strategies: Dict com nome da estratégia -> Series de sinais

        Returns:
            Dict com nome -> BacktestResult
        """
        results = {}

        for name, signals in strategies.items():
            logger.info(f"Testando estratégia: {name}")
            results[name] = self.run_vectorized_backtest(df, signals)

        # Ranking por Sharpe Ratio
        ranked = sorted(
            results.items(),
            key=lambda x: x[1].sharpe_ratio,
            reverse=True
        )

        logger.info("\n=== RANKING DE ESTRATÉGIAS (por Sharpe Ratio) ===")
        for i, (name, result) in enumerate(ranked, 1):
            logger.info(f"{i}. {name}: Sharpe={result.sharpe_ratio:.2f}, Win Rate={result.win_rate:.2f}%, Profit=${result.final_balance - result.initial_balance:.2f}")

        return results
