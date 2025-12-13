"""
Sistema de Backtesting Integrado com Risk Management
Valida todos os trades usando RiskManager antes de executar
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import logging
from risk_manager import RiskManager, RiskLimits
from backtesting import BacktestResult

logger = logging.getLogger(__name__)


class RiskManagedBacktester:
    """
    Backtester que integra validação de risco em cada trade

    Compara performance com e sem risk management para demonstrar
    o valor da gestão de risco.
    """

    def __init__(
        self,
        initial_capital: float = 1000.0,
        use_risk_management: bool = True,
        risk_limits: Optional[RiskLimits] = None
    ):
        """
        Args:
            initial_capital: Capital inicial
            use_risk_management: Se True, valida trades com RiskManager
            risk_limits: Limites customizados (opcional)
        """
        self.initial_capital = initial_capital
        self.use_risk_management = use_risk_management

        if use_risk_management:
            self.risk_manager = RiskManager(
                initial_capital=initial_capital,
                risk_limits=risk_limits
            )
        else:
            self.risk_manager = None

        self.results_with_risk = None
        self.results_without_risk = None

    def run_backtest(
        self,
        signals: List[Dict],
        candles: pd.DataFrame,
        atr_period: int = 14
    ) -> Dict:
        """
        Executa backtest com validação de risco

        Args:
            signals: Lista de sinais de trading gerados
            candles: DataFrame com dados de candles (OHLCV + ATR)
            atr_period: Período do ATR para stop loss

        Returns:
            Dicionário com resultados e comparação
        """
        logger.info(f"Iniciando backtest com {len(signals)} sinais")
        logger.info(f"Risk Management: {'ATIVO' if self.use_risk_management else 'DESATIVADO'}")

        # Calcular ATR se não existir
        if 'atr' not in candles.columns:
            candles = self._calculate_atr(candles, atr_period)

        # Executar backtest COM risk management
        result_with = self._execute_backtest(
            signals=signals,
            candles=candles,
            use_risk=True
        )

        # Executar backtest SEM risk management (para comparação)
        result_without = self._execute_backtest(
            signals=signals,
            candles=candles,
            use_risk=False
        )

        # Comparar resultados
        comparison = self._compare_results(result_with, result_without)

        return {
            'with_risk_management': result_with.to_dict(),
            'without_risk_management': result_without.to_dict(),
            'comparison': comparison,
            'risk_management_enabled': self.use_risk_management
        }

    def _execute_backtest(
        self,
        signals: List[Dict],
        candles: pd.DataFrame,
        use_risk: bool
    ) -> BacktestResult:
        """Executa backtest com ou sem validação de risco"""
        result = BacktestResult()
        result.initial_balance = self.initial_capital

        current_balance = self.initial_capital
        peak_balance = self.initial_capital

        # Reset risk manager se usar
        if use_risk and self.risk_manager:
            self.risk_manager = RiskManager(
                initial_capital=self.initial_capital,
                risk_limits=self.risk_manager.limits
            )

        trades_executed = 0
        trades_rejected = 0

        for signal in signals:
            try:
                # Buscar candle correspondente
                candle_idx = candles[candles['time'] == signal['timestamp']].index
                if len(candle_idx) == 0:
                    continue

                candle = candles.loc[candle_idx[0]]

                # Calcular stop loss usando ATR
                atr = candle['atr']
                entry_price = signal['entry_price']
                is_long = signal['signal'] == 'BUY'

                if use_risk and self.risk_manager:
                    # Usar RiskManager para calcular stop loss
                    stop_loss = self.risk_manager.calculate_atr_stop_loss(
                        current_price=entry_price,
                        atr=atr,
                        is_long=is_long,
                        multiplier=2.0
                    )
                else:
                    # Stop loss fixo 2%
                    stop_loss = entry_price * (0.98 if is_long else 1.02)

                # Calcular take profit
                if use_risk and self.risk_manager:
                    tp1, tp2 = self.risk_manager.calculate_take_profit(
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        is_long=is_long,
                        risk_reward_ratio=2.0
                    )
                    take_profit = tp2  # Usar TP2 para simplicidade
                else:
                    # Take profit fixo 4%
                    take_profit = entry_price * (1.04 if is_long else 0.96)

                # Calcular position size
                if use_risk and self.risk_manager:
                    position_size = self.risk_manager.calculate_position_size(
                        entry_price=entry_price,
                        stop_loss=stop_loss
                    )
                else:
                    # Position size fixo 10% do capital
                    position_size = current_balance * 0.10

                # Validar trade se usar risk management
                if use_risk and self.risk_manager:
                    is_valid, reason = self.risk_manager.validate_trade(
                        symbol=signal.get('symbol', 'UNKNOWN'),
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        position_size=position_size
                    )

                    if not is_valid:
                        logger.warning(f"Trade rejeitado: {reason}")
                        trades_rejected += 1
                        continue

                # Simular execução do trade
                profit = self._simulate_trade(
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=position_size,
                    is_long=is_long,
                    candles=candles,
                    start_idx=candle_idx[0]
                )

                current_balance += profit

                # Atualizar risk manager
                if use_risk and self.risk_manager:
                    self.risk_manager.current_capital = current_balance
                    if profit > 0:
                        self.risk_manager.consecutive_losses = 0
                        self.risk_manager.is_circuit_breaker_active = False
                    else:
                        self.risk_manager.consecutive_losses += 1
                        if self.risk_manager.consecutive_losses >= self.risk_manager.limits.circuit_breaker_losses:
                            self.risk_manager.is_circuit_breaker_active = True

                # Atualizar peak
                if current_balance > peak_balance:
                    peak_balance = current_balance

                # Registrar trade
                result.trades.append({
                    'timestamp': signal['timestamp'],
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'position_size': position_size,
                    'is_long': is_long,
                    'profit': profit,
                    'balance_after': current_balance
                })

                trades_executed += 1

            except Exception as e:
                logger.error(f"Erro ao processar sinal: {e}", exc_info=True)
                continue

        # Calcular métricas finais
        result.calculate_metrics()

        logger.info(f"Backtest concluído: {trades_executed} trades executados, {trades_rejected} rejeitados")
        logger.info(f"Balance final: ${current_balance:.2f} (Profit: {(current_balance/self.initial_capital - 1)*100:.2f}%)")

        return result

    def _simulate_trade(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        position_size: float,
        is_long: bool,
        candles: pd.DataFrame,
        start_idx: int,
        max_bars: int = 100
    ) -> float:
        """
        Simula execução de um trade nos próximos candles

        Returns:
            Lucro/prejuízo do trade
        """
        for i in range(start_idx + 1, min(start_idx + max_bars, len(candles))):
            candle = candles.iloc[i]
            high = candle['high']
            low = candle['low']

            if is_long:
                # Trade LONG
                if low <= stop_loss:
                    # Stop loss atingido
                    loss = (stop_loss - entry_price) * (position_size / entry_price)
                    return loss
                elif high >= take_profit:
                    # Take profit atingido
                    profit = (take_profit - entry_price) * (position_size / entry_price)
                    return profit
            else:
                # Trade SHORT
                if high >= stop_loss:
                    # Stop loss atingido
                    loss = (entry_price - stop_loss) * (position_size / entry_price)
                    return loss
                elif low <= take_profit:
                    # Take profit atingido
                    profit = (entry_price - take_profit) * (position_size / entry_price)
                    return profit

        # Trade não fechou - fechar no último candle
        last_price = candles.iloc[min(start_idx + max_bars - 1, len(candles) - 1)]['close']
        if is_long:
            return (last_price - entry_price) * (position_size / entry_price)
        else:
            return (entry_price - last_price) * (position_size / entry_price)

    def _calculate_atr(self, candles: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calcula Average True Range"""
        df = candles.copy()

        df['h_l'] = df['high'] - df['low']
        df['h_pc'] = abs(df['high'] - df['close'].shift(1))
        df['l_pc'] = abs(df['low'] - df['close'].shift(1))

        df['tr'] = df[['h_l', 'h_pc', 'l_pc']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=period).mean()

        # Preencher NaN com primeira ATR válida
        first_valid_atr = df['atr'].dropna().iloc[0] if len(df['atr'].dropna()) > 0 else 0.01
        df['atr'].fillna(first_valid_atr, inplace=True)

        return df

    def _compare_results(
        self,
        with_risk: BacktestResult,
        without_risk: BacktestResult
    ) -> Dict:
        """
        Compara resultados COM e SEM risk management

        Demonstra o valor da gestão de risco
        """
        comparison = {
            'profit_improvement': {
                'with_risk': with_risk.final_balance - with_risk.initial_balance,
                'without_risk': without_risk.final_balance - without_risk.initial_balance,
                'difference': (with_risk.final_balance - without_risk.final_balance),
                'difference_percent': ((with_risk.final_balance / without_risk.final_balance - 1) * 100) if without_risk.final_balance > 0 else 0
            },
            'drawdown_reduction': {
                'with_risk': with_risk.max_drawdown,
                'without_risk': without_risk.max_drawdown,
                'reduction': without_risk.max_drawdown - with_risk.max_drawdown,
                'reduction_percent': ((1 - with_risk.max_drawdown / without_risk.max_drawdown) * 100) if without_risk.max_drawdown > 0 else 0
            },
            'sharpe_improvement': {
                'with_risk': with_risk.sharpe_ratio,
                'without_risk': without_risk.sharpe_ratio,
                'improvement': with_risk.sharpe_ratio - without_risk.sharpe_ratio,
                'improvement_percent': ((with_risk.sharpe_ratio / without_risk.sharpe_ratio - 1) * 100) if without_risk.sharpe_ratio > 0 else 0
            },
            'trades': {
                'with_risk': len(with_risk.trades),
                'without_risk': len(without_risk.trades),
                'trades_filtered': len(without_risk.trades) - len(with_risk.trades),
                'filter_rate': ((len(without_risk.trades) - len(with_risk.trades)) / len(without_risk.trades) * 100) if len(without_risk.trades) > 0 else 0
            },
            'conclusion': self._generate_conclusion(with_risk, without_risk)
        }

        return comparison

    def _generate_conclusion(
        self,
        with_risk: BacktestResult,
        without_risk: BacktestResult
    ) -> str:
        """Gera conclusão sobre o impacto do risk management"""
        profit_diff = with_risk.final_balance - without_risk.final_balance
        dd_reduction = without_risk.max_drawdown - with_risk.max_drawdown

        if profit_diff > 0 and dd_reduction > 0:
            return f"Risk Management AUMENTOU lucro em ${profit_diff:.2f} e REDUZIU drawdown em {dd_reduction:.2f}%. Altamente recomendado!"
        elif profit_diff > 0:
            return f"Risk Management AUMENTOU lucro em ${profit_diff:.2f}. Recomendado!"
        elif dd_reduction > 0:
            return f"Risk Management REDUZIU drawdown em {dd_reduction:.2f}%, protegendo capital. Recomendado!"
        else:
            return "Risk Management filtrou trades arriscados mas pode ter reduzido profit. Ajustar parâmetros."
