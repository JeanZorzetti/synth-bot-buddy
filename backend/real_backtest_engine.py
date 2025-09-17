"""
Real Backtest Engine - Phase 12 Real Infrastructure
Motor de backtest com dados históricos reais
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import logging

class RealBacktestEngine:
    def __init__(self, strategy, initial_capital: float):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(__name__)

        # Backtest state
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.current_datetime = None

    async def run_backtest(
        self,
        historical_data: Dict[str, List[Dict[str, Any]]],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Execute backtest using real historical market data"""

        try:
            # Initialize backtest
            self.current_capital = self.initial_capital
            self.current_datetime = start_date

            # Prepare time-aligned data
            aligned_data = self._align_market_data(historical_data, start_date, end_date)

            if not aligned_data:
                raise ValueError("No aligned market data available")

            # Execute strategy for each time step
            for timestamp, market_snapshot in aligned_data:
                self.current_datetime = timestamp

                # Update open positions
                await self._update_positions(market_snapshot)

                # Execute strategy decision
                signals = await self._execute_strategy_logic(market_snapshot)

                # Process trading signals
                await self._process_signals(signals, market_snapshot)

                # Record equity
                current_equity = self._calculate_current_equity(market_snapshot)
                self.equity_curve.append((timestamp, current_equity))

            # Close remaining positions at end
            await self._close_all_positions(aligned_data[-1][1] if aligned_data else {})

            return {
                'trades': self.trades,
                'equity_curve': self.equity_curve,
                'final_capital': self.current_capital,
                'total_return': (self.current_capital - self.initial_capital) / self.initial_capital
            }

        except Exception as e:
            self.logger.error(f"Backtest execution failed: {e}")
            return {
                'trades': [],
                'equity_curve': [(start_date, self.initial_capital), (end_date, self.initial_capital)],
                'final_capital': self.initial_capital,
                'total_return': 0
            }

    def _align_market_data(
        self,
        historical_data: Dict[str, List[Dict[str, Any]]],
        start_date: datetime,
        end_date: datetime
    ) -> List[Tuple[datetime, Dict[str, Dict[str, Any]]]]:
        """Align market data across symbols by timestamp"""

        if not historical_data:
            return []

        # Get all unique timestamps across all symbols
        all_timestamps = set()
        for symbol_data in historical_data.values():
            for data_point in symbol_data:
                timestamp = data_point.get('timestamp')
                if timestamp and start_date <= timestamp <= end_date:
                    all_timestamps.add(timestamp)

        sorted_timestamps = sorted(all_timestamps)

        # Create aligned data structure
        aligned_data = []
        for timestamp in sorted_timestamps:
            market_snapshot = {}

            for symbol, symbol_data in historical_data.items():
                # Find closest data point for this timestamp
                closest_data = None
                min_time_diff = float('inf')

                for data_point in symbol_data:
                    data_timestamp = data_point.get('timestamp')
                    if data_timestamp:
                        time_diff = abs((data_timestamp - timestamp).total_seconds())
                        if time_diff < min_time_diff and time_diff <= 3600:  # Within 1 hour
                            min_time_diff = time_diff
                            closest_data = data_point

                if closest_data:
                    market_snapshot[symbol] = closest_data

            if market_snapshot:  # Only add if we have data for at least one symbol
                aligned_data.append((timestamp, market_snapshot))

        return aligned_data

    async def _execute_strategy_logic(self, market_snapshot: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute strategy logic and generate trading signals"""

        signals = []

        try:
            # Prepare market data for strategy
            market_data_for_strategy = {}
            for symbol, data in market_snapshot.items():
                market_data_for_strategy[symbol] = {
                    'price': data.get('close_price', data.get('bid', 0)),
                    'bid': data.get('bid', 0),
                    'ask': data.get('ask', 0),
                    'volume': data.get('volume', 0),
                    'timestamp': data.get('timestamp'),
                    'high': data.get('high_price', 0),
                    'low': data.get('low_price', 0),
                    'open': data.get('open_price', 0)
                }

            # Basic strategy logic (can be extended)
            for symbol, data in market_data_for_strategy.items():
                signal = await self._generate_basic_signal(symbol, data, market_data_for_strategy)
                if signal:
                    signals.append(signal)

        except Exception as e:
            self.logger.error(f"Strategy logic error: {e}")

        return signals

    async def _generate_basic_signal(
        self,
        symbol: str,
        current_data: Dict[str, Any],
        all_market_data: Dict[str, Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Generate basic trading signal based on simple strategy"""

        try:
            current_price = current_data.get('price', 0)
            if current_price <= 0:
                return None

            # Simple momentum strategy
            if len(self.equity_curve) >= 10:  # Need some history
                recent_prices = [data.get('close_price', 0) for _, market_data in
                               [(ts, ms) for ts, ms in self.equity_curve[-10:]]
                               if symbol in ms]

                if len(recent_prices) >= 5:
                    price_change = (current_price - recent_prices[0]) / recent_prices[0]

                    # Buy signal: price up > 2%
                    if price_change > 0.02 and symbol not in self.positions:
                        return {
                            'action': 'buy',
                            'symbol': symbol,
                            'price': current_price,
                            'quantity': min(0.1, self.current_capital * 0.1 / current_price),  # Risk 10% of capital
                            'timestamp': self.current_datetime
                        }

                    # Sell signal: price down > 2% or profit > 5%
                    elif symbol in self.positions:
                        position = self.positions[symbol]
                        entry_price = position.get('entry_price', current_price)
                        profit_pct = (current_price - entry_price) / entry_price

                        if price_change < -0.02 or profit_pct > 0.05:
                            return {
                                'action': 'sell',
                                'symbol': symbol,
                                'price': current_price,
                                'quantity': position.get('quantity', 0),
                                'timestamp': self.current_datetime
                            }

        except Exception as e:
            self.logger.error(f"Signal generation error for {symbol}: {e}")

        return None

    async def _process_signals(self, signals: List[Dict[str, Any]], market_snapshot: Dict[str, Dict[str, Any]]):
        """Process trading signals and execute trades"""

        for signal in signals:
            try:
                action = signal.get('action')
                symbol = signal.get('symbol')
                price = signal.get('price', 0)
                quantity = signal.get('quantity', 0)

                if action == 'buy' and symbol not in self.positions:
                    await self._execute_buy(symbol, price, quantity)

                elif action == 'sell' and symbol in self.positions:
                    await self._execute_sell(symbol, price, quantity)

            except Exception as e:
                self.logger.error(f"Signal processing error: {e}")

    async def _execute_buy(self, symbol: str, price: float, quantity: float):
        """Execute buy order"""

        if quantity <= 0 or price <= 0:
            return

        trade_value = price * quantity
        commission = trade_value * 0.001  # 0.1% commission

        if self.current_capital >= trade_value + commission:
            # Execute trade
            self.current_capital -= (trade_value + commission)

            # Create position
            self.positions[symbol] = {
                'symbol': symbol,
                'quantity': quantity,
                'entry_price': price,
                'entry_time': self.current_datetime,
                'commission_paid': commission
            }

            # Record trade
            trade = {
                'trade_id': len(self.trades) + 1,
                'symbol': symbol,
                'entry_time': self.current_datetime,
                'exit_time': None,
                'side': 'buy',
                'quantity': quantity,
                'entry_price': price,
                'exit_price': None,
                'pnl': -commission,  # Initial PnL is negative due to commission
                'commission': commission,
                'status': 'open'
            }

            self.trades.append(trade)

    async def _execute_sell(self, symbol: str, price: float, quantity: float):
        """Execute sell order"""

        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        position_quantity = position.get('quantity', 0)
        entry_price = position.get('entry_price', price)
        entry_commission = position.get('commission_paid', 0)

        # Use position quantity if not specified
        if quantity <= 0:
            quantity = position_quantity

        trade_value = price * quantity
        exit_commission = trade_value * 0.001  # 0.1% commission

        # Calculate PnL
        gross_pnl = (price - entry_price) * quantity
        net_pnl = gross_pnl - entry_commission - exit_commission

        # Execute trade
        self.current_capital += (trade_value - exit_commission)

        # Update position
        if quantity >= position_quantity:
            # Close entire position
            del self.positions[symbol]
        else:
            # Partial close
            self.positions[symbol]['quantity'] -= quantity

        # Find and update the corresponding trade record
        for trade in reversed(self.trades):
            if (trade['symbol'] == symbol and
                trade.get('status') == 'open' and
                trade.get('side') == 'buy'):

                trade.update({
                    'exit_time': self.current_datetime,
                    'exit_price': price,
                    'pnl': net_pnl,
                    'commission': entry_commission + exit_commission,
                    'status': 'closed'
                })
                break

    async def _update_positions(self, market_snapshot: Dict[str, Dict[str, Any]]):
        """Update position values based on current market data"""

        for symbol, position in self.positions.items():
            if symbol in market_snapshot:
                current_price = market_snapshot[symbol].get('close_price',
                               market_snapshot[symbol].get('bid', 0))
                if current_price > 0:
                    position['current_price'] = current_price

    async def _close_all_positions(self, final_market_data: Dict[str, Dict[str, Any]]):
        """Close all remaining positions at the end of backtest"""

        for symbol in list(self.positions.keys()):
            if symbol in final_market_data:
                final_price = final_market_data[symbol].get('close_price',
                             final_market_data[symbol].get('bid', 0))
                if final_price > 0:
                    position = self.positions[symbol]
                    quantity = position.get('quantity', 0)
                    await self._execute_sell(symbol, final_price, quantity)

    def _calculate_current_equity(self, market_snapshot: Dict[str, Dict[str, Any]]) -> float:
        """Calculate current total equity including open positions"""

        total_equity = self.current_capital

        # Add value of open positions
        for symbol, position in self.positions.items():
            if symbol in market_snapshot:
                current_price = market_snapshot[symbol].get('close_price',
                               market_snapshot[symbol].get('bid', 0))
                if current_price > 0:
                    quantity = position.get('quantity', 0)
                    position_value = current_price * quantity
                    total_equity += position_value

        return total_equity