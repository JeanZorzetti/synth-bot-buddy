#!/usr/bin/env python3
"""
Core Trading Engine for Synth Bot Buddy
Implements automated trading with technical analysis and signal detection
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import deque
import statistics

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Trading signal types"""
    CALL = "CALL"
    PUT = "PUT"
    NONE = "NONE"

class TradeStatus(Enum):
    """Trade status enumeration"""
    PENDING = "pending"
    ACTIVE = "active"
    WON = "won"
    LOST = "lost"
    CANCELLED = "cancelled"

@dataclass
class MarketTick:
    """Market tick data structure"""
    symbol: str
    price: float
    timestamp: float
    volume: Optional[float] = None

@dataclass
class TradingSignal:
    """Trading signal data structure"""
    symbol: str
    signal_type: SignalType
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    timestamp: float
    indicators: Dict[str, float]
    reason: str

@dataclass
class Trade:
    """Trade data structure"""
    trade_id: str
    symbol: str
    contract_type: str
    amount: float
    entry_price: float
    entry_time: float
    duration: int
    status: TradeStatus
    exit_price: Optional[float] = None
    exit_time: Optional[float] = None
    pnl: Optional[float] = None
    contract_id: Optional[str] = None

class TechnicalIndicators:
    """Technical analysis indicators calculator"""
    
    def __init__(self, period: int = 14):
        self.period = period
        self.price_history: Dict[str, deque] = {}
    
    def add_tick(self, symbol: str, price: float, timestamp: float):
        """Add new price tick to history"""
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=100)  # Keep last 100 prices
        
        self.price_history[symbol].append({
            'price': price,
            'timestamp': timestamp
        })
    
    def get_prices(self, symbol: str, count: Optional[int] = None) -> List[float]:
        """Get price history for symbol"""
        if symbol not in self.price_history:
            return []
        
        prices = [tick['price'] for tick in self.price_history[symbol]]
        if count:
            return prices[-count:]
        return prices
    
    def calculate_sma(self, symbol: str, period: int) -> Optional[float]:
        """Calculate Simple Moving Average"""
        prices = self.get_prices(symbol, period)
        if len(prices) < period:
            return None
        return statistics.mean(prices)
    
    def calculate_ema(self, symbol: str, period: int) -> Optional[float]:
        """Calculate Exponential Moving Average"""
        prices = self.get_prices(symbol)
        if len(prices) < period:
            return None
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def calculate_rsi(self, symbol: str, period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index"""
        prices = self.get_prices(symbol, period + 1)
        if len(prices) < period + 1:
            return None
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(-change)
        
        avg_gain = statistics.mean(gains)
        avg_loss = statistics.mean(losses)
        
        if avg_loss == 0:
            return 100  # No losses, RSI = 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_bollinger_bands(self, symbol: str, period: int = 20, std_dev: float = 2) -> Optional[Tuple[float, float, float]]:
        """Calculate Bollinger Bands (upper, middle, lower)"""
        prices = self.get_prices(symbol, period)
        if len(prices) < period:
            return None
        
        sma = statistics.mean(prices)
        std = statistics.stdev(prices)
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    def calculate_macd(self, symbol: str, fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[Tuple[float, float, float]]:
        """Calculate MACD (MACD line, Signal line, Histogram)"""
        ema_fast = self.calculate_ema(symbol, fast)
        ema_slow = self.calculate_ema(symbol, slow)
        
        if ema_fast is None or ema_slow is None:
            return None
        
        macd_line = ema_fast - ema_slow
        
        # For simplicity, using SMA instead of EMA for signal line
        # In production, you'd want to maintain MACD history and calculate proper EMA
        signal_line = macd_line  # Simplified
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram

class SignalDetector:
    """Trading signal detection system"""
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.indicators = TechnicalIndicators()
        self.last_signals: Dict[str, TradingSignal] = {}
        self.signal_history: Dict[str, List[TradingSignal]] = {}
    
    def analyze_market(self, symbol: str) -> TradingSignal:
        """Analyze market and generate trading signal"""
        try:
            # Get current price and indicators
            prices = self.indicators.get_prices(symbol, 50)
            if len(prices) < 20:
                return TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.NONE,
                    strength=0.0,
                    confidence=0.0,
                    timestamp=time.time(),
                    indicators={},
                    reason="Insufficient price history"
                )
            
            current_price = prices[-1]
            indicators = {}
            
            # Calculate technical indicators
            rsi = self.indicators.calculate_rsi(symbol)
            sma_20 = self.indicators.calculate_sma(symbol, 20)
            sma_50 = self.indicators.calculate_sma(symbol, 50)
            bollinger = self.indicators.calculate_bollinger_bands(symbol)
            
            if rsi is not None:
                indicators['rsi'] = rsi
            if sma_20 is not None:
                indicators['sma_20'] = sma_20
            if sma_50 is not None:
                indicators['sma_50'] = sma_50
            if bollinger is not None:
                indicators['bb_upper'] = bollinger[0]
                indicators['bb_middle'] = bollinger[1]
                indicators['bb_lower'] = bollinger[2]
            
            # Generate signal based on configured indicators
            signal_type, strength, confidence, reason = self._evaluate_signals(
                current_price, indicators
            )
            
            signal = TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                timestamp=time.time(),
                indicators=indicators,
                reason=reason
            )
            
            # Store signal in history
            if symbol not in self.signal_history:
                self.signal_history[symbol] = []
            self.signal_history[symbol].append(signal)
            
            # Keep only last 100 signals per symbol
            if len(self.signal_history[symbol]) > 100:
                self.signal_history[symbol] = self.signal_history[symbol][-100:]
            
            self.last_signals[symbol] = signal
            
            logger.info(f"Signal generated for {symbol}: {signal_type.value} "
                       f"(strength: {strength:.2f}, confidence: {confidence:.2f}) - {reason}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing market for {symbol}: {e}")
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.NONE,
                strength=0.0,
                confidence=0.0,
                timestamp=time.time(),
                indicators={},
                reason=f"Analysis error: {str(e)}"
            )
    
    def _evaluate_signals(self, current_price: float, indicators: Dict[str, float]) -> Tuple[SignalType, float, float, str]:
        """Evaluate indicators and generate trading signals"""
        signals = []
        reasons = []
        
        # RSI-based signals
        if 'rsi' in indicators and self.settings.get('indicators', {}).get('use_rsi', True):
            rsi = indicators['rsi']
            if rsi < 30:  # Oversold - potential CALL
                signals.append(('CALL', 0.7, f"RSI oversold ({rsi:.1f})"))
            elif rsi > 70:  # Overbought - potential PUT
                signals.append(('PUT', 0.7, f"RSI overbought ({rsi:.1f})"))
        
        # Moving Average signals
        if ('sma_20' in indicators and 'sma_50' in indicators and 
            self.settings.get('indicators', {}).get('use_moving_averages', True)):
            sma_20 = indicators['sma_20']
            sma_50 = indicators['sma_50']
            
            if current_price > sma_20 > sma_50:  # Bullish trend
                signals.append(('CALL', 0.6, "Price above MA20 > MA50 (bullish)"))
            elif current_price < sma_20 < sma_50:  # Bearish trend
                signals.append(('PUT', 0.6, "Price below MA20 < MA50 (bearish)"))
        
        # Bollinger Bands signals
        if (all(k in indicators for k in ['bb_upper', 'bb_lower']) and 
            self.settings.get('indicators', {}).get('use_bollinger', False)):
            bb_upper = indicators['bb_upper']
            bb_lower = indicators['bb_lower']
            
            if current_price <= bb_lower:  # Price at lower band - potential CALL
                signals.append(('CALL', 0.8, "Price at Bollinger lower band"))
            elif current_price >= bb_upper:  # Price at upper band - potential PUT
                signals.append(('PUT', 0.8, "Price at Bollinger upper band"))
        
        # Aggregate signals
        if not signals:
            return SignalType.NONE, 0.0, 0.0, "No clear signals detected"
        
        # Count signal types
        call_signals = [s for s in signals if s[0] == 'CALL']
        put_signals = [s for s in signals if s[0] == 'PUT']
        
        if len(call_signals) > len(put_signals):
            strength = statistics.mean([s[1] for s in call_signals])
            confidence = min(0.9, len(call_signals) / len(signals) * strength)
            reason = "; ".join([s[2] for s in call_signals])
            return SignalType.CALL, strength, confidence, reason
        elif len(put_signals) > len(call_signals):
            strength = statistics.mean([s[1] for s in put_signals])
            confidence = min(0.9, len(put_signals) / len(signals) * strength)
            reason = "; ".join([s[2] for s in put_signals])
            return SignalType.PUT, strength, confidence, reason
        else:
            return SignalType.NONE, 0.0, 0.0, "Conflicting signals"
    
    def should_trade(self, signal: TradingSignal) -> bool:
        """Determine if we should execute a trade based on the signal"""
        if signal.signal_type == SignalType.NONE:
            return False
        
        # Check minimum confidence threshold
        min_confidence = {
            'conservative': 0.8,
            'moderate': 0.6,
            'aggressive': 0.4
        }.get(self.settings.get('aggressiveness', 'moderate'), 0.6)
        
        if signal.confidence < min_confidence:
            logger.debug(f"Signal confidence {signal.confidence:.2f} below threshold {min_confidence}")
            return False
        
        # Check minimum strength threshold
        min_strength = 0.5
        if signal.strength < min_strength:
            logger.debug(f"Signal strength {signal.strength:.2f} below threshold {min_strength}")
            return False
        
        # Avoid rapid repeated trades on same symbol
        if signal.symbol in self.last_signals:
            last_signal = self.last_signals[signal.symbol]
            if signal.timestamp - last_signal.timestamp < 60:  # 1 minute cooldown
                logger.debug(f"Trade cooldown active for {signal.symbol}")
                return False
        
        return True

class TradingEngine:
    """Main trading engine coordinating all components"""
    
    def __init__(self, websocket_manager, settings: Dict[str, Any]):
        self.ws_manager = websocket_manager
        self.settings = settings
        self.signal_detector = SignalDetector(settings)
        self.active_trades: Dict[str, Trade] = {}
        self.trade_history: List[Trade] = []
        self.is_running = False
        self.current_balance = 0.0
        self.session_pnl = 0.0
        self.trades_count = 0
        
        # Performance tracking
        self.wins = 0
        self.losses = 0
        self.total_invested = 0.0
        self.total_returned = 0.0
        
        logger.info("Trading engine initialized")
    
    def add_market_tick(self, tick: MarketTick):
        """Process new market tick"""
        self.signal_detector.indicators.add_tick(
            tick.symbol, tick.price, tick.timestamp
        )
        
        # Update balance from tick if available
        # In real implementation, balance would come from account updates
    
    async def analyze_and_trade(self, symbol: str) -> Optional[TradingSignal]:
        """Analyze market and potentially execute trade"""
        if not self.is_running:
            return None
        
        try:
            # Generate signal
            signal = self.signal_detector.analyze_market(symbol)
            
            # Check if we should trade
            if self.signal_detector.should_trade(signal):
                await self._execute_trade(signal)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in analyze_and_trade for {symbol}: {e}")
            return None
    
    async def _execute_trade(self, signal: TradingSignal):
        """Execute trade based on signal"""
        try:
            # Check if symbol is enabled for trading
            if not self.settings.get('selected_assets', {}).get(
                self._symbol_to_asset_key(signal.symbol), False
            ):
                logger.debug(f"Trading disabled for {signal.symbol}")
                return
            
            # Calculate position size
            stake_amount = self.settings.get('stake_amount', 10.0)
            
            # Create trade
            trade_id = f"trade_{int(time.time() * 1000)}"
            trade = Trade(
                trade_id=trade_id,
                symbol=signal.symbol,
                contract_type=signal.signal_type.value,
                amount=stake_amount,
                entry_price=self.signal_detector.indicators.get_prices(signal.symbol, 1)[-1],
                entry_time=time.time(),
                duration=5,  # 5 ticks duration
                status=TradeStatus.PENDING
            )
            
            # Execute trade via WebSocket
            success = await self._send_buy_order(trade)
            
            if success:
                self.active_trades[trade_id] = trade
                self.trades_count += 1
                self.total_invested += stake_amount
                
                logger.info(f"Trade executed: {trade.contract_type} {trade.symbol} "
                          f"${trade.amount} at {trade.entry_price}")
            else:
                logger.error(f"Failed to execute trade for {signal.symbol}")
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    async def _send_buy_order(self, trade: Trade) -> bool:
        """Send buy order through WebSocket"""
        try:
            # This would integrate with the WebSocket manager
            # For now, we'll simulate the trade execution
            
            buy_request = {
                "buy": 1,
                "price": trade.amount,
                "parameters": {
                    "contract_type": trade.contract_type,
                    "symbol": trade.symbol,
                    "duration": trade.duration,
                    "duration_unit": "t"  # ticks
                }
            }
            
            # In real implementation:
            # response = await self.ws_manager.send_request(buy_request)
            # trade.contract_id = response.get('buy', {}).get('contract_id')
            
            # Simulate successful trade
            trade.status = TradeStatus.ACTIVE
            trade.contract_id = f"contract_{int(time.time())}"
            
            return True
            
        except Exception as e:
            logger.error(f"Error sending buy order: {e}")
            return False
    
    def _symbol_to_asset_key(self, symbol: str) -> str:
        """Convert symbol to settings asset key"""
        symbol_mapping = {
            'R_10': 'volatility10',
            'R_25': 'volatility25', 
            'R_50': 'volatility50',
            'R_75': 'volatility75',
            'R_100': 'volatility100',
            'RDBULL': 'boom1000',
            'RDBEAR': 'crash1000'
        }
        return symbol_mapping.get(symbol, 'volatility75')
    
    def update_trade_result(self, contract_id: str, won: bool, exit_price: float, payout: float = 0.0):
        """Update trade result when contract expires"""
        try:
            # Find trade by contract_id
            trade = None
            for t in self.active_trades.values():
                if t.contract_id == contract_id:
                    trade = t
                    break
            
            if not trade:
                logger.warning(f"Trade not found for contract {contract_id}")
                return
            
            # Update trade
            trade.exit_price = exit_price
            trade.exit_time = time.time()
            trade.status = TradeStatus.WON if won else TradeStatus.LOST
            
            if won:
                trade.pnl = payout - trade.amount
                self.wins += 1
                self.total_returned += payout
            else:
                trade.pnl = -trade.amount
                self.losses += 1
            
            self.session_pnl += trade.pnl
            
            # Move to history
            self.trade_history.append(trade)
            del self.active_trades[trade.trade_id]
            
            logger.info(f"Trade result: {trade.status.value} - P/L: ${trade.pnl:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating trade result: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current trading engine status"""
        win_rate = (self.wins / max(1, self.wins + self.losses)) * 100
        
        return {
            'is_running': self.is_running,
            'session_pnl': round(self.session_pnl, 2),
            'trades_count': self.trades_count,
            'active_trades': len(self.active_trades),
            'win_rate': round(win_rate, 1),
            'wins': self.wins,
            'losses': self.losses,
            'total_invested': round(self.total_invested, 2),
            'total_returned': round(self.total_returned, 2),
            'capital_management': {
                'next_amount': self.settings.get('stake_amount', 10.0),
                'current_sequence': 1,
                'is_in_loss_sequence': self.losses > self.wins,
                'accumulated_profit': round(self.session_pnl, 2),
                'risk_level': self._calculate_risk_level()
            }
        }
    
    def _calculate_risk_level(self) -> str:
        """Calculate current risk level"""
        if abs(self.session_pnl) < self.settings.get('stake_amount', 10.0):
            return 'LOW'
        elif abs(self.session_pnl) < self.settings.get('stop_loss', 50.0) * 0.5:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def start(self):
        """Start the trading engine"""
        self.is_running = True
        logger.info("Trading engine started")
    
    def stop(self):
        """Stop the trading engine"""
        self.is_running = False
        logger.info("Trading engine stopped")
    
    def reset_session(self):
        """Reset session statistics"""
        self.session_pnl = 0.0
        self.trades_count = 0
        self.wins = 0
        self.losses = 0
        self.total_invested = 0.0
        self.total_returned = 0.0
        self.trade_history.clear()
        logger.info("Trading session reset")