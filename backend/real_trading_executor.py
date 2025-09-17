"""
Real Trading Executor - Phase 14 Trading Execution Integration
Sistema completo de execução de trading real integrado com Deriv API
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import json
import uuid

from real_deriv_websocket import get_deriv_websocket, RealDerivWebSocket
from realtime_feature_processor import get_feature_processor, ProcessedFeatures
from real_model_trainer import get_model_trainer
from database_config import get_db_manager
from redis_cache_manager import get_cache_manager, CacheNamespace
from real_logging_system import logging_system, LogComponent, LogLevel

class OrderType(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"

class TradingMode(Enum):
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"

@dataclass
class TradingSignal:
    signal_id: str
    model_id: str
    symbol: str
    timestamp: datetime

    # Signal details
    signal_type: OrderType  # BUY or SELL
    confidence: float
    probability_up: float
    probability_down: float

    # Risk parameters
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    position_size: float

    # Model context
    features_used: Dict[str, float]
    model_version: str
    prediction_horizon_minutes: int

@dataclass
class TradingOrder:
    order_id: str
    signal_id: str
    symbol: str
    order_type: OrderType
    quantity: float

    # Price details
    entry_price: Optional[float]
    current_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]

    # Status and timing
    status: OrderStatus
    created_at: datetime
    executed_at: Optional[datetime]
    closed_at: Optional[datetime]

    # Execution details
    fill_price: Optional[float]
    commission: float
    slippage: float

    # P&L
    unrealized_pnl: float
    realized_pnl: float

@dataclass
class TradingPosition:
    position_id: str
    symbol: str
    side: OrderType
    quantity: float

    # Price tracking
    entry_price: float
    current_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]

    # Status
    status: PositionStatus
    opened_at: datetime
    closed_at: Optional[datetime]

    # P&L tracking
    unrealized_pnl: float
    realized_pnl: float
    total_commission: float

    # Risk metrics
    max_favorable: float
    max_adverse: float
    current_drawdown: float

class RealTradingExecutor:
    """Advanced real trading execution system with full Deriv API integration"""

    def __init__(self):
        # Trading configuration
        self.trading_mode = TradingMode.PAPER  # Start with paper trading
        self.max_concurrent_positions = 5
        self.max_daily_trades = 50
        self.max_risk_per_trade = 0.02  # 2% of capital
        self.default_stop_loss_pct = 0.02  # 2%
        self.default_take_profit_pct = 0.04  # 4%

        # Account management
        self.account_balance = 10000.0  # Starting balance
        self.available_balance = 10000.0
        self.equity = 10000.0
        self.margin_used = 0.0
        self.free_margin = 10000.0

        # Trading state
        self.trading_active = False
        self.active_positions: Dict[str, TradingPosition] = {}
        self.pending_orders: Dict[str, TradingOrder] = {}
        self.closed_positions: List[TradingPosition] = []
        self.trading_history: List[TradingOrder] = []

        # Signal processing
        self.signal_queue: List[TradingSignal] = []
        self.processed_signals: Dict[str, TradingSignal] = {}

        # Risk management
        self.daily_pnl = 0.0
        self.daily_trades_count = 0
        self.max_daily_loss = 500.0  # $500 max daily loss
        self.max_drawdown_pct = 0.10  # 10% max drawdown

        # Dependencies
        self.websocket_client: Optional[RealDerivWebSocket] = None
        self.feature_processor = None
        self.model_trainer = None
        self.db_manager = None
        self.cache_manager = None

        # Model management
        self.active_models: Dict[str, Any] = {}  # model_id -> model_info
        self.model_weights: Dict[str, float] = {}  # model_id -> weight

        # Logging
        self.logger = logging_system.loggers.get('trading', logging.getLogger(__name__))

    async def initialize(self):
        """Initialize trading executor"""
        try:
            # Initialize dependencies
            self.websocket_client = await get_deriv_websocket()
            self.feature_processor = await get_feature_processor()
            self.model_trainer = await get_model_trainer()
            self.db_manager = await get_db_manager()
            self.cache_manager = await get_cache_manager()

            # Load active models
            await self._load_active_models()

            # Setup callbacks
            self.feature_processor.add_feature_callback(self._process_features)

            logging_system.log(
                LogComponent.TRADING,
                LogLevel.INFO,
                "Real trading executor initialized",
                {
                    'trading_mode': self.trading_mode.value,
                    'account_balance': self.account_balance,
                    'active_models': len(self.active_models)
                }
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.TRADING,
                e,
                {'action': 'initialize_trading_executor'}
            )
            raise

    async def start_trading(self, symbols: List[str], trading_mode: TradingMode = TradingMode.PAPER):
        """Start automated trading"""
        try:
            self.trading_mode = trading_mode
            self.trading_active = True

            # Reset daily counters
            self.daily_pnl = 0.0
            self.daily_trades_count = 0

            # Start feature processing for symbols
            await self.feature_processor.start_processing(symbols)

            # Start periodic tasks
            asyncio.create_task(self._position_monitoring_loop())
            asyncio.create_task(self._signal_processing_loop())
            asyncio.create_task(self._risk_monitoring_loop())

            logging_system.log(
                LogComponent.TRADING,
                LogLevel.INFO,
                f"Trading started in {trading_mode.value} mode",
                {
                    'symbols': symbols,
                    'max_positions': self.max_concurrent_positions,
                    'account_balance': self.account_balance
                }
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.TRADING,
                e,
                {'action': 'start_trading', 'symbols': symbols}
            )
            raise

    async def stop_trading(self):
        """Stop automated trading"""
        try:
            self.trading_active = False

            # Close all open positions (if in live mode)
            if self.trading_mode == TradingMode.LIVE:
                await self._close_all_positions("Trading stopped")

            await self.feature_processor.stop_processing()

            logging_system.log(
                LogComponent.TRADING,
                LogLevel.INFO,
                "Trading stopped",
                {
                    'final_balance': self.account_balance,
                    'daily_pnl': self.daily_pnl,
                    'open_positions': len(self.active_positions)
                }
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.TRADING,
                e,
                {'action': 'stop_trading'}
            )

    async def _process_features(self, features: ProcessedFeatures):
        """Process new features and generate trading signals"""
        try:
            if not self.trading_active:
                return

            # Generate signals from all active models
            signals = await self._generate_signals(features)

            # Add signals to queue
            for signal in signals:
                self.signal_queue.append(signal)
                self.processed_signals[signal.signal_id] = signal

        except Exception as e:
            logging_system.log_error(
                LogComponent.TRADING,
                e,
                {'action': 'process_features', 'symbol': features.symbol}
            )

    async def _generate_signals(self, features: ProcessedFeatures) -> List[TradingSignal]:
        """Generate trading signals from AI models"""
        try:
            signals = []

            for model_id, model_info in self.active_models.items():
                try:
                    # Load model and scaler
                    model = model_info['model']
                    scaler = model_info['scaler']

                    # Prepare feature vector
                    feature_vector = self._prepare_feature_vector(features)
                    if feature_vector is None:
                        continue

                    # Scale features
                    feature_vector_scaled = scaler.transform([feature_vector])

                    # Make prediction
                    prediction = model.predict(feature_vector_scaled)[0]

                    # Get prediction probability if available
                    probability = 0.5
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(feature_vector_scaled)[0]
                        probability = proba[1] if len(proba) > 1 else 0.5

                    # Calculate confidence
                    confidence = abs(probability - 0.5) * 2  # Scale to 0-1

                    # Only generate signal if confidence is high enough
                    min_confidence = 0.6
                    if confidence < min_confidence:
                        continue

                    # Determine signal type
                    signal_type = OrderType.BUY if prediction > 0 else OrderType.SELL

                    # Calculate position size based on confidence and risk
                    position_size = self._calculate_position_size(
                        features.symbol, confidence, features.price
                    )

                    if position_size <= 0:
                        continue

                    # Calculate risk parameters
                    stop_loss, take_profit = self._calculate_risk_parameters(
                        signal_type, features.price, features.atr_14
                    )

                    # Create signal
                    signal = TradingSignal(
                        signal_id=f"signal_{uuid.uuid4().hex[:8]}",
                        model_id=model_id,
                        symbol=features.symbol,
                        timestamp=features.timestamp,
                        signal_type=signal_type,
                        confidence=confidence,
                        probability_up=probability,
                        probability_down=1 - probability,
                        entry_price=features.price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        position_size=position_size,
                        features_used=self._extract_features_dict(features),
                        model_version=model_info.get('version', '1.0'),
                        prediction_horizon_minutes=5
                    )

                    signals.append(signal)

                    logging_system.log(
                        LogComponent.TRADING,
                        LogLevel.INFO,
                        f"Generated {signal_type.value} signal for {features.symbol}",
                        {
                            'model_id': model_id,
                            'confidence': confidence,
                            'position_size': position_size,
                            'entry_price': features.price
                        }
                    )

                except Exception as model_error:
                    logging_system.log_error(
                        LogComponent.TRADING,
                        model_error,
                        {'action': 'model_prediction', 'model_id': model_id}
                    )

            return signals

        except Exception as e:
            logging_system.log_error(
                LogComponent.TRADING,
                e,
                {'action': 'generate_signals', 'symbol': features.symbol}
            )
            return []

    def _prepare_feature_vector(self, features: ProcessedFeatures) -> Optional[List[float]]:
        """Prepare feature vector for model prediction"""
        try:
            feature_dict = asdict(features)

            # Remove non-numeric fields
            feature_dict.pop('symbol', None)
            feature_dict.pop('timestamp', None)

            # Create feature vector in consistent order
            feature_vector = []
            expected_features = [
                'price', 'sma_5', 'sma_10', 'sma_20', 'sma_50',
                'ema_5', 'ema_10', 'ema_20', 'ema_50',
                'rsi_14', 'rsi_21', 'macd', 'macd_signal', 'macd_histogram',
                'momentum_10', 'roc_10', 'atr_14',
                'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
                'bollinger_width', 'bollinger_position',
                'stoch_k', 'stoch_d', 'williams_r',
                'volume_sma_10', 'volume_ratio', 'vwap',
                'price_change_1', 'price_change_5', 'price_change_10',
                'volatility_5', 'volatility_10',
                'bid_ask_spread', 'spread_ratio', 'tick_direction', 'tick_intensity',
                'hurst_exponent', 'fractal_dimension', 'entropy',
                'autocorr_1', 'autocorr_5'
            ]

            for feature_name in expected_features:
                value = feature_dict.get(feature_name, 0.0)
                if np.isnan(value) or np.isinf(value):
                    value = 0.0
                feature_vector.append(float(value))

            return feature_vector

        except Exception as e:
            logging_system.log_error(
                LogComponent.TRADING,
                e,
                {'action': 'prepare_feature_vector'}
            )
            return None

    def _calculate_position_size(self, symbol: str, confidence: float, price: float) -> float:
        """Calculate position size based on risk management"""
        try:
            # Base position size on available capital and risk per trade
            max_risk_amount = self.available_balance * self.max_risk_per_trade

            # Adjust based on confidence
            confidence_multiplier = min(confidence * 1.5, 1.0)  # Max 1.0

            # Calculate position size
            position_value = max_risk_amount * confidence_multiplier
            position_size = position_value / price

            # Apply minimum and maximum limits
            min_position_size = 0.01  # Minimum trade size
            max_position_size = self.available_balance * 0.1 / price  # Max 10% of balance

            position_size = max(min_position_size, min(position_size, max_position_size))

            return round(position_size, 2)

        except Exception as e:
            logging_system.log_error(
                LogComponent.TRADING,
                e,
                {'action': 'calculate_position_size', 'symbol': symbol}
            )
            return 0.0

    def _calculate_risk_parameters(self, signal_type: OrderType, entry_price: float, atr: float) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        try:
            # Use ATR for dynamic stop loss and take profit
            atr_multiplier_sl = 2.0  # 2x ATR for stop loss
            atr_multiplier_tp = 3.0  # 3x ATR for take profit

            if signal_type == OrderType.BUY:
                stop_loss = entry_price - (atr * atr_multiplier_sl)
                take_profit = entry_price + (atr * atr_multiplier_tp)
            else:  # SELL
                stop_loss = entry_price + (atr * atr_multiplier_sl)
                take_profit = entry_price - (atr * atr_multiplier_tp)

            # Fallback to percentage-based if ATR is too small
            if abs(stop_loss - entry_price) / entry_price < 0.005:  # Less than 0.5%
                sl_pct = self.default_stop_loss_pct
                tp_pct = self.default_take_profit_pct

                if signal_type == OrderType.BUY:
                    stop_loss = entry_price * (1 - sl_pct)
                    take_profit = entry_price * (1 + tp_pct)
                else:
                    stop_loss = entry_price * (1 + sl_pct)
                    take_profit = entry_price * (1 - tp_pct)

            return round(stop_loss, 5), round(take_profit, 5)

        except Exception as e:
            logging_system.log_error(
                LogComponent.TRADING,
                e,
                {'action': 'calculate_risk_parameters'}
            )
            return entry_price * 0.98, entry_price * 1.02  # Default 2% levels

    def _extract_features_dict(self, features: ProcessedFeatures) -> Dict[str, float]:
        """Extract key features for signal context"""
        return {
            'price': features.price,
            'rsi_14': features.rsi_14,
            'macd': features.macd,
            'bollinger_position': features.bollinger_position,
            'atr_14': features.atr_14,
            'volatility_10': features.volatility_10
        }

    async def _signal_processing_loop(self):
        """Process signals and execute trades"""
        while self.trading_active:
            try:
                if self.signal_queue:
                    # Process signals in FIFO order
                    signal = self.signal_queue.pop(0)
                    await self._process_signal(signal)

                await asyncio.sleep(1)  # Check every second

            except asyncio.CancelledError:
                break
            except Exception as e:
                logging_system.log_error(
                    LogComponent.TRADING,
                    e,
                    {'action': 'signal_processing_loop'}
                )
                await asyncio.sleep(5)

    async def _process_signal(self, signal: TradingSignal):
        """Process individual trading signal"""
        try:
            # Check if we can trade
            if not await self._can_execute_trade(signal):
                return

            # Check for conflicting positions
            if await self._has_conflicting_position(signal):
                logging_system.log(
                    LogComponent.TRADING,
                    LogLevel.DEBUG,
                    f"Skipping signal due to conflicting position: {signal.symbol}"
                )
                return

            # Execute trade
            order = await self._execute_trade(signal)
            if order:
                logging_system.log_trading_activity(
                    'signal_executed',
                    signal.symbol,
                    {
                        'signal_id': signal.signal_id,
                        'order_id': order.order_id,
                        'order_type': order.order_type.value,
                        'quantity': order.quantity,
                        'confidence': signal.confidence
                    }
                )

        except Exception as e:
            logging_system.log_error(
                LogComponent.TRADING,
                e,
                {'action': 'process_signal', 'signal_id': signal.signal_id}
            )

    async def _can_execute_trade(self, signal: TradingSignal) -> bool:
        """Check if trade can be executed"""
        try:
            # Check if trading is active
            if not self.trading_active:
                return False

            # Check daily trade limit
            if self.daily_trades_count >= self.max_daily_trades:
                return False

            # Check maximum concurrent positions
            if len(self.active_positions) >= self.max_concurrent_positions:
                return False

            # Check available balance
            required_margin = signal.position_size * signal.entry_price * 0.1  # 10% margin
            if required_margin > self.available_balance:
                return False

            # Check daily loss limit
            if self.daily_pnl < -self.max_daily_loss:
                return False

            # Check maximum drawdown
            current_drawdown = (self.account_balance - self.equity) / self.account_balance
            if current_drawdown > self.max_drawdown_pct:
                return False

            return True

        except Exception as e:
            logging_system.log_error(
                LogComponent.TRADING,
                e,
                {'action': 'can_execute_trade'}
            )
            return False

    async def _has_conflicting_position(self, signal: TradingSignal) -> bool:
        """Check if there's a conflicting position for the symbol"""
        for position in self.active_positions.values():
            if (position.symbol == signal.symbol and
                position.side != signal.signal_type and
                position.status == PositionStatus.OPEN):
                return True
        return False

    async def _execute_trade(self, signal: TradingSignal) -> Optional[TradingOrder]:
        """Execute trading order"""
        try:
            order_id = f"order_{uuid.uuid4().hex[:8]}"

            # Create order
            order = TradingOrder(
                order_id=order_id,
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                order_type=signal.signal_type,
                quantity=signal.position_size,
                entry_price=signal.entry_price,
                current_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                status=OrderStatus.PENDING,
                created_at=datetime.utcnow(),
                executed_at=None,
                closed_at=None,
                fill_price=None,
                commission=0.0,
                slippage=0.0,
                unrealized_pnl=0.0,
                realized_pnl=0.0
            )

            if self.trading_mode == TradingMode.LIVE:
                # Execute real trade via Deriv API
                success = await self._execute_real_trade(order, signal)
            else:
                # Execute paper trade
                success = await self._execute_paper_trade(order, signal)

            if success:
                self.pending_orders[order_id] = order
                self.daily_trades_count += 1

                # Store in database
                await self._store_order(order)

                return order

            return None

        except Exception as e:
            logging_system.log_error(
                LogComponent.TRADING,
                e,
                {'action': 'execute_trade', 'signal_id': signal.signal_id}
            )
            return None

    async def _execute_paper_trade(self, order: TradingOrder, signal: TradingSignal) -> bool:
        """Execute paper trade (simulation)"""
        try:
            # Simulate execution delay and slippage
            await asyncio.sleep(0.1)  # 100ms execution delay

            # Add realistic slippage (0.1-0.5 pips)
            slippage_pips = np.random.uniform(0.1, 0.5) * 0.0001
            if order.order_type == OrderType.BUY:
                fill_price = order.entry_price + slippage_pips
            else:
                fill_price = order.entry_price - slippage_pips

            order.status = OrderStatus.FILLED
            order.executed_at = datetime.utcnow()
            order.fill_price = fill_price
            order.slippage = abs(fill_price - order.entry_price)

            # Calculate commission
            order.commission = order.quantity * fill_price * 0.0005  # 0.05% for paper trading

            # Create position
            await self._create_position_from_order(order)

            return True

        except Exception as e:
            logging_system.log_error(
                LogComponent.TRADING,
                e,
                {'action': 'execute_paper_trade', 'order_id': order.order_id}
            )
            return False

    async def _create_position_from_order(self, order: TradingOrder, contract_id: Optional[str] = None):
        """Create position from executed order"""
        try:
            position_id = f"pos_{uuid.uuid4().hex[:8]}"

            position = TradingPosition(
                position_id=position_id,
                symbol=order.symbol,
                side=order.order_type,
                quantity=order.quantity,
                entry_price=order.fill_price,
                current_price=order.fill_price,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                status=PositionStatus.OPEN,
                opened_at=order.executed_at,
                closed_at=None,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                total_commission=order.commission,
                max_favorable=0.0,
                max_adverse=0.0,
                current_drawdown=0.0
            )

            self.active_positions[position_id] = position

            # Update account balance
            self.available_balance -= (order.quantity * order.fill_price + order.commission)
            self.margin_used += order.quantity * order.fill_price * 0.1  # 10% margin
            self.free_margin = self.available_balance - self.margin_used

            # Store position
            await self._store_position(position)

            logging_system.log(
                LogComponent.TRADING,
                LogLevel.INFO,
                f"Position opened: {position.side.value} {position.quantity} {position.symbol}",
                {
                    'position_id': position_id,
                    'entry_price': position.entry_price,
                    'stop_loss': position.stop_loss,
                    'take_profit': position.take_profit
                }
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.TRADING,
                e,
                {'action': 'create_position_from_order', 'order_id': order.order_id}
            )

    async def _position_monitoring_loop(self):
        """Monitor open positions and manage risk"""
        while self.trading_active:
            try:
                for position_id, position in list(self.active_positions.items()):
                    await self._update_position(position)

                await asyncio.sleep(5)  # Check every 5 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logging_system.log_error(
                    LogComponent.TRADING,
                    e,
                    {'action': 'position_monitoring_loop'}
                )
                await asyncio.sleep(10)

    async def _update_position(self, position: TradingPosition):
        """Update position with current market price"""
        try:
            # Get current market price
            latest_tick = await self.cache_manager.get_latest_tick(position.symbol)
            if not latest_tick:
                return

            current_price = latest_tick.get('price', position.current_price)
            position.current_price = current_price

            # Calculate unrealized P&L
            if position.side == OrderType.BUY:
                price_diff = current_price - position.entry_price
            else:  # SELL
                price_diff = position.entry_price - current_price

            position.unrealized_pnl = price_diff * position.quantity

            # Update max favorable/adverse
            if position.unrealized_pnl > position.max_favorable:
                position.max_favorable = position.unrealized_pnl
            elif position.unrealized_pnl < position.max_adverse:
                position.max_adverse = position.unrealized_pnl

            # Calculate current drawdown from max favorable
            if position.max_favorable > 0:
                position.current_drawdown = (position.max_favorable - position.unrealized_pnl) / position.max_favorable

            # Update equity
            self._update_account_equity()

        except Exception as e:
            logging_system.log_error(
                LogComponent.TRADING,
                e,
                {'action': 'update_position', 'position_id': position.position_id}
            )

    async def _risk_monitoring_loop(self):
        """Monitor overall risk and account health"""
        while self.trading_active:
            try:
                # Update account equity
                self._update_account_equity()

                # Update risk metrics in cache
                await self._store_account_status()

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logging_system.log_error(
                    LogComponent.TRADING,
                    e,
                    {'action': 'risk_monitoring_loop'}
                )
                await asyncio.sleep(60)

    def _update_account_equity(self):
        """Update account equity based on unrealized P&L"""
        try:
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.active_positions.values())
            self.equity = self.account_balance + total_unrealized_pnl

        except Exception as e:
            logging_system.log_error(
                LogComponent.TRADING,
                e,
                {'action': 'update_account_equity'}
            )

    async def _close_all_positions(self, reason: str):
        """Close all open positions"""
        try:
            for position in list(self.active_positions.values()):
                position.status = PositionStatus.CLOSED
                position.closed_at = datetime.utcnow()
                self.closed_positions.append(position)

            self.active_positions.clear()

        except Exception as e:
            logging_system.log_error(
                LogComponent.TRADING,
                e,
                {'action': 'close_all_positions'}
            )

    async def _load_active_models(self):
        """Load active trading models"""
        try:
            # Get trained models from model trainer
            if hasattr(self.model_trainer, 'trained_models'):
                for model_id, model_info in self.model_trainer.trained_models.items():
                    try:
                        # Load model and scaler
                        import joblib
                        model = joblib.load(model_info['model_path'])
                        scaler = None

                        if model_info.get('scaler_path'):
                            scaler = joblib.load(model_info['scaler_path'])

                        self.active_models[model_id] = {
                            'model': model,
                            'scaler': scaler,
                            'metadata': model_info['metadata'],
                            'version': '1.0'
                        }

                        # Set equal weights for all models
                        self.model_weights[model_id] = 1.0 / len(self.model_trainer.trained_models)

                    except Exception as model_error:
                        logging_system.log_error(
                            LogComponent.TRADING,
                            model_error,
                            {'action': 'load_model', 'model_id': model_id}
                        )

            logging_system.log(
                LogComponent.TRADING,
                LogLevel.INFO,
                f"Loaded {len(self.active_models)} trading models"
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.TRADING,
                e,
                {'action': 'load_active_models'}
            )

    async def _store_order(self, order: TradingOrder):
        """Store order in database"""
        try:
            if self.db_manager:
                order_data = {
                    'order_id': order.order_id,
                    'signal_id': order.signal_id,
                    'symbol': order.symbol,
                    'order_type': order.order_type.value,
                    'quantity': order.quantity,
                    'entry_price': order.entry_price,
                    'status': order.status.value,
                    'created_at': order.created_at,
                    'executed_at': order.executed_at,
                    'fill_price': order.fill_price,
                    'commission': order.commission,
                    'realized_pnl': order.realized_pnl
                }

                await self.db_manager.store_trading_session([order_data])

        except Exception as e:
            logging_system.log_error(
                LogComponent.TRADING,
                e,
                {'action': 'store_order', 'order_id': order.order_id}
            )

    async def _store_position(self, position: TradingPosition):
        """Store position in database"""
        try:
            if self.db_manager:
                position_data = {
                    'position_id': position.position_id,
                    'symbol': position.symbol,
                    'side': position.side.value,
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'status': position.status.value,
                    'opened_at': position.opened_at,
                    'closed_at': position.closed_at,
                    'unrealized_pnl': position.unrealized_pnl,
                    'realized_pnl': position.realized_pnl,
                    'total_commission': position.total_commission
                }

                await self.db_manager.store_trading_positions([position_data])

        except Exception as e:
            logging_system.log_error(
                LogComponent.TRADING,
                e,
                {'action': 'store_position', 'position_id': position.position_id}
            )

    async def _store_account_status(self):
        """Store account status in cache"""
        try:
            account_status = {
                'timestamp': datetime.utcnow().isoformat(),
                'account_balance': self.account_balance,
                'available_balance': self.available_balance,
                'equity': self.equity,
                'margin_used': self.margin_used,
                'free_margin': self.free_margin,
                'daily_pnl': self.daily_pnl,
                'daily_trades_count': self.daily_trades_count,
                'open_positions_count': len(self.active_positions),
                'trading_mode': self.trading_mode.value,
                'trading_active': self.trading_active
            }

            await self.cache_manager.set(
                CacheNamespace.TRADING_POSITIONS,
                'account_status',
                account_status,
                ttl=300  # 5 minutes
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.TRADING,
                e,
                {'action': 'store_account_status'}
            )

    # Public API methods

    async def get_account_status(self) -> Dict[str, Any]:
        """Get current account status"""
        self._update_account_equity()

        return {
            'account_balance': self.account_balance,
            'available_balance': self.available_balance,
            'equity': self.equity,
            'margin_used': self.margin_used,
            'free_margin': self.free_margin,
            'daily_pnl': self.daily_pnl,
            'daily_trades_count': self.daily_trades_count,
            'open_positions': len(self.active_positions),
            'trading_mode': self.trading_mode.value,
            'trading_active': self.trading_active,
            'active_models': len(self.active_models)
        }

    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions"""
        positions = []
        for position in self.active_positions.values():
            await self._update_position(position)
            positions.append(asdict(position))

        return positions

    def get_trading_status(self) -> Dict[str, Any]:
        """Get overall trading status"""
        return {
            'trading_active': self.trading_active,
            'trading_mode': self.trading_mode.value,
            'active_positions_count': len(self.active_positions),
            'pending_orders_count': len(self.pending_orders),
            'signals_in_queue': len(self.signal_queue),
            'daily_trades_count': self.daily_trades_count,
            'daily_pnl': self.daily_pnl,
            'active_models_count': len(self.active_models),
            'risk_limits': {
                'max_concurrent_positions': self.max_concurrent_positions,
                'max_daily_trades': self.max_daily_trades,
                'max_daily_loss': self.max_daily_loss,
                'max_drawdown_pct': self.max_drawdown_pct
            }
        }

# Global trading executor instance
trading_executor = RealTradingExecutor()

async def get_trading_executor() -> RealTradingExecutor:
    """Get initialized trading executor"""
    if not trading_executor.websocket_client:
        await trading_executor.initialize()
    return trading_executor