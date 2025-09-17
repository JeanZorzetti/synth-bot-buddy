"""
Data Quality Monitoring System - Phase 13 Real-Time Data Pipeline
Sistema de monitoramento de qualidade de dados em tempo real
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import logging

from real_deriv_websocket import TickData, CandleData, get_deriv_websocket
from influxdb_timeseries import get_influxdb_manager
from redis_cache_manager import get_cache_manager, CacheNamespace
from real_logging_system import logging_system, LogComponent, LogLevel

class DataQualityLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DataQualityMetrics:
    symbol: str
    timestamp: datetime

    # Completeness metrics
    data_completeness_score: float
    missing_data_percentage: float
    data_gap_count: int
    longest_data_gap_minutes: float

    # Accuracy metrics
    price_consistency_score: float
    spread_anomaly_count: int
    outlier_percentage: float
    data_integrity_score: float

    # Timeliness metrics
    latency_ms: float
    update_frequency_score: float
    real_time_score: float

    # Volatility and pattern metrics
    volatility_score: float
    pattern_consistency_score: float
    trend_stability_score: float

    # Overall quality
    overall_quality_score: float
    quality_level: DataQualityLevel

    # Alerts and issues
    active_alerts: List[str]
    critical_issues: List[str]

    # Data volume metrics
    tick_count_1h: int
    tick_count_24h: int
    average_ticks_per_minute: float

@dataclass
class QualityAlert:
    alert_id: str
    symbol: str
    severity: AlertSeverity
    alert_type: str
    message: str
    timestamp: datetime
    threshold_breached: float
    current_value: float
    auto_resolve: bool = True
    resolved: bool = False

class DataQualityMonitor:
    """Real-time data quality monitoring and alerting system"""

    def __init__(self):
        # Quality thresholds
        self.thresholds = {
            'completeness_min': 95.0,  # Minimum data completeness %
            'max_gap_minutes': 5.0,    # Maximum allowed data gap
            'max_latency_ms': 1000.0,  # Maximum acceptable latency
            'max_spread_ratio': 5.0,   # Maximum spread vs average ratio
            'max_outlier_pct': 2.0,    # Maximum outlier percentage
            'min_ticks_per_minute': 10, # Minimum ticks per minute
            'max_volatility_z_score': 3.0, # Maximum volatility z-score
            'price_jump_threshold': 0.05  # 5% price jump threshold
        }

        # Data storage
        self.quality_history: Dict[str, List[DataQualityMetrics]] = {}
        self.active_alerts: Dict[str, List[QualityAlert]] = {}
        self.tick_timestamps: Dict[str, List[datetime]] = {}
        self.price_history: Dict[str, List[float]] = {}
        self.spread_history: Dict[str, List[float]] = {}

        # Monitoring configuration
        self.monitoring_active = False
        self.symbols_monitored: set = set()
        self.max_history_size = 100
        self.max_price_history = 1000
        self.alert_id_counter = 0

        # Dependencies
        self.websocket_client = None
        self.influxdb_manager = None
        self.cache_manager = None

        # Logging
        self.logger = logging_system.loggers.get('system', logging.getLogger(__name__))

    async def initialize(self):
        """Initialize data quality monitor"""
        try:
            # Initialize dependencies
            self.websocket_client = await get_deriv_websocket()
            self.influxdb_manager = await get_influxdb_manager()
            self.cache_manager = await get_cache_manager()

            # Register callbacks
            self.websocket_client.add_tick_callback(self._monitor_tick_quality)

            logging_system.log(
                LogComponent.SYSTEM,
                LogLevel.INFO,
                "Data quality monitor initialized"
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.SYSTEM,
                e,
                {'action': 'initialize_quality_monitor'}
            )
            raise

    async def start_monitoring(self, symbols: List[str]):
        """Start monitoring data quality for symbols"""
        try:
            self.monitoring_active = True

            for symbol in symbols:
                if symbol not in self.symbols_monitored:
                    # Initialize monitoring structures
                    self.quality_history[symbol] = []
                    self.active_alerts[symbol] = []
                    self.tick_timestamps[symbol] = []
                    self.price_history[symbol] = []
                    self.spread_history[symbol] = []

                    self.symbols_monitored.add(symbol)

            # Start periodic quality assessment
            asyncio.create_task(self._periodic_quality_assessment())

            logging_system.log(
                LogComponent.SYSTEM,
                LogLevel.INFO,
                f"Data quality monitoring started for {len(self.symbols_monitored)} symbols"
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.SYSTEM,
                e,
                {'action': 'start_monitoring', 'symbols': symbols}
            )

    async def stop_monitoring(self):
        """Stop data quality monitoring"""
        try:
            self.monitoring_active = False
            self.symbols_monitored.clear()

            logging_system.log(
                LogComponent.SYSTEM,
                LogLevel.INFO,
                "Data quality monitoring stopped"
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.SYSTEM,
                e,
                {'action': 'stop_monitoring'}
            )

    async def _monitor_tick_quality(self, tick: TickData):
        """Monitor individual tick data quality"""
        try:
            if not self.monitoring_active or tick.symbol not in self.symbols_monitored:
                return

            # Record tick timestamp
            self.tick_timestamps[tick.symbol].append(tick.timestamp)
            if len(self.tick_timestamps[tick.symbol]) > 1000:
                self.tick_timestamps[tick.symbol] = self.tick_timestamps[tick.symbol][-1000:]

            # Record price
            self.price_history[tick.symbol].append(tick.price)
            if len(self.price_history[tick.symbol]) > self.max_price_history:
                self.price_history[tick.symbol] = self.price_history[tick.symbol][-self.max_price_history:]

            # Record spread
            self.spread_history[tick.symbol].append(tick.spread)
            if len(self.spread_history[tick.symbol]) > self.max_price_history:
                self.spread_history[tick.symbol] = self.spread_history[tick.symbol][-self.max_price_history:]

            # Check for immediate quality issues
            await self._check_immediate_issues(tick)

        except Exception as e:
            logging_system.log_error(
                LogComponent.SYSTEM,
                e,
                {'action': 'monitor_tick_quality', 'symbol': tick.symbol}
            )

    async def _check_immediate_issues(self, tick: TickData):
        """Check for immediate quality issues in tick data"""
        try:
            symbol = tick.symbol
            current_time = datetime.utcnow()

            # Check for price jumps
            if len(self.price_history[symbol]) >= 2:
                prev_price = self.price_history[symbol][-2]
                price_change_pct = abs(tick.price - prev_price) / prev_price

                if price_change_pct > self.thresholds['price_jump_threshold']:
                    await self._create_alert(
                        symbol,
                        AlertSeverity.HIGH,
                        "price_jump",
                        f"Large price jump detected: {price_change_pct:.2%}",
                        price_change_pct,
                        self.thresholds['price_jump_threshold']
                    )

            # Check spread anomalies
            if len(self.spread_history[symbol]) >= 10:
                avg_spread = statistics.mean(self.spread_history[symbol][-10:])
                if avg_spread > 0:
                    spread_ratio = tick.spread / avg_spread

                    if spread_ratio > self.thresholds['max_spread_ratio']:
                        await self._create_alert(
                            symbol,
                            AlertSeverity.MEDIUM,
                            "spread_anomaly",
                            f"Abnormal spread detected: {spread_ratio:.2f}x average",
                            spread_ratio,
                            self.thresholds['max_spread_ratio']
                        )

            # Check data latency (if timestamp available)
            if hasattr(tick, 'server_timestamp'):
                latency_ms = (current_time - tick.timestamp).total_seconds() * 1000

                if latency_ms > self.thresholds['max_latency_ms']:
                    await self._create_alert(
                        symbol,
                        AlertSeverity.MEDIUM,
                        "high_latency",
                        f"High data latency: {latency_ms:.0f}ms",
                        latency_ms,
                        self.thresholds['max_latency_ms']
                    )

        except Exception as e:
            logging_system.log_error(
                LogComponent.SYSTEM,
                e,
                {'action': 'check_immediate_issues', 'symbol': tick.symbol}
            )

    async def _periodic_quality_assessment(self):
        """Perform periodic comprehensive quality assessment"""
        while self.monitoring_active:
            try:
                for symbol in list(self.symbols_monitored):
                    metrics = await self._calculate_quality_metrics(symbol)
                    if metrics:
                        await self._store_quality_metrics(metrics)
                        await self._check_quality_thresholds(metrics)

                # Auto-resolve old alerts
                await self._auto_resolve_alerts()

                # Wait 5 minutes before next assessment
                await asyncio.sleep(300)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logging_system.log_error(
                    LogComponent.SYSTEM,
                    e,
                    {'action': 'periodic_quality_assessment'}
                )
                await asyncio.sleep(60)  # Wait 1 minute on error

    async def _calculate_quality_metrics(self, symbol: str) -> Optional[DataQualityMetrics]:
        """Calculate comprehensive quality metrics for symbol"""
        try:
            current_time = datetime.utcnow()

            # Get recent data
            timestamps = self.tick_timestamps.get(symbol, [])
            prices = self.price_history.get(symbol, [])
            spreads = self.spread_history.get(symbol, [])

            if not timestamps or not prices:
                return None

            # Filter data for different time windows
            one_hour_ago = current_time - timedelta(hours=1)
            twenty_four_hours_ago = current_time - timedelta(hours=24)

            timestamps_1h = [ts for ts in timestamps if ts >= one_hour_ago]
            timestamps_24h = [ts for ts in timestamps if ts >= twenty_four_hours_ago]

            # Completeness metrics
            data_completeness_score, missing_data_pct, gap_count, longest_gap = self._calculate_completeness_metrics(timestamps_1h)

            # Accuracy metrics
            consistency_score, outlier_pct, integrity_score = self._calculate_accuracy_metrics(prices[-100:] if len(prices) > 100 else prices)
            spread_anomaly_count = self._count_spread_anomalies(spreads[-100:] if len(spreads) > 100 else spreads)

            # Timeliness metrics
            latency_ms, frequency_score, realtime_score = self._calculate_timeliness_metrics(timestamps_1h)

            # Volatility metrics
            volatility_score, pattern_score, trend_score = self._calculate_volatility_metrics(prices[-100:] if len(prices) > 100 else prices)

            # Overall quality calculation
            overall_score = self._calculate_overall_quality(
                data_completeness_score, consistency_score, frequency_score,
                volatility_score, integrity_score
            )

            quality_level = self._determine_quality_level(overall_score)

            # Get active alerts
            active_alerts = [alert.message for alert in self.active_alerts.get(symbol, []) if not alert.resolved]
            critical_issues = [alert.message for alert in self.active_alerts.get(symbol, [])
                             if not alert.resolved and alert.severity == AlertSeverity.CRITICAL]

            metrics = DataQualityMetrics(
                symbol=symbol,
                timestamp=current_time,
                data_completeness_score=data_completeness_score,
                missing_data_percentage=missing_data_pct,
                data_gap_count=gap_count,
                longest_data_gap_minutes=longest_gap,
                price_consistency_score=consistency_score,
                spread_anomaly_count=spread_anomaly_count,
                outlier_percentage=outlier_pct,
                data_integrity_score=integrity_score,
                latency_ms=latency_ms,
                update_frequency_score=frequency_score,
                real_time_score=realtime_score,
                volatility_score=volatility_score,
                pattern_consistency_score=pattern_score,
                trend_stability_score=trend_score,
                overall_quality_score=overall_score,
                quality_level=quality_level,
                active_alerts=active_alerts,
                critical_issues=critical_issues,
                tick_count_1h=len(timestamps_1h),
                tick_count_24h=len(timestamps_24h),
                average_ticks_per_minute=len(timestamps_1h) / 60.0 if timestamps_1h else 0.0
            )

            # Store in history
            if symbol not in self.quality_history:
                self.quality_history[symbol] = []

            self.quality_history[symbol].append(metrics)
            if len(self.quality_history[symbol]) > self.max_history_size:
                self.quality_history[symbol] = self.quality_history[symbol][-self.max_history_size:]

            return metrics

        except Exception as e:
            logging_system.log_error(
                LogComponent.SYSTEM,
                e,
                {'action': 'calculate_quality_metrics', 'symbol': symbol}
            )
            return None

    def _calculate_completeness_metrics(self, timestamps: List[datetime]) -> Tuple[float, float, int, float]:
        """Calculate data completeness metrics"""
        try:
            if not timestamps:
                return 0.0, 100.0, 0, 0.0

            # Expected vs actual data points
            time_span_minutes = (timestamps[-1] - timestamps[0]).total_seconds() / 60
            expected_points = max(1, time_span_minutes * self.thresholds['min_ticks_per_minute'])
            actual_points = len(timestamps)

            completeness_score = min(100.0, (actual_points / expected_points) * 100)
            missing_percentage = max(0.0, 100.0 - completeness_score)

            # Calculate gaps
            gaps = []
            for i in range(1, len(timestamps)):
                gap_minutes = (timestamps[i] - timestamps[i-1]).total_seconds() / 60
                gaps.append(gap_minutes)

            gap_count = sum(1 for gap in gaps if gap > 1.0)  # Gaps > 1 minute
            longest_gap = max(gaps) if gaps else 0.0

            return completeness_score, missing_percentage, gap_count, longest_gap

        except Exception:
            return 0.0, 100.0, 0, 0.0

    def _calculate_accuracy_metrics(self, prices: List[float]) -> Tuple[float, float, float]:
        """Calculate data accuracy metrics"""
        try:
            if len(prices) < 10:
                return 100.0, 0.0, 100.0

            # Price consistency (measure of smoothness)
            returns = np.diff(prices) / prices[:-1]
            return_std = np.std(returns)
            consistency_score = max(0.0, 100.0 - (return_std * 1000))  # Scale appropriately

            # Outlier detection using IQR method
            q1, q3 = np.percentile(prices, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = [p for p in prices if p < lower_bound or p > upper_bound]
            outlier_percentage = (len(outliers) / len(prices)) * 100

            # Data integrity (no negative prices, reasonable values)
            invalid_count = sum(1 for p in prices if p <= 0 or np.isnan(p) or np.isinf(p))
            integrity_score = max(0.0, ((len(prices) - invalid_count) / len(prices)) * 100)

            return consistency_score, outlier_percentage, integrity_score

        except Exception:
            return 100.0, 0.0, 100.0

    def _count_spread_anomalies(self, spreads: List[float]) -> int:
        """Count spread anomalies"""
        try:
            if len(spreads) < 10:
                return 0

            avg_spread = statistics.mean(spreads)
            anomaly_count = sum(1 for spread in spreads if spread > avg_spread * self.thresholds['max_spread_ratio'])

            return anomaly_count

        except Exception:
            return 0

    def _calculate_timeliness_metrics(self, timestamps: List[datetime]) -> Tuple[float, float, float]:
        """Calculate timeliness metrics"""
        try:
            if len(timestamps) < 2:
                return 0.0, 0.0, 0.0

            current_time = datetime.utcnow()

            # Average latency (time since last update)
            latest_timestamp = timestamps[-1]
            latency_ms = (current_time - latest_timestamp).total_seconds() * 1000

            # Update frequency score
            intervals = [(timestamps[i] - timestamps[i-1]).total_seconds() for i in range(1, len(timestamps))]
            avg_interval = statistics.mean(intervals)
            expected_interval = 60 / self.thresholds['min_ticks_per_minute']  # seconds

            frequency_score = min(100.0, (expected_interval / max(avg_interval, 0.1)) * 100)

            # Real-time score (combination of latency and frequency)
            latency_score = max(0.0, 100.0 - (latency_ms / self.thresholds['max_latency_ms']) * 100)
            realtime_score = (latency_score + frequency_score) / 2

            return latency_ms, frequency_score, realtime_score

        except Exception:
            return 0.0, 0.0, 0.0

    def _calculate_volatility_metrics(self, prices: List[float]) -> Tuple[float, float, float]:
        """Calculate volatility and pattern metrics"""
        try:
            if len(prices) < 20:
                return 50.0, 50.0, 50.0

            # Volatility score (normalized)
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns)

            # Compare with historical volatility (simple version)
            volatility_score = max(0.0, min(100.0, 50.0 + (50.0 * (0.02 - volatility) / 0.02)))

            # Pattern consistency (autocorrelation)
            if len(returns) > 10:
                autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                if not np.isnan(autocorr):
                    pattern_score = 50.0 + autocorr * 50.0
                else:
                    pattern_score = 50.0
            else:
                pattern_score = 50.0

            # Trend stability
            if len(prices) >= 10:
                trend_changes = 0
                for i in range(2, len(prices)):
                    if ((prices[i] > prices[i-1]) != (prices[i-1] > prices[i-2])):
                        trend_changes += 1

                stability_ratio = 1.0 - (trend_changes / max(len(prices) - 2, 1))
                trend_score = stability_ratio * 100
            else:
                trend_score = 50.0

            return volatility_score, pattern_score, trend_score

        except Exception:
            return 50.0, 50.0, 50.0

    def _calculate_overall_quality(self, completeness: float, consistency: float,
                                 frequency: float, volatility: float, integrity: float) -> float:
        """Calculate overall quality score"""
        try:
            # Weighted average of different quality aspects
            weights = {
                'completeness': 0.3,
                'consistency': 0.2,
                'frequency': 0.2,
                'volatility': 0.15,
                'integrity': 0.15
            }

            overall = (
                completeness * weights['completeness'] +
                consistency * weights['consistency'] +
                frequency * weights['frequency'] +
                volatility * weights['volatility'] +
                integrity * weights['integrity']
            )

            return max(0.0, min(100.0, overall))

        except Exception:
            return 50.0

    def _determine_quality_level(self, score: float) -> DataQualityLevel:
        """Determine quality level based on score"""
        if score >= 90:
            return DataQualityLevel.EXCELLENT
        elif score >= 80:
            return DataQualityLevel.GOOD
        elif score >= 70:
            return DataQualityLevel.FAIR
        elif score >= 50:
            return DataQualityLevel.POOR
        else:
            return DataQualityLevel.CRITICAL

    async def _create_alert(self, symbol: str, severity: AlertSeverity, alert_type: str,
                          message: str, current_value: float, threshold: float):
        """Create quality alert"""
        try:
            self.alert_id_counter += 1
            alert_id = f"qa_{self.alert_id_counter}_{int(datetime.utcnow().timestamp())}"

            alert = QualityAlert(
                alert_id=alert_id,
                symbol=symbol,
                severity=severity,
                alert_type=alert_type,
                message=message,
                timestamp=datetime.utcnow(),
                threshold_breached=threshold,
                current_value=current_value
            )

            if symbol not in self.active_alerts:
                self.active_alerts[symbol] = []

            self.active_alerts[symbol].append(alert)

            # Log alert
            logging_system.log(
                LogComponent.SYSTEM,
                LogLevel.WARNING if severity in [AlertSeverity.LOW, AlertSeverity.MEDIUM] else LogLevel.ERROR,
                f"Data quality alert: {message}",
                {
                    'symbol': symbol,
                    'alert_type': alert_type,
                    'severity': severity.value,
                    'current_value': current_value,
                    'threshold': threshold
                }
            )

            # Cache alert for API access
            await self.cache_manager.set(
                CacheNamespace.SYSTEM_METRICS,
                f"quality_alert:{symbol}:{alert_id}",
                asdict(alert),
                ttl=3600  # 1 hour
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.SYSTEM,
                e,
                {'action': 'create_alert', 'symbol': symbol, 'alert_type': alert_type}
            )

    async def _check_quality_thresholds(self, metrics: DataQualityMetrics):
        """Check quality metrics against thresholds"""
        try:
            symbol = metrics.symbol

            # Check completeness
            if metrics.data_completeness_score < self.thresholds['completeness_min']:
                await self._create_alert(
                    symbol,
                    AlertSeverity.HIGH,
                    "low_completeness",
                    f"Data completeness below threshold: {metrics.data_completeness_score:.1f}%",
                    metrics.data_completeness_score,
                    self.thresholds['completeness_min']
                )

            # Check data gaps
            if metrics.longest_data_gap_minutes > self.thresholds['max_gap_minutes']:
                await self._create_alert(
                    symbol,
                    AlertSeverity.MEDIUM,
                    "data_gap",
                    f"Long data gap detected: {metrics.longest_data_gap_minutes:.1f} minutes",
                    metrics.longest_data_gap_minutes,
                    self.thresholds['max_gap_minutes']
                )

            # Check overall quality
            if metrics.overall_quality_score < 50:
                await self._create_alert(
                    symbol,
                    AlertSeverity.CRITICAL,
                    "poor_quality",
                    f"Overall data quality is poor: {metrics.overall_quality_score:.1f}%",
                    metrics.overall_quality_score,
                    50.0
                )

            # Check tick frequency
            if metrics.average_ticks_per_minute < self.thresholds['min_ticks_per_minute']:
                await self._create_alert(
                    symbol,
                    AlertSeverity.MEDIUM,
                    "low_frequency",
                    f"Low tick frequency: {metrics.average_ticks_per_minute:.1f} ticks/min",
                    metrics.average_ticks_per_minute,
                    self.thresholds['min_ticks_per_minute']
                )

        except Exception as e:
            logging_system.log_error(
                LogComponent.SYSTEM,
                e,
                {'action': 'check_quality_thresholds', 'symbol': metrics.symbol}
            )

    async def _auto_resolve_alerts(self):
        """Auto-resolve old alerts"""
        try:
            current_time = datetime.utcnow()
            resolution_age = timedelta(hours=1)  # Auto-resolve after 1 hour

            for symbol in list(self.active_alerts.keys()):
                alerts_to_remove = []

                for i, alert in enumerate(self.active_alerts[symbol]):
                    if (alert.auto_resolve and
                        not alert.resolved and
                        current_time - alert.timestamp > resolution_age):

                        alert.resolved = True
                        alerts_to_remove.append(i)

                # Remove resolved alerts
                for i in reversed(alerts_to_remove):
                    del self.active_alerts[symbol][i]

        except Exception as e:
            logging_system.log_error(
                LogComponent.SYSTEM,
                e,
                {'action': 'auto_resolve_alerts'}
            )

    async def _store_quality_metrics(self, metrics: DataQualityMetrics):
        """Store quality metrics in InfluxDB and cache"""
        try:
            # Store in InfluxDB
            quality_data = {
                'overall_quality_score': metrics.overall_quality_score,
                'data_completeness_score': metrics.data_completeness_score,
                'price_consistency_score': metrics.price_consistency_score,
                'update_frequency_score': metrics.update_frequency_score,
                'volatility_score': metrics.volatility_score,
                'data_integrity_score': metrics.data_integrity_score,
                'tick_count_1h': metrics.tick_count_1h,
                'average_ticks_per_minute': metrics.average_ticks_per_minute,
                'latency_ms': metrics.latency_ms
            }

            if self.influxdb_manager:
                await self.influxdb_manager.write_feature_data(
                    symbol=metrics.symbol,
                    features=quality_data,
                    timestamp=metrics.timestamp
                )

            # Cache latest metrics
            await self.cache_manager.set(
                CacheNamespace.SYSTEM_METRICS,
                f"quality_metrics:{metrics.symbol}",
                asdict(metrics),
                ttl=900  # 15 minutes
            )

        except Exception as e:
            logging_system.log_error(
                LogComponent.SYSTEM,
                e,
                {'action': 'store_quality_metrics', 'symbol': metrics.symbol}
            )

    async def get_quality_metrics(self, symbol: str) -> Optional[DataQualityMetrics]:
        """Get latest quality metrics for symbol"""
        try:
            # Try cache first
            cached_metrics = await self.cache_manager.get(
                CacheNamespace.SYSTEM_METRICS,
                f"quality_metrics:{symbol}"
            )

            if cached_metrics:
                return DataQualityMetrics(**cached_metrics)

            # Fallback to history
            if symbol in self.quality_history and self.quality_history[symbol]:
                return self.quality_history[symbol][-1]

            return None

        except Exception as e:
            logging_system.log_error(
                LogComponent.SYSTEM,
                e,
                {'action': 'get_quality_metrics', 'symbol': symbol}
            )
            return None

    async def get_quality_summary(self) -> Dict[str, Any]:
        """Get quality summary for all monitored symbols"""
        try:
            summary = {
                'monitoring_active': self.monitoring_active,
                'symbols_monitored': list(self.symbols_monitored),
                'total_alerts': sum(len(alerts) for alerts in self.active_alerts.values()),
                'symbols_quality': {}
            }

            for symbol in self.symbols_monitored:
                metrics = await self.get_quality_metrics(symbol)
                if metrics:
                    summary['symbols_quality'][symbol] = {
                        'overall_score': metrics.overall_quality_score,
                        'quality_level': metrics.quality_level.value,
                        'active_alerts': len(metrics.active_alerts),
                        'critical_issues': len(metrics.critical_issues),
                        'tick_count_1h': metrics.tick_count_1h
                    }

            return summary

        except Exception as e:
            logging_system.log_error(
                LogComponent.SYSTEM,
                e,
                {'action': 'get_quality_summary'}
            )
            return {'error': str(e)}

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get monitoring status"""
        return {
            'monitoring_active': self.monitoring_active,
            'symbols_monitored': list(self.symbols_monitored),
            'thresholds': self.thresholds,
            'alert_counts': {
                symbol: len([a for a in alerts if not a.resolved])
                for symbol, alerts in self.active_alerts.items()
            },
            'data_buffer_sizes': {
                symbol: {
                    'timestamps': len(self.tick_timestamps.get(symbol, [])),
                    'prices': len(self.price_history.get(symbol, [])),
                    'spreads': len(self.spread_history.get(symbol, []))
                }
                for symbol in self.symbols_monitored
            }
        }

# Global data quality monitor instance
quality_monitor = DataQualityMonitor()

async def get_quality_monitor() -> DataQualityMonitor:
    """Get initialized data quality monitor"""
    if not quality_monitor.websocket_client:
        await quality_monitor.initialize()
    return quality_monitor