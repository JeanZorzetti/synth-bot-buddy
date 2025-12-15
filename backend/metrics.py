"""
Sistema de Métricas com Prometheus
Expõe métricas de performance e trading para monitoramento
"""

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, REGISTRY
from prometheus_client import CollectorRegistry, multiprocess, generate_latest
from typing import Optional
import time
import logging

logger = logging.getLogger(__name__)


# === MÉTRICAS DE TRADING ===

# Contador de trades executados
trades_total = Counter(
    'deriv_bot_trades_total',
    'Total de trades executados',
    ['signal_type', 'status']  # labels: BUY/SELL, success/failed
)

# Histograma de duração de trades
trade_duration_seconds = Histogram(
    'deriv_bot_trade_duration_seconds',
    'Duração dos trades em segundos',
    ['signal_type'],
    buckets=[60, 300, 600, 1800, 3600, 7200, 14400]  # 1min, 5min, 10min, 30min, 1h, 2h, 4h
)

# Gauge de P&L atual
current_pnl = Gauge(
    'deriv_bot_current_pnl',
    'Profit & Loss atual em USD',
    ['timeframe']  # daily, weekly, monthly, total
)

# Gauge de win rate
win_rate = Gauge(
    'deriv_bot_win_rate',
    'Taxa de acerto (%) dos trades',
    ['timeframe']  # daily, weekly, monthly, total
)

# Contador de lucros/perdas
profit_loss_total = Counter(
    'deriv_bot_profit_loss_total',
    'Total de lucro/perda acumulado',
    ['outcome']  # profit, loss
)


# === MÉTRICAS DE SINAIS ML ===

# Histograma de latência de geração de sinais
signal_latency_ms = Histogram(
    'deriv_bot_signal_latency_milliseconds',
    'Latência para gerar sinal de trading em ms',
    ['model'],
    buckets=[10, 25, 50, 100, 250, 500, 1000]
)

# Contador de sinais gerados
signals_generated = Counter(
    'deriv_bot_signals_generated_total',
    'Total de sinais de trading gerados',
    ['signal_type', 'confidence_level']  # BUY/SELL/NEUTRAL, low/medium/high
)

# Gauge de confiança do modelo
model_confidence = Gauge(
    'deriv_bot_model_confidence',
    'Confiança do modelo ML (0-100)',
    ['model']
)

# Gauge de accuracy do modelo
model_accuracy = Gauge(
    'deriv_bot_model_accuracy',
    'Accuracy histórica do modelo (%)',
    ['model']
)


# === MÉTRICAS DE PERFORMANCE ===

# Histograma de processamento de ticks
tick_processing_ms = Histogram(
    'deriv_bot_tick_processing_milliseconds',
    'Tempo para processar um tick de mercado em ms',
    buckets=[1, 5, 10, 25, 50, 100, 250]
)

# Counter de ticks processados
ticks_processed = Counter(
    'deriv_bot_ticks_processed_total',
    'Total de ticks de mercado processados',
    ['symbol']
)

# Gauge de throughput
ticks_per_second = Gauge(
    'deriv_bot_ticks_per_second',
    'Taxa de processamento de ticks/segundo',
    ['symbol']
)


# === MÉTRICAS DE CACHE ===

# Counter de cache hits/misses
cache_operations = Counter(
    'deriv_bot_cache_operations_total',
    'Operações de cache',
    ['operation', 'result']  # get/set, hit/miss
)

# Gauge de hit rate do cache
cache_hit_rate = Gauge(
    'deriv_bot_cache_hit_rate',
    'Taxa de acerto do cache (%)'
)


# === MÉTRICAS DE API ===

# Counter de chamadas à API Deriv
api_calls_total = Counter(
    'deriv_bot_api_calls_total',
    'Total de chamadas à API Deriv',
    ['endpoint', 'status']  # ticks_history/authorize/etc, success/error
)

# Histograma de latência da API
api_latency_ms = Histogram(
    'deriv_bot_api_latency_milliseconds',
    'Latência das chamadas à API Deriv em ms',
    ['endpoint'],
    buckets=[50, 100, 250, 500, 1000, 2000, 5000]
)


# === MÉTRICAS DE BACKTESTING ===

# Histogram de tempo de execução de backtest
backtest_duration_seconds = Histogram(
    'deriv_bot_backtest_duration_seconds',
    'Duração de execução de backtest',
    ['method'],  # iterativo, vetorizado
    buckets=[0.1, 0.5, 1, 5, 10, 30, 60]
)

# Gauge de métricas de backtest
backtest_sharpe_ratio = Gauge(
    'deriv_bot_backtest_sharpe_ratio',
    'Sharpe Ratio do último backtest',
    ['strategy']
)

backtest_max_drawdown = Gauge(
    'deriv_bot_backtest_max_drawdown',
    'Max Drawdown do último backtest (%)',
    ['strategy']
)


# === MÉTRICAS DE SISTEMA ===

# Info sobre versão do bot
bot_info = Info(
    'deriv_bot_info',
    'Informações sobre o bot de trading'
)

# Gauge de uptime
bot_uptime_seconds = Gauge(
    'deriv_bot_uptime_seconds',
    'Tempo de execução do bot em segundos'
)

# Counter de erros
errors_total = Counter(
    'deriv_bot_errors_total',
    'Total de erros ocorridos',
    ['error_type', 'severity']  # api/trading/ml/cache, critical/error/warning
)


class MetricsManager:
    """
    Gerenciador centralizado de métricas
    """

    def __init__(self):
        self.start_time = time.time()
        self._update_bot_info()

    def _update_bot_info(self):
        """Atualiza informações do bot"""
        bot_info.info({
            'version': '1.0.0',
            'model': 'XGBoost',
            'phase': '6_optimization',
            'features': 'ml,cache,vectorized_backtest,prometheus'
        })

    def get_uptime(self) -> float:
        """Retorna uptime em segundos"""
        return time.time() - self.start_time

    def update_uptime(self):
        """Atualiza métrica de uptime"""
        bot_uptime_seconds.set(self.get_uptime())

    def record_trade(self, signal_type: str, duration_seconds: float, profit: float, success: bool = True):
        """
        Registra execução de trade

        Args:
            signal_type: BUY ou SELL
            duration_seconds: Duração do trade
            profit: Lucro/prejuízo do trade
            success: Se o trade foi executado com sucesso
        """
        status = 'success' if success else 'failed'

        trades_total.labels(signal_type=signal_type, status=status).inc()

        if success:
            trade_duration_seconds.labels(signal_type=signal_type).observe(duration_seconds)

            # Registrar P&L
            if profit > 0:
                profit_loss_total.labels(outcome='profit').inc(profit)
            else:
                profit_loss_total.labels(outcome='loss').inc(abs(profit))

    def record_signal(self, signal_type: str, confidence: float, latency_ms: float, model: str = 'xgboost'):
        """
        Registra geração de sinal

        Args:
            signal_type: BUY, SELL ou NEUTRAL
            confidence: Confiança (0-100)
            latency_ms: Latência em ms
            model: Nome do modelo
        """
        # Classificar confiança
        if confidence >= 70:
            confidence_level = 'high'
        elif confidence >= 50:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'

        signals_generated.labels(signal_type=signal_type, confidence_level=confidence_level).inc()
        signal_latency_ms.labels(model=model).observe(latency_ms)
        model_confidence.labels(model=model).set(confidence)

    def record_tick_processing(self, symbol: str, processing_time_ms: float):
        """
        Registra processamento de tick

        Args:
            symbol: Símbolo do ativo
            processing_time_ms: Tempo de processamento em ms
        """
        ticks_processed.labels(symbol=symbol).inc()
        tick_processing_ms.observe(processing_time_ms)

    def record_api_call(self, endpoint: str, latency_ms: float, success: bool = True):
        """
        Registra chamada à API

        Args:
            endpoint: Endpoint chamado
            latency_ms: Latência em ms
            success: Se a chamada foi bem-sucedida
        """
        status = 'success' if success else 'error'
        api_calls_total.labels(endpoint=endpoint, status=status).inc()

        if success:
            api_latency_ms.labels(endpoint=endpoint).observe(latency_ms)

    def record_cache_operation(self, operation: str, hit: bool):
        """
        Registra operação de cache

        Args:
            operation: get ou set
            hit: Se foi cache hit (apenas para get)
        """
        result = 'hit' if hit else 'miss'
        cache_operations.labels(operation=operation, result=result).inc()

    def record_error(self, error_type: str, severity: str = 'error'):
        """
        Registra erro

        Args:
            error_type: Tipo do erro (api, trading, ml, cache)
            severity: Severidade (critical, error, warning)
        """
        errors_total.labels(error_type=error_type, severity=severity).inc()

    def record_backtest(self, strategy: str, duration_seconds: float, sharpe: float, max_dd: float, method: str = 'vectorized'):
        """
        Registra execução de backtest

        Args:
            strategy: Nome da estratégia
            duration_seconds: Duração do backtest
            sharpe: Sharpe ratio
            max_dd: Max drawdown
            method: iterativo ou vetorizado
        """
        backtest_duration_seconds.labels(method=method).observe(duration_seconds)
        backtest_sharpe_ratio.labels(strategy=strategy).set(sharpe)
        backtest_max_drawdown.labels(strategy=strategy).set(max_dd)

    def update_pnl(self, daily: float = 0, weekly: float = 0, monthly: float = 0, total: float = 0):
        """
        Atualiza P&L por timeframe

        Args:
            daily: P&L diário
            weekly: P&L semanal
            monthly: P&L mensal
            total: P&L total
        """
        if daily != 0:
            current_pnl.labels(timeframe='daily').set(daily)
        if weekly != 0:
            current_pnl.labels(timeframe='weekly').set(weekly)
        if monthly != 0:
            current_pnl.labels(timeframe='monthly').set(monthly)
        if total != 0:
            current_pnl.labels(timeframe='total').set(total)

    def update_win_rate(self, daily: float = 0, weekly: float = 0, monthly: float = 0, total: float = 0):
        """
        Atualiza win rate por timeframe

        Args:
            daily: Win rate diário (%)
            weekly: Win rate semanal (%)
            monthly: Win rate mensal (%)
            total: Win rate total (%)
        """
        if daily != 0:
            win_rate.labels(timeframe='daily').set(daily)
        if weekly != 0:
            win_rate.labels(timeframe='weekly').set(weekly)
        if monthly != 0:
            win_rate.labels(timeframe='monthly').set(monthly)
        if total != 0:
            win_rate.labels(timeframe='total').set(total)


# Instância global do metrics manager
_metrics_manager: Optional[MetricsManager] = None


def get_metrics_manager() -> MetricsManager:
    """
    Retorna instância global do metrics manager (singleton)

    Returns:
        MetricsManager instance
    """
    global _metrics_manager
    if _metrics_manager is None:
        _metrics_manager = MetricsManager()
    return _metrics_manager


def initialize_metrics_manager():
    """Inicializa metrics manager global"""
    global _metrics_manager
    _metrics_manager = MetricsManager()
    logger.info("Metrics manager inicializado")


if __name__ == "__main__":
    # Teste do metrics manager
    logging.basicConfig(level=logging.INFO)

    metrics = MetricsManager()

    # Simular algumas métricas
    metrics.record_trade('BUY', 300, 10.5, success=True)
    metrics.record_trade('SELL', 600, -5.2, success=True)
    metrics.record_signal('BUY', 75.5, 45.2)
    metrics.record_tick_processing('R_100', 5.5)
    metrics.record_api_call('ticks_history', 150, success=True)
    metrics.record_cache_operation('get', hit=True)
    metrics.record_backtest('SMA_CrossOver', 2.5, 1.45, 12.3)
    metrics.update_pnl(total=150.75)
    metrics.update_win_rate(total=68.5)

    logger.info("Métricas de teste registradas com sucesso!")

    # Gerar output Prometheus
    output = generate_latest(REGISTRY).decode('utf-8')
    print("\n=== AMOSTRA DE MÉTRICAS PROMETHEUS ===")
    print(output[:1000] + "...\n")

    logger.info("Sistema de métricas Prometheus funcionando corretamente!")
