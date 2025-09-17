"""
Trading System Orchestrator - Sistema de Orquestração Completo da Fase 14
Sistema principal que integra todos os componentes de trading em tempo real
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import signal
import sys

from database_config import DatabaseManager
from redis_cache_manager import RedisCacheManager
from real_logging_system import RealLoggingSystem
from real_trading_executor import RealTradingExecutor, TradingMode
from real_position_manager import RealPositionManager
from ai_signal_connector import AISignalConnector
from real_risk_manager import RealRiskManager
from order_execution_monitor import OrderExecutionMonitor
from portfolio_tracker import PortfolioTracker
from realtime_alerts import RealtimeAlerts
from trading_test_suite import TradingTestSuite
from real_deriv_websocket import RealDerivWebSocket

class SystemStatus(Enum):
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

class ComponentStatus(Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    ERROR = "error"
    STOPPED = "stopped"

@dataclass
class SystemMetrics:
    """Métricas gerais do sistema"""
    uptime_seconds: int
    total_trades: int
    active_positions: int
    current_balance: float
    daily_pnl: float
    system_load: float
    memory_usage: float
    error_count: int
    last_update: datetime

@dataclass
class ComponentInfo:
    """Informações de um componente"""
    name: str
    status: ComponentStatus
    last_heartbeat: datetime
    error_count: int
    metrics: Dict[str, Any]

class TradingSystemOrchestrator:
    """Orquestrador principal do sistema de trading"""

    def __init__(self, initial_capital: float = 10000.0, trading_mode: TradingMode = TradingMode.PAPER):
        # Configurações principais
        self.initial_capital = initial_capital
        self.trading_mode = trading_mode
        self.symbols_to_trade = [
            "R_10", "R_25", "R_50", "R_75", "R_100",
            "FRXEURUSD", "FRXGBPUSD", "FRXUSDJPY"
        ]

        # Estado do sistema
        self.system_status = SystemStatus.STOPPED
        self.start_time = None
        self.components: Dict[str, ComponentInfo] = {}

        # Componentes principais
        self.db_manager = DatabaseManager()
        self.cache_manager = RedisCacheManager()
        self.logger = RealLoggingSystem()
        self.deriv_client = RealDerivWebSocket()

        # Componentes de trading
        self.trading_executor = RealTradingExecutor()
        self.position_manager = RealPositionManager(self.deriv_client)
        self.risk_manager = RealRiskManager(initial_capital)
        self.order_monitor = OrderExecutionMonitor()
        self.portfolio_tracker = PortfolioTracker(initial_capital)
        self.alert_system = RealtimeAlerts()
        self.ai_connector = AISignalConnector()

        # Sistema de testes
        self.test_suite = TradingTestSuite()

        # Tasks de monitoramento
        self.system_monitor_task = None
        self.heartbeat_task = None
        self.metrics_task = None

        # Callbacks para shutdown graceful
        self.shutdown_event = asyncio.Event()

        logging.basicConfig(level=logging.INFO)
        self.logger_py = logging.getLogger(__name__)

    async def initialize(self):
        """Inicializa todo o sistema de trading"""
        try:
            print("🚀 Inicializando Trading System Orchestrator...")
            self.system_status = SystemStatus.STARTING

            # 1. Componentes base
            print("📊 Inicializando componentes base...")
            await self._initialize_component("database", self.db_manager.initialize())
            await self._initialize_component("cache", self.cache_manager.initialize())
            await self._initialize_component("logger", self.logger.initialize())
            await self._initialize_component("deriv_client", self.deriv_client.initialize())

            # 2. Componentes de trading
            print("💼 Inicializando componentes de trading...")
            await self._initialize_component("trading_executor", self.trading_executor.initialize())
            await self._initialize_component("position_manager", self.position_manager.initialize())
            await self._initialize_component("risk_manager", self.risk_manager.initialize(self.position_manager))
            await self._initialize_component("order_monitor", self.order_monitor.initialize(self.risk_manager))
            await self._initialize_component("portfolio_tracker", self.portfolio_tracker.initialize(self.position_manager, self.order_monitor))

            # 3. Sistemas avançados
            print("🤖 Inicializando sistemas avançados...")
            await self._initialize_component("alert_system", self.alert_system.initialize())
            await self._initialize_component("ai_connector", self.ai_connector.initialize())

            # 4. Sistema de testes
            print("🧪 Inicializando sistema de testes...")
            await self._initialize_component("test_suite", self.test_suite.initialize())

            # 5. Configurar callbacks entre componentes
            await self._setup_component_callbacks()

            # 6. Configurar handlers de shutdown
            self._setup_signal_handlers()

            self.system_status = SystemStatus.READY
            await self.logger.log_activity("trading_system_initialized", {
                "components": len(self.components),
                "trading_mode": self.trading_mode.value,
                "initial_capital": self.initial_capital
            })

            print("✅ Trading System Orchestrator inicializado com sucesso!")
            self._print_system_status()

        except Exception as e:
            self.system_status = SystemStatus.ERROR
            await self.logger.log_error("system_initialization_error", str(e))
            print(f"❌ Erro na inicialização: {e}")
            raise

    async def _initialize_component(self, name: str, init_coro):
        """Inicializa um componente específico"""
        try:
            component_info = ComponentInfo(
                name=name,
                status=ComponentStatus.INITIALIZING,
                last_heartbeat=datetime.utcnow(),
                error_count=0,
                metrics={}
            )
            self.components[name] = component_info

            await init_coro
            component_info.status = ComponentStatus.READY
            component_info.last_heartbeat = datetime.utcnow()

            print(f"  ✅ {name} inicializado")

        except Exception as e:
            component_info.status = ComponentStatus.ERROR
            component_info.error_count += 1
            print(f"  ❌ Erro ao inicializar {name}: {e}")
            raise

    async def start_trading(self):
        """Inicia o sistema de trading"""
        try:
            if self.system_status != SystemStatus.READY:
                raise RuntimeError("Sistema não está pronto para iniciar")

            print(f"🎯 Iniciando trading em modo: {self.trading_mode.value}")
            self.system_status = SystemStatus.RUNNING
            self.start_time = datetime.utcnow()

            # Iniciar componentes de monitoramento
            print("📊 Iniciando monitoramento de sistema...")
            await self.risk_manager.start_monitoring()
            await self.position_manager.start_position_monitoring()
            await self.order_monitor.start_monitoring()
            await self.portfolio_tracker.start_tracking()
            await self.alert_system.start_monitoring()

            # Iniciar execução de trading
            print("💰 Iniciando execução de trading...")
            await self.trading_executor.start_trading(self.symbols_to_trade, self.trading_mode)

            # Iniciar sinais AI
            print("🤖 Iniciando geração de sinais AI...")
            await self.ai_connector.start_signal_generation(self.trading_mode)

            # Iniciar tasks de sistema
            print("🔄 Iniciando tasks de monitoramento...")
            self.system_monitor_task = asyncio.create_task(self._system_monitoring_loop())
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self.metrics_task = asyncio.create_task(self._metrics_collection_loop())

            # Alertas de início
            await self.alert_system.create_alert(
                alert_type="system_status",
                title="Sistema de Trading Iniciado",
                message=f"Trading iniciado em modo {self.trading_mode.value} com capital inicial ${self.initial_capital:,.2f}",
                priority="medium"
            )

            await self.logger.log_activity("trading_started", {
                "mode": self.trading_mode.value,
                "symbols": self.symbols_to_trade,
                "initial_capital": self.initial_capital
            })

            print("🚀 Trading System está ATIVO!")
            self._print_system_status()

        except Exception as e:
            self.system_status = SystemStatus.ERROR
            await self.logger.log_error("trading_start_error", str(e))
            print(f"❌ Erro ao iniciar trading: {e}")
            raise

    async def stop_trading(self):
        """Para o sistema de trading"""
        try:
            print("⏹️ Parando sistema de trading...")
            self.system_status = SystemStatus.STOPPING

            # Parar geração de sinais AI
            await self.ai_connector.stop_signal_generation()

            # Parar execução de trading
            await self.trading_executor.stop_trading()

            # Fechar todas as posições (se em modo paper)
            if self.trading_mode == TradingMode.PAPER:
                await self.position_manager.force_close_all_positions()

            # Parar componentes de monitoramento
            await self.risk_manager.stop_monitoring()
            await self.position_manager.stop_position_monitoring()
            await self.order_monitor.stop_monitoring()
            await self.portfolio_tracker.stop_tracking()
            await self.alert_system.stop_monitoring()

            # Cancelar tasks de sistema
            for task in [self.system_monitor_task, self.heartbeat_task, self.metrics_task]:
                if task:
                    task.cancel()

            # Alerta de parada
            await self.alert_system.create_alert(
                alert_type="system_status",
                title="Sistema de Trading Parado",
                message="Trading foi interrompido pelo usuário",
                priority="medium"
            )

            self.system_status = SystemStatus.STOPPED
            await self.logger.log_activity("trading_stopped", {})

            print("✅ Trading System parado com sucesso!")

        except Exception as e:
            self.system_status = SystemStatus.ERROR
            await self.logger.log_error("trading_stop_error", str(e))
            print(f"❌ Erro ao parar trading: {e}")

    async def run_system_tests(self) -> Dict[str, Any]:
        """Executa testes do sistema"""
        try:
            print("🧪 Executando testes do sistema...")

            await self.alert_system.create_alert(
                alert_type="system_status",
                title="Testes do Sistema Iniciados",
                message="Executando suite completa de testes",
                priority="low"
            )

            # Executar todos os testes
            results = await self.test_suite.run_all_tests()

            # Alerta com resultados
            success_rate = results.get("success_rate", 0)
            alert_priority = "high" if success_rate < 80 else "medium"

            await self.alert_system.create_alert(
                alert_type="system_status",
                title="Testes do Sistema Concluídos",
                message=f"Taxa de sucesso: {success_rate:.1f}% ({results.get('passed', 0)}/{results.get('total_tests', 0)} testes)",
                priority=alert_priority
            )

            return results

        except Exception as e:
            await self.logger.log_error("system_tests_error", str(e))
            raise

    async def _system_monitoring_loop(self):
        """Loop de monitoramento do sistema"""
        while self.system_status == SystemStatus.RUNNING:
            try:
                # Verificar saúde dos componentes
                await self._check_component_health()

                # Verificar uso de recursos
                await self._check_system_resources()

                # Verificar condições de emergência
                await self._check_emergency_conditions()

                await asyncio.sleep(10)  # A cada 10 segundos

            except Exception as e:
                await self.logger.log_error("system_monitoring_error", str(e))
                await asyncio.sleep(30)

    async def _heartbeat_loop(self):
        """Loop de heartbeat dos componentes"""
        while self.system_status == SystemStatus.RUNNING:
            try:
                for component_name, component_info in self.components.items():
                    component_info.last_heartbeat = datetime.utcnow()

                await asyncio.sleep(30)  # A cada 30 segundos

            except Exception as e:
                await self.logger.log_error("heartbeat_error", str(e))
                await asyncio.sleep(60)

    async def _metrics_collection_loop(self):
        """Loop de coleta de métricas"""
        while self.system_status == SystemStatus.RUNNING:
            try:
                metrics = await self._collect_system_metrics()

                # Salvar métricas no cache
                await self.cache_manager.set(
                    "system_metrics",
                    "current",
                    asdict(metrics),
                    ttl=300
                )

                await asyncio.sleep(60)  # A cada minuto

            except Exception as e:
                await self.logger.log_error("metrics_collection_error", str(e))
                await asyncio.sleep(120)

    async def _collect_system_metrics(self) -> SystemMetrics:
        """Coleta métricas do sistema"""
        try:
            uptime = (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0

            # Métricas do portfolio
            portfolio_value = await self.portfolio_tracker.get_current_portfolio_value()
            current_balance = portfolio_value.get("total_value", self.initial_capital)
            daily_pnl = portfolio_value.get("realized_pnl", 0.0)

            # Métricas das posições
            position_summary = await self.position_manager.get_position_summary()
            active_positions = position_summary.get("active_positions_count", 0)
            total_trades = position_summary.get("positions_opened_today", 0)

            # Contagem de erros
            error_count = sum(comp.error_count for comp in self.components.values())

            return SystemMetrics(
                uptime_seconds=int(uptime),
                total_trades=total_trades,
                active_positions=active_positions,
                current_balance=current_balance,
                daily_pnl=daily_pnl,
                system_load=0.0,  # Simplificado
                memory_usage=0.0,  # Simplificado
                error_count=error_count,
                last_update=datetime.utcnow()
            )

        except Exception as e:
            await self.logger.log_error("metrics_collection_individual_error", str(e))
            return SystemMetrics(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0, datetime.utcnow())

    async def _check_component_health(self):
        """Verifica saúde dos componentes"""
        for component_name, component_info in self.components.items():
            # Verificar se componente está respondendo
            time_since_heartbeat = (datetime.utcnow() - component_info.last_heartbeat).total_seconds()

            if time_since_heartbeat > 300:  # 5 minutos sem heartbeat
                component_info.status = ComponentStatus.ERROR
                component_info.error_count += 1

                await self.alert_system.create_alert(
                    alert_type="system_error",
                    title=f"Componente {component_name} não está respondendo",
                    message=f"Último heartbeat há {time_since_heartbeat:.0f} segundos",
                    priority="high",
                    action_required=True
                )

    async def _check_system_resources(self):
        """Verifica recursos do sistema"""
        # Implementação simplificada
        pass

    async def _check_emergency_conditions(self):
        """Verifica condições de emergência"""
        try:
            # Verificar se risk manager ativou emergência
            risk_summary = await self.risk_manager.get_risk_summary()
            if risk_summary.get("emergency_status", {}).get("emergency_stop", False):
                await self._trigger_emergency_shutdown("Risk Manager ativou parada de emergência")

        except Exception as e:
            await self.logger.log_error("emergency_conditions_check_error", str(e))

    async def _trigger_emergency_shutdown(self, reason: str):
        """Aciona shutdown de emergência"""
        try:
            print(f"🚨 EMERGÊNCIA: {reason}")

            await self.alert_system.create_alert(
                alert_type="system_error",
                title="PARADA DE EMERGÊNCIA",
                message=f"Sistema parado por emergência: {reason}",
                priority="emergency",
                action_required=True
            )

            # Fechar todas as posições imediatamente
            await self.position_manager.force_close_all_positions()

            # Parar trading
            await self.stop_trading()

            await self.logger.log_activity("emergency_shutdown", {"reason": reason})

        except Exception as e:
            await self.logger.log_error("emergency_shutdown_error", str(e))

    async def _setup_component_callbacks(self):
        """Configura callbacks entre componentes"""
        try:
            # Registrar callback para alertas de trading
            await self.ai_connector.register_callback(
                "trading_signal",
                self._on_trading_signal
            )

            # Registrar callback para alertas de risco
            await self.risk_manager.register_callback(
                "risk_warning",
                self._on_risk_alert
            )

        except Exception as e:
            await self.logger.log_error("callbacks_setup_error", str(e))

    async def _on_trading_signal(self, signal_data):
        """Callback para sinais de trading"""
        await self.alert_system.alert_trading_signal(
            signal_data.get("symbol", ""),
            signal_data.get("signal_type", ""),
            signal_data.get("confidence", 0.0)
        )

    async def _on_risk_alert(self, risk_data):
        """Callback para alertas de risco"""
        await self.alert_system.alert_risk_warning(
            risk_data.get("message", ""),
            risk_data.get("value", 0.0),
            risk_data.get("threshold", 0.0)
        )

    def _setup_signal_handlers(self):
        """Configura handlers para shutdown graceful"""
        def signal_handler(signum, frame):
            print(f"\n🛑 Recebido sinal {signum}. Iniciando shutdown graceful...")
            asyncio.create_task(self._graceful_shutdown())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def _graceful_shutdown(self):
        """Shutdown graceful do sistema"""
        try:
            print("🔄 Iniciando shutdown graceful...")

            # Parar trading se estiver ativo
            if self.system_status == SystemStatus.RUNNING:
                await self.stop_trading()

            # Shutdown de todos os componentes
            components_to_shutdown = [
                self.test_suite, self.ai_connector, self.alert_system,
                self.portfolio_tracker, self.order_monitor, self.risk_manager,
                self.position_manager, self.trading_executor, self.deriv_client,
                self.logger, self.cache_manager, self.db_manager
            ]

            for component in components_to_shutdown:
                if hasattr(component, 'shutdown'):
                    try:
                        await component.shutdown()
                    except Exception as e:
                        print(f"Erro no shutdown de {component.__class__.__name__}: {e}")

            self.shutdown_event.set()
            print("✅ Shutdown graceful concluído")

        except Exception as e:
            print(f"❌ Erro no shutdown graceful: {e}")

    def _print_system_status(self):
        """Imprime status do sistema"""
        print("\n" + "="*60)
        print("📊 TRADING SYSTEM STATUS")
        print("="*60)
        print(f"Status: {self.system_status.value.upper()}")
        print(f"Modo: {self.trading_mode.value}")
        print(f"Capital Inicial: ${self.initial_capital:,.2f}")
        print(f"Símbolos: {', '.join(self.symbols_to_trade)}")
        print(f"Componentes: {len(self.components)}")

        if self.start_time:
            uptime = datetime.utcnow() - self.start_time
            print(f"Uptime: {uptime}")

        print("\n🔧 COMPONENTES:")
        for name, info in self.components.items():
            status_icon = "✅" if info.status == ComponentStatus.READY else "❌"
            print(f"  {status_icon} {name}: {info.status.value}")

        print("="*60 + "\n")

    async def get_system_status(self) -> Dict[str, Any]:
        """Retorna status completo do sistema"""
        try:
            metrics = await self._collect_system_metrics()

            return {
                "system": {
                    "status": self.system_status.value,
                    "trading_mode": self.trading_mode.value,
                    "uptime_seconds": metrics.uptime_seconds,
                    "start_time": self.start_time.isoformat() if self.start_time else None
                },
                "components": {
                    name: asdict(info) for name, info in self.components.items()
                },
                "metrics": asdict(metrics),
                "trading": {
                    "symbols": self.symbols_to_trade,
                    "initial_capital": self.initial_capital
                }
            }

        except Exception as e:
            await self.logger.log_error("system_status_query_error", str(e))
            return {"error": str(e)}

    async def wait_for_shutdown(self):
        """Aguarda sinal de shutdown"""
        await self.shutdown_event.wait()

# Função principal para executar o sistema
async def main():
    """Função principal"""
    orchestrator = TradingSystemOrchestrator(
        initial_capital=10000.0,
        trading_mode=TradingMode.PAPER
    )

    try:
        # Inicializar sistema
        await orchestrator.initialize()

        # Executar testes (opcional)
        if len(sys.argv) > 1 and sys.argv[1] == "--test":
            print("\n🧪 Executando testes do sistema...")
            test_results = await orchestrator.run_system_tests()
            print(f"\n📋 Resultados dos testes: {test_results['success_rate']:.1f}% de sucesso")

        # Iniciar trading
        await orchestrator.start_trading()

        # Aguardar shutdown
        await orchestrator.wait_for_shutdown()

    except KeyboardInterrupt:
        print("\n⚠️ Interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro no sistema: {e}")
    finally:
        await orchestrator._graceful_shutdown()

if __name__ == "__main__":
    asyncio.run(main())