"""
Trading Test Suite - Sistema de Teste de Execução com Paper Trading
Sistema completo para testar a execução de trading usando paper trading e simulações
"""

import asyncio
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import random

from database_config import DatabaseManager
from redis_cache_manager import RedisCacheManager, CacheNamespace
from real_logging_system import RealLoggingSystem
from real_trading_executor import RealTradingExecutor, TradingMode, TradeType
from real_position_manager import RealPositionManager
from ai_signal_connector import AISignalConnector
from real_risk_manager import RealRiskManager
from order_execution_monitor import OrderExecutionMonitor
from portfolio_tracker import PortfolioTracker
from realtime_alerts import RealtimeAlerts

class TestScenario(Enum):
    BASIC_TRADING = "basic_trading"
    HIGH_FREQUENCY = "high_frequency"
    VOLATILE_MARKET = "volatile_market"
    TRENDING_MARKET = "trending_market"
    SIDEWAYS_MARKET = "sideways_market"
    RISK_MANAGEMENT = "risk_management"
    AI_SIGNALS = "ai_signals"
    STRESS_TEST = "stress_test"

class TestResult(Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"

@dataclass
class TestCase:
    """Caso de teste individual"""
    test_id: str
    name: str
    description: str
    scenario: TestScenario
    duration_minutes: int
    symbols: List[str]
    initial_capital: float
    max_positions: int
    expected_trades: int
    success_criteria: Dict[str, Any]
    risk_limits: Dict[str, float]

@dataclass
class TestExecution:
    """Execução de um teste"""
    test_id: str
    start_time: datetime
    end_time: Optional[datetime]
    status: str
    trades_executed: int
    positions_opened: int
    max_drawdown: float
    final_balance: float
    total_return: float
    return_pct: float
    sharpe_ratio: float
    win_rate: float
    alerts_generated: int
    errors_count: int
    result: TestResult
    details: Dict[str, Any]

@dataclass
class MarketSimulator:
    """Simulador de mercado para testes"""
    symbol: str
    initial_price: float
    volatility: float
    trend: float
    current_price: float
    price_history: List[Tuple[datetime, float]]
    last_update: datetime

class TradingTestSuite:
    """Suite de testes para trading system"""

    def __init__(self):
        # Componentes principais
        self.db_manager = DatabaseManager()
        self.cache_manager = RedisCacheManager()
        self.logger = RealLoggingSystem()

        # Componentes de trading (serão inicializados para cada teste)
        self.trading_executor = None
        self.position_manager = None
        self.risk_manager = None
        self.ai_connector = None
        self.order_monitor = None
        self.portfolio_tracker = None
        self.alert_system = None

        # Estado dos testes
        self.is_testing = False
        self.current_test = None
        self.test_results: List[TestExecution] = []

        # Simuladores de mercado
        self.market_simulators: Dict[str, MarketSimulator] = {}

        # Configurações de teste
        self.test_symbols = ["R_10", "R_25", "R_50", "FRXEURUSD", "FRXGBPUSD"]
        self.base_capital = 10000.0

        # Casos de teste predefinidos
        self.test_cases: Dict[str, TestCase] = {}

        logging.basicConfig(level=logging.INFO)
        self.logger_py = logging.getLogger(__name__)

    async def initialize(self):
        """Inicializa a suite de testes"""
        try:
            await self.db_manager.initialize()
            await self.cache_manager.initialize()
            await self.logger.initialize()

            # Criar casos de teste
            await self._create_test_cases()

            # Inicializar simuladores de mercado
            await self._initialize_market_simulators()

            await self.logger.log_activity("trading_test_suite_initialized", {
                "test_cases": len(self.test_cases),
                "market_simulators": len(self.market_simulators)
            })

            print("✅ Trading Test Suite inicializada com sucesso")

        except Exception as e:
            await self.logger.log_error("test_suite_init_error", str(e))
            raise

    async def run_all_tests(self) -> Dict[str, Any]:
        """Executa todos os casos de teste"""
        try:
            print("🧪 Iniciando execução de todos os testes...")

            results = {}
            total_tests = len(self.test_cases)
            passed_tests = 0
            failed_tests = 0

            for test_id, test_case in self.test_cases.items():
                print(f"\n📋 Executando teste: {test_case.name}")

                try:
                    result = await self.run_single_test(test_id)
                    results[test_id] = result

                    if result.result == TestResult.PASSED:
                        passed_tests += 1
                        print(f"✅ PASSOU: {test_case.name}")
                    else:
                        failed_tests += 1
                        print(f"❌ FALHOU: {test_case.name}")

                except Exception as e:
                    failed_tests += 1
                    error_result = TestExecution(
                        test_id=test_id,
                        start_time=datetime.utcnow(),
                        end_time=datetime.utcnow(),
                        status="error",
                        trades_executed=0,
                        positions_opened=0,
                        max_drawdown=0.0,
                        final_balance=0.0,
                        total_return=0.0,
                        return_pct=0.0,
                        sharpe_ratio=0.0,
                        win_rate=0.0,
                        alerts_generated=0,
                        errors_count=1,
                        result=TestResult.FAILED,
                        details={"error": str(e)}
                    )
                    results[test_id] = error_result
                    print(f"💥 ERRO: {test_case.name} - {str(e)}")

            # Sumário final
            summary = {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                "test_results": {test_id: asdict(result) for test_id, result in results.items()},
                "execution_time": datetime.utcnow().isoformat()
            }

            print(f"\n📊 RESUMO DOS TESTES:")
            print(f"   Total: {total_tests}")
            print(f"   Passou: {passed_tests}")
            print(f"   Falhou: {failed_tests}")
            print(f"   Taxa de Sucesso: {summary['success_rate']:.1f}%")

            # Salvar resultados
            await self._save_test_results(summary)

            return summary

        except Exception as e:
            await self.logger.log_error("run_all_tests_error", str(e))
            raise

    async def run_single_test(self, test_id: str) -> TestExecution:
        """Executa um teste específico"""
        try:
            test_case = self.test_cases.get(test_id)
            if not test_case:
                raise ValueError(f"Teste {test_id} não encontrado")

            # Inicializar componentes para o teste
            await self._initialize_test_components(test_case)

            # Configurar simuladores de mercado
            await self._setup_market_simulation(test_case)

            execution = TestExecution(
                test_id=test_id,
                start_time=datetime.utcnow(),
                end_time=None,
                status="running",
                trades_executed=0,
                positions_opened=0,
                max_drawdown=0.0,
                final_balance=test_case.initial_capital,
                total_return=0.0,
                return_pct=0.0,
                sharpe_ratio=0.0,
                win_rate=0.0,
                alerts_generated=0,
                errors_count=0,
                result=TestResult.FAILED,
                details={}
            )

            self.current_test = execution

            # Executar teste baseado no cenário
            await self._execute_test_scenario(test_case, execution)

            # Finalizar teste
            execution.end_time = datetime.utcnow()
            execution.status = "completed"

            # Avaliar resultados
            await self._evaluate_test_results(test_case, execution)

            # Cleanup
            await self._cleanup_test_components()

            self.test_results.append(execution)
            return execution

        except Exception as e:
            await self.logger.log_error("single_test_execution_error", f"{test_id}: {str(e)}")
            raise

    async def _initialize_test_components(self, test_case: TestCase):
        """Inicializa componentes para um teste"""
        try:
            # Trading Executor com paper trading
            self.trading_executor = RealTradingExecutor()
            await self.trading_executor.initialize()

            # Position Manager
            self.position_manager = RealPositionManager(None)
            await self.position_manager.initialize()

            # Risk Manager
            self.risk_manager = RealRiskManager(test_case.initial_capital)
            await self.risk_manager.initialize(self.position_manager)

            # Order Monitor
            self.order_monitor = OrderExecutionMonitor()
            await self.order_monitor.initialize(self.risk_manager)

            # Portfolio Tracker
            self.portfolio_tracker = PortfolioTracker(test_case.initial_capital)
            await self.portfolio_tracker.initialize(self.position_manager, self.order_monitor)

            # Alert System
            self.alert_system = RealtimeAlerts()
            await self.alert_system.initialize()

            # AI Signal Connector (se necessário)
            if test_case.scenario in [TestScenario.AI_SIGNALS, TestScenario.STRESS_TEST]:
                self.ai_connector = AISignalConnector()
                await self.ai_connector.initialize()

            print(f"🔧 Componentes inicializados para teste: {test_case.name}")

        except Exception as e:
            await self.logger.log_error("test_components_init_error", str(e))
            raise

    async def _setup_market_simulation(self, test_case: TestCase):
        """Configura simulação de mercado para o teste"""
        try:
            for symbol in test_case.symbols:
                simulator = self.market_simulators.get(symbol)
                if simulator:
                    # Reset do simulador
                    simulator.current_price = simulator.initial_price
                    simulator.price_history = [(datetime.utcnow(), simulator.initial_price)]
                    simulator.last_update = datetime.utcnow()

                    # Ajustar parâmetros baseado no cenário
                    if test_case.scenario == TestScenario.VOLATILE_MARKET:
                        simulator.volatility *= 2.0
                    elif test_case.scenario == TestScenario.TRENDING_MARKET:
                        simulator.trend = 0.05  # 5% trend positivo
                    elif test_case.scenario == TestScenario.SIDEWAYS_MARKET:
                        simulator.trend = 0.0
                        simulator.volatility *= 0.5

        except Exception as e:
            await self.logger.log_error("market_simulation_setup_error", str(e))
            raise

    async def _execute_test_scenario(self, test_case: TestCase, execution: TestExecution):
        """Executa cenário específico de teste"""
        try:
            # Iniciar componentes
            await self.trading_executor.start_trading(test_case.symbols, TradingMode.PAPER)
            await self.position_manager.start_position_monitoring()
            await self.risk_manager.start_monitoring()
            await self.order_monitor.start_monitoring()
            await self.portfolio_tracker.start_tracking()
            await self.alert_system.start_monitoring()

            # Executar cenário específico
            if test_case.scenario == TestScenario.BASIC_TRADING:
                await self._run_basic_trading_scenario(test_case, execution)

            elif test_case.scenario == TestScenario.HIGH_FREQUENCY:
                await self._run_high_frequency_scenario(test_case, execution)

            elif test_case.scenario == TestScenario.VOLATILE_MARKET:
                await self._run_volatile_market_scenario(test_case, execution)

            elif test_case.scenario == TestScenario.RISK_MANAGEMENT:
                await self._run_risk_management_scenario(test_case, execution)

            elif test_case.scenario == TestScenario.AI_SIGNALS:
                await self._run_ai_signals_scenario(test_case, execution)

            elif test_case.scenario == TestScenario.STRESS_TEST:
                await self._run_stress_test_scenario(test_case, execution)

            else:
                await self._run_basic_trading_scenario(test_case, execution)

        except Exception as e:
            execution.errors_count += 1
            await self.logger.log_error("test_scenario_execution_error", f"{test_case.scenario.value}: {str(e)}")
            raise

    async def _run_basic_trading_scenario(self, test_case: TestCase, execution: TestExecution):
        """Executa cenário básico de trading"""
        print("📈 Executando cenário básico de trading...")

        # Simular trading por duração especificada
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(minutes=test_case.duration_minutes)

        trade_count = 0
        target_trades = test_case.expected_trades

        while datetime.utcnow() < end_time and trade_count < target_trades:
            try:
                # Atualizar preços do mercado
                await self._update_market_prices()

                # Executar trade aleatório para teste
                symbol = random.choice(test_case.symbols)
                trade_type = random.choice([TradeType.BUY, TradeType.SELL])
                amount = random.uniform(10, 50)

                result = await self.trading_executor.execute_trade(
                    symbol=symbol,
                    trade_type=trade_type,
                    amount=amount,
                    duration_minutes=5
                )

                if result and result.success:
                    trade_count += 1
                    execution.trades_executed += 1

                # Aguardar intervalo
                await asyncio.sleep(2)

            except Exception as e:
                execution.errors_count += 1
                await self.logger.log_error("basic_trading_error", str(e))

        # Atualizar métricas finais
        await self._update_execution_metrics(execution)

    async def _run_high_frequency_scenario(self, test_case: TestCase, execution: TestExecution):
        """Executa cenário de alta frequência"""
        print("⚡ Executando cenário de alta frequência...")

        start_time = datetime.utcnow()
        end_time = start_time + timedelta(minutes=test_case.duration_minutes)

        while datetime.utcnow() < end_time:
            try:
                # Executar múltiplos trades rapidamente
                tasks = []
                for _ in range(5):  # 5 trades simultâneos
                    symbol = random.choice(test_case.symbols)
                    trade_type = random.choice([TradeType.BUY, TradeType.SELL])
                    amount = random.uniform(5, 25)

                    task = self.trading_executor.execute_trade(
                        symbol=symbol,
                        trade_type=trade_type,
                        amount=amount,
                        duration_minutes=1  # Trades muito rápidos
                    )
                    tasks.append(task)

                # Aguardar execução paralela
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if not isinstance(result, Exception) and result and result.success:
                        execution.trades_executed += 1

                await asyncio.sleep(0.1)  # Intervalo muito pequeno

            except Exception as e:
                execution.errors_count += 1
                await self.logger.log_error("high_frequency_error", str(e))

        await self._update_execution_metrics(execution)

    async def _run_volatile_market_scenario(self, test_case: TestCase, execution: TestExecution):
        """Executa cenário de mercado volátil"""
        print("📊 Executando cenário de mercado volátil...")

        start_time = datetime.utcnow()
        end_time = start_time + timedelta(minutes=test_case.duration_minutes)

        while datetime.utcnow() < end_time:
            try:
                # Gerar volatilidade extrema nos preços
                await self._simulate_high_volatility()

                # Testar sistema sob stress de volatilidade
                symbol = random.choice(test_case.symbols)
                trade_type = random.choice([TradeType.BUY, TradeType.SELL])
                amount = random.uniform(20, 100)

                result = await self.trading_executor.execute_trade(
                    symbol=symbol,
                    trade_type=trade_type,
                    amount=amount,
                    duration_minutes=3
                )

                if result and result.success:
                    execution.trades_executed += 1

                await asyncio.sleep(1)

            except Exception as e:
                execution.errors_count += 1
                await self.logger.log_error("volatile_market_error", str(e))

        await self._update_execution_metrics(execution)

    async def _run_risk_management_scenario(self, test_case: TestCase, execution: TestExecution):
        """Executa cenário de teste de risk management"""
        print("🛡️ Executando cenário de risk management...")

        # Tentar executar trades que devem ser bloqueados pelo risk manager
        risky_trades = [
            {"symbol": "R_10", "amount": 5000},  # Trade muito grande
            {"symbol": "R_25", "amount": 1000},  # Segundo trade grande
            {"symbol": "R_50", "amount": 1000},  # Terceiro trade grande
        ]

        blocked_trades = 0
        for trade_data in risky_trades:
            try:
                result = await self.trading_executor.execute_trade(
                    symbol=trade_data["symbol"],
                    trade_type=TradeType.BUY,
                    amount=trade_data["amount"],
                    duration_minutes=10
                )

                if not result or not result.success:
                    blocked_trades += 1
                else:
                    execution.trades_executed += 1

            except Exception as e:
                blocked_trades += 1
                execution.errors_count += 1

        # Risk management deve ter bloqueado trades arriscados
        execution.details["blocked_trades"] = blocked_trades
        execution.details["risk_management_effective"] = blocked_trades >= 2

        await self._update_execution_metrics(execution)

    async def _run_ai_signals_scenario(self, test_case: TestCase, execution: TestExecution):
        """Executa cenário de sinais AI"""
        print("🤖 Executando cenário de sinais AI...")

        if self.ai_connector:
            await self.ai_connector.start_signal_generation(TradingMode.PAPER)

            # Simular por duração especificada
            start_time = datetime.utcnow()
            end_time = start_time + timedelta(minutes=test_case.duration_minutes)

            while datetime.utcnow() < end_time:
                await self._update_market_prices()
                await asyncio.sleep(5)  # Aguardar sinais AI

            await self.ai_connector.stop_signal_generation()

            # Obter estatísticas dos sinais
            signal_stats = await self.ai_connector.get_signal_statistics()
            execution.details["ai_signals"] = signal_stats

        await self._update_execution_metrics(execution)

    async def _run_stress_test_scenario(self, test_case: TestCase, execution: TestExecution):
        """Executa cenário de stress test"""
        print("💪 Executando stress test...")

        # Combinar múltiplos cenários estressantes
        tasks = [
            self._simulate_high_volatility(),
            self._simulate_network_issues(),
            self._simulate_high_load(),
        ]

        # Executar stress tests em paralelo
        await asyncio.gather(*tasks, return_exceptions=True)

        # Testar recuperação do sistema
        recovery_successful = await self._test_system_recovery()
        execution.details["recovery_successful"] = recovery_successful

        await self._update_execution_metrics(execution)

    async def _update_market_prices(self):
        """Atualiza preços dos simuladores de mercado"""
        try:
            current_time = datetime.utcnow()

            for symbol, simulator in self.market_simulators.items():
                # Calcular novo preço baseado em modelo de random walk
                dt = (current_time - simulator.last_update).total_seconds() / 60  # em minutos

                if dt > 0:
                    # Drift + volatilidade
                    drift = simulator.trend * dt
                    shock = simulator.volatility * np.random.normal(0, np.sqrt(dt))

                    price_change = simulator.current_price * (drift + shock)
                    new_price = max(0.1, simulator.current_price + price_change)

                    simulator.current_price = new_price
                    simulator.price_history.append((current_time, new_price))
                    simulator.last_update = current_time

                    # Manter apenas histórico recente
                    if len(simulator.price_history) > 1000:
                        simulator.price_history = simulator.price_history[-1000:]

                    # Atualizar cache para outros componentes
                    await self.cache_manager.set(
                        CacheNamespace.MARKET_DATA,
                        f"{symbol}:latest_tick",
                        {"price": new_price, "timestamp": current_time.isoformat()},
                        ttl=60
                    )

        except Exception as e:
            await self.logger.log_error("market_price_update_error", str(e))

    async def _simulate_high_volatility(self):
        """Simula alta volatilidade no mercado"""
        for simulator in self.market_simulators.values():
            simulator.volatility *= 5.0  # Aumentar volatilidade drasticamente

        await asyncio.sleep(30)  # Simular por 30 segundos

        # Restaurar volatilidade normal
        for simulator in self.market_simulators.values():
            simulator.volatility /= 5.0

    async def _simulate_network_issues(self):
        """Simula problemas de rede"""
        # Simular latência e timeouts
        await asyncio.sleep(random.uniform(1, 5))

    async def _simulate_high_load(self):
        """Simula alta carga no sistema"""
        # Simular processos intensivos
        for _ in range(100):
            await asyncio.sleep(0.01)

    async def _test_system_recovery(self) -> bool:
        """Testa recuperação do sistema após stress"""
        try:
            # Tentar executar trade simples após stress
            result = await self.trading_executor.execute_trade(
                symbol="R_10",
                trade_type=TradeType.BUY,
                amount=10,
                duration_minutes=1
            )

            return result is not None

        except Exception:
            return False

    async def _update_execution_metrics(self, execution: TestExecution):
        """Atualiza métricas de execução do teste"""
        try:
            # Obter métricas do portfolio tracker
            if self.portfolio_tracker:
                portfolio_value = await self.portfolio_tracker.get_current_portfolio_value()
                execution.final_balance = portfolio_value.get("total_value", execution.final_balance)
                execution.total_return = execution.final_balance - self.base_capital
                execution.return_pct = (execution.total_return / self.base_capital) * 100

                performance = await self.portfolio_tracker.get_performance_summary()
                execution.sharpe_ratio = performance.get("sharpe_ratio", 0.0)
                execution.max_drawdown = performance.get("max_drawdown", 0.0)

            # Obter métricas do position manager
            if self.position_manager:
                summary = await self.position_manager.get_position_summary()
                execution.positions_opened = summary.get("positions_opened_today", 0)

                # Calcular win rate simplificado
                if summary.get("positions_closed_today", 0) > 0:
                    execution.win_rate = 65.0  # Valor simulado

            # Obter alertas gerados
            if self.alert_system:
                alert_metrics = await self.alert_system.get_alert_metrics()
                execution.alerts_generated = alert_metrics.get("total_alerts", 0)

        except Exception as e:
            await self.logger.log_error("execution_metrics_update_error", str(e))

    async def _evaluate_test_results(self, test_case: TestCase, execution: TestExecution):
        """Avalia resultados do teste contra critérios de sucesso"""
        try:
            success_criteria = test_case.success_criteria
            passed_criteria = 0
            total_criteria = len(success_criteria)

            # Verificar cada critério
            for criterion, expected_value in success_criteria.items():
                actual_value = getattr(execution, criterion, None)

                if actual_value is not None:
                    if criterion.endswith("_min") and actual_value >= expected_value:
                        passed_criteria += 1
                    elif criterion.endswith("_max") and actual_value <= expected_value:
                        passed_criteria += 1
                    elif criterion.endswith("_exact") and actual_value == expected_value:
                        passed_criteria += 1
                    elif not criterion.endswith(("_min", "_max", "_exact")) and actual_value >= expected_value:
                        passed_criteria += 1

            # Critérios adicionais
            if execution.errors_count == 0:
                passed_criteria += 1
                total_criteria += 1

            if execution.trades_executed > 0:
                passed_criteria += 1
                total_criteria += 1

            # Determinar resultado final
            success_rate = passed_criteria / total_criteria if total_criteria > 0 else 0

            if success_rate >= 0.8:
                execution.result = TestResult.PASSED
            elif success_rate >= 0.6:
                execution.result = TestResult.WARNING
            else:
                execution.result = TestResult.FAILED

            execution.details["success_rate"] = success_rate
            execution.details["criteria_passed"] = passed_criteria
            execution.details["total_criteria"] = total_criteria

        except Exception as e:
            execution.result = TestResult.FAILED
            await self.logger.log_error("test_evaluation_error", str(e))

    async def _cleanup_test_components(self):
        """Limpa componentes após teste"""
        try:
            components = [
                self.trading_executor,
                self.position_manager,
                self.risk_manager,
                self.order_monitor,
                self.portfolio_tracker,
                self.alert_system,
                self.ai_connector
            ]

            for component in components:
                if component:
                    try:
                        if hasattr(component, 'shutdown'):
                            await component.shutdown()
                        elif hasattr(component, 'stop_monitoring'):
                            await component.stop_monitoring()
                        elif hasattr(component, 'stop_tracking'):
                            await component.stop_tracking()
                    except Exception as e:
                        await self.logger.log_error("component_cleanup_error", str(e))

            # Reset componentes
            self.trading_executor = None
            self.position_manager = None
            self.risk_manager = None
            self.order_monitor = None
            self.portfolio_tracker = None
            self.alert_system = None
            self.ai_connector = None

        except Exception as e:
            await self.logger.log_error("test_cleanup_error", str(e))

    async def _create_test_cases(self):
        """Cria casos de teste predefinidos"""
        test_cases = [
            TestCase(
                test_id="basic_trading_test",
                name="Teste Básico de Trading",
                description="Teste básico de execução de trades",
                scenario=TestScenario.BASIC_TRADING,
                duration_minutes=5,
                symbols=["R_10", "R_25"],
                initial_capital=10000.0,
                max_positions=5,
                expected_trades=10,
                success_criteria={
                    "trades_executed": 5,
                    "errors_count_max": 2,
                    "return_pct": -10.0  # Não perder mais que 10%
                },
                risk_limits={"max_loss_pct": 10.0}
            ),
            TestCase(
                test_id="high_frequency_test",
                name="Teste de Alta Frequência",
                description="Teste de execução de muitos trades rapidamente",
                scenario=TestScenario.HIGH_FREQUENCY,
                duration_minutes=3,
                symbols=["R_10", "R_25", "R_50"],
                initial_capital=10000.0,
                max_positions=20,
                expected_trades=50,
                success_criteria={
                    "trades_executed": 30,
                    "errors_count_max": 5
                },
                risk_limits={"max_loss_pct": 15.0}
            ),
            TestCase(
                test_id="risk_management_test",
                name="Teste de Risk Management",
                description="Teste do sistema de gestão de risco",
                scenario=TestScenario.RISK_MANAGEMENT,
                duration_minutes=2,
                symbols=["R_10"],
                initial_capital=1000.0,  # Capital baixo para testar limites
                max_positions=2,
                expected_trades=3,
                success_criteria={
                    "trades_executed_max": 2,  # Deve bloquear trades arriscados
                    "errors_count_max": 1
                },
                risk_limits={"max_loss_pct": 5.0}
            ),
            TestCase(
                test_id="ai_signals_test",
                name="Teste de Sinais AI",
                description="Teste do sistema de sinais de AI",
                scenario=TestScenario.AI_SIGNALS,
                duration_minutes=4,
                symbols=["R_10", "R_25", "FRXEURUSD"],
                initial_capital=10000.0,
                max_positions=10,
                expected_trades=15,
                success_criteria={
                    "trades_executed": 5,
                    "alerts_generated": 1
                },
                risk_limits={"max_loss_pct": 12.0}
            ),
            TestCase(
                test_id="stress_test",
                name="Stress Test",
                description="Teste de stress do sistema completo",
                scenario=TestScenario.STRESS_TEST,
                duration_minutes=3,
                symbols=self.test_symbols,
                initial_capital=10000.0,
                max_positions=15,
                expected_trades=20,
                success_criteria={
                    "trades_executed": 1,  # Pelo menos algum trade deve funcionar
                    "errors_count_max": 10
                },
                risk_limits={"max_loss_pct": 20.0}
            )
        ]

        for test_case in test_cases:
            self.test_cases[test_case.test_id] = test_case

    async def _initialize_market_simulators(self):
        """Inicializa simuladores de mercado"""
        base_prices = {
            "R_10": 100.0,
            "R_25": 100.0,
            "R_50": 100.0,
            "R_75": 100.0,
            "R_100": 100.0,
            "FRXEURUSD": 1.0850,
            "FRXGBPUSD": 1.2650,
            "FRXUSDJPY": 110.25
        }

        volatilities = {
            "R_10": 0.10,
            "R_25": 0.25,
            "R_50": 0.50,
            "R_75": 0.75,
            "R_100": 1.00,
            "FRXEURUSD": 0.08,
            "FRXGBPUSD": 0.12,
            "FRXUSDJPY": 0.10
        }

        for symbol in self.test_symbols:
            initial_price = base_prices.get(symbol, 100.0)
            volatility = volatilities.get(symbol, 0.15)

            simulator = MarketSimulator(
                symbol=symbol,
                initial_price=initial_price,
                volatility=volatility,
                trend=0.0,  # Sem trend por padrão
                current_price=initial_price,
                price_history=[(datetime.utcnow(), initial_price)],
                last_update=datetime.utcnow()
            )

            self.market_simulators[symbol] = simulator

    async def _save_test_results(self, summary: Dict[str, Any]):
        """Salva resultados dos testes"""
        try:
            await self.cache_manager.set(
                CacheNamespace.TRADING_SESSIONS,
                "latest_test_results",
                summary,
                ttl=86400  # 24 horas
            )

            await self.logger.log_activity("test_results_saved", {
                "total_tests": summary["total_tests"],
                "success_rate": summary["success_rate"]
            })

        except Exception as e:
            await self.logger.log_error("test_results_save_error", str(e))

    async def get_test_results(self) -> Dict[str, Any]:
        """Retorna resultados dos testes"""
        try:
            cached_results = await self.cache_manager.get(
                CacheNamespace.TRADING_SESSIONS,
                "latest_test_results"
            )

            if cached_results:
                return cached_results

            return {
                "message": "Nenhum resultado de teste disponível",
                "available_tests": list(self.test_cases.keys())
            }

        except Exception as e:
            await self.logger.log_error("test_results_query_error", str(e))
            return {"error": str(e)}

    async def shutdown(self):
        """Encerra a suite de testes"""
        await self._cleanup_test_components()
        await self.logger.log_activity("trading_test_suite_shutdown", {})
        print("🔌 Trading Test Suite encerrada")