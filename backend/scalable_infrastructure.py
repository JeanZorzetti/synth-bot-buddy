"""
Scalable Infrastructure - Infraestrutura Escalável
Sistema de infraestrutura escalável com microserviços, load balancing e auto-scaling.
"""

import asyncio
import logging
import time
import psutil
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import deque
import aiohttp
import aioredis
from contextlib import asynccontextmanager
import weakref
import gc

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceType(Enum):
    DATA_INGESTION = "data_ingestion"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_INFERENCE = "model_inference"
    TRADING_EXECUTION = "trading_execution"
    RISK_MANAGEMENT = "risk_management"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    MONITORING = "monitoring"

class ScalingPolicy(Enum):
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    REQUEST_BASED = "request_based"
    LATENCY_BASED = "latency_based"
    QUEUE_BASED = "queue_based"

class ServiceStatus(Enum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    SCALING = "scaling"

@dataclass
class ServiceMetrics:
    """Métricas de um serviço"""
    service_id: str
    cpu_usage: float
    memory_usage: float
    request_count: int
    average_latency: float
    error_rate: float
    queue_size: int
    throughput: float
    timestamp: datetime

@dataclass
class ScalingRule:
    """Regra de auto-scaling"""
    service_type: ServiceType
    scaling_policy: ScalingPolicy
    scale_up_threshold: float
    scale_down_threshold: float
    min_instances: int
    max_instances: int
    cooldown_period: int  # seconds
    metric_window: int    # seconds

@dataclass
class ServiceInstance:
    """Instância de um serviço"""
    instance_id: str
    service_type: ServiceType
    status: ServiceStatus
    host: str
    port: int
    start_time: datetime
    cpu_limit: float
    memory_limit: float
    current_load: float

@dataclass
class LoadBalancerConfig:
    """Configuração do load balancer"""
    algorithm: str  # round_robin, weighted, least_connections
    health_check_interval: int
    health_check_timeout: int
    failure_threshold: int
    sticky_sessions: bool

class ResourceMonitor:
    """Monitor de recursos do sistema"""

    def __init__(self, monitoring_interval: int = 5):
        self.monitoring_interval = monitoring_interval
        self.metrics_history: deque = deque(maxlen=1000)
        self.alert_thresholds = {
            "cpu": 80.0,
            "memory": 85.0,
            "disk": 90.0,
            "network": 70.0
        }

        # Estado
        self.is_monitoring = False
        self.monitoring_task = None

    async def start_monitoring(self):
        """Inicia monitoramento de recursos"""
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Monitor de recursos iniciado")

    async def stop_monitoring(self):
        """Para monitoramento"""
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()

    async def _monitoring_loop(self):
        """Loop principal de monitoramento"""
        while self.is_monitoring:
            try:
                metrics = await self._collect_system_metrics()
                self.metrics_history.append(metrics)

                # Verificar alertas
                await self._check_alerts(metrics)

                await asyncio.sleep(self.monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erro no monitoramento: {e}")
                await asyncio.sleep(self.monitoring_interval)

    async def _collect_system_metrics(self) -> Dict[str, float]:
        """Coleta métricas do sistema"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memória
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disco
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent

            # Rede
            network = psutil.net_io_counters()
            network_utilization = (network.bytes_sent + network.bytes_recv) / (1024 * 1024)  # MB

            # Processos
            process_count = len(psutil.pids())

            return {
                "timestamp": time.time(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent,
                "network_mb": network_utilization,
                "process_count": process_count,
                "available_memory_gb": memory.available / (1024**3)
            }

        except Exception as e:
            logger.error(f"Erro ao coletar métricas: {e}")
            return {}

    async def _check_alerts(self, metrics: Dict[str, float]):
        """Verifica condições de alerta"""
        alerts = []

        if metrics.get("cpu_percent", 0) > self.alert_thresholds["cpu"]:
            alerts.append(f"CPU usage high: {metrics['cpu_percent']:.1f}%")

        if metrics.get("memory_percent", 0) > self.alert_thresholds["memory"]:
            alerts.append(f"Memory usage high: {metrics['memory_percent']:.1f}%")

        if metrics.get("disk_percent", 0) > self.alert_thresholds["disk"]:
            alerts.append(f"Disk usage high: {metrics['disk_percent']:.1f}%")

        if alerts:
            logger.warning(f"System alerts: {', '.join(alerts)}")

    def get_current_metrics(self) -> Optional[Dict[str, float]]:
        """Obtém métricas atuais"""
        return list(self.metrics_history)[-1] if self.metrics_history else None

    def get_metrics_history(self, duration_minutes: int = 60) -> List[Dict[str, float]]:
        """Obtém histórico de métricas"""
        cutoff_time = time.time() - (duration_minutes * 60)
        return [m for m in self.metrics_history if m.get("timestamp", 0) > cutoff_time]

class ServiceRegistry:
    """Registro de serviços para descoberta"""

    def __init__(self):
        self.services: Dict[str, ServiceInstance] = {}
        self.service_types: Dict[ServiceType, List[str]] = {
            service_type: [] for service_type in ServiceType
        }
        self.health_status: Dict[str, bool] = {}

    async def register_service(self, instance: ServiceInstance):
        """Registra uma instância de serviço"""
        self.services[instance.instance_id] = instance
        self.service_types[instance.service_type].append(instance.instance_id)
        self.health_status[instance.instance_id] = True

        logger.info(f"Serviço registrado: {instance.instance_id} ({instance.service_type.value})")

    async def unregister_service(self, instance_id: str):
        """Desregistra uma instância de serviço"""
        if instance_id in self.services:
            instance = self.services[instance_id]
            self.service_types[instance.service_type].remove(instance_id)
            del self.services[instance_id]
            self.health_status.pop(instance_id, None)

            logger.info(f"Serviço desregistrado: {instance_id}")

    async def get_healthy_instances(self, service_type: ServiceType) -> List[ServiceInstance]:
        """Obtém instâncias saudáveis de um tipo de serviço"""
        healthy_instances = []

        for instance_id in self.service_types[service_type]:
            if self.health_status.get(instance_id, False):
                instance = self.services.get(instance_id)
                if instance and instance.status == ServiceStatus.RUNNING:
                    healthy_instances.append(instance)

        return healthy_instances

    async def update_health_status(self, instance_id: str, is_healthy: bool):
        """Atualiza status de saúde de uma instância"""
        self.health_status[instance_id] = is_healthy

        if not is_healthy:
            logger.warning(f"Serviço unhealthy: {instance_id}")

    def get_service_count(self, service_type: ServiceType) -> int:
        """Obtém contagem de instâncias de um serviço"""
        return len([
            instance_id for instance_id in self.service_types[service_type]
            if self.health_status.get(instance_id, False)
        ])

class LoadBalancer:
    """Load balancer para distribuição de carga"""

    def __init__(self, config: LoadBalancerConfig, service_registry: ServiceRegistry):
        self.config = config
        self.service_registry = service_registry

        # Algoritmos de balanceamento
        self.algorithms = {
            "round_robin": self._round_robin,
            "weighted": self._weighted_round_robin,
            "least_connections": self._least_connections
        }

        # Estado para round robin
        self.round_robin_counters: Dict[ServiceType, int] = {
            service_type: 0 for service_type in ServiceType
        }

        # Conexões ativas por instância
        self.active_connections: Dict[str, int] = {}

    async def get_instance(self, service_type: ServiceType) -> Optional[ServiceInstance]:
        """Obtém instância baseada no algoritmo de load balancing"""
        healthy_instances = await self.service_registry.get_healthy_instances(service_type)

        if not healthy_instances:
            return None

        algorithm = self.algorithms.get(self.config.algorithm, self._round_robin)
        return await algorithm(service_type, healthy_instances)

    async def _round_robin(self, service_type: ServiceType,
                          instances: List[ServiceInstance]) -> ServiceInstance:
        """Algoritmo round robin"""
        counter = self.round_robin_counters[service_type]
        selected_instance = instances[counter % len(instances)]
        self.round_robin_counters[service_type] = (counter + 1) % len(instances)
        return selected_instance

    async def _weighted_round_robin(self, service_type: ServiceType,
                                   instances: List[ServiceInstance]) -> ServiceInstance:
        """Algoritmo round robin ponderado (baseado na carga atual)"""
        # Calcular pesos baseados na carga inversa
        weights = []
        for instance in instances:
            # Peso maior para instâncias com menor carga
            weight = max(1.0 - instance.current_load, 0.1)
            weights.append(weight)

        # Seleção ponderada
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Seleção aleatória ponderada
        import random
        return random.choices(instances, weights=normalized_weights)[0]

    async def _least_connections(self, service_type: ServiceType,
                                instances: List[ServiceInstance]) -> ServiceInstance:
        """Algoritmo least connections"""
        min_connections = float('inf')
        selected_instance = instances[0]

        for instance in instances:
            connections = self.active_connections.get(instance.instance_id, 0)
            if connections < min_connections:
                min_connections = connections
                selected_instance = instance

        return selected_instance

    async def register_connection(self, instance_id: str):
        """Registra nova conexão para uma instância"""
        self.active_connections[instance_id] = self.active_connections.get(instance_id, 0) + 1

    async def unregister_connection(self, instance_id: str):
        """Remove conexão de uma instância"""
        if instance_id in self.active_connections:
            self.active_connections[instance_id] = max(0, self.active_connections[instance_id] - 1)

class AutoScaler:
    """Auto-scaler para gerenciamento automático de capacidade"""

    def __init__(self, service_registry: ServiceRegistry, resource_monitor: ResourceMonitor):
        self.service_registry = service_registry
        self.resource_monitor = resource_monitor

        # Regras de scaling
        self.scaling_rules: Dict[ServiceType, ScalingRule] = {}

        # Estado
        self.last_scaling_actions: Dict[ServiceType, datetime] = {}
        self.is_scaling_enabled = True

        # Task de monitoramento
        self.scaling_task = None

    async def add_scaling_rule(self, rule: ScalingRule):
        """Adiciona regra de auto-scaling"""
        self.scaling_rules[rule.service_type] = rule
        logger.info(f"Regra de scaling adicionada para {rule.service_type.value}")

    async def start_auto_scaling(self):
        """Inicia auto-scaling"""
        self.scaling_task = asyncio.create_task(self._scaling_loop())
        logger.info("Auto-scaler iniciado")

    async def stop_auto_scaling(self):
        """Para auto-scaling"""
        if self.scaling_task:
            self.scaling_task.cancel()

    async def _scaling_loop(self):
        """Loop principal de auto-scaling"""
        while self.is_scaling_enabled:
            try:
                for service_type, rule in self.scaling_rules.items():
                    await self._evaluate_scaling(service_type, rule)

                await asyncio.sleep(30)  # Verificar a cada 30 segundos

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erro no auto-scaling: {e}")
                await asyncio.sleep(60)

    async def _evaluate_scaling(self, service_type: ServiceType, rule: ScalingRule):
        """Avalia necessidade de scaling para um serviço"""
        try:
            # Verificar cooldown
            last_action = self.last_scaling_actions.get(service_type)
            if last_action:
                time_since_action = (datetime.now() - last_action).total_seconds()
                if time_since_action < rule.cooldown_period:
                    return

            # Obter métricas atuais
            current_metrics = await self._get_service_metrics(service_type, rule)
            if not current_metrics:
                return

            current_instances = self.service_registry.get_service_count(service_type)

            # Decidir ação de scaling
            action = await self._decide_scaling_action(
                current_metrics, rule, current_instances
            )

            if action == "scale_up":
                await self._scale_up(service_type, rule, current_instances)
            elif action == "scale_down":
                await self._scale_down(service_type, rule, current_instances)

        except Exception as e:
            logger.error(f"Erro na avaliação de scaling para {service_type.value}: {e}")

    async def _get_service_metrics(self, service_type: ServiceType,
                                 rule: ScalingRule) -> Optional[float]:
        """Obtém métrica relevante para o serviço"""
        try:
            if rule.scaling_policy == ScalingPolicy.CPU_BASED:
                metrics = self.resource_monitor.get_current_metrics()
                return metrics.get("cpu_percent", 0) if metrics else None

            elif rule.scaling_policy == ScalingPolicy.MEMORY_BASED:
                metrics = self.resource_monitor.get_current_metrics()
                return metrics.get("memory_percent", 0) if metrics else None

            elif rule.scaling_policy == ScalingPolicy.REQUEST_BASED:
                # Seria obtido de métricas de aplicação
                return 50.0  # Simulado

            elif rule.scaling_policy == ScalingPolicy.LATENCY_BASED:
                # Seria obtido de métricas de aplicação
                return 100.0  # Simulado (ms)

            else:
                return None

        except Exception:
            return None

    async def _decide_scaling_action(self, metric_value: float,
                                   rule: ScalingRule,
                                   current_instances: int) -> Optional[str]:
        """Decide ação de scaling baseada nas métricas"""
        if metric_value > rule.scale_up_threshold and current_instances < rule.max_instances:
            return "scale_up"
        elif metric_value < rule.scale_down_threshold and current_instances > rule.min_instances:
            return "scale_down"
        else:
            return None

    async def _scale_up(self, service_type: ServiceType, rule: ScalingRule, current_instances: int):
        """Escala serviço para cima"""
        try:
            # Simular criação de nova instância
            new_instance = ServiceInstance(
                instance_id=f"{service_type.value}_{current_instances + 1}",
                service_type=service_type,
                status=ServiceStatus.STARTING,
                host="localhost",
                port=8000 + current_instances + 1,
                start_time=datetime.now(),
                cpu_limit=2.0,
                memory_limit=4.0,
                current_load=0.0
            )

            await self.service_registry.register_service(new_instance)
            self.last_scaling_actions[service_type] = datetime.now()

            logger.info(f"Scaled up {service_type.value}: {current_instances} -> {current_instances + 1}")

        except Exception as e:
            logger.error(f"Erro ao escalar {service_type.value}: {e}")

    async def _scale_down(self, service_type: ServiceType, rule: ScalingRule, current_instances: int):
        """Escala serviço para baixo"""
        try:
            # Obter instâncias do serviço
            instances = await self.service_registry.get_healthy_instances(service_type)

            if instances:
                # Remover instância com menor carga
                instance_to_remove = min(instances, key=lambda x: x.current_load)
                await self.service_registry.unregister_service(instance_to_remove.instance_id)
                self.last_scaling_actions[service_type] = datetime.now()

                logger.info(f"Scaled down {service_type.value}: {current_instances} -> {current_instances - 1}")

        except Exception as e:
            logger.error(f"Erro ao reduzir escala de {service_type.value}: {e}")

class CircuitBreaker:
    """Circuit breaker para proteção contra falhas"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout

        # Estado
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open

    async def call(self, func: Callable, *args, **kwargs):
        """Executa função com circuit breaker"""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
            else:
                raise Exception("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)

            # Reset em caso de sucesso
            if self.state == "half_open":
                self.state = "closed"
                self.failure_count = 0

            return result

        except Exception as e:
            self._record_failure()
            raise e

    def _record_failure(self):
        """Registra falha"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"

    def _should_attempt_reset(self) -> bool:
        """Verifica se deve tentar reset"""
        if self.last_failure_time is None:
            return False

        return (time.time() - self.last_failure_time) >= self.recovery_timeout

class ServiceMesh:
    """Service mesh para comunicação entre microserviços"""

    def __init__(self):
        self.service_registry = ServiceRegistry()
        self.load_balancer = LoadBalancer(
            LoadBalancerConfig(
                algorithm="round_robin",
                health_check_interval=30,
                health_check_timeout=5,
                failure_threshold=3,
                sticky_sessions=False
            ),
            self.service_registry
        )
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Cache distribuído
        self.redis_client = None

        # Session HTTP
        self.http_session = None

    async def initialize(self):
        """Inicializa service mesh"""
        try:
            # Inicializar Redis para cache distribuído
            self.redis_client = await aioredis.from_url("redis://localhost:6379")

            # Inicializar session HTTP
            self.http_session = aiohttp.ClientSession()

            logger.info("Service mesh inicializado")

        except Exception as e:
            logger.warning(f"Erro ao inicializar service mesh: {e}")

    async def call_service(self, service_type: ServiceType,
                          endpoint: str,
                          method: str = "GET",
                          data: Any = None) -> Any:
        """Chama um serviço através do service mesh"""
        try:
            # Obter instância através do load balancer
            instance = await self.load_balancer.get_instance(service_type)
            if not instance:
                raise Exception(f"No healthy instances for {service_type.value}")

            # Circuit breaker
            circuit_breaker_key = f"{service_type.value}_{instance.instance_id}"
            if circuit_breaker_key not in self.circuit_breakers:
                self.circuit_breakers[circuit_breaker_key] = CircuitBreaker()

            circuit_breaker = self.circuit_breakers[circuit_breaker_key]

            # Fazer chamada com circuit breaker
            return await circuit_breaker.call(
                self._make_http_call,
                instance, endpoint, method, data
            )

        except Exception as e:
            logger.error(f"Erro na chamada para {service_type.value}: {e}")
            raise

    async def _make_http_call(self, instance: ServiceInstance,
                             endpoint: str,
                             method: str,
                             data: Any) -> Any:
        """Faz chamada HTTP para uma instância"""
        url = f"http://{instance.host}:{instance.port}{endpoint}"

        # Registrar conexão
        await self.load_balancer.register_connection(instance.instance_id)

        try:
            if method.upper() == "GET":
                async with self.http_session.get(url) as response:
                    return await response.json()
            elif method.upper() == "POST":
                async with self.http_session.post(url, json=data) as response:
                    return await response.json()
            else:
                raise ValueError(f"Método HTTP não suportado: {method}")

        finally:
            # Desregistrar conexão
            await self.load_balancer.unregister_connection(instance.instance_id)

    async def shutdown(self):
        """Encerra service mesh"""
        if self.http_session:
            await self.http_session.close()

        if self.redis_client:
            await self.redis_client.close()

class ScalableInfrastructure:
    """Infraestrutura escalável principal"""

    def __init__(self):
        # Componentes
        self.resource_monitor = ResourceMonitor()
        self.service_mesh = ServiceMesh()
        self.auto_scaler = AutoScaler(
            self.service_mesh.service_registry,
            self.resource_monitor
        )

        # Configurações padrão de scaling
        self._setup_default_scaling_rules()

        # Estado
        self.is_running = False

    def _setup_default_scaling_rules(self):
        """Configura regras padrão de auto-scaling"""
        rules = [
            ScalingRule(
                service_type=ServiceType.DATA_INGESTION,
                scaling_policy=ScalingPolicy.CPU_BASED,
                scale_up_threshold=70.0,
                scale_down_threshold=30.0,
                min_instances=2,
                max_instances=10,
                cooldown_period=300,
                metric_window=60
            ),
            ScalingRule(
                service_type=ServiceType.MODEL_INFERENCE,
                scaling_policy=ScalingPolicy.LATENCY_BASED,
                scale_up_threshold=200.0,  # 200ms
                scale_down_threshold=50.0,   # 50ms
                min_instances=3,
                max_instances=15,
                cooldown_period=180,
                metric_window=30
            ),
            ScalingRule(
                service_type=ServiceType.TRADING_EXECUTION,
                scaling_policy=ScalingPolicy.REQUEST_BASED,
                scale_up_threshold=80.0,
                scale_down_threshold=20.0,
                min_instances=2,
                max_instances=8,
                cooldown_period=120,
                metric_window=45
            )
        ]

        for rule in rules:
            asyncio.create_task(self.auto_scaler.add_scaling_rule(rule))

    async def start(self):
        """Inicia infraestrutura"""
        try:
            # Inicializar componentes
            await self.service_mesh.initialize()
            await self.resource_monitor.start_monitoring()
            await self.auto_scaler.start_auto_scaling()

            self.is_running = True
            logger.info("Infraestrutura escalável iniciada")

        except Exception as e:
            logger.error(f"Erro ao iniciar infraestrutura: {e}")
            raise

    async def register_service_instance(self, service_type: ServiceType,
                                      host: str = "localhost",
                                      port: int = 8000) -> str:
        """Registra nova instância de serviço"""
        instance_id = f"{service_type.value}_{int(time.time())}"

        instance = ServiceInstance(
            instance_id=instance_id,
            service_type=service_type,
            status=ServiceStatus.RUNNING,
            host=host,
            port=port,
            start_time=datetime.now(),
            cpu_limit=2.0,
            memory_limit=4.0,
            current_load=0.0
        )

        await self.service_mesh.service_registry.register_service(instance)
        return instance_id

    async def get_infrastructure_status(self) -> Dict[str, Any]:
        """Obtém status da infraestrutura"""
        service_counts = {}
        for service_type in ServiceType:
            service_counts[service_type.value] = self.service_mesh.service_registry.get_service_count(service_type)

        return {
            "is_running": self.is_running,
            "resource_metrics": self.resource_monitor.get_current_metrics(),
            "service_counts": service_counts,
            "scaling_rules": len(self.auto_scaler.scaling_rules),
            "circuit_breakers": len(self.service_mesh.circuit_breakers),
            "active_connections": sum(self.service_mesh.load_balancer.active_connections.values())
        }

    async def scale_service_manually(self, service_type: ServiceType, target_instances: int):
        """Escala serviço manualmente"""
        current_count = self.service_mesh.service_registry.get_service_count(service_type)

        if target_instances > current_count:
            # Scale up
            for i in range(target_instances - current_count):
                await self.register_service_instance(service_type, port=8000 + current_count + i)
        elif target_instances < current_count:
            # Scale down
            instances = await self.service_mesh.service_registry.get_healthy_instances(service_type)
            instances_to_remove = instances[:current_count - target_instances]

            for instance in instances_to_remove:
                await self.service_mesh.service_registry.unregister_service(instance.instance_id)

        logger.info(f"Manual scaling: {service_type.value} {current_count} -> {target_instances}")

    async def shutdown(self):
        """Encerra infraestrutura"""
        self.is_running = False

        await self.auto_scaler.stop_auto_scaling()
        await self.resource_monitor.stop_monitoring()
        await self.service_mesh.shutdown()

        logger.info("Infraestrutura escalável encerrada")