"""
High-Frequency Optimizer - Otimizador de Alta Frequência
Sistema de otimização para trading de alta frequência com latência ultra-baixa.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import numba
from numba import jit, cuda, vectorize
import cupy as cp  # GPU acceleration
import cython
from memory_profiler import profile
import gc
import mmap
import os

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationTarget(Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    CPU = "cpu"
    NETWORK = "network"

class ComputeBackend(Enum):
    CPU = "cpu"
    GPU = "gpu"
    FPGA = "fpga"
    VECTORIZED = "vectorized"

@dataclass
class PerformanceMetrics:
    """Métricas de performance"""
    operation: str
    execution_time_ns: int
    memory_usage_mb: float
    cpu_usage: float
    throughput_ops_sec: float
    cache_hit_rate: float
    timestamp: datetime

@dataclass
class OptimizationResult:
    """Resultado de otimização"""
    target: OptimizationTarget
    improvement_factor: float
    before_metrics: PerformanceMetrics
    after_metrics: PerformanceMetrics
    optimization_method: str
    backend_used: ComputeBackend

class MemoryManager:
    """Gerenciador de memória otimizado"""

    def __init__(self, pool_size_mb: int = 1024):
        self.pool_size = pool_size_mb * 1024 * 1024  # Convert to bytes
        self.memory_pools: Dict[str, List[bytearray]] = {}
        self.allocated_blocks: Dict[int, Tuple[str, int]] = {}
        self.pool_lock = threading.Lock()

        # Estatísticas
        self.allocations = 0
        self.deallocations = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def allocate_buffer(self, size: int, buffer_type: str = "default") -> bytearray:
        """Aloca buffer otimizado"""
        with self.pool_lock:
            if buffer_type not in self.memory_pools:
                self.memory_pools[buffer_type] = []

            # Procurar buffer reutilizável
            for i, buffer in enumerate(self.memory_pools[buffer_type]):
                if len(buffer) >= size:
                    reused_buffer = self.memory_pools[buffer_type].pop(i)
                    self.cache_hits += 1
                    self.allocations += 1
                    return reused_buffer

            # Criar novo buffer
            new_buffer = bytearray(size)
            self.cache_misses += 1
            self.allocations += 1
            return new_buffer

    def deallocate_buffer(self, buffer: bytearray, buffer_type: str = "default"):
        """Desaloca buffer para o pool"""
        with self.pool_lock:
            if buffer_type not in self.memory_pools:
                self.memory_pools[buffer_type] = []

            # Retornar ao pool se não exceder limite
            if len(self.memory_pools[buffer_type]) < 100:  # Máximo 100 buffers por tipo
                self.memory_pools[buffer_type].append(buffer)

            self.deallocations += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Obtém estatísticas de uso"""
        total_pooled = sum(len(pool) for pool in self.memory_pools.values())
        cache_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0

        return {
            "allocations": self.allocations,
            "deallocations": self.deallocations,
            "cache_hit_rate": cache_rate,
            "pooled_buffers": total_pooled,
            "pool_types": len(self.memory_pools)
        }

class CacheManager:
    """Gerenciador de cache otimizado"""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.cache: Dict[str, Tuple[Any, float]] = {}  # key -> (value, timestamp)
        self.access_order: deque = deque()
        self.cache_lock = threading.Lock()

        # Estatísticas
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Obtém valor do cache"""
        with self.cache_lock:
            if key in self.cache:
                value, timestamp = self.cache[key]

                # Atualizar ordem de acesso
                try:
                    self.access_order.remove(key)
                except ValueError:
                    pass
                self.access_order.append(key)

                self.hits += 1
                return value
            else:
                self.misses += 1
                return None

    def set(self, key: str, value: Any):
        """Define valor no cache"""
        with self.cache_lock:
            # Remover itens antigos se necessário
            while len(self.cache) >= self.max_size:
                oldest_key = self.access_order.popleft()
                self.cache.pop(oldest_key, None)

            self.cache[key] = (value, time.time())

            # Atualizar ordem de acesso
            try:
                self.access_order.remove(key)
            except ValueError:
                pass
            self.access_order.append(key)

    def clear_expired(self, max_age_seconds: int = 300):
        """Remove itens expirados"""
        current_time = time.time()
        expired_keys = []

        for key, (value, timestamp) in self.cache.items():
            if current_time - timestamp > max_age_seconds:
                expired_keys.append(key)

        for key in expired_keys:
            self.cache.pop(key, None)
            try:
                self.access_order.remove(key)
            except ValueError:
                pass

    def get_hit_rate(self) -> float:
        """Obtém taxa de acerto do cache"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

@jit(nopython=True, cache=True)
def calculate_technical_indicators_numba(prices: np.ndarray, volumes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calcula indicadores técnicos otimizado com Numba"""
    n = len(prices)

    # SMA
    sma = np.zeros(n)
    window = 20
    for i in range(window-1, n):
        sma[i] = np.mean(prices[i-window+1:i+1])

    # RSI
    rsi = np.zeros(n)
    if n > 14:
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.mean(gains[:14])
        avg_loss = np.mean(losses[:14])

        for i in range(14, n-1):
            avg_gain = (avg_gain * 13 + gains[i]) / 14
            avg_loss = (avg_loss * 13 + losses[i]) / 14

            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi[i+1] = 100 - (100 / (1 + rs))

    # VWAP
    vwap = np.zeros(n)
    for i in range(1, n):
        cum_volume = np.sum(volumes[:i+1])
        cum_price_volume = np.sum(prices[:i+1] * volumes[:i+1])
        if cum_volume > 0:
            vwap[i] = cum_price_volume / cum_volume

    return sma, rsi, vwap

class GPUAccelerator:
    """Acelerador GPU para cálculos"""

    def __init__(self):
        self.gpu_available = self._check_gpu_availability()
        self.memory_pool = None

        if self.gpu_available:
            self._initialize_gpu()

    def _check_gpu_availability(self) -> bool:
        """Verifica disponibilidade de GPU"""
        try:
            import cupy
            device_count = cupy.cuda.runtime.getDeviceCount()
            return device_count > 0
        except:
            return False

    def _initialize_gpu(self):
        """Inicializa GPU"""
        try:
            import cupy
            self.memory_pool = cupy.get_default_memory_pool()
            logger.info("GPU inicializada para cálculos")
        except Exception as e:
            logger.warning(f"Erro ao inicializar GPU: {e}")
            self.gpu_available = False

    def calculate_features_gpu(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, np.ndarray]:
        """Calcula features usando GPU"""
        if not self.gpu_available:
            return self._fallback_cpu_calculation(prices, volumes)

        try:
            import cupy as cp

            # Transferir dados para GPU
            gpu_prices = cp.asarray(prices)
            gpu_volumes = cp.asarray(volumes)

            # Cálculos na GPU
            results = {}

            # SMA
            window = 20
            gpu_sma = cp.convolve(gpu_prices, cp.ones(window)/window, mode='same')
            results['sma'] = cp.asnumpy(gpu_sma)

            # Volatilidade
            returns = cp.diff(gpu_prices) / gpu_prices[:-1]
            gpu_volatility = cp.std(returns) * cp.sqrt(252)
            results['volatility'] = float(cp.asnumpy(gpu_volatility))

            # VWAP
            cumsum_pv = cp.cumsum(gpu_prices * gpu_volumes)
            cumsum_v = cp.cumsum(gpu_volumes)
            gpu_vwap = cumsum_pv / cp.where(cumsum_v > 0, cumsum_v, 1)
            results['vwap'] = cp.asnumpy(gpu_vwap)

            # Correlação entre preço e volume
            if len(gpu_prices) > 1 and len(gpu_volumes) > 1:
                correlation_matrix = cp.corrcoef(gpu_prices, gpu_volumes)
                results['price_volume_correlation'] = float(cp.asnumpy(correlation_matrix[0, 1]))

            return results

        except Exception as e:
            logger.error(f"Erro no cálculo GPU: {e}")
            return self._fallback_cpu_calculation(prices, volumes)

    def _fallback_cpu_calculation(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, np.ndarray]:
        """Cálculo fallback usando CPU"""
        sma, rsi, vwap = calculate_technical_indicators_numba(prices, volumes)

        return {
            'sma': sma,
            'rsi': rsi,
            'vwap': vwap,
            'volatility': np.std(np.diff(prices) / prices[:-1]) * np.sqrt(252) if len(prices) > 1 else 0.0,
            'price_volume_correlation': np.corrcoef(prices, volumes)[0, 1] if len(prices) > 1 else 0.0
        }

    def cleanup_gpu_memory(self):
        """Limpa memória GPU"""
        if self.gpu_available and self.memory_pool:
            self.memory_pool.free_all_blocks()

class NetworkOptimizer:
    """Otimizador de rede para baixa latência"""

    def __init__(self):
        self.connection_pools: Dict[str, List[Any]] = {}
        self.persistent_connections: Dict[str, Any] = {}

        # Configurações de rede otimizadas
        self.socket_options = {
            'TCP_NODELAY': True,    # Disable Nagle's algorithm
            'SO_REUSEADDR': True,   # Reuse address
            'SO_KEEPALIVE': True,   # Keep connections alive
        }

    async def send_optimized(self, endpoint: str, data: bytes) -> bytes:
        """Envia dados com otimizações de rede"""
        try:
            # Usar conexão persistente se disponível
            if endpoint in self.persistent_connections:
                connection = self.persistent_connections[endpoint]
                return await self._send_via_connection(connection, data)

            # Criar nova conexão otimizada
            connection = await self._create_optimized_connection(endpoint)
            self.persistent_connections[endpoint] = connection

            return await self._send_via_connection(connection, data)

        except Exception as e:
            logger.error(f"Erro no envio otimizado: {e}")
            raise

    async def _create_optimized_connection(self, endpoint: str) -> Any:
        """Cria conexão otimizada"""
        import socket
        import asyncio

        # Parse endpoint
        host, port = endpoint.split(':')
        port = int(port)

        # Criar socket otimizado
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Aplicar opções de otimização
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

        # Configurar buffer sizes
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)

        # Conectar
        sock.setblocking(False)
        await asyncio.get_event_loop().sock_connect(sock, (host, port))

        return sock

    async def _send_via_connection(self, connection: Any, data: bytes) -> bytes:
        """Envia dados via conexão existente"""
        loop = asyncio.get_event_loop()

        # Enviar dados
        await loop.sock_sendall(connection, data)

        # Receber resposta
        response = await loop.sock_recv(connection, 4096)

        return response

    def close_connections(self):
        """Fecha todas as conexões"""
        for connection in self.persistent_connections.values():
            try:
                connection.close()
            except:
                pass
        self.persistent_connections.clear()

class DataStructureOptimizer:
    """Otimizador de estruturas de dados"""

    def __init__(self):
        # Pools de objetos reutilizáveis
        self.array_pools: Dict[Tuple[int, type], List[np.ndarray]] = {}
        self.list_pools: Dict[int, List[List]] = {}

    def get_optimized_array(self, size: int, dtype: type = np.float64) -> np.ndarray:
        """Obtém array otimizado do pool"""
        key = (size, dtype)

        if key in self.array_pools and self.array_pools[key]:
            array = self.array_pools[key].pop()
            array.fill(0)  # Reset valores
            return array
        else:
            return np.zeros(size, dtype=dtype)

    def return_array(self, array: np.ndarray):
        """Retorna array ao pool"""
        key = (len(array), array.dtype.type)

        if key not in self.array_pools:
            self.array_pools[key] = []

        if len(self.array_pools[key]) < 50:  # Máximo 50 arrays por tipo
            self.array_pools[key].append(array)

    def get_optimized_list(self, initial_size: int = 100) -> List:
        """Obtém lista otimizada do pool"""
        if initial_size in self.list_pools and self.list_pools[initial_size]:
            list_obj = self.list_pools[initial_size].pop()
            list_obj.clear()
            return list_obj
        else:
            return [None] * initial_size

    def return_list(self, list_obj: List):
        """Retorna lista ao pool"""
        size = len(list_obj)

        if size not in self.list_pools:
            self.list_pools[size] = []

        if len(self.list_pools[size]) < 20:  # Máximo 20 listas por tamanho
            self.list_pools[size].append(list_obj)

class HighFrequencyOptimizer:
    """Otimizador principal para alta frequência"""

    def __init__(self):
        # Componentes
        self.memory_manager = MemoryManager()
        self.cache_manager = CacheManager()
        self.gpu_accelerator = GPUAccelerator()
        self.network_optimizer = NetworkOptimizer()
        self.data_optimizer = DataStructureOptimizer()

        # Métricas
        self.performance_history: deque = deque(maxlen=1000)

        # Threading
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        # Estado
        self.optimization_enabled = True

        logger.info("HighFrequencyOptimizer inicializado")

    async def optimize_calculation(self, operation: str,
                                 data: np.ndarray,
                                 backend: ComputeBackend = ComputeBackend.CPU) -> Tuple[Any, PerformanceMetrics]:
        """Otimiza cálculo com backend especificado"""
        start_time = time.perf_counter_ns()
        memory_before = self._get_memory_usage()

        try:
            # Verificar cache primeiro
            cache_key = f"{operation}_{hash(data.tobytes())}"
            cached_result = self.cache_manager.get(cache_key)

            if cached_result is not None:
                end_time = time.perf_counter_ns()

                metrics = PerformanceMetrics(
                    operation=operation,
                    execution_time_ns=end_time - start_time,
                    memory_usage_mb=0.0,  # Cache hit - no memory allocation
                    cpu_usage=0.0,
                    throughput_ops_sec=1e9 / (end_time - start_time),
                    cache_hit_rate=1.0,
                    timestamp=datetime.now()
                )

                return cached_result, metrics

            # Executar cálculo baseado no backend
            if backend == ComputeBackend.GPU and self.gpu_accelerator.gpu_available:
                result = await self._execute_gpu_calculation(operation, data)
            elif backend == ComputeBackend.VECTORIZED:
                result = await self._execute_vectorized_calculation(operation, data)
            else:
                result = await self._execute_cpu_calculation(operation, data)

            # Cache resultado
            self.cache_manager.set(cache_key, result)

            # Calcular métricas
            end_time = time.perf_counter_ns()
            memory_after = self._get_memory_usage()

            metrics = PerformanceMetrics(
                operation=operation,
                execution_time_ns=end_time - start_time,
                memory_usage_mb=memory_after - memory_before,
                cpu_usage=self._get_cpu_usage(),
                throughput_ops_sec=len(data) / ((end_time - start_time) / 1e9),
                cache_hit_rate=self.cache_manager.get_hit_rate(),
                timestamp=datetime.now()
            )

            self.performance_history.append(metrics)
            return result, metrics

        except Exception as e:
            logger.error(f"Erro na otimização de cálculo: {e}")
            raise

    async def _execute_gpu_calculation(self, operation: str, data: np.ndarray) -> Any:
        """Executa cálculo na GPU"""
        if operation == "technical_indicators":
            return self.gpu_accelerator.calculate_features_gpu(data, np.ones_like(data))
        else:
            # Fallback para CPU
            return await self._execute_cpu_calculation(operation, data)

    async def _execute_vectorized_calculation(self, operation: str, data: np.ndarray) -> Any:
        """Executa cálculo vetorizado"""
        if operation == "technical_indicators":
            # Usar numba JIT
            volumes = np.ones_like(data)
            return calculate_technical_indicators_numba(data, volumes)
        else:
            return await self._execute_cpu_calculation(operation, data)

    async def _execute_cpu_calculation(self, operation: str, data: np.ndarray) -> Any:
        """Executa cálculo na CPU"""
        loop = asyncio.get_event_loop()

        if operation == "technical_indicators":
            return await loop.run_in_executor(
                self.thread_pool,
                self._cpu_technical_indicators,
                data
            )
        else:
            # Operação genérica
            return np.mean(data)

    def _cpu_technical_indicators(self, data: np.ndarray) -> Dict[str, Any]:
        """Calcula indicadores técnicos na CPU"""
        volumes = np.ones_like(data)
        sma, rsi, vwap = calculate_technical_indicators_numba(data, volumes)

        return {
            'sma': sma,
            'rsi': rsi,
            'vwap': vwap
        }

    async def optimize_data_pipeline(self, data_stream: List[np.ndarray]) -> List[np.ndarray]:
        """Otimiza pipeline de dados"""
        optimized_results = []

        # Usar memory mapping para datasets grandes
        if len(data_stream) > 100:
            return await self._optimize_with_memory_mapping(data_stream)

        # Pipeline normal otimizado
        for i, data in enumerate(data_stream):
            # Usar arrays do pool
            optimized_array = self.data_optimizer.get_optimized_array(len(data))
            optimized_array[:] = data

            # Processar
            result, _ = await self.optimize_calculation("technical_indicators", optimized_array)
            optimized_results.append(result)

            # Retornar ao pool
            self.data_optimizer.return_array(optimized_array)

            # Limpeza periódica
            if i % 100 == 0:
                self._periodic_cleanup()

        return optimized_results

    async def _optimize_with_memory_mapping(self, data_stream: List[np.ndarray]) -> List[np.ndarray]:
        """Otimiza usando memory mapping para datasets grandes"""
        try:
            # Criar arquivo temporário
            temp_file = f"/tmp/hft_data_{int(time.time())}.dat"

            # Escrever dados
            with open(temp_file, 'wb') as f:
                for data in data_stream:
                    f.write(data.tobytes())

            # Memory map
            with open(temp_file, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    # Processar dados mapeados
                    results = []
                    data_size = len(data_stream[0]) * data_stream[0].itemsize

                    for i in range(len(data_stream)):
                        start_pos = i * data_size
                        end_pos = start_pos + data_size

                        # Ler chunk
                        chunk_data = np.frombuffer(
                            mm[start_pos:end_pos],
                            dtype=data_stream[0].dtype
                        )

                        # Processar
                        result, _ = await self.optimize_calculation("technical_indicators", chunk_data)
                        results.append(result)

            # Limpar arquivo temporário
            os.unlink(temp_file)

            return results

        except Exception as e:
            logger.error(f"Erro no memory mapping: {e}")
            # Fallback para método normal
            return await self.optimize_data_pipeline(data_stream)

    def _periodic_cleanup(self):
        """Limpeza periódica de recursos"""
        # Garbage collection
        gc.collect()

        # Limpar cache expirado
        self.cache_manager.clear_expired()

        # Limpar memória GPU se disponível
        self.gpu_accelerator.cleanup_gpu_memory()

    def _get_memory_usage(self) -> float:
        """Obtém uso atual de memória em MB"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def _get_cpu_usage(self) -> float:
        """Obtém uso atual de CPU"""
        import psutil
        return psutil.cpu_percent()

    async def benchmark_backends(self, test_data: np.ndarray) -> Dict[ComputeBackend, PerformanceMetrics]:
        """Benchmarks diferentes backends"""
        results = {}

        backends_to_test = [ComputeBackend.CPU, ComputeBackend.VECTORIZED]

        if self.gpu_accelerator.gpu_available:
            backends_to_test.append(ComputeBackend.GPU)

        for backend in backends_to_test:
            try:
                _, metrics = await self.optimize_calculation("technical_indicators", test_data, backend)
                results[backend] = metrics
                logger.info(f"Backend {backend.value}: {metrics.execution_time_ns/1e6:.2f}ms")
            except Exception as e:
                logger.error(f"Erro no benchmark {backend.value}: {e}")

        return results

    async def auto_select_optimal_backend(self, test_data: np.ndarray) -> ComputeBackend:
        """Seleciona automaticamente o backend ótimo"""
        benchmark_results = await self.benchmark_backends(test_data)

        if not benchmark_results:
            return ComputeBackend.CPU

        # Selecionar backend com menor latência
        optimal_backend = min(
            benchmark_results.keys(),
            key=lambda backend: benchmark_results[backend].execution_time_ns
        )

        logger.info(f"Backend ótimo selecionado: {optimal_backend.value}")
        return optimal_backend

    async def get_optimization_report(self) -> Dict[str, Any]:
        """Obtém relatório completo de otimização"""
        recent_metrics = list(self.performance_history)[-100:] if self.performance_history else []

        if recent_metrics:
            avg_latency = np.mean([m.execution_time_ns for m in recent_metrics]) / 1e6  # ms
            avg_throughput = np.mean([m.throughput_ops_sec for m in recent_metrics])
            avg_memory = np.mean([m.memory_usage_mb for m in recent_metrics])
        else:
            avg_latency = avg_throughput = avg_memory = 0.0

        return {
            "performance_summary": {
                "average_latency_ms": avg_latency,
                "average_throughput_ops_sec": avg_throughput,
                "average_memory_usage_mb": avg_memory,
                "cache_hit_rate": self.cache_manager.get_hit_rate()
            },
            "memory_manager": self.memory_manager.get_statistics(),
            "gpu_available": self.gpu_accelerator.gpu_available,
            "total_operations": len(self.performance_history),
            "optimization_enabled": self.optimization_enabled
        }

    async def shutdown(self):
        """Encerra otimizador"""
        self.optimization_enabled = False

        # Fechar conexões de rede
        self.network_optimizer.close_connections()

        # Limpar GPU
        self.gpu_accelerator.cleanup_gpu_memory()

        # Encerrar thread pool
        self.thread_pool.shutdown(wait=True)

        logger.info("HighFrequencyOptimizer encerrado")