"""
🚀 PERFORMANCE OPTIMIZER
System optimization for production deployment
"""

import asyncio
import time
import psutil
import gc
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import cProfile
import pstats
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """📊 Métricas de Performance"""
    cpu_usage: float
    memory_usage_mb: float
    memory_usage_percent: float
    processing_time_ms: float
    throughput_per_second: float
    latency_p95_ms: float
    error_rate: float
    active_connections: int


@dataclass
class OptimizationConfig:
    """⚙️ Configuração de Otimização"""
    max_cpu_usage: float = 80.0
    max_memory_usage_mb: float = 512.0
    target_latency_ms: float = 100.0
    target_throughput: float = 200.0
    gc_threshold: int = 1000
    enable_profiling: bool = False
    enable_caching: bool = True
    cache_size: int = 10000


class MemoryOptimizer:
    """💾 Otimizador de Memória"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.memory_usage_history = []
        self.gc_counter = 0

    def monitor_memory(self) -> Dict:
        """Monitorar uso de memória"""
        process = psutil.Process()
        memory_info = process.memory_info()

        memory_data = {
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024),
            'percent': process.memory_percent(),
            'timestamp': datetime.now().isoformat()
        }

        self.memory_usage_history.append(memory_data)

        # Manter apenas últimas 100 medições
        if len(self.memory_usage_history) > 100:
            self.memory_usage_history = self.memory_usage_history[-100:]

        return memory_data

    def optimize_memory(self) -> Dict:
        """Otimizar uso de memória"""
        initial_memory = self.monitor_memory()

        # 1. Forçar garbage collection
        gc.collect()

        # 2. Limpar caches se necessário
        if initial_memory['rss_mb'] > self.config.max_memory_usage_mb:
            self._clear_internal_caches()

        # 3. Ajustar thresholds do GC
        gc.set_threshold(self.config.gc_threshold, 10, 10)

        final_memory = self.monitor_memory()

        optimization_result = {
            'memory_freed_mb': initial_memory['rss_mb'] - final_memory['rss_mb'],
            'initial_memory_mb': initial_memory['rss_mb'],
            'final_memory_mb': final_memory['rss_mb'],
            'optimization_time': datetime.now().isoformat()
        }

        logger.info(f"Memory optimized: {optimization_result['memory_freed_mb']:.2f}MB freed")

        return optimization_result

    def _clear_internal_caches(self):
        """Limpar caches internos"""
        # Implementar limpeza de caches específicos do sistema
        pass


class LatencyOptimizer:
    """⚡ Otimizador de Latência"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.latency_measurements = []
        self.optimization_cache = {}

    def measure_latency(self, func: Callable) -> Callable:
        """Decorator para medir latência de funções"""
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            self.latency_measurements.append({
                'function': func.__name__,
                'latency_ms': latency_ms,
                'timestamp': datetime.now().isoformat()
            })

            # Manter apenas últimas 1000 medições
            if len(self.latency_measurements) > 1000:
                self.latency_measurements = self.latency_measurements[-1000:]

            return result
        return wrapper

    async def measure_async_latency(self, coro_func: Callable) -> Callable:
        """Decorator para medir latência de funções async"""
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = await coro_func(*args, **kwargs)
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            self.latency_measurements.append({
                'function': coro_func.__name__,
                'latency_ms': latency_ms,
                'timestamp': datetime.now().isoformat()
            })

            return result
        return wrapper

    def get_latency_stats(self) -> Dict:
        """Obter estatísticas de latência"""
        if not self.latency_measurements:
            return {}

        recent_measurements = [m['latency_ms'] for m in self.latency_measurements[-100:]]

        return {
            'avg_latency_ms': np.mean(recent_measurements),
            'p50_latency_ms': np.percentile(recent_measurements, 50),
            'p95_latency_ms': np.percentile(recent_measurements, 95),
            'p99_latency_ms': np.percentile(recent_measurements, 99),
            'max_latency_ms': np.max(recent_measurements),
            'min_latency_ms': np.min(recent_measurements),
            'total_measurements': len(self.latency_measurements)
        }


class ThroughputOptimizer:
    """📈 Otimizador de Throughput"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.throughput_history = []
        self.processing_queue = asyncio.Queue()
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

    async def optimize_async_processing(self, tasks: List[Callable]) -> List:
        """Otimizar processamento assíncrono"""
        # 1. Agrupar tarefas em batches
        batch_size = min(50, len(tasks))
        batches = [tasks[i:i+batch_size] for i in range(0, len(tasks), batch_size)]

        results = []
        start_time = time.perf_counter()

        # 2. Processar batches concorrentemente
        for batch in batches:
            batch_results = await asyncio.gather(*[task() for task in batch], return_exceptions=True)
            results.extend(batch_results)

        end_time = time.perf_counter()

        # 3. Calcular throughput
        processing_time = end_time - start_time
        throughput = len(tasks) / processing_time if processing_time > 0 else 0

        self.throughput_history.append({
            'throughput': throughput,
            'processing_time': processing_time,
            'task_count': len(tasks),
            'timestamp': datetime.now().isoformat()
        })

        logger.info(f"Processed {len(tasks)} tasks in {processing_time:.3f}s (throughput: {throughput:.1f}/s)")

        return results

    def optimize_cpu_bound_tasks(self, tasks: List[Callable]) -> List:
        """Otimizar tarefas CPU-intensivas"""
        start_time = time.perf_counter()

        # Usar thread pool para tarefas CPU-intensivas
        futures = [self.thread_pool.submit(task) for task in tasks]
        results = [future.result() for future in futures]

        end_time = time.perf_counter()
        processing_time = end_time - start_time
        throughput = len(tasks) / processing_time if processing_time > 0 else 0

        logger.info(f"CPU-bound tasks completed: {throughput:.1f} tasks/s")

        return results


class CacheOptimizer:
    """🗄️ Otimizador de Cache"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0, 'evictions': 0}
        self.cache_access_times = {}

    def get_cached(self, key: str) -> Optional[any]:
        """Obter item do cache"""
        if key in self.cache:
            self.cache_stats['hits'] += 1
            self.cache_access_times[key] = time.time()
            return self.cache[key]
        else:
            self.cache_stats['misses'] += 1
            return None

    def set_cached(self, key: str, value: any) -> None:
        """Armazenar item no cache"""
        # Verificar limite de tamanho
        if len(self.cache) >= self.config.cache_size:
            self._evict_oldest()

        self.cache[key] = value
        self.cache_access_times[key] = time.time()

    def _evict_oldest(self) -> None:
        """Remover item mais antigo do cache"""
        if not self.cache_access_times:
            return

        oldest_key = min(self.cache_access_times, key=self.cache_access_times.get)
        del self.cache[oldest_key]
        del self.cache_access_times[oldest_key]
        self.cache_stats['evictions'] += 1

    def get_cache_stats(self) -> Dict:
        """Obter estatísticas do cache"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0

        return {
            'cache_size': len(self.cache),
            'max_cache_size': self.config.cache_size,
            'hit_rate': hit_rate,
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'evictions': self.cache_stats['evictions']
        }


class DatabaseOptimizer:
    """🗃️ Otimizador de Database"""

    def __init__(self):
        self.connection_pool_size = 10
        self.query_cache = {}
        self.slow_query_threshold_ms = 100

    async def optimize_queries(self, queries: List[Dict]) -> List[Dict]:
        """Otimizar consultas de database"""
        optimized_queries = []

        for query in queries:
            # 1. Verificar cache de consultas
            cache_key = self._generate_query_cache_key(query)
            cached_result = self.query_cache.get(cache_key)

            if cached_result:
                optimized_queries.append(cached_result)
                continue

            # 2. Otimizar estrutura da query
            optimized_query = self._optimize_query_structure(query)

            # 3. Execute real query with database manager
            start_time = time.perf_counter()
            result = await self._execute_real_query(optimized_query)
            execution_time_ms = (time.perf_counter() - start_time) * 1000

            # 4. Cache resultado se consulta for lenta
            if execution_time_ms > self.slow_query_threshold_ms:
                self.query_cache[cache_key] = result

            optimized_queries.append(result)

        return optimized_queries

    def _generate_query_cache_key(self, query: Dict) -> str:
        """Gerar chave de cache para consulta"""
        return f"{query.get('table', '')}-{query.get('conditions', '')}"

    def _optimize_query_structure(self, query: Dict) -> Dict:
        """Otimizar estrutura da consulta"""
        # Implementar otimizações específicas
        optimized = query.copy()

        # Adicionar índices sugeridos
        if 'timestamp' in query.get('conditions', ''):
            optimized['suggested_index'] = 'idx_timestamp'

        return optimized

    async def _execute_real_query(self, query: Dict) -> Dict:
        """Execute real database query with performance monitoring"""
        try:
            # Import database manager
            from database_config import get_db_manager

            db_manager = await get_db_manager()

            # Extract query parameters
            table = query.get('table', 'market_data')
            conditions = query.get('conditions', '')
            limit = query.get('limit', 100)
            fields = query.get('fields', '*')

            # Construct and execute real SQL query
            if table == 'market_data':
                if 'timestamp' in conditions:
                    # Time-based query
                    sql_query = f"""
                        SELECT {fields} FROM market_data
                        WHERE {conditions}
                        ORDER BY timestamp DESC
                        LIMIT {limit}
                    """
                else:
                    # General market data query
                    sql_query = f"""
                        SELECT {fields} FROM market_data
                        ORDER BY timestamp DESC
                        LIMIT {limit}
                    """

                async with db_manager.postgres_pool.acquire() as conn:
                    start_time = time.perf_counter()
                    rows = await conn.fetch(sql_query)
                    execution_time_ms = (time.perf_counter() - start_time) * 1000

                    return {
                        'query': query,
                        'result_count': len(rows),
                        'execution_time_ms': execution_time_ms,
                        'results': [dict(row) for row in rows[:10]]  # Sample results
                    }

            elif table == 'system_metrics':
                sql_query = f"""
                    SELECT {fields} FROM system_metrics
                    WHERE timestamp >= NOW() - INTERVAL '1 hour'
                    ORDER BY timestamp DESC
                    LIMIT {limit}
                """

                async with db_manager.postgres_pool.acquire() as conn:
                    start_time = time.perf_counter()
                    rows = await conn.fetch(sql_query)
                    execution_time_ms = (time.perf_counter() - start_time) * 1000

                    return {
                        'query': query,
                        'result_count': len(rows),
                        'execution_time_ms': execution_time_ms,
                        'results': [dict(row) for row in rows[:10]]
                    }

            else:
                # Default query execution
                return {
                    'query': query,
                    'result_count': 0,
                    'execution_time_ms': 1.0,
                    'error': f'Unsupported table: {table}'
                }

        except Exception as e:
            # Fallback to estimated performance metrics
            estimated_time = len(query.get('conditions', '')) * 10  # 10ms per condition
            return {
                'query': query,
                'result_count': 0,
                'execution_time_ms': estimated_time,
                'error': str(e),
                'fallback': True
            }


class SystemProfiler:
    """🔍 Profiler do Sistema"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.profiling_enabled = config.enable_profiling
        self.profile_results = []

    def profile_function(self, func: Callable) -> Callable:
        """Decorator para profiling de funções"""
        def wrapper(*args, **kwargs):
            if not self.profiling_enabled:
                return func(*args, **kwargs)

            profiler = cProfile.Profile()
            profiler.enable()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                profiler.disable()

                # Analisar resultados
                stats_stream = io.StringIO()
                stats = pstats.Stats(profiler, stream=stats_stream)
                stats.sort_stats('cumulative')
                stats.print_stats(10)  # Top 10 funções

                profile_data = {
                    'function': func.__name__,
                    'stats': stats_stream.getvalue(),
                    'timestamp': datetime.now().isoformat()
                }

                self.profile_results.append(profile_data)

        return wrapper

    def get_profile_summary(self) -> Dict:
        """Obter resumo do profiling"""
        if not self.profile_results:
            return {'message': 'No profiling data available'}

        return {
            'total_profiles': len(self.profile_results),
            'latest_profile': self.profile_results[-1],
            'profiling_enabled': self.profiling_enabled
        }


class PerformanceOptimizer:
    """🚀 Otimizador Principal de Performance"""

    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()

        # Inicializar otimizadores
        self.memory_optimizer = MemoryOptimizer(self.config)
        self.latency_optimizer = LatencyOptimizer(self.config)
        self.throughput_optimizer = ThroughputOptimizer(self.config)
        self.cache_optimizer = CacheOptimizer(self.config)
        self.db_optimizer = DatabaseOptimizer()
        self.profiler = SystemProfiler(self.config)

        # Monitoramento contínuo
        self.monitoring_active = False
        self.monitoring_task = None

    async def start_monitoring(self) -> None:
        """Iniciar monitoramento contínuo"""
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitor_continuously())
        logger.info("Performance monitoring started")

    async def stop_monitoring(self) -> None:
        """Parar monitoramento"""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
        logger.info("Performance monitoring stopped")

    async def _monitor_continuously(self) -> None:
        """Loop de monitoramento contínuo"""
        while self.monitoring_active:
            try:
                # Coletar métricas
                metrics = await self.collect_metrics()

                # Verificar se otimização é necessária
                if self._should_optimize(metrics):
                    await self.optimize_system()

                # Aguardar próximo ciclo
                await asyncio.sleep(30)  # Monitorar a cada 30s

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)

    async def collect_metrics(self) -> PerformanceMetrics:
        """Coletar métricas de performance"""
        # CPU e memória
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_data = self.memory_optimizer.monitor_memory()

        # Latência
        latency_stats = self.latency_optimizer.get_latency_stats()

        # Cache
        cache_stats = self.cache_optimizer.get_cache_stats()

        return PerformanceMetrics(
            cpu_usage=cpu_usage,
            memory_usage_mb=memory_data['rss_mb'],
            memory_usage_percent=memory_data['percent'],
            processing_time_ms=latency_stats.get('avg_latency_ms', 0),
            throughput_per_second=len(self.throughput_optimizer.throughput_history),
            latency_p95_ms=latency_stats.get('p95_latency_ms', 0),
            error_rate=0.0,  # Implementar cálculo de taxa de erro
            active_connections=10  # Implementar contagem de conexões ativas
        )

    def _should_optimize(self, metrics: PerformanceMetrics) -> bool:
        """Verificar se otimização é necessária"""
        return (
            metrics.cpu_usage > self.config.max_cpu_usage or
            metrics.memory_usage_mb > self.config.max_memory_usage_mb or
            metrics.latency_p95_ms > self.config.target_latency_ms
        )

    async def optimize_system(self) -> Dict:
        """Executar otimização completa do sistema"""
        logger.info("Starting system optimization...")

        optimization_results = {}

        # 1. Otimizar memória
        memory_result = self.memory_optimizer.optimize_memory()
        optimization_results['memory'] = memory_result

        # 2. Limpar caches antigos
        cache_stats = self.cache_optimizer.get_cache_stats()
        optimization_results['cache'] = cache_stats

        # 3. Forçar garbage collection
        gc.collect()

        logger.info(f"System optimization completed: {optimization_results}")

        return optimization_results

    async def run_performance_benchmark(self) -> Dict:
        """Executar benchmark de performance"""
        logger.info("Running performance benchmark...")

        benchmark_results = {}

        # 1. Benchmark de latência
        start_time = time.perf_counter()

        # Simular operações típicas
        tasks = []
        for i in range(1000):
            async def dummy_task():
                await asyncio.sleep(0.001)  # 1ms de "trabalho"
                return i * 2
            tasks.append(dummy_task)

        results = await self.throughput_optimizer.optimize_async_processing(tasks)

        latency_benchmark = time.perf_counter() - start_time
        benchmark_results['latency_test'] = {
            'total_time_s': latency_benchmark,
            'tasks_completed': len(results),
            'avg_latency_ms': (latency_benchmark / len(results)) * 1000
        }

        # 2. Benchmark de memória
        initial_memory = self.memory_optimizer.monitor_memory()

        # Alocar e liberar memória
        large_data = [list(range(1000)) for _ in range(1000)]
        peak_memory = self.memory_optimizer.monitor_memory()
        del large_data
        gc.collect()
        final_memory = self.memory_optimizer.monitor_memory()

        benchmark_results['memory_test'] = {
            'initial_memory_mb': initial_memory['rss_mb'],
            'peak_memory_mb': peak_memory['rss_mb'],
            'final_memory_mb': final_memory['rss_mb'],
            'memory_recovered_mb': peak_memory['rss_mb'] - final_memory['rss_mb']
        }

        # 3. Benchmark de throughput
        cpu_tasks = [lambda: sum(range(10000)) for _ in range(100)]
        start_cpu_time = time.perf_counter()
        cpu_results = self.throughput_optimizer.optimize_cpu_bound_tasks(cpu_tasks)
        cpu_benchmark_time = time.perf_counter() - start_cpu_time

        benchmark_results['throughput_test'] = {
            'cpu_tasks_completed': len(cpu_results),
            'cpu_processing_time_s': cpu_benchmark_time,
            'cpu_throughput_tasks_per_s': len(cpu_results) / cpu_benchmark_time
        }

        logger.info(f"Benchmark completed: {benchmark_results}")

        return benchmark_results

    def get_optimization_report(self) -> Dict:
        """Gerar relatório de otimização"""
        return {
            'memory_optimizer': {
                'current_usage_mb': self.memory_optimizer.monitor_memory()['rss_mb'],
                'optimization_history': len(self.memory_optimizer.memory_usage_history)
            },
            'latency_optimizer': self.latency_optimizer.get_latency_stats(),
            'cache_optimizer': self.cache_optimizer.get_cache_stats(),
            'profiler': self.profiler.get_profile_summary(),
            'config': {
                'max_cpu_usage': self.config.max_cpu_usage,
                'max_memory_usage_mb': self.config.max_memory_usage_mb,
                'target_latency_ms': self.config.target_latency_ms,
                'cache_enabled': self.config.enable_caching
            }
        }


# 🧪 Funções de Teste
async def test_performance_optimizer():
    """Testar otimizador de performance"""
    config = OptimizationConfig(
        max_memory_usage_mb=256,
        target_latency_ms=50,
        enable_profiling=True
    )

    optimizer = PerformanceOptimizer(config)

    # Iniciar monitoramento
    await optimizer.start_monitoring()

    # Executar benchmark
    benchmark_results = await optimizer.run_performance_benchmark()

    # Obter relatório
    report = optimizer.get_optimization_report()

    # Parar monitoramento
    await optimizer.stop_monitoring()

    print("🚀 Performance Optimizer Test Results:")
    print(f"Benchmark: {benchmark_results}")
    print(f"Report: {report}")


class HighFrequencyOptimizer:
    """⚡ Ultra-Low Latency Optimizer para High-Frequency Trading"""

    def __init__(self):
        self.memory_pools = {}
        self.circular_buffers = {}
        self.optimization_stats = {
            'memory_pool_hits': 0,
            'serialization_optimizations': 0,
            'cpu_optimizations': 0,
            'network_optimizations': 0
        }

        # Configurar uvloop para performance máxima
        try:
            import uvloop
            uvloop.install()
            logger.info("✅ uvloop installed for maximum async performance")
        except ImportError:
            logger.warning("⚠️ uvloop not available, using default asyncio")

    def create_memory_pool(self, pool_name: str, object_size: int, pool_size: int = 1000):
        """Criar pool de memória pré-alocada"""
        self.memory_pools[pool_name] = {
            'objects': [bytearray(object_size) for _ in range(pool_size)],
            'available': list(range(pool_size)),
            'in_use': set(),
            'object_size': object_size,
            'total_size': pool_size
        }
        logger.info(f"Memory pool '{pool_name}' created: {pool_size} objects of {object_size} bytes")

    def get_from_pool(self, pool_name: str) -> Optional[bytearray]:
        """Obter objeto do pool de memória"""
        if pool_name not in self.memory_pools:
            return None

        pool = self.memory_pools[pool_name]
        if not pool['available']:
            return None

        index = pool['available'].pop()
        pool['in_use'].add(index)
        self.optimization_stats['memory_pool_hits'] += 1

        return pool['objects'][index]

    def return_to_pool(self, pool_name: str, obj_index: int):
        """Retornar objeto ao pool"""
        if pool_name not in self.memory_pools:
            return

        pool = self.memory_pools[pool_name]
        if obj_index in pool['in_use']:
            pool['in_use'].remove(obj_index)
            pool['available'].append(obj_index)
            # Limpar o objeto para reuso
            pool['objects'][obj_index][:] = b'\x00' * pool['object_size']

    def optimize_serialization(self, data: dict) -> bytes:
        """Serialização ultra-rápida com orjson/msgpack"""
        try:
            import orjson
            self.optimization_stats['serialization_optimizations'] += 1
            return orjson.dumps(data)
        except ImportError:
            try:
                import msgpack
                self.optimization_stats['serialization_optimizations'] += 1
                return msgpack.packb(data)
            except ImportError:
                import json
                return json.dumps(data).encode('utf-8')

    async def benchmark_latency(self, iterations: int = 10000) -> Dict:
        """Benchmark de latência ultra-baixa"""
        import statistics

        test_data = {
            'symbol': 'WDOU25',
            'price': 129.456,
            'volume': 1000,
            'timestamp': time.time()
        }

        serialization_times = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            serialized = self.optimize_serialization(test_data)
            end = time.perf_counter_ns()
            serialization_times.append((end - start) / 1000)  # microseconds

        return {
            'serialization': {
                'avg_microseconds': statistics.mean(serialization_times),
                'p95_microseconds': sorted(serialization_times)[int(0.95 * len(serialization_times))],
                'p99_microseconds': sorted(serialization_times)[int(0.99 * len(serialization_times))],
                'min_microseconds': min(serialization_times)
            },
            'optimization_stats': self.optimization_stats.copy()
        }


class EnhancedPerformanceOptimizer(PerformanceOptimizer):
    """🚀 Performance Optimizer com otimizações HFT"""

    def __init__(self, config: OptimizationConfig = None):
        super().__init__(config)
        self.hft_optimizer = HighFrequencyOptimizer()
        logger.info("Enhanced Performance Optimizer initialized with HFT optimizations")

    async def run_hft_benchmark(self) -> Dict:
        """Executar benchmark específico para HFT"""
        logger.info("Running HFT performance benchmark...")

        standard_benchmark = await super().run_performance_benchmark()
        hft_benchmark = await self.hft_optimizer.benchmark_latency(iterations=50000)

        return {
            'standard_benchmark': standard_benchmark,
            'hft_benchmark': hft_benchmark
        }


async def test_hft_performance_optimizer():
    """Testar otimizador HFT"""
    config = OptimizationConfig(
        max_memory_usage_mb=512,
        target_latency_ms=1,
        enable_profiling=True
    )

    optimizer = EnhancedPerformanceOptimizer(config)
    benchmark_results = await optimizer.run_hft_benchmark()

    print("\n🏎️  HIGH-FREQUENCY TRADING PERFORMANCE RESULTS")
    print("="*60)

    hft_bench = benchmark_results['hft_benchmark']
    print(f"📊 SERIALIZATION PERFORMANCE:")
    print(f"   Average: {hft_bench['serialization']['avg_microseconds']:.2f} μs")
    print(f"   P95:     {hft_bench['serialization']['p95_microseconds']:.2f} μs")
    print(f"   P99:     {hft_bench['serialization']['p99_microseconds']:.2f} μs")
    print(f"   Min:     {hft_bench['serialization']['min_microseconds']:.2f} μs")

    print("✅ HFT Performance Test Completed")


if __name__ == "__main__":
    # Testar otimizador padrão
    print("Running standard performance test...")
    asyncio.run(test_performance_optimizer())

    print("\n" + "="*50)

    # Testar otimizador HFT
    print("Running HFT performance test...")
    asyncio.run(test_hft_performance_optimizer())