"""
Sistema de Cache para Indicadores Técnicos
Reduz cálculos redundantes com caching in-memory (LRU) e opcional Redis
"""

import hashlib
import json
import logging
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Gerenciador centralizado de cache para indicadores técnicos

    Suporta:
    - Cache in-memory com LRU (functools.lru_cache)
    - Cache Redis (opcional, para persistência entre restarts)
    - TTL configurável
    - Invalidação de cache
    """

    def __init__(self, use_redis: bool = False, redis_url: Optional[str] = None):
        """
        Inicializa cache manager

        Args:
            use_redis: Se True, usa Redis além do cache in-memory
            redis_url: URL de conexão Redis (ex: redis://localhost:6379/0)
        """
        self.use_redis = use_redis
        self.redis_client = None
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0

        if use_redis:
            try:
                import redis
                self.redis_client = redis.from_url(
                    redis_url or "redis://localhost:6379/0",
                    decode_responses=True
                )
                logger.info("Redis cache habilitado")
            except ImportError:
                logger.warning("Redis não disponível, usando apenas cache in-memory")
                self.use_redis = False
            except Exception as e:
                logger.warning(f"Erro ao conectar Redis: {e}. Usando apenas cache in-memory")
                self.use_redis = False

    @staticmethod
    def _hash_dataframe(df: pd.DataFrame) -> str:
        """
        Gera hash único de um DataFrame para usar como cache key

        Args:
            df: DataFrame a ser hasheado

        Returns:
            String hash MD5
        """
        # Usar últimas 50 linhas + primeiras 10 linhas para gerar hash
        # (balanceia performance vs unicidade)
        sample_data = pd.concat([df.head(10), df.tail(50)])

        # Converter para string e gerar hash
        data_str = sample_data.to_json()
        return hashlib.md5(data_str.encode()).hexdigest()

    @staticmethod
    def _serialize_value(value: Any) -> str:
        """
        Serializa valor para armazenamento em cache

        Args:
            value: Valor a serializar (pode ser Series, array, dict, etc)

        Returns:
            String JSON
        """
        if isinstance(value, pd.Series):
            return json.dumps({
                'type': 'series',
                'data': value.tolist(),
                'index': value.index.tolist() if hasattr(value.index, 'tolist') else list(value.index)
            })
        elif isinstance(value, np.ndarray):
            return json.dumps({
                'type': 'ndarray',
                'data': value.tolist()
            })
        elif isinstance(value, dict):
            # Recursivamente serializar dicionários
            serialized = {}
            for k, v in value.items():
                if isinstance(v, (pd.Series, np.ndarray)):
                    serialized[k] = CacheManager._serialize_value(v)
                else:
                    serialized[k] = v
            return json.dumps({
                'type': 'dict',
                'data': serialized
            })
        else:
            return json.dumps({
                'type': 'simple',
                'data': value
            })

    @staticmethod
    def _deserialize_value(serialized: str) -> Any:
        """
        Deserializa valor do cache

        Args:
            serialized: String JSON

        Returns:
            Valor original (Series, array, dict, etc)
        """
        obj = json.loads(serialized)

        if obj['type'] == 'series':
            return pd.Series(obj['data'], index=obj['index'])
        elif obj['type'] == 'ndarray':
            return np.array(obj['data'])
        elif obj['type'] == 'dict':
            # Recursivamente deserializar
            result = {}
            for k, v in obj['data'].items():
                if isinstance(v, str) and v.startswith('{"type":'):
                    result[k] = CacheManager._deserialize_value(v)
                else:
                    result[k] = v
            return result
        else:
            return obj['data']

    def get(self, key: str) -> Optional[Any]:
        """
        Busca valor no cache Redis

        Args:
            key: Chave do cache

        Returns:
            Valor ou None se não encontrado
        """
        if not self.use_redis or not self.redis_client:
            return None

        try:
            cached = self.redis_client.get(key)
            if cached:
                self.cache_hits += 1
                return self._deserialize_value(cached)
            else:
                self.cache_misses += 1
                return None
        except Exception as e:
            logger.warning(f"Erro ao buscar cache Redis: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int = 300):
        """
        Armazena valor no cache Redis

        Args:
            key: Chave do cache
            value: Valor a armazenar
            ttl: Time-to-live em segundos (padrão 5 minutos)
        """
        if not self.use_redis or not self.redis_client:
            return

        try:
            serialized = self._serialize_value(value)
            self.redis_client.setex(key, ttl, serialized)
        except Exception as e:
            logger.warning(f"Erro ao salvar cache Redis: {e}")

    def invalidate(self, pattern: str = "*"):
        """
        Invalida cache matching pattern

        Args:
            pattern: Padrão de chaves a invalidar (ex: "indicator:*")
        """
        if not self.use_redis or not self.redis_client:
            return

        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
                logger.info(f"Invalidado {len(keys)} chaves de cache")
        except Exception as e:
            logger.warning(f"Erro ao invalidar cache: {e}")

    def get_stats(self) -> Dict:
        """
        Retorna estatísticas do cache

        Returns:
            Dict com hits, misses e hit rate
        """
        hit_rate = (self.cache_hits / self.total_requests * 100) if self.total_requests > 0 else 0

        return {
            'total_requests': self.total_requests,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': round(hit_rate, 2)
        }


# Instância global do cache manager
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """
    Retorna instância global do cache manager (singleton)

    Returns:
        CacheManager instance
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(use_redis=False)  # Redis disabled por padrão
    return _cache_manager


def initialize_cache_manager(use_redis: bool = False, redis_url: Optional[str] = None):
    """
    Inicializa cache manager global

    Args:
        use_redis: Se True, habilita cache Redis
        redis_url: URL de conexão Redis
    """
    global _cache_manager
    _cache_manager = CacheManager(use_redis=use_redis, redis_url=redis_url)
    logger.info("Cache manager inicializado")


def cached_indicator(ttl: int = 300):
    """
    Decorator para cache automático de indicadores técnicos

    Args:
        ttl: Time-to-live em segundos (padrão 5 minutos)

    Usage:
        @cached_indicator(ttl=300)
        def calculate_sma(df, period):
            return df['close'].rolling(period).mean()
    """
    def decorator(func: Callable) -> Callable:
        # Cache in-memory usando hash do DataFrame como key
        memory_cache = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache_manager()
            cache.total_requests += 1

            # Gerar cache key baseado em função + argumentos
            # args[0] normalmente é 'self', args[1] é o DataFrame
            df = None
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    df = arg
                    break

            if df is not None:
                df_hash = CacheManager._hash_dataframe(df)
                cache_key = f"indicator:{func.__name__}:{df_hash}:{str(kwargs)}"

                # Tentar buscar no cache in-memory primeiro (mais rápido)
                if cache_key in memory_cache:
                    logger.debug(f"Cache HIT (memory): {func.__name__}")
                    cache.cache_hits += 1
                    return memory_cache[cache_key]

                # Tentar buscar no cache Redis
                cached_value = cache.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache HIT (redis): {func.__name__}")
                    # Salvar também em memory cache
                    memory_cache[cache_key] = cached_value
                    return cached_value
            else:
                cache_key = None

            # Cache miss - calcular valor
            logger.debug(f"Cache MISS: {func.__name__}")
            cache.cache_misses += 1
            result = func(*args, **kwargs)

            # Salvar nos caches
            if cache_key:
                memory_cache[cache_key] = result
                cache.set(cache_key, result, ttl=ttl)

                # Limitar tamanho do memory cache
                if len(memory_cache) > 128:
                    # Remover entrada mais antiga (FIFO simples)
                    first_key = next(iter(memory_cache))
                    del memory_cache[first_key]

            return result

        return wrapper

    return decorator


# Funções de cache específicas para indicadores comuns
@lru_cache(maxsize=256)
def cached_sma(close_hash: str, period: int) -> float:
    """
    Cache helper para SMA

    Nota: Não deve ser chamado diretamente - use através de indicators
    """
    pass


@lru_cache(maxsize=256)
def cached_ema(close_hash: str, period: int) -> float:
    """
    Cache helper para EMA
    """
    pass


@lru_cache(maxsize=256)
def cached_rsi(close_hash: str, period: int) -> float:
    """
    Cache helper para RSI
    """
    pass


def clear_all_caches():
    """
    Limpa todos os caches (in-memory + Redis)
    """
    # Limpar LRU caches
    cached_sma.cache_clear()
    cached_ema.cache_clear()
    cached_rsi.cache_clear()

    # Limpar Redis cache
    cache = get_cache_manager()
    cache.invalidate("indicator:*")

    logger.info("Todos os caches limpos")


if __name__ == "__main__":
    # Teste do cache manager
    logging.basicConfig(level=logging.INFO)

    # Criar dados de teste
    test_df = pd.DataFrame({
        'close': [100 + i for i in range(100)]
    })

    # Criar cache manager
    cache = CacheManager()

    # Testar hash
    hash1 = cache._hash_dataframe(test_df)
    hash2 = cache._hash_dataframe(test_df)
    assert hash1 == hash2, "Hash deve ser consistente"

    # Testar serialização
    test_series = pd.Series([1, 2, 3, 4, 5])
    serialized = cache._serialize_value(test_series)
    deserialized = cache._deserialize_value(serialized)
    assert list(test_series) == list(deserialized), "Serialização/deserialização deve preservar dados"

    logger.info("Cache manager: Todos os testes passaram!")
