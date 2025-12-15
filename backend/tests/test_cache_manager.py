"""
Testes para o sistema de cache de indicadores
"""

import pytest
import pandas as pd
import numpy as np
from cache_manager import CacheManager, get_cache_manager, cached_indicator, clear_all_caches


def test_cache_manager_hash_consistency():
    """Testa se hash de DataFrame é consistente"""
    cache = CacheManager()

    df = pd.DataFrame({'close': [100 + i for i in range(100)]})

    hash1 = cache._hash_dataframe(df)
    hash2 = cache._hash_dataframe(df)

    assert hash1 == hash2, "Hash deve ser consistente para o mesmo DataFrame"


def test_cache_manager_hash_different():
    """Testa se DataFrames diferentes geram hashes diferentes"""
    cache = CacheManager()

    df1 = pd.DataFrame({'close': [100 + i for i in range(100)]})
    df2 = pd.DataFrame({'close': [200 + i for i in range(100)]})

    hash1 = cache._hash_dataframe(df1)
    hash2 = cache._hash_dataframe(df2)

    assert hash1 != hash2, "Hashes devem ser diferentes para DataFrames diferentes"


def test_cache_serialize_series():
    """Testa serialização e deserialização de Series"""
    cache = CacheManager()

    original = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5])
    serialized = cache._serialize_value(original)
    deserialized = cache._deserialize_value(serialized)

    assert isinstance(deserialized, pd.Series), "Deve retornar Series"
    assert list(original) == list(deserialized), "Valores devem ser preservados"


def test_cache_serialize_ndarray():
    """Testa serialização e deserialização de NumPy array"""
    cache = CacheManager()

    original = np.array([1, 2, 3, 4, 5])
    serialized = cache._serialize_value(original)
    deserialized = cache._deserialize_value(serialized)

    assert isinstance(deserialized, np.ndarray), "Deve retornar ndarray"
    assert np.array_equal(original, deserialized), "Valores devem ser preservados"


def test_cache_serialize_dict():
    """Testa serialização de dicionário com Series"""
    cache = CacheManager()

    original = {
        'rsi': pd.Series([30.5, 40.2, 50.1]),
        'value': 123.45
    }
    serialized = cache._serialize_value(original)
    deserialized = cache._deserialize_value(serialized)

    assert isinstance(deserialized, dict), "Deve retornar dict"
    assert 'rsi' in deserialized
    assert deserialized['value'] == 123.45


def test_cached_indicator_decorator():
    """Testa decorator de cache para indicadores"""

    # Contador de execuções
    call_count = 0

    @cached_indicator(ttl=60)
    def expensive_calculation(df, period):
        nonlocal call_count
        call_count += 1
        return df['close'].rolling(period).mean()

    # Criar DataFrame de teste
    df = pd.DataFrame({'close': [100 + i for i in range(100)]})

    # Primeira chamada - deve executar
    result1 = expensive_calculation(df, 20)
    assert call_count == 1, "Deve executar cálculo na primeira chamada"

    # Segunda chamada com mesmo DataFrame - deve usar cache (LRU)
    result2 = expensive_calculation(df, 20)
    assert call_count == 1, "Deve usar cache na segunda chamada"

    # Verificar que resultados são iguais
    assert result1.equals(result2), "Resultados devem ser iguais"


def test_cache_stats():
    """Testa estatísticas do cache"""
    cache = get_cache_manager()

    # Resetar estatísticas
    cache.cache_hits = 0
    cache.cache_misses = 0
    cache.total_requests = 10

    cache.cache_hits = 7
    cache.cache_misses = 3

    stats = cache.get_stats()

    assert stats['total_requests'] == 10
    assert stats['cache_hits'] == 7
    assert stats['cache_misses'] == 3
    assert stats['hit_rate'] == 70.0


def test_clear_caches():
    """Testa limpeza de caches"""
    # Não deve dar erro
    clear_all_caches()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
