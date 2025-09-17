"""
Redis Cache Manager - Phase 12 Real Infrastructure
Sistema completo de cache Redis para alta performance
"""

import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
import redis.asyncio as redis
import pickle
import logging
from enum import Enum

class CacheNamespace(Enum):
    MARKET_DATA = "market_data"
    AI_PREDICTIONS = "ai_predictions"
    USER_SESSIONS = "user_sessions"
    TRADING_POSITIONS = "trading_positions"
    SYSTEM_METRICS = "system_metrics"
    FEATURE_CACHE = "feature_cache"
    MODEL_CACHE = "model_cache"
    STRATEGY_CACHE = "strategy_cache"

class RedisCacheManager:
    def __init__(self):
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.redis_client: Optional[redis.Redis] = None
        self.logger = logging.getLogger(__name__)

        # Cache configuration
        self.default_ttl = 3600  # 1 hour
        self.namespace_ttls = {
            CacheNamespace.MARKET_DATA: 300,  # 5 minutes
            CacheNamespace.AI_PREDICTIONS: 60,  # 1 minute
            CacheNamespace.USER_SESSIONS: 86400,  # 24 hours
            CacheNamespace.TRADING_POSITIONS: 30,  # 30 seconds
            CacheNamespace.SYSTEM_METRICS: 60,  # 1 minute
            CacheNamespace.FEATURE_CACHE: 600,  # 10 minutes
            CacheNamespace.MODEL_CACHE: 3600,  # 1 hour
            CacheNamespace.STRATEGY_CACHE: 1800,  # 30 minutes
        }

    async def initialize(self) -> bool:
        """Initialize Redis connection with retry logic"""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )

            # Test connection
            await self.redis_client.ping()
            self.logger.info("Redis cache manager initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize Redis: {e}")
            return False

    def _generate_key(self, namespace: CacheNamespace, key: str) -> str:
        """Generate namespaced cache key"""
        return f"{namespace.value}:{key}"

    async def set(
        self,
        namespace: CacheNamespace,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set cache value with automatic serialization"""
        if not self.redis_client:
            await self.initialize()

        try:
            cache_key = self._generate_key(namespace, key)
            ttl = ttl or self.namespace_ttls.get(namespace, self.default_ttl)

            # Serialize complex objects
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value, default=str)
            elif isinstance(value, (int, float, str, bool)):
                serialized_value = str(value)
            else:
                # Use pickle for complex objects
                serialized_value = pickle.dumps(value).decode('latin-1')
                cache_key += ":pickle"

            await self.redis_client.setex(cache_key, ttl, serialized_value)
            return True

        except Exception as e:
            self.logger.error(f"Cache set error: {e}")
            return False

    async def get(self, namespace: CacheNamespace, key: str) -> Optional[Any]:
        """Get cache value with automatic deserialization"""
        if not self.redis_client:
            await self.initialize()

        try:
            cache_key = self._generate_key(namespace, key)

            # Try regular key first
            value = await self.redis_client.get(cache_key)
            if value is not None:
                # Try JSON deserialization
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value

            # Try pickle key
            pickle_key = cache_key + ":pickle"
            pickle_value = await self.redis_client.get(pickle_key)
            if pickle_value is not None:
                return pickle.loads(pickle_value.encode('latin-1'))

            return None

        except Exception as e:
            self.logger.error(f"Cache get error: {e}")
            return None

    async def delete(self, namespace: CacheNamespace, key: str) -> bool:
        """Delete cache key"""
        if not self.redis_client:
            await self.initialize()

        try:
            cache_key = self._generate_key(namespace, key)
            deleted = await self.redis_client.delete(cache_key)

            # Also try to delete pickle version
            pickle_key = cache_key + ":pickle"
            await self.redis_client.delete(pickle_key)

            return deleted > 0

        except Exception as e:
            self.logger.error(f"Cache delete error: {e}")
            return False

    async def exists(self, namespace: CacheNamespace, key: str) -> bool:
        """Check if cache key exists"""
        if not self.redis_client:
            await self.initialize()

        try:
            cache_key = self._generate_key(namespace, key)
            exists = await self.redis_client.exists(cache_key)

            if not exists:
                pickle_key = cache_key + ":pickle"
                exists = await self.redis_client.exists(pickle_key)

            return exists > 0

        except Exception as e:
            self.logger.error(f"Cache exists error: {e}")
            return False

    async def expire(self, namespace: CacheNamespace, key: str, ttl: int) -> bool:
        """Set expiration time for cache key"""
        if not self.redis_client:
            await self.initialize()

        try:
            cache_key = self._generate_key(namespace, key)
            return await self.redis_client.expire(cache_key, ttl)

        except Exception as e:
            self.logger.error(f"Cache expire error: {e}")
            return False

    async def increment(self, namespace: CacheNamespace, key: str, amount: int = 1) -> Optional[int]:
        """Increment numeric cache value"""
        if not self.redis_client:
            await self.initialize()

        try:
            cache_key = self._generate_key(namespace, key)
            return await self.redis_client.incrby(cache_key, amount)

        except Exception as e:
            self.logger.error(f"Cache increment error: {e}")
            return None

    async def get_keys_by_pattern(self, namespace: CacheNamespace, pattern: str = "*") -> List[str]:
        """Get all keys matching pattern in namespace"""
        if not self.redis_client:
            await self.initialize()

        try:
            search_pattern = self._generate_key(namespace, pattern)
            keys = await self.redis_client.keys(search_pattern)
            return [key.replace(f"{namespace.value}:", "") for key in keys if not key.endswith(":pickle")]

        except Exception as e:
            self.logger.error(f"Cache keys error: {e}")
            return []

    async def flush_namespace(self, namespace: CacheNamespace) -> bool:
        """Delete all keys in namespace"""
        if not self.redis_client:
            await self.initialize()

        try:
            pattern = f"{namespace.value}:*"
            keys = await self.redis_client.keys(pattern)

            if keys:
                deleted = await self.redis_client.delete(*keys)
                self.logger.info(f"Deleted {deleted} keys from namespace {namespace.value}")
                return True
            return True

        except Exception as e:
            self.logger.error(f"Cache flush namespace error: {e}")
            return False

    # Specialized cache methods for trading bot

    async def cache_market_tick(self, symbol: str, tick_data: Dict[str, Any]) -> bool:
        """Cache latest market tick data"""
        key = f"{symbol}:latest_tick"
        return await self.set(CacheNamespace.MARKET_DATA, key, tick_data, ttl=30)

    async def get_latest_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest cached tick data"""
        key = f"{symbol}:latest_tick"
        return await self.get(CacheNamespace.MARKET_DATA, key)

    async def cache_ai_prediction(self, model_id: str, symbol: str, prediction: Dict[str, Any]) -> bool:
        """Cache AI model prediction"""
        key = f"{model_id}:{symbol}:prediction"
        return await self.set(CacheNamespace.AI_PREDICTIONS, key, prediction, ttl=60)

    async def get_ai_prediction(self, model_id: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached AI prediction"""
        key = f"{model_id}:{symbol}:prediction"
        return await self.get(CacheNamespace.AI_PREDICTIONS, key)

    async def cache_user_session(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """Cache user session data"""
        return await self.set(CacheNamespace.USER_SESSIONS, session_id, session_data, ttl=86400)

    async def get_user_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get cached user session"""
        return await self.get(CacheNamespace.USER_SESSIONS, session_id)

    async def cache_trading_position(self, position_id: str, position_data: Dict[str, Any]) -> bool:
        """Cache trading position data"""
        return await self.set(CacheNamespace.TRADING_POSITIONS, position_id, position_data, ttl=30)

    async def get_trading_position(self, position_id: str) -> Optional[Dict[str, Any]]:
        """Get cached trading position"""
        return await self.get(CacheNamespace.TRADING_POSITIONS, position_id)

    async def cache_system_metric(self, metric_name: str, value: Union[int, float, Dict[str, Any]]) -> bool:
        """Cache system metric"""
        return await self.set(CacheNamespace.SYSTEM_METRICS, metric_name, value, ttl=60)

    async def get_system_metric(self, metric_name: str) -> Optional[Union[int, float, Dict[str, Any]]]:
        """Get cached system metric"""
        return await self.get(CacheNamespace.SYSTEM_METRICS, metric_name)

    async def cache_features(self, symbol: str, timestamp: str, features: Dict[str, Any]) -> bool:
        """Cache processed features"""
        key = f"{symbol}:{timestamp}"
        return await self.set(CacheNamespace.FEATURE_CACHE, key, features, ttl=600)

    async def get_cached_features(self, symbol: str, timestamp: str) -> Optional[Dict[str, Any]]:
        """Get cached features"""
        key = f"{symbol}:{timestamp}"
        return await self.get(CacheNamespace.FEATURE_CACHE, key)

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache usage statistics"""
        if not self.redis_client:
            await self.initialize()

        try:
            info = await self.redis_client.info()
            stats = {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory_human', '0B'),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'hit_rate': 0
            }

            hits = stats['keyspace_hits']
            misses = stats['keyspace_misses']
            total = hits + misses

            if total > 0:
                stats['hit_rate'] = round((hits / total) * 100, 2)

            # Get namespace statistics
            namespace_stats = {}
            for namespace in CacheNamespace:
                keys = await self.get_keys_by_pattern(namespace, "*")
                namespace_stats[namespace.value] = len(keys)

            stats['namespace_counts'] = namespace_stats

            return stats

        except Exception as e:
            self.logger.error(f"Cache stats error: {e}")
            return {}

    async def health_check(self) -> Dict[str, Any]:
        """Redis health check"""
        health_status = {
            'status': 'unhealthy',
            'timestamp': datetime.utcnow().isoformat(),
            'latency_ms': None
        }

        try:
            if not self.redis_client:
                await self.initialize()

            start_time = datetime.utcnow()
            await self.redis_client.ping()
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000

            health_status.update({
                'status': 'healthy',
                'latency_ms': round(latency, 2)
            })

        except Exception as e:
            health_status['error'] = str(e)

        return health_status

    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()

# Global cache manager instance
cache_manager = RedisCacheManager()

async def get_cache_manager() -> RedisCacheManager:
    """Get cache manager instance"""
    if not cache_manager.redis_client:
        await cache_manager.initialize()
    return cache_manager