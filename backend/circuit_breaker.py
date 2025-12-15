"""
Circuit Breaker Pattern
Proteção contra falhas em cascata e degradação de serviços
"""

import time
import logging
import asyncio
from enum import Enum
from typing import Callable, Optional, Any
from functools import wraps
from dataclasses import dataclass
from datetime import datetime, timedelta

from metrics import get_metrics_manager

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Estados do circuit breaker"""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Failing - reject calls
    HALF_OPEN = "half_open"    # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuração do circuit breaker"""
    failure_threshold: int = 5           # Número de falhas para abrir
    success_threshold: int = 2           # Sucessos consecutivos para fechar (de half-open)
    timeout_seconds: float = 60.0        # Tempo até tentar half-open
    half_open_max_calls: int = 3         # Max chamadas em half-open
    expected_exception: type = Exception # Tipo de exceção a tratar


class CircuitBreaker:
    """
    Implementação do padrão Circuit Breaker

    Estados:
    - CLOSED: Operação normal, chamadas passam
    - OPEN: Sistema falhou, chamadas são rejeitadas
    - HALF_OPEN: Testando recuperação, algumas chamadas passam
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """
        Inicializa circuit breaker

        Args:
            name: Nome do circuit breaker (para logs/métricas)
            config: Configuração personalizada
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0

        self.metrics = get_metrics_manager()

        logger.info(f"Circuit Breaker '{name}' inicializado: {self.config}")

    def _record_success(self):
        """Registra sucesso de chamada"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            logger.debug(f"[{self.name}] Sucesso em HALF_OPEN ({self.success_count}/{self.config.success_threshold})")

            if self.success_count >= self.config.success_threshold:
                self._close()
        elif self.state == CircuitState.CLOSED:
            # Reset failure count em CLOSED
            self.failure_count = 0

    def _record_failure(self):
        """Registra falha de chamada"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        logger.warning(f"[{self.name}] Falha #{self.failure_count}")

        # Registrar métrica de erro
        self.metrics.record_error(f'circuit_breaker_{self.name}', 'error')

        if self.state == CircuitState.HALF_OPEN:
            # Uma falha em HALF_OPEN volta para OPEN
            self._open()
        elif self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self._open()

    def _open(self):
        """Abre o circuit (rejeita chamadas)"""
        self.state = CircuitState.OPEN
        self.half_open_calls = 0
        self.success_count = 0

        logger.error(f"[{self.name}] Circuit ABERTO - rejeitando chamadas por {self.config.timeout_seconds}s")

        # Registrar métrica crítica
        self.metrics.record_error(f'circuit_breaker_{self.name}', 'critical')

    def _half_open(self):
        """Coloca circuit em half-open (teste de recuperação)"""
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        self.success_count = 0
        self.failure_count = 0

        logger.warning(f"[{self.name}] Circuit HALF-OPEN - testando recuperação")

    def _close(self):
        """Fecha o circuit (operação normal)"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0

        logger.info(f"[{self.name}] Circuit FECHADO - operação normal restaurada")

    def _can_attempt_call(self) -> bool:
        """Verifica se pode tentar fazer chamada"""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Verificar se já passou tempo suficiente para tentar half-open
            if self.last_failure_time:
                elapsed = time.time() - self.last_failure_time
                if elapsed >= self.config.timeout_seconds:
                    self._half_open()
                    return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            # Limitar número de chamadas em half-open
            if self.half_open_calls < self.config.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False

        return False

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Executa função através do circuit breaker

        Args:
            func: Função a executar
            *args, **kwargs: Argumentos da função

        Returns:
            Resultado da função

        Raises:
            CircuitBreakerOpenError: Se circuit está aberto
        """
        if not self._can_attempt_call():
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' está {self.state.value}"
            )

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except self.config.expected_exception as e:
            self._record_failure()
            raise

    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        Executa função assíncrona através do circuit breaker

        Args:
            func: Função assíncrona a executar
            *args, **kwargs: Argumentos da função

        Returns:
            Resultado da função

        Raises:
            CircuitBreakerOpenError: Se circuit está aberto
        """
        if not self._can_attempt_call():
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' está {self.state.value}"
            )

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except self.config.expected_exception as e:
            self._record_failure()
            raise

    def __call__(self, func: Callable) -> Callable:
        """
        Decorator para circuit breaker

        Usage:
            @circuit_breaker
            def my_function():
                pass
        """
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self.call_async(func, *args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self.call(func, *args, **kwargs)
            return wrapper

    def get_status(self) -> dict:
        """Retorna status atual do circuit breaker"""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'half_open_calls': self.half_open_calls,
            'last_failure_time': datetime.fromtimestamp(self.last_failure_time).isoformat() if self.last_failure_time else None
        }


class CircuitBreakerOpenError(Exception):
    """Exceção lançada quando circuit breaker está aberto"""
    pass


# Circuit breakers pré-configurados para serviços comuns
_circuit_breakers = {}


def get_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None
) -> CircuitBreaker:
    """
    Retorna circuit breaker para um serviço

    Args:
        name: Nome do serviço
        config: Configuração personalizada

    Returns:
        CircuitBreaker instance
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]


def get_all_circuit_breakers() -> dict:
    """Retorna status de todos os circuit breakers"""
    return {
        name: cb.get_status()
        for name, cb in _circuit_breakers.items()
    }


# Circuit breakers pré-configurados
def get_deriv_api_circuit_breaker() -> CircuitBreaker:
    """Circuit breaker para API Deriv"""
    return get_circuit_breaker(
        'deriv_api',
        CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=30.0,
            half_open_max_calls=2
        )
    )


def get_ml_predictor_circuit_breaker() -> CircuitBreaker:
    """Circuit breaker para ML predictor"""
    return get_circuit_breaker(
        'ml_predictor',
        CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=3,
            timeout_seconds=60.0,
            half_open_max_calls=3
        )
    )


def get_trading_engine_circuit_breaker() -> CircuitBreaker:
    """Circuit breaker para trading engine"""
    return get_circuit_breaker(
        'trading_engine',
        CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=2,
            timeout_seconds=120.0,
            half_open_max_calls=1
        )
    )


if __name__ == "__main__":
    # Teste do circuit breaker
    logging.basicConfig(level=logging.INFO)

    # Criar circuit breaker
    cb = CircuitBreaker('test_service', CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout_seconds=5.0
    ))

    # Função que falha
    call_count = 0
    def faulty_function():
        global call_count
        call_count += 1
        if call_count <= 5:
            raise Exception("Serviço indisponível")
        return "Sucesso!"

    # Testar circuit breaker
    print("Testando circuit breaker...")

    # Chamadas que falham
    for i in range(5):
        try:
            result = cb.call(faulty_function)
            print(f"Chamada {i+1}: {result}")
        except CircuitBreakerOpenError as e:
            print(f"Chamada {i+1}: Circuit ABERTO - {e}")
        except Exception as e:
            print(f"Chamada {i+1}: Falhou - {e}")

        print(f"  Status: {cb.get_status()}")
        time.sleep(1)

    # Aguardar timeout para half-open
    print(f"\nAguardando {cb.config.timeout_seconds}s para half-open...")
    time.sleep(cb.config.timeout_seconds + 1)

    # Chamadas que recuperam
    for i in range(3):
        try:
            result = cb.call(faulty_function)
            print(f"Chamada recuperação {i+1}: {result}")
        except CircuitBreakerOpenError as e:
            print(f"Chamada recuperação {i+1}: Circuit ABERTO - {e}")
        except Exception as e:
            print(f"Chamada recuperação {i+1}: Falhou - {e}")

        print(f"  Status: {cb.get_status()}")
        time.sleep(0.5)

    print("\n[OK] Teste de circuit breaker completo!")
