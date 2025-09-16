"""
Contract Proposals Engine - Sistema avançado de proposals da Deriv API
Implementa cálculo de preços em tempo real, validação de barriers e otimização de performance
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class ContractType(Enum):
    """Tipos de contratos suportados"""
    CALL = "CALL"
    PUT = "PUT"
    DIGITEVEN = "DIGITEVEN"
    DIGITODD = "DIGITODD"
    DIGITOVER = "DIGITOVER"
    DIGITUNDER = "DIGITUNDER"
    ONETOUCH = "ONETOUCH"
    NOTOUCH = "NOTOUCH"

@dataclass
class ProposalRequest:
    """Estrutura de uma solicitação de proposal"""
    contract_type: str
    symbol: str
    amount: float
    duration: int
    duration_unit: str = "m"
    barrier: Optional[str] = None
    basis: str = "stake"
    currency: str = "USD"
    req_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

@dataclass
class ProposalResponse:
    """Estrutura de resposta de uma proposal"""
    id: str
    ask_price: float
    payout: float
    spot: Optional[float] = None
    barrier: Optional[str] = None
    contract_type: str = ""
    symbol: str = ""
    display_value: str = ""
    timestamp: float = field(default_factory=time.time)
    valid_until: Optional[float] = None

@dataclass
class CachedProposal:
    """Proposal em cache com controle de validade"""
    response: ProposalResponse
    created_at: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

class ProposalCache:
    """Sistema de cache inteligente para proposals"""

    def __init__(self, max_size: int = 1000, ttl: float = 30.0):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, CachedProposal] = {}
        self.access_order: deque = deque()

    def _generate_key(self, request: ProposalRequest) -> str:
        """Gera chave única para o cache baseada nos parâmetros da proposal"""
        barrier_key = f"_{request.barrier}" if request.barrier else ""
        return f"{request.contract_type}_{request.symbol}_{request.amount}_{request.duration}_{request.duration_unit}{barrier_key}"

    def get(self, request: ProposalRequest) -> Optional[ProposalResponse]:
        """Busca proposal no cache"""
        key = self._generate_key(request)

        if key not in self.cache:
            return None

        cached = self.cache[key]
        current_time = time.time()

        # Verifica se ainda é válida
        if current_time - cached.created_at > self.ttl:
            del self.cache[key]
            if key in self.access_order:
                self.access_order.remove(key)
            return None

        # Atualiza estatísticas de acesso
        cached.access_count += 1
        cached.last_accessed = current_time

        # Move para o final da fila (LRU)
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

        return cached.response

    def put(self, request: ProposalRequest, response: ProposalResponse):
        """Armazena proposal no cache"""
        key = self._generate_key(request)
        current_time = time.time()

        # Remove item mais antigo se necessário
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_oldest()

        self.cache[key] = CachedProposal(
            response=response,
            created_at=current_time
        )

        # Adiciona na fila de acesso
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

    def _evict_oldest(self):
        """Remove o item menos recentemente usado"""
        if self.access_order:
            oldest_key = self.access_order.popleft()
            if oldest_key in self.cache:
                del self.cache[oldest_key]

    def clear_expired(self):
        """Remove proposals expiradas"""
        current_time = time.time()
        expired_keys = [
            key for key, cached in self.cache.items()
            if current_time - cached.created_at > self.ttl
        ]

        for key in expired_keys:
            del self.cache[key]
            if key in self.access_order:
                self.access_order.remove(key)

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": self._calculate_hit_rate(),
            "oldest_entry": min([c.created_at for c in self.cache.values()]) if self.cache else None
        }

    def _calculate_hit_rate(self) -> float:
        """Calcula taxa de acerto do cache"""
        total_access = sum(c.access_count for c in self.cache.values())
        return total_access / max(len(self.cache), 1)

class BarrierValidator:
    """Validador de barriers para diferentes tipos de contrato"""

    @staticmethod
    def validate_barrier(contract_type: str, barrier: Optional[str], current_price: Optional[float] = None) -> Tuple[bool, Optional[str]]:
        """
        Valida se a barrier é apropriada para o tipo de contrato
        Returns: (is_valid, error_message)
        """
        contract_type = contract_type.upper()

        # Contratos que não precisam de barrier
        if contract_type in ['CALL', 'PUT', 'DIGITEVEN', 'DIGITODD']:
            if barrier is not None:
                return False, f"Contrato {contract_type} não aceita barrier"
            return True, None

        # Contratos digit que precisam de barrier
        if contract_type in ['DIGITOVER', 'DIGITUNDER']:
            if barrier is None:
                return False, f"Contrato {contract_type} requer barrier"

            # Barrier deve ser um dígito de 0-9
            try:
                barrier_int = int(barrier)
                if barrier_int < 0 or barrier_int > 9:
                    return False, "Barrier deve ser um dígito entre 0 e 9"
            except ValueError:
                return False, "Barrier deve ser um número válido"

            return True, None

        # Contratos touch que precisam de barrier de preço
        if contract_type in ['ONETOUCH', 'NOTOUCH']:
            if barrier is None:
                return False, f"Contrato {contract_type} requer barrier de preço"

            try:
                barrier_float = float(barrier)
                if current_price and barrier_float == current_price:
                    return False, "Barrier não pode ser igual ao preço atual"
            except ValueError:
                return False, "Barrier deve ser um preço válido"

            return True, None

        return False, f"Tipo de contrato {contract_type} não suportado"

class ProposalBatchProcessor:
    """Processador de proposals em lote para otimização"""

    def __init__(self, batch_size: int = 10, batch_timeout: float = 0.1):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.pending_requests: List[Tuple[ProposalRequest, asyncio.Future]] = []
        self.processing_lock = asyncio.Lock()

    async def add_request(self, request: ProposalRequest, deriv_api) -> ProposalResponse:
        """Adiciona request ao lote para processamento"""
        future = asyncio.Future()

        async with self.processing_lock:
            self.pending_requests.append((request, future))

            # Processa lote se atingiu o limite ou timeout
            if len(self.pending_requests) >= self.batch_size:
                await self._process_batch(deriv_api)

        # Inicia timer para processar lote por timeout
        if len(self.pending_requests) == 1:
            asyncio.create_task(self._timeout_processor(deriv_api))

        return await future

    async def _timeout_processor(self, deriv_api):
        """Processa lote por timeout"""
        await asyncio.sleep(self.batch_timeout)

        async with self.processing_lock:
            if self.pending_requests:
                await self._process_batch(deriv_api)

    async def _process_batch(self, deriv_api):
        """Processa lote de requests"""
        if not self.pending_requests:
            return

        current_batch = self.pending_requests.copy()
        self.pending_requests.clear()

        # Processa requests em paralelo
        tasks = []
        for request, future in current_batch:
            task = asyncio.create_task(self._single_proposal(deriv_api, request, future))
            tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _single_proposal(self, deriv_api, request: ProposalRequest, future: asyncio.Future):
        """Processa uma única proposal"""
        try:
            # Chama API da Deriv para obter proposal
            response = await deriv_api.get_proposal(
                contract_type=request.contract_type,
                symbol=request.symbol,
                amount=request.amount,
                duration=request.duration,
                duration_unit=request.duration_unit,
                barrier=request.barrier,
                basis=request.basis,
                currency=request.currency
            )

            # Converte resposta para ProposalResponse
            proposal_response = ProposalResponse(
                id=response.get('id', ''),
                ask_price=response.get('ask_price', 0.0),
                payout=response.get('payout', 0.0),
                spot=response.get('spot'),
                barrier=request.barrier,
                contract_type=request.contract_type,
                symbol=request.symbol,
                display_value=response.get('display_value', ''),
                valid_until=time.time() + 30  # Válida por 30 segundos
            )

            future.set_result(proposal_response)

        except Exception as e:
            future.set_exception(e)

class ContractProposalsEngine:
    """Engine principal para gerenciamento de proposals de contratos"""

    def __init__(self, deriv_api, cache_size: int = 1000, cache_ttl: float = 30.0):
        self.deriv_api = deriv_api
        self.cache = ProposalCache(max_size=cache_size, ttl=cache_ttl)
        self.validator = BarrierValidator()
        self.batch_processor = ProposalBatchProcessor()
        self.active_subscriptions: Set[str] = set()
        self.price_cache: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.running = False

        # Estatísticas
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "validation_errors": 0,
            "api_errors": 0
        }

    async def start(self):
        """Inicia o engine"""
        self.running = True
        # Inicia limpeza periódica do cache
        asyncio.create_task(self._periodic_cleanup())
        logger.info("Contract Proposals Engine iniciado")

    async def stop(self):
        """Para o engine"""
        self.running = False
        logger.info("Contract Proposals Engine parado")

    async def get_proposal(self, request: ProposalRequest) -> ProposalResponse:
        """
        Obtém proposal com cálculo em tempo real
        Inclui validação, cache e otimização de performance
        """
        self.stats["total_requests"] += 1

        try:
            # 1. Validação de barrier
            is_valid, error_msg = self.validator.validate_barrier(
                request.contract_type,
                request.barrier,
                self._get_current_price(request.symbol)
            )

            if not is_valid:
                self.stats["validation_errors"] += 1
                raise ValueError(f"Erro de validação: {error_msg}")

            # 2. Verificar cache primeiro
            cached_response = self.cache.get(request)
            if cached_response:
                self.stats["cache_hits"] += 1
                # Atualiza preço em tempo real se disponível
                current_price = self._get_current_price(request.symbol)
                if current_price:
                    cached_response.spot = current_price
                return cached_response

            self.stats["cache_misses"] += 1

            # 3. Processar via batch para otimização
            response = await self.batch_processor.add_request(request, self.deriv_api)

            # 4. Armazenar no cache
            self.cache.put(request, response)

            return response

        except Exception as e:
            self.stats["api_errors"] += 1
            logger.error(f"Erro ao obter proposal: {e}")
            raise

    async def get_realtime_proposal(self, request: ProposalRequest) -> ProposalResponse:
        """
        Obtém proposal com preços em tempo real garantido
        Força nova requisição sem usar cache
        """
        self.stats["total_requests"] += 1

        # Validação obrigatória
        is_valid, error_msg = self.validator.validate_barrier(
            request.contract_type,
            request.barrier,
            self._get_current_price(request.symbol)
        )

        if not is_valid:
            self.stats["validation_errors"] += 1
            raise ValueError(f"Erro de validação: {error_msg}")

        try:
            # Força nova requisição para preços atualizados
            response = await self.batch_processor.add_request(request, self.deriv_api)

            # Atualiza cache com nova resposta
            self.cache.put(request, response)

            return response

        except Exception as e:
            self.stats["api_errors"] += 1
            logger.error(f"Erro ao obter proposal em tempo real: {e}")
            raise

    async def get_multiple_proposals(self, requests: List[ProposalRequest]) -> List[ProposalResponse]:
        """
        Obtém múltiplas proposals de forma otimizada
        Usa processamento em lote para melhor performance
        """
        tasks = []
        for request in requests:
            task = asyncio.create_task(self.get_proposal(request))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filtra exceções e retorna apenas resultados válidos
        valid_results = []
        for result in results:
            if isinstance(result, ProposalResponse):
                valid_results.append(result)
            else:
                logger.error(f"Erro em proposal: {result}")

        return valid_results

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Obtém preço atual do símbolo do cache de preços"""
        price_data = self.price_cache.get(symbol, {})
        return price_data.get('price')

    def update_price(self, symbol: str, price: float, timestamp: float = None):
        """Atualiza preço atual de um símbolo"""
        self.price_cache[symbol] = {
            'price': price,
            'timestamp': timestamp or time.time()
        }

    async def _periodic_cleanup(self):
        """Limpeza periódica do cache"""
        while self.running:
            try:
                await asyncio.sleep(60)  # Executa a cada minuto
                self.cache.clear_expired()

                # Remove preços antigos (mais de 5 minutos)
                current_time = time.time()
                expired_symbols = [
                    symbol for symbol, data in self.price_cache.items()
                    if current_time - data.get('timestamp', 0) > 300
                ]

                for symbol in expired_symbols:
                    del self.price_cache[symbol]

            except Exception as e:
                logger.error(f"Erro na limpeza periódica: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do engine"""
        cache_stats = self.cache.get_stats()

        return {
            **self.stats,
            "cache": cache_stats,
            "price_cache_symbols": len(self.price_cache),
            "hit_rate": self.stats["cache_hits"] / max(self.stats["total_requests"], 1),
            "error_rate": (self.stats["validation_errors"] + self.stats["api_errors"]) / max(self.stats["total_requests"], 1)
        }

    def reset_stats(self):
        """Reseta estatísticas"""
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "validation_errors": 0,
            "api_errors": 0
        }

# Factory function para criar engine singleton
_engine_instance: Optional[ContractProposalsEngine] = None

def get_proposals_engine(deriv_api) -> ContractProposalsEngine:
    """Retorna instância singleton do engine"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ContractProposalsEngine(deriv_api)
    return _engine_instance

async def initialize_proposals_engine(deriv_api):
    """Inicializa o engine de proposals"""
    engine = get_proposals_engine(deriv_api)
    await engine.start()
    return engine