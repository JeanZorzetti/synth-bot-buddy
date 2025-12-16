#!/usr/bin/env python3
"""
Deriv API WebSocket Client
Implementa as 16 funcionalidades essenciais da API do Deriv para operação real
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import websockets
import ujson
from websockets.exceptions import ConnectionClosed, WebSocketException

logger = logging.getLogger(__name__)

class DerivAPIStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    ERROR = "error"

@dataclass
class DerivSymbol:
    symbol: str
    display_name: str
    market: str
    submarket: str
    symbol_type: str
    is_trading_suspended: bool
    pip: float
    decimal_places: int

@dataclass
class DerivTick:
    epoch: int
    quote: float
    symbol: str
    timestamp: datetime

@dataclass
class DerivBalance:
    currency: str
    balance: float
    loginid: str

@dataclass
class DerivContract:
    contract_id: int
    longcode: str
    contract_type: str
    currency: str
    date_start: int
    date_expiry: int
    barrier: Optional[str]
    buy_price: float
    payout: float
    profit: float
    current_spot: Optional[float]
    is_sold: bool
    sold_for: Optional[float]

class DerivAPI:
    """Cliente WebSocket para API do Deriv com as 16 funcionalidades essenciais"""
    
    def __init__(self, app_id: int = 1089, endpoint: str = "wss://ws.derivws.com/websockets/v3", demo: bool = True):
        self.app_id = app_id
        self.endpoint = f"{endpoint}?app_id={app_id}" if demo else f"{endpoint}?app_id={app_id}"
        self.websocket = None
        self.status = DerivAPIStatus.DISCONNECTED
        self.auth_token = None
        self.loginid = None
        self.subscriptions = {}
        self.request_id_counter = 0
        self.pending_requests = {}
        self.event_handlers = {}
        
        # Controle de reconexão
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 5
        self.reconnect_attempts = 0
        
    def _generate_request_id(self) -> int:
        """Gera ID único para requisições"""
        self.request_id_counter += 1
        return self.request_id_counter
    
    async def connect(self) -> bool:
        """Conecta ao WebSocket do Deriv"""
        try:
            self.status = DerivAPIStatus.CONNECTING
            logger.info(f"Conectando ao Deriv API: {self.endpoint}")
            
            self.websocket = await websockets.connect(
                self.endpoint,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )
            
            self.status = DerivAPIStatus.CONNECTED
            logger.info("Conectado ao Deriv API com sucesso")
            
            # Inicia task de escuta de mensagens
            asyncio.create_task(self._listen_messages())
            
            # Reset contador de reconexão
            self.reconnect_attempts = 0
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao conectar com Deriv API: {e}")
            self.status = DerivAPIStatus.ERROR
            return False
    
    async def disconnect(self):
        """Desconecta do WebSocket"""
        try:
            if self.websocket:
                await self.websocket.close()
            self.status = DerivAPIStatus.DISCONNECTED
            logger.info("Desconectado do Deriv API")
        except Exception as e:
            logger.error(f"Erro ao desconectar: {e}")
    
    async def _listen_messages(self):
        """Escuta mensagens do WebSocket"""
        try:
            async for message in self.websocket:
                try:
                    data = ujson.loads(message)
                    await self._handle_message(data)
                except ujson.JSONDecodeError as e:
                    logger.error(f"Erro ao decodificar JSON: {e}")
                except Exception as e:
                    logger.error(f"Erro ao processar mensagem: {e}")
                    
        except ConnectionClosed:
            logger.warning("Conexão WebSocket fechada")
            await self._handle_reconnection()
        except Exception as e:
            logger.error(f"Erro na escuta de mensagens: {e}")
            await self._handle_reconnection()
    
    async def _handle_message(self, data: Dict[str, Any]):
        """Processa mensagens recebidas"""
        msg_type = data.get('msg_type')
        req_id = data.get('req_id')
        
        # Se é resposta a uma requisição pendente
        if req_id and req_id in self.pending_requests:
            future = self.pending_requests.pop(req_id)
            if not future.cancelled():
                future.set_result(data)
        
        # Se é uma subscrição (tick, balance, etc)
        elif msg_type in self.subscriptions:
            handlers = self.subscriptions[msg_type]
            for handler in handlers:
                try:
                    await handler(data)
                except Exception as e:
                    logger.error(f"Erro no handler de {msg_type}: {e}")
        
        # Handler genérico por tipo de mensagem
        if msg_type in self.event_handlers:
            handler = self.event_handlers[msg_type]
            try:
                await handler(data)
            except Exception as e:
                logger.error(f"Erro no handler genérico de {msg_type}: {e}")
    
    async def _handle_reconnection(self):
        """Tenta reconectar automaticamente"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Máximo de tentativas de reconexão atingido")
            self.status = DerivAPIStatus.ERROR
            return
        
        self.reconnect_attempts += 1
        logger.info(f"Tentativa de reconexão {self.reconnect_attempts}/{self.max_reconnect_attempts}")
        
        await asyncio.sleep(self.reconnect_delay)
        
        if await self.connect():
            # Reautenticar se necessário
            if self.auth_token:
                await self.authorize(self.auth_token)
    
    async def _send_request(self, request: Dict[str, Any], expect_response: bool = True) -> Optional[Dict[str, Any]]:
        """Envia requisição e aguarda resposta"""
        if self.status not in [DerivAPIStatus.CONNECTED, DerivAPIStatus.AUTHENTICATED]:
            raise Exception("WebSocket não conectado")
        
        req_id = self._generate_request_id()
        request['req_id'] = req_id
        
        if expect_response:
            future = asyncio.Future()
            self.pending_requests[req_id] = future
        
        try:
            message = ujson.dumps(request)
            await self.websocket.send(message)
            logger.debug(f"Enviado: {message}")
            
            if expect_response:
                # Timeout de 30 segundos para resposta
                response = await asyncio.wait_for(future, timeout=30.0)
                
                # Verificar se houve erro na resposta
                if 'error' in response:
                    error_info = response['error']
                    raise Exception(f"Deriv API Error: {error_info.get('message', 'Unknown error')}")
                
                return response
            return None
            
        except asyncio.TimeoutError:
            self.pending_requests.pop(req_id, None)
            raise Exception("Timeout ao aguardar resposta da API")
        except Exception as e:
            self.pending_requests.pop(req_id, None)
            raise e
    
    # =====================================================
    # FUNCIONALIDADES ESSENCIAIS DA API (16 FUNÇÕES)
    # =====================================================
    
    async def ping(self) -> Dict[str, Any]:
        """1. Ping - Manter conexão ativa"""
        request = {"ping": 1}
        return await self._send_request(request)
    
    async def time(self) -> Dict[str, Any]:
        """2. Time - Obter horário do servidor"""
        request = {"time": 1}
        return await self._send_request(request)
    
    async def website_status(self) -> Dict[str, Any]:
        """3. Website Status - Status do sistema"""
        request = {"website_status": 1}
        return await self._send_request(request)
    
    async def authorize(self, token: str) -> Dict[str, Any]:
        """4. Authorize - Autenticar sessão"""
        request = {"authorize": token}
        response = await self._send_request(request)
        
        if response and 'authorize' in response:
            self.auth_token = token
            self.loginid = response['authorize'].get('loginid')
            self.status = DerivAPIStatus.AUTHENTICATED
            logger.info(f"Autenticado com sucesso - LoginID: {self.loginid}")
        
        return response
    
    async def balance(self, subscribe: bool = False) -> Dict[str, Any]:
        """5. Balance - Obter saldo da conta"""
        request = {"balance": 1}
        if subscribe:
            request["subscribe"] = 1
            # Adicionar handler para atualizações de saldo
            if 'balance' not in self.subscriptions:
                self.subscriptions['balance'] = []
        
        return await self._send_request(request)
    
    async def get_limits(self) -> Dict[str, Any]:
        """6. Get Limits - Limites de trading"""
        request = {"get_limits": 1}
        return await self._send_request(request)
    
    async def get_settings(self) -> Dict[str, Any]:
        """7. Get Settings - Configurações da conta"""
        request = {"get_settings": 1}
        return await self._send_request(request)
    
    async def statement(self, limit: int = 50) -> Dict[str, Any]:
        """8. Statement - Histórico de transações"""
        request = {
            "statement": 1,
            "limit": limit
        }
        return await self._send_request(request)
    
    async def active_symbols(self, product_type: str = "basic") -> Dict[str, Any]:
        """9. Active Symbols - Símbolos disponíveis"""
        request = {
            "active_symbols": product_type
        }
        return await self._send_request(request)
    
    async def ticks(self, symbol: str, subscribe: bool = True) -> Dict[str, Any]:
        """10. Ticks - Preços em tempo real"""
        request = {"ticks": symbol}
        if subscribe:
            request["subscribe"] = 1
            # Adicionar handler para ticks
            if 'tick' not in self.subscriptions:
                self.subscriptions['tick'] = []

        return await self._send_request(request)

    async def get_latest_tick(self, symbol: str) -> Dict[str, Any]:
        """
        Obter último tick sem criar subscrição
        Usa ticks_history com count=1 para evitar problemas de subscrição duplicada
        """
        request = {
            "ticks_history": symbol,
            "count": 1,
            "end": "latest",
            "style": "ticks"
        }
        return await self._send_request(request)
    
    async def trading_durations(self, symbol: str, contract_type: str = "CALL") -> Dict[str, Any]:
        """11. Trading Durations - Durações disponíveis"""
        request = {
            "trading_durations": 1,
            "symbol": symbol,
            "contract_type": contract_type
        }
        return await self._send_request(request)
    
    async def get_proposal(self, contract_type: str, symbol: str, amount: float,
                          duration: int, duration_unit: str = "m", barrier: Optional[str] = None,
                          basis: str = "stake", currency: str = "USD") -> Dict[str, Any]:
        """11. Proposal - Obter cotação para contrato"""
        request = {
            "proposal": 1,
            "amount": amount,
            "basis": basis,
            "contract_type": contract_type,
            "currency": currency,
            "duration": duration,
            "duration_unit": duration_unit,
            "symbol": symbol
        }

        if barrier:
            request["barrier"] = barrier

        response = await self._send_request(request)

        # Retorna apenas os dados da proposal
        return response.get('proposal', response)

    async def buy(self, contract_type: str, symbol: str, amount: float,
                  duration: int, duration_unit: str = "m", barrier: Optional[str] = None,
                  basis: str = "stake", currency: str = "USD",
                  subscribe: bool = True) -> Dict[str, Any]:
        """12. Buy - Comprar contratos"""
        request = {
            "buy": 1,
            "price": amount * 2,  # Max price willing to pay (payout potential)
            "parameters": {
                "amount": amount,
                "basis": basis,
                "contract_type": contract_type,
                "currency": currency,
                "symbol": symbol,
                "duration": duration,
                "duration_unit": duration_unit
            }
        }

        # Adicionar subscrição se solicitado
        if subscribe:
            request["subscribe"] = 1

        # Adicionar barreira se especificada
        if barrier:
            request["parameters"]["barrier"] = barrier

        return await self._send_request(request)
    
    async def sell(self, contract_id: int, price: Optional[float] = None) -> Dict[str, Any]:
        """13. Sell - Vender contratos"""
        request = {"sell": contract_id}
        if price:
            request["price"] = price
        
        return await self._send_request(request)
    
    async def portfolio(self) -> Dict[str, Any]:
        """14. Portfolio - Posições abertas"""
        request = {"portfolio": 1}
        return await self._send_request(request)
    
    async def profit_table(self, limit: int = 50) -> Dict[str, Any]:
        """15. Profit Table - Histórico de lucros"""
        request = {
            "profit_table": 1,
            "limit": limit
        }
        return await self._send_request(request)
    
    async def forget(self, subscription_id: str) -> Dict[str, Any]:
        """16. Forget - Cancelar subscrição específica"""
        request = {"forget": subscription_id}
        return await self._send_request(request)
    
    async def forget_all(self, types: List[str] = None) -> Dict[str, Any]:
        """16. Forget All - Cancelar todas as subscrições"""
        request = {"forget_all": types or ["ticks", "balance", "portfolio"]}
        return await self._send_request(request)

    async def switch_account(self, loginid: str) -> Dict[str, Any]:
        """Switch to a different account using the same token

        Deriv API doesn't support direct account switching via WebSocket.
        The workaround is to:
        1. Disconnect current session
        2. Reconnect
        3. Authorize with token (will use default account)

        For now, we'll inform the user they need a token from the Demo account.
        """
        raise NotImplementedError(
            "Account switching is not supported by Deriv WebSocket API.\n"
            "To use a Demo account, generate a new token while logged into the Demo account:\n"
            "1. Go to https://app.deriv.com/\n"
            f"2. Switch to Demo account ({loginid})\n"
            "3. Generate a new API token at https://app.deriv.com/account/api-token\n"
            "4. Use the new token in this script"
        )

    # =====================================================
    # MÉTODOS AUXILIARES E HANDLERS
    # =====================================================
    
    def on_tick(self, handler: Callable):
        """Adicionar handler para ticks"""
        if 'tick' not in self.subscriptions:
            self.subscriptions['tick'] = []
        self.subscriptions['tick'].append(handler)
    
    def on_balance_update(self, handler: Callable):
        """Adicionar handler para atualizações de saldo"""
        if 'balance' not in self.subscriptions:
            self.subscriptions['balance'] = []
        self.subscriptions['balance'].append(handler)
    
    def on_message(self, msg_type: str, handler: Callable):
        """Adicionar handler genérico para tipo de mensagem"""
        self.event_handlers[msg_type] = handler
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Obter dados completos do mercado para um símbolo"""
        try:
            # Obter informações do símbolo
            symbols_data = await self.active_symbols()
            symbol_info = None
            
            if 'active_symbols' in symbols_data:
                for s in symbols_data['active_symbols']:
                    if s['symbol'] == symbol:
                        symbol_info = s
                        break
            
            # Obter tick atual
            tick_data = await self.ticks(symbol, subscribe=False)
            
            # Obter durações de trading
            durations_data = await self.trading_durations(symbol)
            
            return {
                'symbol_info': symbol_info,
                'current_tick': tick_data.get('tick'),
                'trading_durations': durations_data.get('trading_durations')
            }
            
        except Exception as e:
            logger.error(f"Erro ao obter dados do mercado para {symbol}: {e}")
            return {}
    
    async def is_market_open(self, symbol: str) -> bool:
        """Verificar se o mercado está aberto para um símbolo"""
        try:
            symbols_data = await self.active_symbols()
            if 'active_symbols' in symbols_data:
                for s in symbols_data['active_symbols']:
                    if s['symbol'] == symbol:
                        return not s.get('is_trading_suspended', True)
            return False
        except Exception as e:
            logger.error(f"Erro ao verificar status do mercado para {symbol}: {e}")
            return False
    
    async def get_account_summary(self) -> Dict[str, Any]:
        """Obter resumo completo da conta"""
        try:
            balance_data = await self.balance()
            limits_data = await self.get_limits()
            settings_data = await self.get_settings()
            portfolio_data = await self.portfolio()
            
            return {
                'balance': balance_data.get('balance'),
                'limits': limits_data.get('get_limits'),
                'settings': settings_data.get('get_settings'),
                'portfolio': portfolio_data.get('portfolio'),
                'loginid': self.loginid
            }
        except Exception as e:
            logger.error(f"Erro ao obter resumo da conta: {e}")
            return {}