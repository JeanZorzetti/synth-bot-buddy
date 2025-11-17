#!/usr/bin/env python3
"""
Deriv Trading Adapter
Adapta o DerivAPI para funcionar com o sistema de trading existente
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass

from deriv_api_legacy import DerivAPI, DerivAPIStatus, DerivTick, DerivBalance, DerivContract
from enhanced_websocket_manager import ConnectionState

logger = logging.getLogger(__name__)

@dataclass
class AdaptedTick:
    """Tick adaptado para compatibilidade com sistema existente"""
    symbol: str
    price: float
    timestamp: int
    bid: Optional[float] = None
    ask: Optional[float] = None

class DerivTradingAdapter:
    """Adapter que conecta DerivAPI com o sistema de trading existente"""
    
    def __init__(self, app_id: int = 1089, demo: bool = True):
        self.deriv_api = DerivAPI(app_id=app_id, demo=demo)
        self.is_connected = False
        self.is_authenticated = False
        self.current_balance = 0.0
        self.tick_handlers = []
        self.balance_handlers = []
        self.portfolio_handlers = []
        
        # Cache de dados
        self.active_symbols_cache = {}
        self.last_ticks = {}
        self.subscribed_symbols = set()
        
        # Configurar handlers do DerivAPI
        self.deriv_api.on_tick(self._handle_tick_update)
        self.deriv_api.on_balance_update(self._handle_balance_update)
    
    async def connect(self) -> bool:
        """Conectar ao Deriv"""
        try:
            success = await self.deriv_api.connect()
            if success:
                self.is_connected = True
                logger.info("DerivTradingAdapter conectado com sucesso")
                
                # Símbolos ativos serão carregados após autenticação
            
            return success
        except Exception as e:
            logger.error(f"Erro ao conectar DerivTradingAdapter: {e}")
            return False
    
    async def authenticate(self, token: str) -> bool:
        """Autenticar com token da API"""
        try:
            if not self.is_connected:
                await self.connect()
            
            response = await self.deriv_api.authorize(token)
            if response and 'authorize' in response:
                self.is_authenticated = True
                logger.info("DerivTradingAdapter autenticado com sucesso")
                
                # Carregar símbolos ativos após autenticação
                await self._load_active_symbols()
                
                # Obter saldo inicial
                await self._update_balance()
                return True
            
            return False
        except Exception as e:
            logger.error(f"Erro ao autenticar: {e}")
            return False
    
    async def disconnect(self):
        """Desconectar do Deriv"""
        try:
            await self.deriv_api.disconnect()
            self.is_connected = False
            self.is_authenticated = False
            logger.info("DerivTradingAdapter desconectado")
        except Exception as e:
            logger.error(f"Erro ao desconectar: {e}")
    
    # =====================================================
    # COMPATIBILIDADE COM ENHANCED_WEBSOCKET_MANAGER
    # =====================================================
    
    @property
    def connection_state(self) -> ConnectionState:
        """Estado da conexão compatível com sistema existente"""
        if self.deriv_api.status == DerivAPIStatus.AUTHENTICATED:
            return ConnectionState.CONNECTED
        elif self.deriv_api.status == DerivAPIStatus.CONNECTED:
            return ConnectionState.CONNECTING
        elif self.deriv_api.status == DerivAPIStatus.CONNECTING:
            return ConnectionState.CONNECTING
        else:
            return ConnectionState.DISCONNECTED
    
    async def subscribe_to_ticks(self, symbol: str) -> bool:
        """Subscrever a ticks de um símbolo"""
        try:
            if not self.is_connected:
                return False
            
            response = await self.deriv_api.ticks(symbol, subscribe=True)
            if response and 'tick' in response:
                self.subscribed_symbols.add(symbol)
                
                # Processar primeiro tick
                tick_data = response['tick']
                adapted_tick = AdaptedTick(
                    symbol=tick_data['symbol'],
                    price=tick_data['quote'],
                    timestamp=tick_data['epoch']
                )
                self.last_ticks[symbol] = adapted_tick
                
                logger.info(f"Subscrito a ticks de {symbol}")
                return True
            
            return False
        except Exception as e:
            logger.error(f"Erro ao subscrever a ticks de {symbol}: {e}")
            return False
    
    async def unsubscribe_from_ticks(self, symbol: str) -> bool:
        """Cancelar subscrição de ticks"""
        try:
            # O Deriv API cancelará automaticamente quando a conexão for fechada
            # Ou podemos usar forget_all()
            if symbol in self.subscribed_symbols:
                self.subscribed_symbols.remove(symbol)
            return True
        except Exception as e:
            logger.error(f"Erro ao cancelar subscrição de {symbol}: {e}")
            return False
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Informações da conexão"""
        return {
            'is_connected': self.is_connected,
            'is_authenticated': self.is_authenticated,
            'status': self.deriv_api.status.value,
            'loginid': self.deriv_api.loginid,
            'subscribed_symbols': list(self.subscribed_symbols),
            'balance': self.current_balance
        }
    
    # =====================================================
    # OPERAÇÕES DE TRADING
    # =====================================================
    
    async def place_trade(self, contract_type: str, symbol: str, amount: float,
                         duration: int, duration_unit: str = "m", barrier: Optional[str] = None,
                         basis: str = "stake", currency: str = "USD") -> Dict[str, Any]:
        """Realizar uma operação de trade"""
        try:
            if not self.is_authenticated:
                raise Exception("Não autenticado")

            # Validar se símbolo está disponível
            if not await self._is_symbol_available(symbol):
                raise Exception(f"Símbolo {symbol} não está disponível")

            # Validar parâmetros
            if amount <= 0:
                raise Exception("Amount deve ser maior que zero")

            if duration <= 0:
                raise Exception("Duration deve ser maior que zero")

            # Executar compra
            response = await self.deriv_api.buy(
                contract_type=contract_type,
                symbol=symbol,
                amount=amount,
                duration=duration,
                duration_unit=duration_unit,
                barrier=barrier,
                basis=basis,
                currency=currency,
                subscribe=True
            )
            
            if response and 'buy' in response:
                buy_data = response['buy']
                logger.info(f"Trade executado - ID: {buy_data.get('contract_id')}")
                
                # Atualizar saldo
                await self._update_balance()
                
                return {
                    'success': True,
                    'contract_id': buy_data.get('contract_id'),
                    'buy_price': buy_data.get('buy_price'),
                    'payout': buy_data.get('payout'),
                    'longcode': buy_data.get('longcode')
                }
            
            return {'success': False, 'error': 'Resposta inválida da API'}
            
        except Exception as e:
            logger.error(f"Erro ao executar trade: {e}")
            return {'success': False, 'error': str(e)}
    
    async def close_position(self, contract_id: int, price: Optional[float] = None) -> Dict[str, Any]:
        """Fechar posição"""
        try:
            response = await self.deriv_api.sell(contract_id, price)
            
            if response and 'sell' in response:
                sell_data = response['sell']
                logger.info(f"Posição fechada - ID: {contract_id}")
                
                # Atualizar saldo
                await self._update_balance()
                
                return {
                    'success': True,
                    'sold_for': sell_data.get('sold_for'),
                    'transaction_id': sell_data.get('transaction_id')
                }
            
            return {'success': False, 'error': 'Falha ao vender contrato'}
            
        except Exception as e:
            logger.error(f"Erro ao fechar posição {contract_id}: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_portfolio(self) -> List[Dict[str, Any]]:
        """Obter portfólio de posições abertas"""
        try:
            response = await self.deriv_api.portfolio()
            
            if response and 'portfolio' in response:
                contracts = response['portfolio']['contracts']
                
                portfolio = []
                for contract in contracts:
                    portfolio.append({
                        'contract_id': contract.get('contract_id'),
                        'symbol': contract.get('symbol'),
                        'contract_type': contract.get('contract_type'),
                        'buy_price': contract.get('buy_price'),
                        'current_spot': contract.get('current_spot'),
                        'profit': contract.get('profit'),
                        'payout': contract.get('payout'),
                        'is_sold': contract.get('is_sold', False),
                        'longcode': contract.get('longcode'),
                        'date_start': contract.get('date_start'),
                        'date_expiry': contract.get('date_expiry')
                    })
                
                return portfolio
            
            return []
            
        except Exception as e:
            logger.error(f"Erro ao obter portfólio: {e}")
            return []
    
    async def get_trade_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Obter histórico de trades"""
        try:
            response = await self.deriv_api.profit_table(limit=limit)
            
            if response and 'profit_table' in response:
                transactions = response['profit_table']['transactions']
                
                history = []
                for tx in transactions:
                    history.append({
                        'transaction_id': tx.get('transaction_id'),
                        'contract_id': tx.get('contract_id'),
                        'symbol': tx.get('symbol'),
                        'contract_type': tx.get('contract_type'),
                        'buy_price': tx.get('buy_price'),
                        'sell_price': tx.get('sell_price'),
                        'profit': tx.get('profit'),
                        'duration': tx.get('duration'),
                        'purchase_time': tx.get('purchase_time'),
                        'sell_time': tx.get('sell_time'),
                        'longcode': tx.get('longcode')
                    })
                
                return history
            
            return []
            
        except Exception as e:
            logger.error(f"Erro ao obter histórico de trades: {e}")
            return []
    
    async def get_balance(self) -> float:
        """Obter saldo atual"""
        try:
            await self._update_balance()
            return self.current_balance
        except Exception as e:
            logger.error(f"Erro ao obter saldo: {e}")
            return 0.0
    
    def get_last_tick(self, symbol: str) -> Optional[AdaptedTick]:
        """Obter último tick de um símbolo"""
        return self.last_ticks.get(symbol)
    
    # =====================================================
    # HANDLERS E CALLBACKS
    # =====================================================
    
    async def _handle_tick_update(self, data: Dict[str, Any]):
        """Processar atualizações de tick"""
        try:
            if 'tick' not in data:
                return
            
            tick_data = data['tick']
            symbol = tick_data['symbol']
            
            adapted_tick = AdaptedTick(
                symbol=symbol,
                price=tick_data['quote'],
                timestamp=tick_data['epoch']
            )
            
            self.last_ticks[symbol] = adapted_tick
            
            # Notificar handlers
            for handler in self.tick_handlers:
                try:
                    await handler(adapted_tick)
                except Exception as e:
                    logger.error(f"Erro em tick handler: {e}")
                    
        except Exception as e:
            logger.error(f"Erro ao processar tick: {e}")
    
    async def _handle_balance_update(self, data: Dict[str, Any]):
        """Processar atualizações de saldo"""
        try:
            if 'balance' in data:
                balance_data = data['balance']
                self.current_balance = balance_data.get('balance', 0.0)
                
                # Notificar handlers
                for handler in self.balance_handlers:
                    try:
                        await handler(self.current_balance)
                    except Exception as e:
                        logger.error(f"Erro em balance handler: {e}")
                        
        except Exception as e:
            logger.error(f"Erro ao processar balance: {e}")
    
    def add_tick_handler(self, handler: Callable):
        """Adicionar handler para ticks"""
        self.tick_handlers.append(handler)
    
    def add_balance_handler(self, handler: Callable):
        """Adicionar handler para saldo"""
        self.balance_handlers.append(handler)
    
    # =====================================================
    # MÉTODOS AUXILIARES
    # =====================================================
    
    async def _load_active_symbols(self):
        """Carregar símbolos ativos"""
        try:
            response = await self.deriv_api.active_symbols()
            
            if response and 'active_symbols' in response:
                for symbol_data in response['active_symbols']:
                    symbol = symbol_data['symbol']
                    self.active_symbols_cache[symbol] = symbol_data
                
                logger.info(f"Carregados {len(self.active_symbols_cache)} símbolos ativos")
                
        except Exception as e:
            logger.error(f"Erro ao carregar símbolos ativos: {e}")
    
    async def _update_balance(self):
        """Atualizar saldo atual"""
        try:
            response = await self.deriv_api.balance()
            
            if response and 'balance' in response:
                balance_data = response['balance']
                self.current_balance = balance_data.get('balance', 0.0)
                
        except Exception as e:
            logger.error(f"Erro ao atualizar saldo: {e}")
    
    async def _is_symbol_available(self, symbol: str) -> bool:
        """Verificar se símbolo está disponível para trading"""
        try:
            if not self.active_symbols_cache:
                await self._load_active_symbols()
            
            symbol_info = self.active_symbols_cache.get(symbol)
            if symbol_info:
                return not symbol_info.get('is_trading_suspended', True)
            
            return False
            
        except Exception as e:
            logger.error(f"Erro ao verificar disponibilidade do símbolo {symbol}: {e}")
            return False
    
    async def get_available_symbols(self) -> List[str]:
        """Obter lista de símbolos disponíveis"""
        try:
            if not self.active_symbols_cache:
                await self._load_active_symbols()
            
            available_symbols = []
            for symbol, info in self.active_symbols_cache.items():
                if not info.get('is_trading_suspended', True):
                    available_symbols.append(symbol)
            
            return available_symbols
            
        except Exception as e:
            logger.error(f"Erro ao obter símbolos disponíveis: {e}")
            return []
    
    async def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Obter informações detalhadas de um símbolo"""
        try:
            if not self.active_symbols_cache:
                await self._load_active_symbols()
            
            return self.active_symbols_cache.get(symbol, {})
            
        except Exception as e:
            logger.error(f"Erro ao obter informações do símbolo {symbol}: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificação de saúde da conexão"""
        try:
            # Fazer ping para verificar conectividade
            ping_response = await self.deriv_api.ping()
            
            # Obter status do website
            status_response = await self.deriv_api.website_status()
            
            return {
                'connection_status': self.deriv_api.status.value,
                'is_connected': self.is_connected,
                'is_authenticated': self.is_authenticated,
                'ping_success': 'ping' in ping_response,
                'website_status': status_response.get('website_status', {}),
                'subscribed_symbols_count': len(self.subscribed_symbols),
                'balance': self.current_balance
            }
            
        except Exception as e:
            logger.error(f"Erro no health check: {e}")
            return {
                'connection_status': 'error',
                'is_connected': False,
                'is_authenticated': False,
                'error': str(e)
            }