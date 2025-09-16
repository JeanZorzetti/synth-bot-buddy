"""
🤖 AUTONOMOUS TRADING ENGINE
Módulo responsável pela tomada de decisões autônomas de trading
Integra com IA/ML para executar trades sem intervenção humana
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json

from prediction_engine import PredictionEngine, TradingSignal
from tick_data_collector import TickData


class TradeAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


@dataclass
class TradingDecision:
    """Decisão de trading gerada pela IA"""
    action: TradeAction
    symbol: str
    confidence: float
    signal_strength: float
    position_size: float
    risk_level: RiskLevel
    reasoning: str
    timestamp: datetime
    expected_holding_period: int  # segundos
    target_profit: float
    stop_loss: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Position:
    """Posição aberta no mercado"""
    id: str
    symbol: str
    action: TradeAction
    entry_price: float
    position_size: float
    entry_time: datetime
    stop_loss: float
    target_profit: float
    current_pnl: float = 0.0
    is_open: bool = True

    def update_pnl(self, current_price: float):
        """Atualiza P&L da posição"""
        if self.action == TradeAction.BUY:
            self.current_pnl = (current_price - self.entry_price) * self.position_size
        else:  # SELL
            self.current_pnl = (self.entry_price - current_price) * self.position_size


@dataclass
class PortfolioState:
    """Estado atual do portfólio"""
    total_balance: float
    available_balance: float
    total_pnl: float
    open_positions: List[Position]
    daily_trades: int
    drawdown_percent: float
    risk_exposure: float

    def get_position_count(self) -> int:
        return len(self.open_positions)

    def get_total_exposure(self) -> float:
        return sum(pos.position_size for pos in self.open_positions)


class AutonomousTradingEngine:
    """
    🤖 Motor de Trading Autônomo

    Responsabilidades:
    1. Analisar sinais da IA em tempo real
    2. Tomar decisões de trading autônomas
    3. Calcular tamanho de posição baseado em risco
    4. Gerenciar portfólio e posições abertas
    5. Aplicar regras de risk management
    """

    def __init__(self,
                 initial_balance: float = 10000.0,
                 max_positions: int = 5,
                 max_daily_trades: int = 50,
                 max_risk_per_trade: float = 0.02,  # 2%
                 max_portfolio_risk: float = 0.10,   # 10%
                 min_confidence: float = 0.75):

        self.logger = logging.getLogger(__name__)

        # Configurações de trading
        self.initial_balance = initial_balance
        self.max_positions = max_positions
        self.max_daily_trades = max_daily_trades
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.min_confidence = min_confidence

        # Estado do portfólio
        self.portfolio = PortfolioState(
            total_balance=initial_balance,
            available_balance=initial_balance,
            total_pnl=0.0,
            open_positions=[],
            daily_trades=0,
            drawdown_percent=0.0,
            risk_exposure=0.0
        )

        # Histórico de decisões
        self.decision_history: List[TradingDecision] = []
        self.trade_log: List[Dict[str, Any]] = []

        # Sistema ativo
        self.is_active = False
        self.emergency_stop = False

        # Estatísticas de performance
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'average_profit': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }

    async def start_autonomous_trading(self):
        """Inicia o sistema de trading autônomo"""
        self.logger.info("🤖 Iniciando sistema de trading autônomo...")
        self.is_active = True
        self.emergency_stop = False

        # Reset daily counters
        self.portfolio.daily_trades = 0

        self.logger.info("✅ Sistema autônomo ATIVO - IA tomando decisões independentes")

    async def stop_autonomous_trading(self):
        """Para o sistema de trading autônomo"""
        self.logger.info("🛑 Parando sistema de trading autônomo...")
        self.is_active = False

        # Fechar todas as posições abertas
        for position in self.portfolio.open_positions:
            await self._close_position(position, "SYSTEM_SHUTDOWN")

        self.logger.info("✅ Sistema autônomo PARADO")

    async def emergency_shutdown(self):
        """Parada de emergência - fecha tudo imediatamente"""
        self.logger.warning("🚨 PARADA DE EMERGÊNCIA ATIVADA!")
        self.emergency_stop = True
        self.is_active = False

        # Fechar todas as posições imediatamente
        for position in self.portfolio.open_positions:
            await self._close_position(position, "EMERGENCY_STOP")

        self.logger.warning("🚨 TODAS AS POSIÇÕES FECHADAS - SISTEMA EM MODO SEGURO")

    async def process_trading_signal(self, signal: TradingSignal) -> Optional[TradingDecision]:
        """
        Processa sinal da IA e toma decisão autônoma de trading

        Este é o coração do sistema autônomo - aqui a IA decide sozinha
        """
        if not self.is_active or self.emergency_stop:
            return None

        try:
            # Analisar estado atual do mercado
            market_state = await self._analyze_market_state(signal)

            # Tomar decisão autônoma
            decision = await self._make_autonomous_decision(signal, market_state)

            if decision:
                # Executar decisão se válida
                await self._execute_decision(decision)

                # Registrar decisão
                self.decision_history.append(decision)

                # Log da decisão da IA
                self.logger.info(f"🤖 IA DECIDIU: {decision.action.value} {decision.symbol} "
                               f"| Confiança: {decision.confidence:.1%} "
                               f"| Força: {decision.signal_strength:.2f} "
                               f"| Tamanho: ${decision.position_size:.2f}")

                self.logger.info(f"💭 Raciocínio da IA: {decision.reasoning}")

            return decision

        except Exception as e:
            self.logger.error(f"❌ Erro ao processar sinal de trading: {e}")
            return None

    async def _analyze_market_state(self, signal: TradingSignal) -> Dict[str, Any]:
        """Analisa estado atual do mercado para decisão"""

        # Calcular volatilidade recente
        volatility = signal.features.price_volatility if signal.features else 0.02

        # Calcular tendência de curto prazo
        momentum = signal.features.momentum_5min if signal.features else 0.0

        # Verificar liquidez estimada
        liquidity_score = 0.8  # Placeholder - seria calculado com dados de volume

        return {
            'volatility': volatility,
            'momentum': momentum,
            'liquidity': liquidity_score,
            'market_hours': self._is_market_hours(),
            'spread_estimate': volatility * 0.1  # Estimativa de spread
        }

    async def _make_autonomous_decision(self,
                                      signal: TradingSignal,
                                      market_state: Dict[str, Any]) -> Optional[TradingDecision]:
        """
        🧠 CORE DA IA - Toma decisão completamente autônoma

        Aqui a IA analisa tudo e decide sozinha, sem perguntar nada
        """

        # 1. Verificar filtros básicos de qualidade do sinal
        if signal.confidence < self.min_confidence:
            return None

        if abs(signal.signal_strength) < 0.6:  # Sinal muito fraco
            return None

        # 2. Verificar limites de portfólio
        if not await self._check_portfolio_limits():
            return None

        # 3. Determinar ação baseada no sinal
        if signal.signal_strength > 0.6:
            action = TradeAction.BUY
        elif signal.signal_strength < -0.6:
            action = TradeAction.SELL
        else:
            action = TradeAction.HOLD

        if action == TradeAction.HOLD:
            return None

        # 4. Calcular tamanho da posição (Kelly Criterion modificado)
        position_size = await self._calculate_position_size(signal, market_state)

        if position_size <= 0:
            return None

        # 5. Calcular níveis de stop loss e take profit
        stop_loss, target_profit = await self._calculate_risk_levels(
            signal, market_state, position_size
        )

        # 6. Avaliar nível de risco da operação
        risk_level = await self._assess_risk_level(signal, market_state)

        # 7. Gerar raciocínio da IA
        reasoning = await self._generate_ai_reasoning(signal, market_state, action)

        # 8. Estimar tempo de holding
        holding_period = int(3600 * (0.5 + signal.confidence))  # 30min a 1.5h

        # 9. Criar decisão final
        decision = TradingDecision(
            action=action,
            symbol=signal.symbol,
            confidence=signal.confidence,
            signal_strength=signal.signal_strength,
            position_size=position_size,
            risk_level=risk_level,
            reasoning=reasoning,
            timestamp=datetime.now(),
            expected_holding_period=holding_period,
            target_profit=target_profit,
            stop_loss=stop_loss
        )

        return decision

    async def _calculate_position_size(self,
                                     signal: TradingSignal,
                                     market_state: Dict[str, Any]) -> float:
        """
        Calcula tamanho ótimo da posição usando Kelly Criterion modificado
        """

        # Kelly Criterion: f = (bp - q) / b
        # onde: b = odds, p = prob win, q = prob loss

        # Estimar probabilidade de sucesso baseada na confiança da IA
        win_prob = 0.5 + (signal.confidence - 0.5) * 0.8  # 0.5 to 0.9
        loss_prob = 1 - win_prob

        # Estimar odds baseado na força do sinal e volatilidade
        volatility = market_state['volatility']
        expected_move = abs(signal.signal_strength) * volatility * 2
        odds = expected_move / volatility if volatility > 0 else 1.0

        # Kelly fraction
        kelly_fraction = (odds * win_prob - loss_prob) / odds
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%

        # Ajustar por confiança e risk management
        confidence_multiplier = signal.confidence ** 2  # Quadratic scaling
        risk_multiplier = 1 - market_state['volatility']  # Menos risco em alta vol

        # Tamanho base da posição
        base_size = self.portfolio.available_balance * self.max_risk_per_trade

        # Tamanho final
        position_size = base_size * kelly_fraction * confidence_multiplier * risk_multiplier

        # Aplicar limites mínimos e máximos
        min_size = 10.0  # $10 mínimo
        max_size = self.portfolio.available_balance * 0.05  # 5% máximo

        position_size = max(min_size, min(position_size, max_size))

        return round(position_size, 2)

    async def _calculate_risk_levels(self,
                                   signal: TradingSignal,
                                   market_state: Dict[str, Any],
                                   position_size: float) -> tuple[float, float]:
        """Calcula níveis de stop loss e take profit"""

        current_price = signal.current_price
        volatility = market_state['volatility']

        # Stop loss baseado em volatilidade e confiança
        stop_distance = volatility * (2.0 - signal.confidence)  # 0.5x to 2x volatility

        # Take profit com ratio risk/reward de 1.5:1 a 3:1
        reward_ratio = 1.5 + signal.confidence  # 1.5 to 2.5
        profit_distance = stop_distance * reward_ratio

        if signal.signal_strength > 0:  # BUY
            stop_loss = current_price * (1 - stop_distance)
            target_profit = current_price * (1 + profit_distance)
        else:  # SELL
            stop_loss = current_price * (1 + stop_distance)
            target_profit = current_price * (1 - profit_distance)

        return round(stop_loss, 5), round(target_profit, 5)

    async def _assess_risk_level(self,
                               signal: TradingSignal,
                               market_state: Dict[str, Any]) -> RiskLevel:
        """Avalia o nível de risco da operação"""

        risk_score = 0

        # Volatilidade
        if market_state['volatility'] > 0.03:
            risk_score += 2
        elif market_state['volatility'] > 0.02:
            risk_score += 1

        # Confiança do sinal
        if signal.confidence < 0.8:
            risk_score += 1
        if signal.confidence < 0.7:
            risk_score += 1

        # Liquidez
        if market_state['liquidity'] < 0.7:
            risk_score += 1

        # Horário de mercado
        if not market_state['market_hours']:
            risk_score += 2

        # Exposição atual do portfólio
        if self.portfolio.risk_exposure > 0.05:
            risk_score += 1

        # Classificar nível de risco
        if risk_score <= 1:
            return RiskLevel.LOW
        elif risk_score <= 3:
            return RiskLevel.MEDIUM
        elif risk_score <= 5:
            return RiskLevel.HIGH
        else:
            return RiskLevel.EXTREME

    async def _generate_ai_reasoning(self,
                                   signal: TradingSignal,
                                   market_state: Dict[str, Any],
                                   action: TradeAction) -> str:
        """Gera explicação do raciocínio da IA"""

        reasons = []

        # Análise do sinal
        reasons.append(f"Sinal {action.value} com confiança {signal.confidence:.1%}")
        reasons.append(f"Força do sinal: {signal.signal_strength:.2f}")

        # Condições de mercado
        vol = market_state['volatility']
        if vol > 0.025:
            reasons.append("Volatilidade elevada detectada")
        elif vol < 0.015:
            reasons.append("Mercado em baixa volatilidade")

        # Momentum
        momentum = market_state['momentum']
        if abs(momentum) > 0.02:
            direction = "alta" if momentum > 0 else "baixa"
            reasons.append(f"Momentum de {direction} confirmado")

        # Gestão de risco
        reasons.append(f"Posição dimensionada para {self.max_risk_per_trade:.1%} de risco")

        return " | ".join(reasons)

    async def _check_portfolio_limits(self) -> bool:
        """Verifica se é possível abrir nova posição"""

        # Máximo de posições
        if len(self.portfolio.open_positions) >= self.max_positions:
            return False

        # Máximo de trades diários
        if self.portfolio.daily_trades >= self.max_daily_trades:
            return False

        # Risco total do portfólio
        if self.portfolio.risk_exposure > self.max_portfolio_risk:
            return False

        # Saldo disponível
        if self.portfolio.available_balance < 100:  # $100 mínimo
            return False

        return True

    async def _execute_decision(self, decision: TradingDecision):
        """Executa a decisão de trading"""

        # Simular execução da ordem (aqui integraria com Deriv API)
        position = Position(
            id=f"pos_{len(self.portfolio.open_positions) + 1}",
            symbol=decision.symbol,
            action=decision.action,
            entry_price=decision.target_profit,  # Placeholder - seria preço atual
            position_size=decision.position_size,
            entry_time=decision.timestamp,
            stop_loss=decision.stop_loss,
            target_profit=decision.target_profit
        )

        # Adicionar à carteira
        self.portfolio.open_positions.append(position)
        self.portfolio.available_balance -= decision.position_size
        self.portfolio.daily_trades += 1
        self.portfolio.risk_exposure += decision.position_size / self.portfolio.total_balance

        # Log da execução
        trade_log = {
            'timestamp': decision.timestamp.isoformat(),
            'action': decision.action.value,
            'symbol': decision.symbol,
            'size': decision.position_size,
            'confidence': decision.confidence,
            'reasoning': decision.reasoning
        }

        self.trade_log.append(trade_log)
        self.stats['total_trades'] += 1

        self.logger.info(f"✅ Ordem executada: {decision.action.value} ${decision.position_size:.2f}")

    async def _close_position(self, position: Position, reason: str):
        """Fecha uma posição"""
        position.is_open = False

        # Remover da lista de posições abertas
        self.portfolio.open_positions.remove(position)

        # Atualizar saldo
        self.portfolio.available_balance += position.position_size + position.current_pnl
        self.portfolio.total_pnl += position.current_pnl

        # Atualizar estatísticas
        if position.current_pnl > 0:
            self.stats['winning_trades'] += 1
        else:
            self.stats['losing_trades'] += 1

        self.stats['win_rate'] = self.stats['winning_trades'] / max(1, self.stats['total_trades'])

        self.logger.info(f"🔄 Posição fechada: {position.symbol} | P&L: ${position.current_pnl:.2f} | Motivo: {reason}")

    def _is_market_hours(self) -> bool:
        """Verifica se está em horário de mercado"""
        # Deriv/forex opera 24/5 - verificar se é dia útil
        now = datetime.now()
        return now.weekday() < 5  # Segunda a sexta

    async def update_positions(self, current_prices: Dict[str, float]):
        """Atualiza P&L de todas as posições abertas"""
        for position in self.portfolio.open_positions[:]:  # Copy list to avoid modification during iteration
            if position.symbol in current_prices:
                position.update_pnl(current_prices[position.symbol])

                # Verificar stop loss / take profit
                current_price = current_prices[position.symbol]

                should_close = False
                close_reason = ""

                if position.action == TradeAction.BUY:
                    if current_price <= position.stop_loss:
                        should_close = True
                        close_reason = "STOP_LOSS"
                    elif current_price >= position.target_profit:
                        should_close = True
                        close_reason = "TAKE_PROFIT"
                else:  # SELL
                    if current_price >= position.stop_loss:
                        should_close = True
                        close_reason = "STOP_LOSS"
                    elif current_price <= position.target_profit:
                        should_close = True
                        close_reason = "TAKE_PROFIT"

                if should_close:
                    await self._close_position(position, close_reason)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de performance"""
        return {
            'portfolio': asdict(self.portfolio),
            'stats': self.stats,
            'total_decisions': len(self.decision_history),
            'is_active': self.is_active,
            'emergency_stop': self.emergency_stop
        }

    def get_recent_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retorna decisões recentes da IA"""
        recent = self.decision_history[-limit:] if limit else self.decision_history
        return [decision.to_dict() for decision in recent]