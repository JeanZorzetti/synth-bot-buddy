"""
Gestão de Risco Inteligente para Trading Bot

Este módulo implementa:
- Kelly Criterion para position sizing
- Kelly Criterion com ML (ajuste dinâmico baseado em padrões)
- Stop Loss dinâmico baseado em ATR
- Trailing Stop para proteger lucros
- Partial Take Profit
- Limites diários/semanais
- Circuit Breaker
- Controle de correlação
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradeRisk:
    """Informações de risco de um trade"""
    symbol: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_amount: float
    reward_amount: float
    risk_reward_ratio: float
    timestamp: datetime


@dataclass
class RiskLimits:
    """Limites de risco configuráveis"""
    max_daily_loss_percent: float = 5.0  # -5% máximo por dia
    max_weekly_loss_percent: float = 10.0  # -10% máximo por semana
    max_drawdown_percent: float = 15.0  # -15% drawdown máximo
    max_position_size_percent: float = 10.0  # 10% do capital por trade
    max_concurrent_trades: int = 3  # Máximo de trades simultâneos
    max_correlation: float = 0.7  # Correlação máxima entre trades
    circuit_breaker_losses: int = 3  # Pausar após X perdas consecutivas
    min_risk_reward_ratio: float = 1.5  # Mínimo 1:1.5 (risco:recompensa)


class RiskManager:
    """
    Gerenciador de Risco Inteligente

    Responsabilidades:
    - Calcular tamanho de posição ideal (Kelly Criterion)
    - Validar trades contra limites de risco
    - Calcular stop loss dinâmico
    - Gerenciar trailing stops
    - Implementar circuit breaker
    - Monitorar drawdown
    """

    def __init__(self,
                 initial_capital: float = 1000.0,
                 risk_limits: Optional[RiskLimits] = None):
        """
        Inicializa RiskManager

        Args:
            initial_capital: Capital inicial em USD
            risk_limits: Limites de risco personalizados
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.limits = risk_limits or RiskLimits()

        # Estado do sistema
        self.active_trades: List[TradeRisk] = []
        self.trade_history: List[Dict] = []
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.peak_capital = initial_capital
        self.consecutive_losses = 0
        self.is_circuit_breaker_active = False

        # Estatísticas para Kelly Criterion
        self.win_rate = 0.6  # Inicializa com 60% (será atualizado)
        self.avg_win = 0.0
        self.avg_loss = 0.0

        # ML Predictor (opcional)
        self.use_ml_kelly = False  # Ativa ML após treinar
        self.ml_predictions: Optional[Dict] = None
        self.auto_retrain_enabled = False  # Re-treino automático
        self.retrain_interval = 20  # Re-treinar a cada X trades
        self.last_train_count = 0  # Contador de trades no último treino

        # Equity Curve Tracking
        self.equity_history: List[Dict] = [{
            'timestamp': datetime.now().isoformat(),
            'capital': initial_capital,
            'pnl': 0.0,
            'drawdown': 0.0,
            'trade_count': 0
        }]

        # Timestamps
        self.last_daily_reset = datetime.now()
        self.last_weekly_reset = datetime.now()

        logger.info(f"RiskManager inicializado - Capital: ${initial_capital}")

    def calculate_kelly_criterion(self,
                                  win_rate: Optional[float] = None,
                                  avg_win: Optional[float] = None,
                                  avg_loss: Optional[float] = None) -> float:
        """
        Calcula % ideal do capital para arriscar usando Kelly Criterion

        Formula: f = (p * b - q) / b
        Onde:
        - f = fração do capital a arriscar
        - p = win rate
        - q = (1 - p) = loss rate
        - b = avg_win / avg_loss (win/loss ratio)

        Args:
            win_rate: Taxa de vitória (0-1), usa self.win_rate se None
            avg_win: Ganho médio, usa self.avg_win se None
            avg_loss: Perda média, usa self.avg_loss se None

        Returns:
            Fração do capital (0-1), conservador (Quarter Kelly)
        """
        p = win_rate if win_rate is not None else self.win_rate
        w = avg_win if avg_win is not None else self.avg_win
        l = avg_loss if avg_loss is not None else self.avg_loss

        # Evitar divisão por zero
        if l == 0 or w == 0:
            logger.warning("Avg win ou avg loss é zero, usando 2% fixo")
            return 0.02

        q = 1 - p
        b = abs(w / l)  # Win/Loss ratio

        # Kelly Criterion
        kelly = (p * b - q) / b

        # Quarter Kelly para segurança (muito conservador)
        conservative_kelly = kelly * 0.25

        # Limitar entre 1% e 5%
        kelly_limited = max(0.01, min(conservative_kelly, 0.05))

        logger.debug(f"Kelly: {kelly:.2%}, Quarter Kelly: {conservative_kelly:.2%}, Limitado: {kelly_limited:.2%}")

        return kelly_limited

    def calculate_kelly_with_ml(self) -> float:
        """
        Calcula Kelly Criterion usando previsões de ML

        Se ML predictions estiver disponível e use_ml_kelly=True,
        usa as previsões do modelo. Caso contrário, usa estatísticas históricas.

        Returns:
            Fração do capital (0-1), conservador (Quarter Kelly)
        """
        if self.use_ml_kelly and self.ml_predictions:
            # Usar previsões ML
            kelly = self.ml_predictions.get('kelly_criterion', 0.02)
            logger.debug(f"Kelly ML: {kelly:.2%} (confiança: {self.ml_predictions.get('confidence', 0):.2%})")
            return kelly
        else:
            # Fallback para cálculo tradicional
            return self.calculate_kelly_criterion()

    def update_ml_predictions(self, predictions: Dict):
        """
        Atualiza previsões ML para uso no Kelly Criterion

        Args:
            predictions: Dict com predicted_win_rate, predicted_avg_win/loss, kelly_criterion
        """
        self.ml_predictions = predictions
        logger.info(f"ML predictions atualizadas: Win Rate={predictions.get('predicted_win_rate', 0):.2%}, "
                   f"Kelly={predictions.get('kelly_criterion', 0):.2%}")

    def enable_ml_kelly(self, enable: bool = True):
        """
        Ativa/desativa uso de ML para Kelly Criterion

        Args:
            enable: True para ativar ML, False para usar estatísticas históricas
        """
        self.use_ml_kelly = enable
        logger.info(f"Kelly ML {'ATIVADO' if enable else 'DESATIVADO'}")

    def enable_auto_retrain(self, enable: bool = True, interval: int = 20):
        """
        Ativa/desativa re-treino automático do modelo ML

        Args:
            enable: True para ativar re-treino automático
            interval: Número de trades entre re-treinos (padrão: 20)
        """
        self.auto_retrain_enabled = enable
        self.retrain_interval = interval
        if enable:
            self.last_train_count = len(self.trade_history)
        logger.info(f"Re-treino automático {'ATIVADO' if enable else 'DESATIVADO'} (intervalo: {interval} trades)")

    def should_retrain(self) -> bool:
        """
        Verifica se deve re-treinar o modelo ML

        Returns:
            True se deve re-treinar (auto_retrain ativado + intervalo atingido)
        """
        if not self.auto_retrain_enabled:
            return False

        trades_since_last_train = len(self.trade_history) - self.last_train_count
        should_retrain = trades_since_last_train >= self.retrain_interval

        if should_retrain:
            logger.info(f"Re-treino necessário: {trades_since_last_train} trades desde último treino")

        return should_retrain

    def mark_retrain_done(self):
        """Marca que re-treino foi realizado"""
        self.last_train_count = len(self.trade_history)
        logger.info(f"Re-treino marcado como completo no trade #{self.last_train_count}")

    def calculate_position_size(self,
                               entry_price: float,
                               stop_loss: float,
                               risk_per_trade_percent: Optional[float] = None) -> float:
        """
        Calcula tamanho da posição usando Fixed Fractional Method

        Args:
            entry_price: Preço de entrada
            stop_loss: Preço de stop loss
            risk_per_trade_percent: % do capital a arriscar (usa Kelly se None)

        Returns:
            Tamanho da posição em USD
        """
        # Usar Kelly Criterion se não especificado (com ou sem ML)
        if risk_per_trade_percent is None:
            risk_per_trade_percent = self.calculate_kelly_with_ml()

        # Valor em risco
        risk_amount = self.current_capital * risk_per_trade_percent

        # Distância até stop loss (em %)
        risk_per_unit = abs((entry_price - stop_loss) / entry_price)

        # Evitar divisão por zero
        if risk_per_unit == 0:
            logger.error("Stop loss igual ao entry price!")
            return 0.0

        # Tamanho da posição
        position_size = risk_amount / risk_per_unit

        # Limitar ao máximo permitido
        max_position = self.current_capital * (self.limits.max_position_size_percent / 100)
        position_size = min(position_size, max_position)

        logger.info(f"Position Size: ${position_size:.2f} (Risk: {risk_per_trade_percent:.2%}, Max: ${max_position:.2f})")

        return position_size

    def calculate_atr_stop_loss(self,
                               current_price: float,
                               atr: float,
                               is_long: bool,
                               multiplier: float = 2.0) -> float:
        """
        Calcula stop loss dinâmico baseado em ATR (Average True Range)

        Args:
            current_price: Preço atual
            atr: Average True Range
            is_long: True para long, False para short
            multiplier: Multiplicador do ATR (padrão: 2.0)

        Returns:
            Preço de stop loss
        """
        if is_long:
            stop_loss = current_price - (atr * multiplier)
        else:
            stop_loss = current_price + (atr * multiplier)

        logger.debug(f"ATR Stop Loss: ${stop_loss:.5f} (Price: ${current_price:.5f}, ATR: {atr:.5f})")

        return stop_loss

    def calculate_take_profit_levels(self,
                                    entry_price: float,
                                    stop_loss: float,
                                    is_long: bool,
                                    risk_reward_ratio: float = 2.0) -> Dict[str, float]:
        """
        Calcula níveis de take profit para partial exits

        Args:
            entry_price: Preço de entrada
            stop_loss: Preço de stop loss
            is_long: True para long, False para short
            risk_reward_ratio: Razão risco:recompensa (padrão: 2.0)

        Returns:
            Dict com TP1 (50% exit) e TP2 (50% exit)
        """
        # Calcular distância de risco
        risk_distance = abs(entry_price - stop_loss)

        if is_long:
            tp1 = entry_price + (risk_distance * risk_reward_ratio * 0.5)  # 1:1 R:R
            tp2 = entry_price + (risk_distance * risk_reward_ratio)  # 1:2 R:R
        else:
            tp1 = entry_price - (risk_distance * risk_reward_ratio * 0.5)
            tp2 = entry_price - (risk_distance * risk_reward_ratio)

        return {
            'tp1': tp1,  # 50% da posição
            'tp2': tp2,  # 50% restante
            'risk_reward_ratio': risk_reward_ratio
        }

    def validate_trade(self,
                      symbol: str,
                      entry_price: float,
                      stop_loss: float,
                      take_profit: float,
                      position_size: float) -> Tuple[bool, str]:
        """
        Valida se um trade pode ser executado baseado em limites de risco

        Args:
            symbol: Símbolo do ativo
            entry_price: Preço de entrada
            stop_loss: Preço de stop loss
            take_profit: Preço de take profit
            position_size: Tamanho da posição em USD

        Returns:
            (is_valid, reason)
        """
        # 1. Verificar circuit breaker
        if self.is_circuit_breaker_active:
            return False, f"Circuit breaker ativo após {self.consecutive_losses} perdas consecutivas"

        # 2. Verificar limites diários
        if self.daily_pnl < 0:
            daily_loss_percent = abs(self.daily_pnl / self.initial_capital * 100)
            if daily_loss_percent >= self.limits.max_daily_loss_percent:
                return False, f"Limite diário atingido: -{daily_loss_percent:.2f}% (max: {self.limits.max_daily_loss_percent}%)"

        # 3. Verificar limites semanais
        if self.weekly_pnl < 0:
            weekly_loss_percent = abs(self.weekly_pnl / self.initial_capital * 100)
            if weekly_loss_percent >= self.limits.max_weekly_loss_percent:
                return False, f"Limite semanal atingido: -{weekly_loss_percent:.2f}% (max: {self.limits.max_weekly_loss_percent}%)"

        # 4. Verificar drawdown
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital * 100
        if current_drawdown >= self.limits.max_drawdown_percent:
            return False, f"Drawdown máximo atingido: {current_drawdown:.2f}% (max: {self.limits.max_drawdown_percent}%)"

        # 5. Verificar número de trades simultâneos
        if len(self.active_trades) >= self.limits.max_concurrent_trades:
            return False, f"Máximo de trades simultâneos atingido: {len(self.active_trades)}/{self.limits.max_concurrent_trades}"

        # 6. Verificar tamanho da posição
        max_position = self.current_capital * (self.limits.max_position_size_percent / 100)
        if position_size > max_position:
            return False, f"Posição muito grande: ${position_size:.2f} (max: ${max_position:.2f})"

        # 7. Verificar risk/reward ratio
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0

        if rr_ratio < self.limits.min_risk_reward_ratio:
            return False, f"R:R ratio muito baixo: {rr_ratio:.2f} (min: {self.limits.min_risk_reward_ratio})"

        return True, "Trade aprovado"

    def record_trade(self,
                    symbol: str,
                    entry_price: float,
                    stop_loss: float,
                    take_profit: float,
                    position_size: float,
                    is_long: bool) -> TradeRisk:
        """
        Registra um novo trade ativo

        Args:
            symbol: Símbolo do ativo
            entry_price: Preço de entrada
            stop_loss: Preço de stop loss
            take_profit: Preço de take profit
            position_size: Tamanho da posição em USD
            is_long: True para long, False para short

        Returns:
            TradeRisk object
        """
        risk_amount = position_size * abs((entry_price - stop_loss) / entry_price)
        reward_amount = position_size * abs((take_profit - entry_price) / entry_price)
        rr_ratio = reward_amount / risk_amount if risk_amount > 0 else 0

        trade = TradeRisk(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            risk_amount=risk_amount,
            reward_amount=reward_amount,
            risk_reward_ratio=rr_ratio,
            timestamp=datetime.now()
        )

        self.active_trades.append(trade)

        logger.info(f"Trade registrado: {symbol} ${position_size:.2f} @ ${entry_price:.5f} (R:R {rr_ratio:.2f})")

        return trade

    def close_trade(self,
                   symbol: str,
                   exit_price: float,
                   is_win: bool) -> float:
        """
        Fecha um trade e atualiza estatísticas

        Args:
            symbol: Símbolo do ativo
            exit_price: Preço de saída
            is_win: True se foi lucrativo

        Returns:
            PnL do trade
        """
        # Encontrar trade ativo
        trade = next((t for t in self.active_trades if t.symbol == symbol), None)

        if not trade:
            logger.error(f"Trade não encontrado: {symbol}")
            return 0.0

        # Calcular PnL
        pnl = trade.position_size * ((exit_price - trade.entry_price) / trade.entry_price)

        # Atualizar capital
        self.current_capital += pnl
        self.daily_pnl += pnl
        self.weekly_pnl += pnl

        # Atualizar peak capital
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital

        # Atualizar estatísticas
        if is_win:
            self.consecutive_losses = 0
            if self.avg_win == 0:
                self.avg_win = abs(pnl)
            else:
                self.avg_win = (self.avg_win + abs(pnl)) / 2
        else:
            self.consecutive_losses += 1
            if self.avg_loss == 0:
                self.avg_loss = abs(pnl)
            else:
                self.avg_loss = (self.avg_loss + abs(pnl)) / 2

            # Verificar circuit breaker
            if self.consecutive_losses >= self.limits.circuit_breaker_losses:
                self.is_circuit_breaker_active = True
                logger.warning(f"⚠️ CIRCUIT BREAKER ATIVADO após {self.consecutive_losses} perdas consecutivas")

        # Atualizar win rate
        total_trades = len(self.trade_history) + 1
        winning_trades = sum(1 for t in self.trade_history if t['is_win']) + (1 if is_win else 0)
        self.win_rate = winning_trades / total_trades

        # Salvar no histórico
        self.trade_history.append({
            'symbol': symbol,
            'entry_price': trade.entry_price,
            'exit_price': exit_price,
            'position_size': trade.position_size,
            'pnl': pnl,
            'is_win': is_win,
            'timestamp': datetime.now()
        })

        # Remover dos trades ativos
        self.active_trades.remove(trade)

        # Registrar ponto na equity curve
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital * 100
        self.equity_history.append({
            'timestamp': datetime.now().isoformat(),
            'capital': self.current_capital,
            'pnl': pnl,
            'drawdown': current_drawdown,
            'trade_count': len(self.trade_history),
            'is_win': is_win
        })

        logger.info(f"Trade fechado: {symbol} PnL: ${pnl:.2f} ({'WIN' if is_win else 'LOSS'})")

        return pnl

    def reset_circuit_breaker(self):
        """Reseta o circuit breaker manualmente"""
        self.is_circuit_breaker_active = False
        self.consecutive_losses = 0
        logger.info("Circuit breaker resetado manualmente")

    def reset_daily_limits(self):
        """Reseta limites diários (chamar a cada 24h)"""
        self.daily_pnl = 0.0
        self.last_daily_reset = datetime.now()
        logger.info("Limites diários resetados")

    def reset_weekly_limits(self):
        """Reseta limites semanais (chamar a cada 7 dias)"""
        self.weekly_pnl = 0.0
        self.last_weekly_reset = datetime.now()
        logger.info("Limites semanais resetados")

    def get_risk_metrics(self) -> Dict:
        """
        Retorna métricas de risco atuais

        Returns:
            Dict com métricas de risco
        """
        drawdown = (self.peak_capital - self.current_capital) / self.peak_capital * 100
        daily_loss_pct = (self.daily_pnl / self.initial_capital * 100) if self.daily_pnl < 0 else 0
        weekly_loss_pct = (self.weekly_pnl / self.initial_capital * 100) if self.weekly_pnl < 0 else 0

        return {
            'current_capital': self.current_capital,
            'initial_capital': self.initial_capital,
            'total_pnl': self.current_capital - self.initial_capital,
            'total_pnl_percent': ((self.current_capital - self.initial_capital) / self.initial_capital * 100),
            'daily_pnl': self.daily_pnl,
            'daily_loss_percent': abs(daily_loss_pct),
            'weekly_pnl': self.weekly_pnl,
            'weekly_loss_percent': abs(weekly_loss_pct),
            'drawdown_percent': drawdown,
            'peak_capital': self.peak_capital,
            'active_trades_count': len(self.active_trades),
            'total_trades': len(self.trade_history),
            'win_rate': self.win_rate * 100,
            'consecutive_losses': self.consecutive_losses,
            'is_circuit_breaker_active': self.is_circuit_breaker_active,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'kelly_criterion': self.calculate_kelly_criterion() * 100,
            'limits': {
                'max_daily_loss': self.limits.max_daily_loss_percent,
                'max_weekly_loss': self.limits.max_weekly_loss_percent,
                'max_drawdown': self.limits.max_drawdown_percent,
                'max_position_size': self.limits.max_position_size_percent,
                'max_concurrent_trades': self.limits.max_concurrent_trades,
                'circuit_breaker_losses': self.limits.circuit_breaker_losses,
                'min_risk_reward': self.limits.min_risk_reward_ratio
            }
        }


class TrailingStop:
    """
    Gerenciador de Trailing Stop

    Atualiza o stop loss conforme o preço move a favor do trade
    """

    def __init__(self,
                 initial_price: float,
                 initial_stop: float,
                 is_long: bool,
                 trailing_percent: float = 2.0):
        """
        Inicializa Trailing Stop

        Args:
            initial_price: Preço de entrada
            initial_stop: Stop loss inicial
            is_long: True para long, False para short
            trailing_percent: % de trailing (padrão: 2%)
        """
        self.initial_price = initial_price
        self.current_stop = initial_stop
        self.is_long = is_long
        self.trailing_percent = trailing_percent / 100
        self.highest_price = initial_price if is_long else None
        self.lowest_price = initial_price if not is_long else None

    def update(self, current_price: float) -> float:
        """
        Atualiza stop loss baseado no preço atual

        Args:
            current_price: Preço atual do ativo

        Returns:
            Novo stop loss (pode ser o mesmo se não moveu)
        """
        if self.is_long:
            # Long: stop sobe quando preço sobe
            if current_price > self.highest_price:
                self.highest_price = current_price
                new_stop = current_price * (1 - self.trailing_percent)

                # Stop só sobe, nunca desce
                if new_stop > self.current_stop:
                    self.current_stop = new_stop
                    logger.debug(f"Trailing Stop atualizado: ${self.current_stop:.5f}")
        else:
            # Short: stop desce quando preço desce
            if current_price < self.lowest_price:
                self.lowest_price = current_price
                new_stop = current_price * (1 + self.trailing_percent)

                # Stop só desce, nunca sobe
                if new_stop < self.current_stop:
                    self.current_stop = new_stop
                    logger.debug(f"Trailing Stop atualizado: ${self.current_stop:.5f}")

        return self.current_stop

    def is_hit(self, current_price: float) -> bool:
        """
        Verifica se o stop foi atingido

        Args:
            current_price: Preço atual

        Returns:
            True se stop foi hit
        """
        if self.is_long:
            return current_price <= self.current_stop
        else:
            return current_price >= self.current_stop
