"""
Sistema de Alertas Inteligentes para Forward Testing

Este módulo detecta situações críticas, avisos e eventos importantes
durante a execução do Forward Testing e gera alertas apropriados.

Níveis de Alerta:
- CRITICAL: Situações que requerem atenção imediata (drawdown alto, perdas seguidas)
- WARNING: Situações que indicam problemas potenciais (win rate baixo, timeout alto)
- INFO: Eventos importantes positivos (TP atingido, novo recorde de capital)
"""

from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Níveis de severidade dos alertas"""
    CRITICAL = "CRITICAL"  # 🔴 Requer atenção imediata
    WARNING = "WARNING"    # 🟡 Situação de alerta
    INFO = "INFO"          # 🟢 Informação importante


class AlertType(str, Enum):
    """Tipos de alertas disponíveis"""
    HIGH_DRAWDOWN = "HIGH_DRAWDOWN"                    # Drawdown > threshold
    CONSECUTIVE_LOSSES = "CONSECUTIVE_LOSSES"          # N perdas seguidas
    LOW_WIN_RATE = "LOW_WIN_RATE"                      # Win rate abaixo do esperado
    HIGH_TIMEOUT_RATE = "HIGH_TIMEOUT_RATE"            # Muitos timeouts
    HIGH_SL_HIT_RATE = "HIGH_SL_HIT_RATE"              # Muitos stop losses
    CAPITAL_RECORD = "CAPITAL_RECORD"                  # Novo recorde de capital
    TAKE_PROFIT_HIT = "TAKE_PROFIT_HIT"                # Take profit atingido
    PROFIT_MILESTONE = "PROFIT_MILESTONE"              # Marco de lucro atingido (ex: +10%)
    POSITION_TIMEOUT = "POSITION_TIMEOUT"              # Posição fechada por timeout


class Alert:
    """Representa um alerta gerado pelo sistema"""

    def __init__(
        self,
        level: AlertLevel,
        alert_type: AlertType,
        message: str,
        details: Optional[Dict] = None,
        timestamp: Optional[datetime] = None
    ):
        self.id = f"{alert_type.value}_{datetime.now().timestamp()}"
        self.level = level
        self.alert_type = alert_type
        self.message = message
        self.details = details or {}
        self.timestamp = timestamp or datetime.now()
        self.read = False  # Para tracking se foi lido pelo usuário

    def to_dict(self) -> Dict:
        """Serializa alerta para dicionário"""
        return {
            "id": self.id,
            "level": self.level.value,
            "type": self.alert_type.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "read": self.read
        }


class AlertSystem:
    """
    Sistema de Alertas para Forward Testing

    Monitora métricas em tempo real e gera alertas quando limiares são atingidos.
    """

    def __init__(self):
        self.alerts: List[Alert] = []
        self.last_capital: Optional[float] = None
        self.peak_capital: Optional[float] = None
        self.consecutive_losses: int = 0

        # Thresholds configuráveis
        self.config = {
            "drawdown_critical_pct": 10.0,      # Drawdown > 10% = CRITICAL
            "drawdown_warning_pct": 7.0,        # Drawdown > 7% = WARNING
            "consecutive_losses_critical": 5,   # 5 perdas seguidas = CRITICAL
            "consecutive_losses_warning": 3,    # 3 perdas seguidas = WARNING
            "win_rate_warning_pct": 50.0,       # Win rate < 50% = WARNING
            "timeout_rate_warning_pct": 30.0,   # Timeout rate > 30% = WARNING
            "sl_hit_rate_warning_pct": 50.0,    # SL hit rate > 50% = WARNING
            "profit_milestones": [5.0, 10.0, 25.0, 50.0, 100.0],  # Marcos de lucro %
        }

        logger.info("AlertSystem inicializado com configurações padrão")

    def check_all_conditions(
        self,
        current_capital: float,
        initial_capital: float,
        win_rate_pct: float,
        timeout_rate_pct: float,
        sl_hit_rate_pct: float,
        last_trade: Optional[Dict] = None,
        total_trades: int = 0
    ) -> List[Alert]:
        """
        Verifica todas as condições de alerta

        Args:
            current_capital: Capital atual
            initial_capital: Capital inicial
            win_rate_pct: Win rate em porcentagem
            timeout_rate_pct: Taxa de timeout em porcentagem
            sl_hit_rate_pct: Taxa de SL em porcentagem
            last_trade: Último trade executado (opcional)
            total_trades: Total de trades executados

        Returns:
            Lista de novos alertas gerados
        """
        new_alerts = []

        # Atualizar peak capital
        if self.peak_capital is None or current_capital > self.peak_capital:
            if self.peak_capital is not None and current_capital > self.peak_capital:
                # Novo recorde de capital!
                new_alerts.append(self._create_capital_record_alert(current_capital))
            self.peak_capital = current_capital

        # 1. Verificar Drawdown
        drawdown_alert = self._check_drawdown(current_capital, self.peak_capital)
        if drawdown_alert:
            new_alerts.append(drawdown_alert)

        # 2. Verificar Perdas Consecutivas
        if last_trade:
            consecutive_alert = self._check_consecutive_losses(last_trade)
            if consecutive_alert:
                new_alerts.append(consecutive_alert)

        # 3. Verificar Win Rate Baixo (só após 20+ trades)
        if total_trades >= 20:
            win_rate_alert = self._check_low_win_rate(win_rate_pct, total_trades)
            if win_rate_alert:
                new_alerts.append(win_rate_alert)

        # 4. Verificar Timeout Rate Alto (só após 10+ trades)
        if total_trades >= 10:
            timeout_alert = self._check_high_timeout_rate(timeout_rate_pct, total_trades)
            if timeout_alert:
                new_alerts.append(timeout_alert)

        # 5. Verificar SL Hit Rate Alto (só após 10+ trades)
        if total_trades >= 10:
            sl_alert = self._check_high_sl_hit_rate(sl_hit_rate_pct, total_trades)
            if sl_alert:
                new_alerts.append(sl_alert)

        # 6. Verificar Marcos de Lucro
        profit_milestone_alert = self._check_profit_milestones(
            current_capital, initial_capital
        )
        if profit_milestone_alert:
            new_alerts.append(profit_milestone_alert)

        # 7. Eventos de Trade (TP, Timeout)
        if last_trade:
            trade_event_alert = self._check_trade_events(last_trade)
            if trade_event_alert:
                new_alerts.append(trade_event_alert)

        # Adicionar novos alertas à lista
        self.alerts.extend(new_alerts)
        self.last_capital = current_capital

        # Log novos alertas
        for alert in new_alerts:
            logger.info(f"[{alert.level.value}] {alert.message}")

        return new_alerts

    def _check_drawdown(self, current_capital: float, peak_capital: float) -> Optional[Alert]:
        """Verifica se drawdown atingiu níveis críticos"""
        if peak_capital is None or peak_capital == 0:
            return None

        drawdown_pct = ((peak_capital - current_capital) / peak_capital) * 100

        if drawdown_pct >= self.config["drawdown_critical_pct"]:
            return Alert(
                level=AlertLevel.CRITICAL,
                alert_type=AlertType.HIGH_DRAWDOWN,
                message=f"⚠️ DRAWDOWN CRÍTICO: {drawdown_pct:.1f}%",
                details={
                    "drawdown_pct": drawdown_pct,
                    "current_capital": current_capital,
                    "peak_capital": peak_capital,
                    "threshold": self.config["drawdown_critical_pct"]
                }
            )
        elif drawdown_pct >= self.config["drawdown_warning_pct"]:
            return Alert(
                level=AlertLevel.WARNING,
                alert_type=AlertType.HIGH_DRAWDOWN,
                message=f"⚠️ Drawdown Elevado: {drawdown_pct:.1f}%",
                details={
                    "drawdown_pct": drawdown_pct,
                    "current_capital": current_capital,
                    "peak_capital": peak_capital,
                    "threshold": self.config["drawdown_warning_pct"]
                }
            )

        return None

    def _check_consecutive_losses(self, last_trade: Dict) -> Optional[Alert]:
        """Verifica perdas consecutivas"""
        is_loss = last_trade.get("profit_loss", 0) < 0

        if is_loss:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        if self.consecutive_losses >= self.config["consecutive_losses_critical"]:
            return Alert(
                level=AlertLevel.CRITICAL,
                alert_type=AlertType.CONSECUTIVE_LOSSES,
                message=f"🔴 {self.consecutive_losses} PERDAS CONSECUTIVAS",
                details={
                    "consecutive_losses": self.consecutive_losses,
                    "threshold": self.config["consecutive_losses_critical"],
                    "last_loss": last_trade.get("profit_loss", 0)
                }
            )
        elif self.consecutive_losses >= self.config["consecutive_losses_warning"]:
            return Alert(
                level=AlertLevel.WARNING,
                alert_type=AlertType.CONSECUTIVE_LOSSES,
                message=f"⚠️ {self.consecutive_losses} perdas consecutivas",
                details={
                    "consecutive_losses": self.consecutive_losses,
                    "threshold": self.config["consecutive_losses_warning"]
                }
            )

        return None

    def _check_low_win_rate(self, win_rate_pct: float, total_trades: int) -> Optional[Alert]:
        """Verifica se win rate está muito baixo"""
        if win_rate_pct < self.config["win_rate_warning_pct"]:
            return Alert(
                level=AlertLevel.WARNING,
                alert_type=AlertType.LOW_WIN_RATE,
                message=f"⚠️ Win Rate Baixo: {win_rate_pct:.1f}%",
                details={
                    "win_rate_pct": win_rate_pct,
                    "threshold": self.config["win_rate_warning_pct"],
                    "total_trades": total_trades
                }
            )
        return None

    def _check_high_timeout_rate(self, timeout_rate_pct: float, total_trades: int) -> Optional[Alert]:
        """Verifica se taxa de timeout está muito alta"""
        if timeout_rate_pct > self.config["timeout_rate_warning_pct"]:
            return Alert(
                level=AlertLevel.WARNING,
                alert_type=AlertType.HIGH_TIMEOUT_RATE,
                message=f"⚠️ Taxa de Timeout Elevada: {timeout_rate_pct:.1f}%",
                details={
                    "timeout_rate_pct": timeout_rate_pct,
                    "threshold": self.config["timeout_rate_warning_pct"],
                    "total_trades": total_trades
                }
            )
        return None

    def _check_high_sl_hit_rate(self, sl_hit_rate_pct: float, total_trades: int) -> Optional[Alert]:
        """Verifica se taxa de SL está muito alta"""
        if sl_hit_rate_pct > self.config["sl_hit_rate_warning_pct"]:
            return Alert(
                level=AlertLevel.WARNING,
                alert_type=AlertType.HIGH_SL_HIT_RATE,
                message=f"⚠️ Taxa de Stop Loss Elevada: {sl_hit_rate_pct:.1f}%",
                details={
                    "sl_hit_rate_pct": sl_hit_rate_pct,
                    "threshold": self.config["sl_hit_rate_warning_pct"],
                    "total_trades": total_trades
                }
            )
        return None

    def _check_profit_milestones(self, current_capital: float, initial_capital: float) -> Optional[Alert]:
        """Verifica se atingiu marcos de lucro"""
        if initial_capital == 0:
            return None

        profit_pct = ((current_capital - initial_capital) / initial_capital) * 100

        for milestone in self.config["profit_milestones"]:
            # Verificar se acabamos de ultrapassar esse marco
            if profit_pct >= milestone:
                # Verificar se não foi já alertado (checar alertas anteriores)
                already_alerted = any(
                    a.alert_type == AlertType.PROFIT_MILESTONE and
                    a.details.get("milestone") == milestone
                    for a in self.alerts
                )

                if not already_alerted:
                    return Alert(
                        level=AlertLevel.INFO,
                        alert_type=AlertType.PROFIT_MILESTONE,
                        message=f"🎉 Marco de Lucro Atingido: +{milestone:.0f}%",
                        details={
                            "milestone": milestone,
                            "profit_pct": profit_pct,
                            "current_capital": current_capital,
                            "initial_capital": initial_capital
                        }
                    )

        return None

    def _check_trade_events(self, last_trade: Dict) -> Optional[Alert]:
        """Verifica eventos importantes de trade"""
        exit_reason = last_trade.get("exit_reason")
        profit_loss = last_trade.get("profit_loss", 0)

        # Take Profit atingido
        if exit_reason == "take_profit":
            return Alert(
                level=AlertLevel.INFO,
                alert_type=AlertType.TAKE_PROFIT_HIT,
                message=f"🎯 Take Profit Atingido: +${profit_loss:.2f}",
                details={
                    "trade_id": last_trade.get("id", "unknown"),
                    "profit_loss": profit_loss,
                    "exit_reason": exit_reason
                }
            )

        # Position Timeout
        elif exit_reason == "timeout":
            return Alert(
                level=AlertLevel.INFO,
                alert_type=AlertType.POSITION_TIMEOUT,
                message=f"⏱️ Posição Fechada por Timeout: ${profit_loss:+.2f}",
                details={
                    "trade_id": last_trade.get("id", "unknown"),
                    "profit_loss": profit_loss,
                    "exit_reason": exit_reason
                }
            )

        return None

    def _create_capital_record_alert(self, new_peak: float) -> Alert:
        """Cria alerta de novo recorde de capital"""
        return Alert(
            level=AlertLevel.INFO,
            alert_type=AlertType.CAPITAL_RECORD,
            message=f"🏆 Novo Recorde de Capital: ${new_peak:,.2f}",
            details={
                "new_peak": new_peak,
                "previous_peak": self.peak_capital
            }
        )

    def get_recent_alerts(self, limit: int = 50) -> List[Dict]:
        """Retorna os alertas mais recentes"""
        recent = sorted(self.alerts, key=lambda a: a.timestamp, reverse=True)[:limit]
        return [alert.to_dict() for alert in recent]

    def get_unread_alerts(self) -> List[Dict]:
        """Retorna alertas não lidos"""
        unread = [a for a in self.alerts if not a.read]
        return [alert.to_dict() for alert in unread]

    def mark_as_read(self, alert_id: str) -> bool:
        """Marca alerta como lido"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.read = True
                return True
        return False

    def mark_all_as_read(self) -> int:
        """Marca todos os alertas como lidos"""
        count = 0
        for alert in self.alerts:
            if not alert.read:
                alert.read = True
                count += 1
        return count

    def clear_old_alerts(self, max_age_hours: int = 24):
        """Remove alertas antigos"""
        cutoff = datetime.now().timestamp() - (max_age_hours * 3600)
        self.alerts = [
            a for a in self.alerts
            if a.timestamp.timestamp() > cutoff
        ]
        logger.info(f"Alertas antigos limpos (mantidos últimas {max_age_hours}h)")

    def reset(self):
        """Reseta o sistema de alertas"""
        self.alerts.clear()
        self.last_capital = None
        self.peak_capital = None
        self.consecutive_losses = 0
        logger.info("AlertSystem resetado")
