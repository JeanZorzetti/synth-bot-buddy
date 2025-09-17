"""
Real-Time Alerts - Sistema de Alertas e Notificações em Tempo Real
Sistema completo para alertas de trading, risco, performance e eventos do sistema
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import websockets

from database_config import DatabaseManager
from redis_cache_manager import RedisCacheManager, CacheNamespace
from real_logging_system import RealLoggingSystem

class AlertType(Enum):
    TRADING_SIGNAL = "trading_signal"
    POSITION_UPDATE = "position_update"
    RISK_WARNING = "risk_warning"
    RISK_CRITICAL = "risk_critical"
    PRICE_ALERT = "price_alert"
    PERFORMANCE_MILESTONE = "performance_milestone"
    SYSTEM_ERROR = "system_error"
    SYSTEM_STATUS = "system_status"
    TRADE_EXECUTION = "trade_execution"
    PROFIT_TARGET = "profit_target"
    STOP_LOSS = "stop_loss"

class AlertPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class NotificationChannel(Enum):
    WEBSOCKET = "websocket"
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    DASHBOARD = "dashboard"
    CONSOLE = "console"

@dataclass
class AlertRule:
    """Regra de alerta configurável"""
    rule_id: str
    name: str
    alert_type: AlertType
    priority: AlertPriority
    enabled: bool
    conditions: Dict[str, Any]
    channels: List[NotificationChannel]
    cooldown_minutes: int
    last_triggered: Optional[datetime]
    trigger_count: int
    metadata: Dict[str, Any]

@dataclass
class Alert:
    """Alerta gerado pelo sistema"""
    alert_id: str
    rule_id: Optional[str]
    alert_type: AlertType
    priority: AlertPriority
    title: str
    message: str
    timestamp: datetime
    symbol: Optional[str]
    value: Optional[float]
    threshold: Optional[float]
    action_required: bool
    acknowledged: bool
    channels_sent: List[NotificationChannel]
    metadata: Dict[str, Any]
    expires_at: Optional[datetime]

@dataclass
class NotificationTemplate:
    """Template de notificação"""
    template_id: str
    alert_type: AlertType
    channel: NotificationChannel
    subject_template: str
    body_template: str
    variables: List[str]

class WebSocketManager:
    """Gerenciador de conexões WebSocket"""

    def __init__(self):
        self.connections = set()
        self.subscriptions = {}  # connection -> [alert_types]

    async def add_connection(self, websocket, alert_types: List[AlertType] = None):
        """Adiciona nova conexão WebSocket"""
        self.connections.add(websocket)
        if alert_types:
            self.subscriptions[websocket] = alert_types
        else:
            self.subscriptions[websocket] = list(AlertType)

    async def remove_connection(self, websocket):
        """Remove conexão WebSocket"""
        self.connections.discard(websocket)
        self.subscriptions.pop(websocket, None)

    async def broadcast_alert(self, alert: Alert):
        """Envia alerta para conexões interessadas"""
        message = {
            "type": "alert",
            "data": asdict(alert)
        }

        disconnected = set()
        for websocket in self.connections:
            try:
                subscribed_types = self.subscriptions.get(websocket, [])
                if alert.alert_type in subscribed_types:
                    await websocket.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(websocket)

        # Limpar conexões mortas
        for websocket in disconnected:
            await self.remove_connection(websocket)

class EmailNotifier:
    """Notificador por email"""

    def __init__(self, smtp_host: str = None, smtp_port: int = 587,
                 username: str = None, password: str = None):
        self.smtp_host = smtp_host or "smtp.gmail.com"
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.enabled = bool(username and password)

    async def send_alert(self, alert: Alert, recipient: str):
        """Envia alerta por email"""
        if not self.enabled:
            return False

        try:
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = recipient
            msg['Subject'] = f"[{alert.priority.value.upper()}] {alert.title}"

            # Corpo do email
            body = self._format_email_body(alert)
            msg.attach(MIMEText(body, 'html'))

            # Enviar email
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.send_message(msg)
            server.quit()

            return True

        except Exception as e:
            logging.error(f"Erro ao enviar email: {e}")
            return False

    def _format_email_body(self, alert: Alert) -> str:
        """Formata corpo do email"""
        return f"""
        <html>
        <body>
            <h2>Alert: {alert.title}</h2>
            <p><strong>Tipo:</strong> {alert.alert_type.value}</p>
            <p><strong>Prioridade:</strong> {alert.priority.value}</p>
            <p><strong>Timestamp:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            {f'<p><strong>Símbolo:</strong> {alert.symbol}</p>' if alert.symbol else ''}
            {f'<p><strong>Valor:</strong> {alert.value}</p>' if alert.value else ''}
            {f'<p><strong>Threshold:</strong> {alert.threshold}</p>' if alert.threshold else ''}
            <p><strong>Mensagem:</strong></p>
            <p>{alert.message}</p>
            {f'<p><strong>Ação Requerida:</strong> Sim</p>' if alert.action_required else ''}
        </body>
        </html>
        """

class RealtimeAlerts:
    """Sistema de alertas em tempo real"""

    def __init__(self):
        # Componentes principais
        self.db_manager = DatabaseManager()
        self.cache_manager = RedisCacheManager()
        self.logger = RealLoggingSystem()

        # Gerenciadores de notificação
        self.websocket_manager = WebSocketManager()
        self.email_notifier = EmailNotifier()

        # Estado do sistema
        self.is_active = False
        self.alert_processing_task = None
        self.monitoring_task = None

        # Configurações
        self.max_alerts_per_minute = 50
        self.alert_retention_days = 30
        self.default_cooldown_minutes = 5

        # Dados em memória
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.alert_queue = asyncio.Queue()

        # Templates de notificação
        self.notification_templates: Dict[str, NotificationTemplate] = {}

        # Callbacks externos
        self.external_callbacks: Dict[AlertType, List[Callable]] = {}

        # Métricas
        self.alert_metrics = {
            "total_alerts": 0,
            "alerts_by_type": {},
            "alerts_by_priority": {},
            "notifications_sent": 0,
            "notifications_failed": 0
        }

        logging.basicConfig(level=logging.INFO)
        self.logger_py = logging.getLogger(__name__)

    async def initialize(self):
        """Inicializa o sistema de alertas"""
        try:
            await self.db_manager.initialize()
            await self.cache_manager.initialize()
            await self.logger.initialize()

            # Carregar regras de alerta
            await self._load_alert_rules()

            # Criar templates padrão
            await self._create_default_templates()

            # Criar regras padrão
            await self._create_default_alert_rules()

            await self.logger.log_activity("realtime_alerts_initialized", {
                "alert_rules": len(self.alert_rules),
                "templates": len(self.notification_templates)
            })

            print("✅ Real-Time Alerts inicializado com sucesso")

        except Exception as e:
            await self.logger.log_error("alerts_init_error", str(e))
            raise

    async def start_monitoring(self):
        """Inicia monitoramento de alertas"""
        if self.is_active:
            return

        self.is_active = True

        # Iniciar tasks
        self.alert_processing_task = asyncio.create_task(self._alert_processing_loop())
        self.monitoring_task = asyncio.create_task(self._alert_monitoring_loop())

        await self.logger.log_activity("alert_monitoring_started", {})
        print("🚨 Monitoramento de alertas iniciado")

    async def stop_monitoring(self):
        """Para o monitoramento"""
        self.is_active = False

        # Cancelar tasks
        for task in [self.alert_processing_task, self.monitoring_task]:
            if task:
                task.cancel()

        await self.logger.log_activity("alert_monitoring_stopped", {})
        print("⏹️ Monitoramento de alertas parado")

    async def create_alert(
        self,
        alert_type: AlertType,
        title: str,
        message: str,
        priority: AlertPriority = AlertPriority.MEDIUM,
        symbol: Optional[str] = None,
        value: Optional[float] = None,
        threshold: Optional[float] = None,
        action_required: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Cria um novo alerta"""
        try:
            alert_id = f"alert_{int(datetime.utcnow().timestamp() * 1000)}"

            alert = Alert(
                alert_id=alert_id,
                rule_id=None,
                alert_type=alert_type,
                priority=priority,
                title=title,
                message=message,
                timestamp=datetime.utcnow(),
                symbol=symbol,
                value=value,
                threshold=threshold,
                action_required=action_required,
                acknowledged=False,
                channels_sent=[],
                metadata=metadata or {},
                expires_at=datetime.utcnow() + timedelta(hours=24)
            )

            # Adicionar à fila de processamento
            await self.alert_queue.put(alert)

            # Atualizar métricas
            self.alert_metrics["total_alerts"] += 1
            type_count = self.alert_metrics["alerts_by_type"].get(alert_type.value, 0)
            self.alert_metrics["alerts_by_type"][alert_type.value] = type_count + 1

            priority_count = self.alert_metrics["alerts_by_priority"].get(priority.value, 0)
            self.alert_metrics["alerts_by_priority"][priority.value] = priority_count + 1

            return alert_id

        except Exception as e:
            await self.logger.log_error("alert_creation_error", str(e))
            raise

    async def _alert_processing_loop(self):
        """Loop de processamento de alertas"""
        while self.is_active:
            try:
                # Processar alertas da fila
                try:
                    alert = await asyncio.wait_for(self.alert_queue.get(), timeout=1.0)
                    await self._process_alert(alert)
                except asyncio.TimeoutError:
                    continue

            except Exception as e:
                await self.logger.log_error("alert_processing_error", str(e))
                await asyncio.sleep(1)

    async def _process_alert(self, alert: Alert):
        """Processa um alerta específico"""
        try:
            # Verificar se alerta já expirou
            if alert.expires_at and datetime.utcnow() > alert.expires_at:
                return

            # Verificar regras de cooldown
            if not await self._check_cooldown(alert):
                return

            # Adicionar aos alertas ativos
            self.active_alerts.append(alert)
            self.alert_history.append(alert)

            # Determinar canais de notificação
            channels = await self._determine_notification_channels(alert)

            # Enviar notificações
            await self._send_notifications(alert, channels)

            # Executar callbacks externos
            await self._execute_external_callbacks(alert)

            # Salvar no banco
            await self._save_alert_to_db(alert)

            # Cache do alerta
            await self.cache_manager.set(
                CacheNamespace.USER_SESSIONS,
                f"alert_{alert.alert_id}",
                asdict(alert),
                ttl=3600
            )

            await self.logger.log_activity("alert_processed", {
                "alert_id": alert.alert_id,
                "type": alert.alert_type.value,
                "priority": alert.priority.value,
                "channels": [c.value for c in channels]
            })

            print(f"🚨 Alerta processado: {alert.title} [{alert.priority.value}]")

        except Exception as e:
            await self.logger.log_error("alert_processing_individual_error", f"{alert.alert_id}: {str(e)}")

    async def _check_cooldown(self, alert: Alert) -> bool:
        """Verifica cooldown para evitar spam de alertas"""
        try:
            # Verificar alertas similares recentes
            cutoff_time = datetime.utcnow() - timedelta(minutes=self.default_cooldown_minutes)

            similar_alerts = [
                a for a in self.active_alerts[-50:]  # Últimos 50 alertas
                if (a.alert_type == alert.alert_type and
                    a.symbol == alert.symbol and
                    a.timestamp > cutoff_time)
            ]

            # Se há muitos alertas similares, aplicar cooldown
            if len(similar_alerts) >= 3:
                return False

            return True

        except Exception as e:
            await self.logger.log_error("cooldown_check_error", str(e))
            return True

    async def _determine_notification_channels(self, alert: Alert) -> List[NotificationChannel]:
        """Determina quais canais usar para o alerta"""
        channels = [NotificationChannel.DASHBOARD, NotificationChannel.CONSOLE]

        # WebSocket sempre para alertas ativos
        channels.append(NotificationChannel.WEBSOCKET)

        # Email para alertas de alta prioridade
        if alert.priority in [AlertPriority.HIGH, AlertPriority.CRITICAL, AlertPriority.EMERGENCY]:
            channels.append(NotificationChannel.EMAIL)

        return channels

    async def _send_notifications(self, alert: Alert, channels: List[NotificationChannel]):
        """Envia notificações pelos canais especificados"""
        for channel in channels:
            try:
                success = False

                if channel == NotificationChannel.WEBSOCKET:
                    await self.websocket_manager.broadcast_alert(alert)
                    success = True

                elif channel == NotificationChannel.EMAIL:
                    # Enviar para admin (em produção, obter da configuração)
                    admin_email = "admin@example.com"
                    success = await self.email_notifier.send_alert(alert, admin_email)

                elif channel == NotificationChannel.CONSOLE:
                    self._print_console_alert(alert)
                    success = True

                elif channel == NotificationChannel.DASHBOARD:
                    # Salvar para dashboard
                    await self._save_dashboard_alert(alert)
                    success = True

                if success:
                    alert.channels_sent.append(channel)
                    self.alert_metrics["notifications_sent"] += 1
                else:
                    self.alert_metrics["notifications_failed"] += 1

            except Exception as e:
                await self.logger.log_error("notification_send_error", f"{channel.value}: {str(e)}")
                self.alert_metrics["notifications_failed"] += 1

    def _print_console_alert(self, alert: Alert):
        """Imprime alerta no console"""
        priority_icons = {
            AlertPriority.LOW: "ℹ️",
            AlertPriority.MEDIUM: "⚠️",
            AlertPriority.HIGH: "🔥",
            AlertPriority.CRITICAL: "🚨",
            AlertPriority.EMERGENCY: "🆘"
        }

        icon = priority_icons.get(alert.priority, "📢")
        symbol_text = f" [{alert.symbol}]" if alert.symbol else ""

        print(f"{icon} {alert.title}{symbol_text}")
        print(f"   {alert.message}")
        if alert.action_required:
            print("   ⚡ AÇÃO REQUERIDA")

    async def _save_dashboard_alert(self, alert: Alert):
        """Salva alerta para o dashboard"""
        try:
            await self.cache_manager.set(
                CacheNamespace.USER_SESSIONS,
                f"dashboard_alert_{alert.alert_id}",
                asdict(alert),
                ttl=86400  # 24 horas
            )
        except Exception as e:
            await self.logger.log_error("dashboard_alert_save_error", str(e))

    async def _execute_external_callbacks(self, alert: Alert):
        """Executa callbacks externos registrados"""
        try:
            callbacks = self.external_callbacks.get(alert.alert_type, [])
            for callback in callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        callback(alert)
                except Exception as e:
                    await self.logger.log_error("external_callback_error", str(e))

        except Exception as e:
            await self.logger.log_error("external_callbacks_execution_error", str(e))

    async def _alert_monitoring_loop(self):
        """Loop de monitoramento e limpeza"""
        while self.is_active:
            try:
                # Limpar alertas expirados
                await self._cleanup_expired_alerts()

                # Atualizar métricas
                await self._update_alert_metrics()

                # Verificar alertas não acknowledgment críticos
                await self._check_unacknowledged_critical_alerts()

                await asyncio.sleep(60)  # A cada minuto

            except Exception as e:
                await self.logger.log_error("alert_monitoring_error", str(e))
                await asyncio.sleep(300)

    async def _cleanup_expired_alerts(self):
        """Remove alertas expirados"""
        try:
            now = datetime.utcnow()

            # Remover alertas ativos expirados
            self.active_alerts = [
                alert for alert in self.active_alerts
                if not alert.expires_at or alert.expires_at > now
            ]

            # Manter apenas histórico recente
            cutoff_date = now - timedelta(days=self.alert_retention_days)
            self.alert_history = [
                alert for alert in self.alert_history
                if alert.timestamp > cutoff_date
            ]

        except Exception as e:
            await self.logger.log_error("alert_cleanup_error", str(e))

    async def _check_unacknowledged_critical_alerts(self):
        """Verifica alertas críticos não acknowledgment"""
        try:
            critical_alerts = [
                alert for alert in self.active_alerts
                if alert.priority in [AlertPriority.CRITICAL, AlertPriority.EMERGENCY]
                and not alert.acknowledged
                and alert.action_required
            ]

            if critical_alerts:
                # Re-enviar notificações para alertas críticos não acknowledgment há mais de 10 minutos
                cutoff_time = datetime.utcnow() - timedelta(minutes=10)

                for alert in critical_alerts:
                    if alert.timestamp < cutoff_time:
                        await self._resend_critical_alert(alert)

        except Exception as e:
            await self.logger.log_error("critical_alerts_check_error", str(e))

    async def _resend_critical_alert(self, alert: Alert):
        """Re-envia alerta crítico"""
        try:
            # Criar novo alerta com prioridade elevada
            escalated_alert = Alert(
                alert_id=f"escalated_{alert.alert_id}",
                rule_id=alert.rule_id,
                alert_type=alert.alert_type,
                priority=AlertPriority.EMERGENCY,
                title=f"ESCALADO: {alert.title}",
                message=f"Alerta crítico não acknowledgment: {alert.message}",
                timestamp=datetime.utcnow(),
                symbol=alert.symbol,
                value=alert.value,
                threshold=alert.threshold,
                action_required=True,
                acknowledged=False,
                channels_sent=[],
                metadata=alert.metadata,
                expires_at=datetime.utcnow() + timedelta(hours=1)
            )

            await self.alert_queue.put(escalated_alert)

        except Exception as e:
            await self.logger.log_error("critical_alert_resend_error", str(e))

    # Métodos de configuração e gerenciamento
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledga um alerta"""
        try:
            alert = next((a for a in self.active_alerts if a.alert_id == alert_id), None)
            if alert:
                alert.acknowledged = True
                await self._update_alert_in_db(alert)

                await self.logger.log_activity("alert_acknowledged", {
                    "alert_id": alert_id,
                    "type": alert.alert_type.value
                })

                return True
            return False

        except Exception as e:
            await self.logger.log_error("alert_acknowledge_error", f"{alert_id}: {str(e)}")
            return False

    async def register_callback(self, alert_type: AlertType, callback: Callable):
        """Registra callback externo para tipo de alerta"""
        if alert_type not in self.external_callbacks:
            self.external_callbacks[alert_type] = []

        self.external_callbacks[alert_type].append(callback)

    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Retorna alertas ativos"""
        try:
            return [asdict(alert) for alert in self.active_alerts]
        except Exception as e:
            await self.logger.log_error("active_alerts_query_error", str(e))
            return []

    async def get_alert_metrics(self) -> Dict[str, Any]:
        """Retorna métricas dos alertas"""
        try:
            return {
                **self.alert_metrics,
                "active_alerts_count": len(self.active_alerts),
                "unacknowledged_critical": len([
                    a for a in self.active_alerts
                    if a.priority in [AlertPriority.CRITICAL, AlertPriority.EMERGENCY]
                    and not a.acknowledged
                ]),
                "system_status": "active" if self.is_active else "stopped"
            }
        except Exception as e:
            await self.logger.log_error("alert_metrics_error", str(e))
            return {}

    # Métodos auxiliares para configuração
    async def _create_default_alert_rules(self):
        """Cria regras de alerta padrão"""
        default_rules = [
            {
                "rule_id": "daily_loss_limit",
                "name": "Limite de Perda Diária",
                "alert_type": AlertType.RISK_CRITICAL,
                "priority": AlertPriority.CRITICAL,
                "conditions": {"daily_loss_pct": 5.0},
                "channels": [NotificationChannel.EMAIL, NotificationChannel.WEBSOCKET],
                "cooldown_minutes": 60
            },
            {
                "rule_id": "large_position_opened",
                "name": "Posição Grande Aberta",
                "alert_type": AlertType.POSITION_UPDATE,
                "priority": AlertPriority.MEDIUM,
                "conditions": {"position_size_pct": 10.0},
                "channels": [NotificationChannel.WEBSOCKET],
                "cooldown_minutes": 5
            }
        ]

        for rule_data in default_rules:
            rule = AlertRule(
                rule_id=rule_data["rule_id"],
                name=rule_data["name"],
                alert_type=rule_data["alert_type"],
                priority=rule_data["priority"],
                enabled=True,
                conditions=rule_data["conditions"],
                channels=rule_data["channels"],
                cooldown_minutes=rule_data["cooldown_minutes"],
                last_triggered=None,
                trigger_count=0,
                metadata={}
            )
            self.alert_rules[rule.rule_id] = rule

    async def _create_default_templates(self):
        """Cria templates padrão de notificação"""
        # Implementação de templates básicos
        pass

    # Métodos auxiliares para banco de dados
    async def _load_alert_rules(self):
        """Carrega regras de alerta do banco"""
        try:
            # Implementar carregamento do banco
            pass
        except Exception as e:
            await self.logger.log_error("alert_rules_load_error", str(e))

    async def _save_alert_to_db(self, alert: Alert):
        """Salva alerta no banco"""
        try:
            # Implementar salvamento no banco
            pass
        except Exception as e:
            await self.logger.log_error("alert_db_save_error", str(e))

    async def _update_alert_in_db(self, alert: Alert):
        """Atualiza alerta no banco"""
        try:
            # Implementar atualização no banco
            pass
        except Exception as e:
            await self.logger.log_error("alert_db_update_error", str(e))

    async def _update_alert_metrics(self):
        """Atualiza e salva métricas"""
        try:
            await self.cache_manager.set(
                CacheNamespace.USER_SESSIONS,
                "alert_metrics",
                self.alert_metrics,
                ttl=300
            )
        except Exception as e:
            await self.logger.log_error("alert_metrics_update_error", str(e))

    # Métodos convenientes para criação de alertas específicos
    async def alert_trading_signal(self, symbol: str, signal_type: str, confidence: float):
        """Alerta de sinal de trading"""
        await self.create_alert(
            AlertType.TRADING_SIGNAL,
            f"Sinal de {signal_type.upper()}",
            f"Sinal {signal_type} detectado para {symbol} com confiança {confidence:.1%}",
            AlertPriority.MEDIUM,
            symbol=symbol,
            value=confidence,
            metadata={"signal_type": signal_type}
        )

    async def alert_risk_warning(self, message: str, value: float, threshold: float):
        """Alerta de aviso de risco"""
        await self.create_alert(
            AlertType.RISK_WARNING,
            "Aviso de Risco",
            message,
            AlertPriority.HIGH,
            value=value,
            threshold=threshold,
            action_required=True
        )

    async def alert_trade_execution(self, symbol: str, trade_type: str, amount: float, price: float):
        """Alerta de execução de trade"""
        await self.create_alert(
            AlertType.TRADE_EXECUTION,
            "Trade Executado",
            f"Trade {trade_type} executado: {amount} {symbol} @ {price}",
            AlertPriority.LOW,
            symbol=symbol,
            value=amount,
            metadata={"trade_type": trade_type, "price": price}
        )

    async def alert_system_error(self, error_message: str, component: str):
        """Alerta de erro do sistema"""
        await self.create_alert(
            AlertType.SYSTEM_ERROR,
            "Erro do Sistema",
            f"Erro em {component}: {error_message}",
            AlertPriority.HIGH,
            action_required=True,
            metadata={"component": component, "error": error_message}
        )

    async def shutdown(self):
        """Encerra o sistema de alertas"""
        await self.stop_monitoring()
        await self.logger.log_activity("realtime_alerts_shutdown", {})
        print("🔌 Real-Time Alerts encerrado")