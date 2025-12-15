"""
Sistema de Alertas
Notificações via Discord, Telegram e Email
"""

import os
import logging
import asyncio
import aiohttp
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Níveis de alerta"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Canais de notificação"""
    DISCORD = "discord"
    TELEGRAM = "telegram"
    EMAIL = "email"


@dataclass
class AlertConfig:
    """Configuração de alertas"""
    # Discord
    discord_webhook_url: Optional[str] = None

    # Telegram
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

    # Email
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    email_from: Optional[str] = None
    email_to: List[str] = None

    # Configurações gerais
    enabled_channels: List[AlertChannel] = None
    min_level: AlertLevel = AlertLevel.WARNING


class AlertsManager:
    """
    Gerenciador de alertas multi-canal
    """

    def __init__(self, config: Optional[AlertConfig] = None):
        """
        Inicializa alerts manager

        Args:
            config: Configuração de alertas (carrega de env vars se None)
        """
        self.config = config or self._load_config_from_env()
        self.alert_history: List[Dict] = []
        self.max_history = 100

        logger.info("Alerts Manager inicializado")
        logger.info(f"  Canais habilitados: {[c.value for c in (self.config.enabled_channels or [])]}")
        logger.info(f"  Nível mínimo: {self.config.min_level.value}")

    def _load_config_from_env(self) -> AlertConfig:
        """Carrega configuração de environment variables"""
        enabled_channels = []

        # Discord
        discord_webhook = os.getenv("DISCORD_WEBHOOK_URL")
        if discord_webhook:
            enabled_channels.append(AlertChannel.DISCORD)

        # Telegram
        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
        telegram_chat = os.getenv("TELEGRAM_CHAT_ID")
        if telegram_token and telegram_chat:
            enabled_channels.append(AlertChannel.TELEGRAM)

        # Email
        smtp_server = os.getenv("SMTP_SERVER")
        smtp_username = os.getenv("SMTP_USERNAME")
        if smtp_server and smtp_username:
            enabled_channels.append(AlertChannel.EMAIL)

        email_to_str = os.getenv("EMAIL_TO", "")
        email_to = [e.strip() for e in email_to_str.split(",") if e.strip()]

        return AlertConfig(
            discord_webhook_url=discord_webhook,
            telegram_bot_token=telegram_token,
            telegram_chat_id=telegram_chat,
            smtp_server=smtp_server,
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            smtp_username=smtp_username,
            smtp_password=os.getenv("SMTP_PASSWORD"),
            email_from=os.getenv("EMAIL_FROM", smtp_username),
            email_to=email_to,
            enabled_channels=enabled_channels,
            min_level=AlertLevel[os.getenv("ALERT_MIN_LEVEL", "WARNING")]
        )

    def _should_send_alert(self, level: AlertLevel) -> bool:
        """Verifica se deve enviar alerta baseado no nível"""
        level_priority = {
            AlertLevel.INFO: 0,
            AlertLevel.WARNING: 1,
            AlertLevel.ERROR: 2,
            AlertLevel.CRITICAL: 3
        }

        return level_priority[level] >= level_priority[self.config.min_level]

    async def send_discord(self, title: str, message: str, level: AlertLevel):
        """
        Envia alerta via Discord webhook

        Args:
            title: Título do alerta
            message: Mensagem do alerta
            level: Nível de severidade
        """
        if not self.config.discord_webhook_url:
            logger.warning("Discord webhook não configurado")
            return

        # Cores por nível
        colors = {
            AlertLevel.INFO: 3447003,      # Azul
            AlertLevel.WARNING: 16776960,  # Amarelo
            AlertLevel.ERROR: 15158332,    # Laranja
            AlertLevel.CRITICAL: 15158332  # Vermelho
        }

        # Emojis por nível
        emojis = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨"
        }

        embed = {
            "title": f"{emojis[level]} {title}",
            "description": message,
            "color": colors[level],
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {
                "text": "Deriv Bot Buddy - Alert System"
            }
        }

        payload = {"embeds": [embed]}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.discord_webhook_url,
                    json=payload
                ) as response:
                    if response.status == 204:
                        logger.info(f"Alerta Discord enviado: {title}")
                    else:
                        logger.error(f"Erro ao enviar Discord: {response.status}")
        except Exception as e:
            logger.error(f"Erro ao enviar Discord: {e}")

    async def send_telegram(self, title: str, message: str, level: AlertLevel):
        """
        Envia alerta via Telegram bot

        Args:
            title: Título do alerta
            message: Mensagem do alerta
            level: Nível de severidade
        """
        if not self.config.telegram_bot_token or not self.config.telegram_chat_id:
            logger.warning("Telegram não configurado")
            return

        # Emojis por nível
        emojis = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨"
        }

        # Formatar mensagem
        text = f"{emojis[level]} *{title}*\n\n{message}"

        url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": self.config.telegram_chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"Alerta Telegram enviado: {title}")
                    else:
                        logger.error(f"Erro ao enviar Telegram: {response.status}")
        except Exception as e:
            logger.error(f"Erro ao enviar Telegram: {e}")

    async def send_email(self, title: str, message: str, level: AlertLevel):
        """
        Envia alerta via Email

        Args:
            title: Título do alerta (subject)
            message: Mensagem do alerta
            level: Nível de severidade
        """
        if not all([
            self.config.smtp_server,
            self.config.smtp_username,
            self.config.email_from,
            self.config.email_to
        ]):
            logger.warning("Email não configurado completamente")
            return

        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            # Criar mensagem
            msg = MIMEMultipart()
            msg['From'] = self.config.email_from
            msg['To'] = ', '.join(self.config.email_to)
            msg['Subject'] = f"[{level.value.upper()}] {title}"

            # HTML body
            html = f"""
            <html>
                <body style="font-family: Arial, sans-serif;">
                    <h2 style="color: {'#f00' if level == AlertLevel.CRITICAL else '#ff9800' if level == AlertLevel.ERROR else '#ffc107' if level == AlertLevel.WARNING else '#2196F3'};">
                        {title}
                    </h2>
                    <p>{message.replace(chr(10), '<br>')}</p>
                    <hr>
                    <p style="color: #666; font-size: 12px;">
                        Deriv Bot Buddy - Alert System<br>
                        {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    </p>
                </body>
            </html>
            """

            msg.attach(MIMEText(html, 'html'))

            # Enviar
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                if self.config.smtp_password:
                    server.login(self.config.smtp_username, self.config.smtp_password)
                server.send_message(msg)

            logger.info(f"Alerta Email enviado: {title}")

        except Exception as e:
            logger.error(f"Erro ao enviar Email: {e}")

    async def send_alert(
        self,
        title: str,
        message: str,
        level: AlertLevel = AlertLevel.WARNING,
        channels: Optional[List[AlertChannel]] = None
    ):
        """
        Envia alerta para os canais especificados

        Args:
            title: Título do alerta
            message: Mensagem do alerta
            level: Nível de severidade
            channels: Canais específicos (usa enabled_channels se None)
        """
        if not self._should_send_alert(level):
            logger.debug(f"Alerta ignorado (nível {level.value} < {self.config.min_level.value})")
            return

        # Usar canais habilitados se não especificado
        if channels is None:
            channels = self.config.enabled_channels or []

        # Registrar no histórico
        alert_record = {
            'timestamp': datetime.now().isoformat(),
            'title': title,
            'message': message,
            'level': level.value,
            'channels': [c.value for c in channels]
        }
        self.alert_history.append(alert_record)

        # Limitar histórico
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history:]

        logger.info(f"Enviando alerta [{level.value}]: {title}")

        # Enviar para cada canal em paralelo
        tasks = []

        if AlertChannel.DISCORD in channels:
            tasks.append(self.send_discord(title, message, level))

        if AlertChannel.TELEGRAM in channels:
            tasks.append(self.send_telegram(title, message, level))

        if AlertChannel.EMAIL in channels:
            tasks.append(self.send_email(title, message, level))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def get_history(self, limit: int = 20) -> List[Dict]:
        """
        Retorna histórico de alertas

        Args:
            limit: Número máximo de alertas

        Returns:
            Lista de alertas recentes
        """
        return self.alert_history[-limit:]


# Instância global
_alerts_manager: Optional[AlertsManager] = None


def get_alerts_manager() -> AlertsManager:
    """
    Retorna instância global do alerts manager (singleton)

    Returns:
        AlertsManager instance
    """
    global _alerts_manager
    if _alerts_manager is None:
        _alerts_manager = AlertsManager()
    return _alerts_manager


def initialize_alerts_manager(config: Optional[AlertConfig] = None):
    """Inicializa alerts manager global"""
    global _alerts_manager
    _alerts_manager = AlertsManager(config)
    logger.info("Alerts manager inicializado")


# Funções de conveniência para alertas comuns
async def alert_trade_executed(symbol: str, signal_type: str, profit: float):
    """Alerta de trade executado"""
    manager = get_alerts_manager()
    level = AlertLevel.INFO if profit >= 0 else AlertLevel.WARNING

    await manager.send_alert(
        title=f"Trade Executado: {symbol}",
        message=f"Tipo: {signal_type}\nLucro: ${profit:.2f}",
        level=level
    )


async def alert_high_win_rate(win_rate: float, trades_count: int):
    """Alerta de alta taxa de acerto"""
    manager = get_alerts_manager()

    await manager.send_alert(
        title="Alta Taxa de Acerto!",
        message=f"Win Rate: {win_rate:.2f}%\nTrades: {trades_count}",
        level=AlertLevel.INFO
    )


async def alert_circuit_breaker_open(service: str):
    """Alerta de circuit breaker aberto"""
    manager = get_alerts_manager()

    await manager.send_alert(
        title=f"Circuit Breaker Aberto: {service}",
        message=f"O serviço {service} está falhando. Circuit breaker foi ativado.",
        level=AlertLevel.ERROR
    )


async def alert_system_error(error_message: str):
    """Alerta de erro crítico do sistema"""
    manager = get_alerts_manager()

    await manager.send_alert(
        title="Erro Crítico do Sistema",
        message=error_message,
        level=AlertLevel.CRITICAL
    )


if __name__ == "__main__":
    # Teste do alerts manager
    logging.basicConfig(level=logging.INFO)

    async def test():
        # Criar manager
        manager = AlertsManager(AlertConfig(
            discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL"),
            enabled_channels=[AlertChannel.DISCORD] if os.getenv("DISCORD_WEBHOOK_URL") else []
        ))

        # Teste de alertas (só funciona se webhook configurado)
        if manager.config.enabled_channels:
            await manager.send_alert(
                title="Teste de Alert System",
                message="Sistema de alertas funcionando corretamente!",
                level=AlertLevel.INFO
            )

            logger.info("Alerta de teste enviado!")
        else:
            logger.info("Nenhum canal configurado. Configure DISCORD_WEBHOOK_URL, TELEGRAM_BOT_TOKEN, etc.")

        # Exibir histórico
        history = manager.get_history()
        logger.info(f"Histórico de alertas: {len(history)} entradas")

    asyncio.run(test())
