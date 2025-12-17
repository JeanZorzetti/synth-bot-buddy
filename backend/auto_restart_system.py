"""
Auto-Restart System - Sistema de Recuperação Automática

Este módulo implementa um watchdog que monitora o Forward Testing
e reinicia automaticamente em caso de crash ou falha.

Features:
1. Health check periódico
2. Detecção de crashes
3. Salvamento de checkpoint (estado do sistema)
4. Restauração automática após crash
5. Logging detalhado de incidentes
"""

import asyncio
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Callable
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class SystemCheckpoint:
    """Estado do sistema para recuperação"""
    timestamp: str
    symbol: str
    capital: float
    total_trades: int
    win_rate: float
    is_running: bool
    last_prediction_time: Optional[str]
    open_positions: list  # IDs das posições abertas

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)


class AutoRestartSystem:
    """
    Sistema de Auto-Restart para Forward Testing

    Monitora a saúde do sistema e reinicia automaticamente em caso de falha.
    """

    def __init__(
        self,
        check_interval: int = 30,  # Verificar a cada 30 segundos
        max_failures: int = 3,  # Número de falhas antes de tentar restart
        checkpoint_dir: str = "checkpoints"
    ):
        """
        Inicializa o sistema de auto-restart

        Args:
            check_interval: Intervalo em segundos entre health checks
            max_failures: Número máximo de falhas consecutivas antes de restart
            checkpoint_dir: Diretório para salvar checkpoints
        """
        self.check_interval = check_interval
        self.max_failures = max_failures
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Estado do watchdog
        self.is_monitoring = False
        self.consecutive_failures = 0
        self.last_successful_check: Optional[datetime] = None
        self.restart_count = 0

        # Callbacks
        self.health_check_callback: Optional[Callable] = None
        self.restart_callback: Optional[Callable] = None

        # Histórico de incidentes
        self.incident_log = []

        logger.info(f"AutoRestartSystem inicializado: check_interval={check_interval}s, max_failures={max_failures}")

    def set_health_check_callback(self, callback: Callable):
        """
        Define callback para verificar saúde do sistema

        Args:
            callback: Função assíncrona que retorna True se sistema está saudável
        """
        self.health_check_callback = callback

    def set_restart_callback(self, callback: Callable):
        """
        Define callback para reiniciar o sistema

        Args:
            callback: Função assíncrona que reinicia o sistema
        """
        self.restart_callback = callback

    async def start_monitoring(self):
        """Inicia monitoramento do sistema"""
        if self.is_monitoring:
            logger.warning("Watchdog já está rodando")
            return

        self.is_monitoring = True
        logger.info("🔍 Watchdog iniciado - Monitorando saúde do sistema")

        try:
            while self.is_monitoring:
                await self._check_system_health()
                await asyncio.sleep(self.check_interval)
        except Exception as e:
            logger.error(f"Erro crítico no watchdog: {e}", exc_info=True)
            self.is_monitoring = False

    def stop_monitoring(self):
        """Para monitoramento do sistema"""
        self.is_monitoring = False
        logger.info("🛑 Watchdog parado")

    async def _check_system_health(self):
        """
        Verifica saúde do sistema

        Chama o health_check_callback e registra resultado
        """
        try:
            if not self.health_check_callback:
                logger.warning("Health check callback não configurado")
                return

            # Executar health check
            is_healthy = await self.health_check_callback()

            if is_healthy:
                # Sistema saudável
                self.consecutive_failures = 0
                self.last_successful_check = datetime.now()
                logger.debug("✅ Health check passou")
            else:
                # Sistema com problema
                self.consecutive_failures += 1
                logger.warning(f"⚠️ Health check falhou ({self.consecutive_failures}/{self.max_failures})")

                # Registrar incidente
                self._log_incident(
                    incident_type="health_check_failed",
                    description=f"Health check falhou {self.consecutive_failures} vez(es) consecutiva(s)",
                    severity="WARNING" if self.consecutive_failures < self.max_failures else "CRITICAL"
                )

                # Se atingiu limite de falhas, tentar restart
                if self.consecutive_failures >= self.max_failures:
                    await self._attempt_restart()

        except Exception as e:
            logger.error(f"Erro durante health check: {e}", exc_info=True)
            self.consecutive_failures += 1

            self._log_incident(
                incident_type="health_check_error",
                description=f"Exceção durante health check: {str(e)}",
                severity="ERROR"
            )

            if self.consecutive_failures >= self.max_failures:
                await self._attempt_restart()

    async def _attempt_restart(self):
        """
        Tenta reiniciar o sistema automaticamente
        """
        try:
            logger.warning(f"🔄 Tentando auto-restart (tentativa #{self.restart_count + 1})")

            self._log_incident(
                incident_type="auto_restart_initiated",
                description=f"Auto-restart iniciado após {self.consecutive_failures} falhas consecutivas",
                severity="CRITICAL"
            )

            if not self.restart_callback:
                logger.error("Restart callback não configurado - não é possível reiniciar")
                return

            # Salvar checkpoint antes de restart
            # (assumindo que o callback de restart irá carregar o checkpoint)

            # Executar restart
            success = await self.restart_callback()

            if success:
                logger.info("✅ Auto-restart completado com sucesso")
                self.consecutive_failures = 0
                self.restart_count += 1

                self._log_incident(
                    incident_type="auto_restart_success",
                    description=f"Sistema reiniciado com sucesso (restart #{self.restart_count})",
                    severity="INFO"
                )
            else:
                logger.error("❌ Auto-restart falhou")

                self._log_incident(
                    incident_type="auto_restart_failed",
                    description="Falha ao reiniciar o sistema automaticamente",
                    severity="CRITICAL"
                )

        except Exception as e:
            logger.error(f"Erro durante auto-restart: {e}", exc_info=True)

            self._log_incident(
                incident_type="auto_restart_error",
                description=f"Exceção durante auto-restart: {str(e)}",
                severity="CRITICAL"
            )

    def save_checkpoint(self, checkpoint: SystemCheckpoint):
        """
        Salva checkpoint do sistema

        Args:
            checkpoint: Estado atual do sistema
        """
        try:
            checkpoint_file = self.checkpoint_dir / "latest_checkpoint.json"

            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint.to_dict(), f, indent=2)

            logger.info(f"💾 Checkpoint salvo: {checkpoint.total_trades} trades, capital=${checkpoint.capital:.2f}")

        except Exception as e:
            logger.error(f"Erro ao salvar checkpoint: {e}", exc_info=True)

    def load_checkpoint(self) -> Optional[SystemCheckpoint]:
        """
        Carrega último checkpoint salvo

        Returns:
            SystemCheckpoint ou None se não existir
        """
        try:
            checkpoint_file = self.checkpoint_dir / "latest_checkpoint.json"

            if not checkpoint_file.exists():
                logger.info("Nenhum checkpoint encontrado")
                return None

            with open(checkpoint_file, 'r') as f:
                data = json.load(f)

            checkpoint = SystemCheckpoint.from_dict(data)
            logger.info(f"📂 Checkpoint carregado: {checkpoint.total_trades} trades, capital=${checkpoint.capital:.2f}")

            return checkpoint

        except Exception as e:
            logger.error(f"Erro ao carregar checkpoint: {e}", exc_info=True)
            return None

    def _log_incident(self, incident_type: str, description: str, severity: str):
        """
        Registra incidente no log

        Args:
            incident_type: Tipo do incidente
            description: Descrição detalhada
            severity: Severidade (INFO, WARNING, ERROR, CRITICAL)
        """
        incident = {
            'timestamp': datetime.now().isoformat(),
            'type': incident_type,
            'description': description,
            'severity': severity,
            'consecutive_failures': self.consecutive_failures,
            'restart_count': self.restart_count
        }

        self.incident_log.append(incident)

        # Salvar em arquivo
        incident_file = self.checkpoint_dir / "incidents.jsonl"
        with open(incident_file, 'a') as f:
            f.write(json.dumps(incident) + '\n')

    def get_status(self) -> Dict:
        """
        Retorna status do watchdog

        Returns:
            Dict com métricas do watchdog
        """
        time_since_last_check = None
        if self.last_successful_check:
            time_since_last_check = (datetime.now() - self.last_successful_check).total_seconds()

        return {
            'is_monitoring': self.is_monitoring,
            'consecutive_failures': self.consecutive_failures,
            'last_successful_check': self.last_successful_check.isoformat() if self.last_successful_check else None,
            'time_since_last_check_seconds': time_since_last_check,
            'restart_count': self.restart_count,
            'total_incidents': len(self.incident_log),
            'recent_incidents': self.incident_log[-10:]  # Últimos 10 incidentes
        }
