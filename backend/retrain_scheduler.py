"""
Scheduler para Retreinamento Automático de Modelos ML

Executa retreinamento em intervalos regulares (semanal/mensal)
usando APScheduler
"""

import logging
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from ml_retrain_service import get_retrain_service

logger = logging.getLogger(__name__)


class RetrainScheduler:
    """
    Scheduler para retreinamento automático

    Executa retreinamento:
    - Semanalmente: Domingo às 3 AM
    - Pode ser configurado para outros intervalos
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.scheduler = BackgroundScheduler()
        self.retrain_service = get_retrain_service()

        if self.enabled:
            self._setup_jobs()

    def _setup_jobs(self):
        """Configura jobs de retreinamento"""

        # Retreinamento semanal: Domingo às 3 AM
        self.scheduler.add_job(
            func=self._weekly_retrain,
            trigger=CronTrigger(day_of_week='sun', hour=3, minute=0),
            id='weekly_retrain',
            name='Weekly Model Retrain',
            replace_existing=True
        )

        logger.info("✅ Scheduler de retreinamento configurado: Domingos às 3 AM")

    def _weekly_retrain(self):
        """Executa retreinamento semanal"""
        try:
            logger.info("🤖 Iniciando retreinamento semanal automático...")

            result = self.retrain_service.execute_retrain(force=False)

            if result['success']:
                logger.info(f"✅ Retreinamento concluído: {result['message']}")
                logger.info(f"   Versão: {result['version']}")
                logger.info(f"   Accuracy: {result['metrics']['accuracy']:.4f}")
            else:
                logger.warning(f"⚠️ Retreinamento não executado: {result['message']}")

        except Exception as e:
            logger.error(f"❌ Erro no retreinamento automático: {e}", exc_info=True)

    def start(self):
        """Inicia o scheduler"""
        if self.enabled and not self.scheduler.running:
            self.scheduler.start()
            logger.info("🚀 Scheduler de retreinamento iniciado")

    def stop(self):
        """Para o scheduler"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("⏹️ Scheduler de retreinamento parado")

    def trigger_manual_retrain(self, force: bool = False):
        """Trigger manual de retreinamento"""
        logger.info("🔧 Retreinamento manual acionado")
        self._weekly_retrain()


# Singleton global
_scheduler = None

def get_retrain_scheduler(enabled: bool = True) -> RetrainScheduler:
    """Retorna instância singleton do RetrainScheduler"""
    global _scheduler
    if _scheduler is None:
        _scheduler = RetrainScheduler(enabled=enabled)
    return _scheduler


def start_retrain_scheduler(enabled: bool = True):
    """Inicializa e inicia o scheduler de retreinamento"""
    scheduler = get_retrain_scheduler(enabled=enabled)
    scheduler.start()
    return scheduler
