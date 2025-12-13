"""
Sistema de tasks em background para processos longos como backtesting
"""
import asyncio
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BackgroundTask:
    """Representa uma task em background"""
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.status = "pending"  # pending, running, completed, error
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.progress: int = 0  # 0-100
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class BackgroundTaskManager:
    """Gerencia tasks em background"""

    def __init__(self):
        self.tasks: Dict[str, BackgroundTask] = {}

    def create_task(self) -> str:
        """Cria uma nova task e retorna o ID"""
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = BackgroundTask(task_id)
        logger.info(f"Task criada: {task_id}")
        return task_id

    def get_task(self, task_id: str) -> Optional[BackgroundTask]:
        """Retorna uma task pelo ID"""
        return self.tasks.get(task_id)

    def update_status(self, task_id: str, status: str, progress: int = 0):
        """Atualiza o status de uma task"""
        task = self.tasks.get(task_id)
        if task:
            task.status = status
            task.progress = progress

            if status == "running" and not task.started_at:
                task.started_at = datetime.now()
            elif status in ["completed", "error"]:
                task.completed_at = datetime.now()

    def set_result(self, task_id: str, result: Any):
        """Define o resultado de uma task"""
        task = self.tasks.get(task_id)
        if task:
            task.result = result
            task.status = "completed"
            task.progress = 100
            task.completed_at = datetime.now()
            logger.info(f"Task completada: {task_id}")

    def set_error(self, task_id: str, error: str):
        """Define um erro em uma task"""
        task = self.tasks.get(task_id)
        if task:
            task.error = error
            task.status = "error"
            task.completed_at = datetime.now()
            logger.error(f"Task com erro: {task_id} - {error}")

    def cleanup_old_tasks(self, max_age_minutes: int = 60):
        """Remove tasks antigas para liberar memória"""
        now = datetime.now()
        to_remove = []

        for task_id, task in self.tasks.items():
            if task.completed_at:
                age = (now - task.completed_at).total_seconds() / 60
                if age > max_age_minutes:
                    to_remove.append(task_id)

        for task_id in to_remove:
            del self.tasks[task_id]
            logger.info(f"Task antiga removida: {task_id}")

        return len(to_remove)


# Instância global do gerenciador
task_manager = BackgroundTaskManager()
