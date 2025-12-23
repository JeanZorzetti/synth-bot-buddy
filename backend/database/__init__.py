"""
Database layer - PostgreSQL ONLY (SQLite removed)
"""
import os
import logging

logger = logging.getLogger(__name__)

# PostgreSQL é OBRIGATÓRIO
DATABASE_URL = os.getenv("DATABASE_URL", "")

if not DATABASE_URL:
    logger.error("❌ DATABASE_URL não configurada! Configure a variável de ambiente.")
    raise RuntimeError("DATABASE_URL environment variable is required. Please configure PostgreSQL connection.")

if not DATABASE_URL.startswith("postgresql"):
    logger.error(f"❌ DATABASE_URL inválida: {DATABASE_URL}")
    raise RuntimeError(f"Only PostgreSQL is supported. DATABASE_URL must start with 'postgresql://', got: {DATABASE_URL}")

logger.info(f"Using PostgreSQL database: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'unknown'}")
from .abutre_repository_postgres import get_abutre_repository

__all__ = ["get_abutre_repository"]
