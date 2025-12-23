"""
Database layer - Auto-detect PostgreSQL or SQLite
"""
import os
import logging

logger = logging.getLogger(__name__)

# Check if PostgreSQL is configured
DATABASE_URL = os.getenv("DATABASE_URL", "")

if DATABASE_URL and DATABASE_URL.startswith("postgresql"):
    logger.info("Using PostgreSQL database")
    from .abutre_repository_postgres import get_abutre_repository
else:
    logger.info("Using SQLite database")
    from .abutre_repository import get_abutre_repository

__all__ = ["get_abutre_repository"]
