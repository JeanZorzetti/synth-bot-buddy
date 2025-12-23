#!/usr/bin/env python3
"""
MIGRAÇÃO AUTOMÁTICA DO BANCO DE DADOS

Script que garante que todas as tabelas do PostgreSQL existem.
Roda automaticamente no startup do servidor.
"""
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_migrations():
    """Executa todas as migrações necessárias"""
    try:
        DATABASE_URL = os.getenv("DATABASE_URL", "")

        if not DATABASE_URL:
            logger.warning("DATABASE_URL não configurada, pulando migrações")
            return False

        if not DATABASE_URL.startswith("postgresql"):
            logger.info("Usando SQLite, não precisa de migrações")
            return True

        logger.info("=" * 60)
        logger.info("INICIANDO MIGRAÇÕES DO BANCO DE DADOS")
        logger.info("=" * 60)
        logger.info(f"Database: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'PostgreSQL'}")

        # Import repository que já tem o _ensure_tables()
        from database.abutre_repository_postgres import AbutreRepositoryPostgres

        # Instanciar repository (isso chama _ensure_tables automaticamente)
        logger.info("Criando tabelas se não existirem...")
        repo = AbutreRepositoryPostgres(database_url=DATABASE_URL)

        logger.info("✅ Migrações completadas com sucesso!")
        logger.info("=" * 60)

        # Verificar se as tabelas foram criadas
        import psycopg2
        from psycopg2.extras import RealDictCursor

        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name LIKE 'abutre_%'
            ORDER BY table_name
        """)

        tables = cursor.fetchall()
        logger.info(f"Tabelas criadas: {len(tables)}")
        for table in tables:
            logger.info(f"  ✓ {table['table_name']}")

        cursor.close()
        conn.close()

        return True

    except Exception as e:
        logger.error(f"❌ Erro ao executar migrações: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = run_migrations()
    sys.exit(0 if success else 1)
