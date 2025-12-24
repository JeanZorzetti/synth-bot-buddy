#!/usr/bin/env python3
"""
MIGRAÇÃO AUTOMÁTICA DO BANCO DE DADOS

Script que garante que todas as tabelas do PostgreSQL existem.
Roda automaticamente no startup do servidor.
"""
import os
import sys
import logging
import asyncio
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


async def ensure_tables(database_url: str):
    """Cria todas as tabelas necessárias usando asyncpg"""
    import asyncpg

    conn = await asyncpg.connect(database_url)

    try:
        # Table: abutre_candles
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS abutre_candles (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                symbol TEXT NOT NULL DEFAULT '1HZ100V',
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                color INTEGER NOT NULL,
                source TEXT DEFAULT 'deriv_bot_xml',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table: abutre_triggers
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS abutre_triggers (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                streak_count INTEGER NOT NULL,
                direction TEXT NOT NULL,
                source TEXT DEFAULT 'deriv_bot_xml',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table: abutre_trades
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS abutre_trades (
                id SERIAL PRIMARY KEY,
                trade_id TEXT UNIQUE NOT NULL,
                contract_id TEXT,
                entry_time TIMESTAMP WITH TIME ZONE NOT NULL,
                direction TEXT NOT NULL,
                initial_stake REAL NOT NULL,
                max_level_reached INTEGER DEFAULT 1,
                total_staked REAL NOT NULL,
                exit_time TIMESTAMP WITH TIME ZONE,
                result TEXT,
                profit REAL,
                balance_after REAL,
                source TEXT DEFAULT 'deriv_bot_xml',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table: abutre_balance_history
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS abutre_balance_history (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                balance REAL NOT NULL,
                peak_balance REAL,
                drawdown_pct REAL,
                total_trades INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                roi_pct REAL DEFAULT 0,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_abutre_candles_timestamp ON abutre_candles(timestamp)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_abutre_triggers_timestamp ON abutre_triggers(timestamp)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_abutre_trades_entry_time ON abutre_trades(entry_time)")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_abutre_trades_trade_id ON abutre_trades(trade_id)")

        logger.info("✅ Tabelas criadas com sucesso!")

        # Verificar quais tabelas foram criadas
        tables = await conn.fetch("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name LIKE 'abutre_%'
            ORDER BY table_name
        """)

        logger.info(f"Tabelas criadas: {len(tables)}")
        for table in tables:
            logger.info(f"  ✓ {table['table_name']}")

    finally:
        await conn.close()


async def run_migrations():
    """Executa todas as migrações necessárias - PostgreSQL OBRIGATÓRIO"""
    try:
        DATABASE_URL = os.getenv("DATABASE_URL", "")

        if not DATABASE_URL:
            logger.error("❌ DATABASE_URL não configurada!")
            logger.error("Configure a variável de ambiente DATABASE_URL com a conexão PostgreSQL.")
            logger.error("Exemplo: DATABASE_URL=postgresql://user:pass@host:5432/database")
            return False

        if not DATABASE_URL.startswith("postgresql"):
            logger.error(f"❌ Apenas PostgreSQL é suportado!")
            logger.error(f"DATABASE_URL deve começar com 'postgresql://', recebido: {DATABASE_URL}")
            return False

        logger.info("=" * 60)
        logger.info("INICIANDO MIGRAÇÕES DO BANCO DE DADOS")
        logger.info("=" * 60)
        logger.info(f"Database: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'PostgreSQL'}")

        # Criar tabelas usando asyncpg
        logger.info("Criando tabelas se não existirem...")
        await ensure_tables(DATABASE_URL)

        logger.info("✅ Migrações completadas com sucesso!")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"❌ Erro ao executar migrações: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = asyncio.run(run_migrations())
    sys.exit(0 if success else 1)
