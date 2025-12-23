#!/usr/bin/env python3
"""
AUTO SYNC DERIV HISTORY
Sincroniza automaticamente o histórico de trades da Deriv no startup do servidor
"""
import asyncio
import websockets
import json
import httpx  # Async HTTP client
from datetime import datetime
import os
import logging
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configuração
DERIV_APP_ID = os.getenv("DERIV_APP_ID", "99188")
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN", "paE5sSemx3oANLE")
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"
ABUTRE_API_URL = os.getenv("ABUTRE_API_URL", "http://127.0.0.1:8000/api/abutre/events")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Removido check_if_needs_sync() - sempre sincroniza para manter dados atualizados em tempo real


async def sync_deriv_history():
    """Sincroniza histórico da Deriv"""

    logger.info("Iniciando sincronizacao de historico Deriv...")

    try:
        async with websockets.connect(DERIV_WS_URL) as ws:
            # 1. Authorize
            await ws.send(json.dumps({"authorize": DERIV_API_TOKEN}))
            auth_response = json.loads(await ws.recv())

            if "error" in auth_response:
                logger.error(f"Erro ao fazer login: {auth_response['error']['message']}")
                return False

            account = auth_response['authorize']['loginid']
            balance = auth_response['authorize']['balance']
            logger.info(f"Login OK - Conta: {account} | Balance: ${balance:.2f}")

            # 2. Buscar profit table
            await ws.send(json.dumps({
                "profit_table": 1,
                "description": 1,
                "limit": 100,
                "sort": "DESC"
            }))

            profit_response = json.loads(await ws.recv())

            if "error" in profit_response:
                logger.error(f"Erro ao buscar historico: {profit_response['error']['message']}")
                return False

            transactions = profit_response.get("profit_table", {}).get("transactions", [])

            if not transactions:
                logger.warning("Nenhum trade encontrado no historico!")
                return True

            logger.info(f"{len(transactions)} trades encontrados. Sincronizando...")

            # 3. Enviar trades (usar async client)
            trades_sent = 0
            trades_failed = 0

            async with httpx.AsyncClient(timeout=30.0) as client:
                for tx in transactions:
                    try:
                        contract_id = str(tx.get("contract_id", ""))
                        buy_time = datetime.fromtimestamp(tx.get("purchase_time", 0))
                        sell_time = datetime.fromtimestamp(tx.get("sell_time", 0)) if tx.get("sell_time") else None

                        contract_type = tx.get("contract_type", "")
                        direction = "CALL" if "CALL" in contract_type.upper() else "PUT"

                        buy_price = float(tx.get("buy_price", 0))
                        profit = float(tx.get("sell_price", 0) - tx.get("buy_price", 0))
                        result = "WIN" if profit > 0 else "LOSS"
                        balance_after = float(balance)

                        # Enviar trade_opened
                        trade_opened = {
                            "timestamp": buy_time.isoformat() + "Z",
                            "trade_id": contract_id,
                            "contract_id": contract_id,
                            "direction": direction,
                            "stake": buy_price,
                            "level": 1
                        }

                        response = await client.post(
                            f"{ABUTRE_API_URL}/trade_opened",
                            json=trade_opened
                        )

                        if response.status_code != 201:
                            trades_failed += 1
                            continue

                        # Enviar trade_closed
                        if sell_time:
                            trade_closed = {
                                "timestamp": sell_time.isoformat() + "Z",
                                "trade_id": contract_id,
                                "result": result,
                                "profit": profit,
                                "balance": balance_after,
                                "max_level_reached": 1
                            }

                            await client.post(
                                f"{ABUTRE_API_URL}/trade_closed",
                                json=trade_closed
                            )

                        trades_sent += 1

                    except Exception as e:
                        logger.error(f"Erro ao processar trade: {e}")
                        trades_failed += 1

            logger.info(f"Sincronizacao concluida! Enviados: {trades_sent} | Erros: {trades_failed}")
            return True

    except Exception as e:
        logger.error(f"Erro fatal na sincronizacao: {e}")
        return False


async def sync_deriv_history_period(date_from: datetime, date_to: datetime, force: bool = False):
    """
    Sincroniza histórico da Deriv para um período específico

    NOTA: A API da Deriv (profit_table) não suporta filtro de data diretamente.
    Esta função busca os últimos trades e filtra por período no backend.
    """
    logger.info(f"Sincronizando período: {date_from.date()} até {date_to.date()}")

    try:
        async with websockets.connect(DERIV_WS_URL) as ws:
            # 1. Authorize
            await ws.send(json.dumps({"authorize": DERIV_API_TOKEN}))
            auth_response = json.loads(await ws.recv())

            if "error" in auth_response:
                logger.error(f"Erro ao fazer login: {auth_response['error']['message']}")
                return {"success": False, "error": "Authentication failed"}

            account = auth_response['authorize']['loginid']
            balance = auth_response['authorize']['balance']
            logger.info(f"Login OK - Conta: {account} | Balance: ${balance:.2f}")

            # 2. Buscar profit table (máximo de trades que conseguimos buscar)
            # A API da Deriv limita o número de trades por request
            # Testado: limit 1000 retorna erro "Input validation failed"
            # Usando 999 que é o máximo real aceito pela API
            logger.info("🔵 COMMIT 23c6fc4: Usando limit=999 (CPU-only PyTorch)")
            await ws.send(json.dumps({
                "profit_table": 1,
                "description": 1,
                "limit": 999,  # Máximo real aceito pela API Deriv
                "sort": "DESC"
            }))

            profit_response = json.loads(await ws.recv())

            if "error" in profit_response:
                logger.error(f"Erro ao buscar histórico: {profit_response['error']['message']}")
                return {"success": False, "error": "Failed to fetch profit table"}

            all_transactions = profit_response.get("profit_table", {}).get("transactions", [])

            if not all_transactions:
                logger.warning("Nenhum trade encontrado no histórico!")
                return {"success": True, "trades_synced": 0, "trades_failed": 0}

            # 3. Filtrar trades pelo período
            filtered_transactions = []
            oldest_trade_time = None

            for tx in all_transactions:
                purchase_time = datetime.fromtimestamp(tx.get("purchase_time", 0))

                # Rastrear o trade mais antigo
                if oldest_trade_time is None or purchase_time < oldest_trade_time:
                    oldest_trade_time = purchase_time

                if date_from <= purchase_time <= date_to:
                    filtered_transactions.append(tx)

            logger.info(f"{len(filtered_transactions)} trades encontrados no período (de {len(all_transactions)} totais)")

            # Avisar se o período solicitado é mais antigo que os trades disponíveis
            if oldest_trade_time and date_from < oldest_trade_time:
                logger.warning(f"⚠️ ATENÇÃO: O período solicitado começa em {date_from.date()}, mas o trade mais antigo disponível é de {oldest_trade_time.date()}")
                logger.warning(f"⚠️ Pode haver trades faltando. A API da Deriv retorna no máximo 1000 trades mais recentes.")

            if not filtered_transactions:
                logger.warning("Nenhum trade encontrado no período especificado!")
                return {"success": True, "trades_synced": 0, "trades_failed": 0}

            # 4. Enviar trades
            trades_sent = 0
            trades_failed = 0

            async with httpx.AsyncClient(timeout=30.0) as client:
                for tx in filtered_transactions:
                    try:
                        contract_id = str(tx.get("contract_id", ""))
                        buy_time = datetime.fromtimestamp(tx.get("purchase_time", 0))
                        sell_time = datetime.fromtimestamp(tx.get("sell_time", 0)) if tx.get("sell_time") else None

                        contract_type = tx.get("contract_type", "")
                        direction = "CALL" if "CALL" in contract_type.upper() else "PUT"

                        buy_price = float(tx.get("buy_price", 0))
                        profit = float(tx.get("sell_price", 0) - tx.get("buy_price", 0))
                        result = "WIN" if profit > 0 else "LOSS"
                        balance_after = float(balance)

                        # Enviar trade_opened
                        trade_opened = {
                            "timestamp": buy_time.isoformat() + "Z",
                            "trade_id": contract_id,
                            "contract_id": contract_id,
                            "direction": direction,
                            "stake": buy_price,
                            "level": 1
                        }

                        response = await client.post(
                            f"{ABUTRE_API_URL}/trade_opened",
                            json=trade_opened
                        )

                        if response.status_code != 201:
                            trades_failed += 1
                            continue

                        # Enviar trade_closed
                        if sell_time:
                            trade_closed = {
                                "timestamp": sell_time.isoformat() + "Z",
                                "trade_id": contract_id,
                                "result": result,
                                "profit": profit,
                                "balance": balance_after,
                                "max_level_reached": 1
                            }

                            await client.post(
                                f"{ABUTRE_API_URL}/trade_closed",
                                json=trade_closed
                            )

                        trades_sent += 1

                    except Exception as e:
                        logger.error(f"Erro ao processar trade: {e}")
                        trades_failed += 1

            logger.info(f"Sincronização de período concluída! Enviados: {trades_sent} | Erros: {trades_failed}")
            return {
                "success": True,
                "trades_synced": trades_sent,
                "trades_failed": trades_failed
            }

    except Exception as e:
        logger.error(f"Erro fatal na sincronização de período: {e}")
        return {"success": False, "error": str(e)}


async def auto_sync_on_startup():
    """
    Função principal chamada no startup do servidor
    """
    logger.info("=" * 60)
    logger.info("AUTO SYNC DERIV - STARTUP")
    logger.info("=" * 60)

    # PASSO 1: Garantir que as tabelas existem
    logger.info("PASSO 1: Verificando/criando tabelas do banco de dados...")
    try:
        from migrate import run_migrations
        migration_success = run_migrations()
        if not migration_success:
            logger.error("Falha ao criar tabelas! Abortando auto-sync.")
            return
        logger.info("✅ Tabelas verificadas/criadas com sucesso!")
    except Exception as e:
        logger.error(f"Erro ao executar migrações: {e}")
        logger.error("Abortando auto-sync.")
        return

    # PASSO 2: Aguardar 5 segundos para garantir que a API está pronta
    logger.info("PASSO 2: Aguardando API ficar pronta...")
    await asyncio.sleep(5)

    # PASSO 3: Sincronizar histórico (sempre executa para manter dados atualizados)
    logger.info("PASSO 3: Sincronizando histórico da Deriv...")
    success = await sync_deriv_history()
    if success:
        logger.info("✅ Sincronização automática completada com sucesso!")
    else:
        logger.error("❌ Falha na sincronização automática!")

    logger.info("=" * 60)


def run_sync():
    """Wrapper síncrono para chamar do FastAPI"""
    asyncio.run(auto_sync_on_startup())


if __name__ == "__main__":
    asyncio.run(auto_sync_on_startup())
