#!/usr/bin/env python3
"""
AUTO SYNC DERIV HISTORY
Sincroniza automaticamente o histórico de trades da Deriv no startup do servidor
"""
import asyncio
import websockets
import json
import requests
from datetime import datetime
import os
import logging

# Configuração
DERIV_WS_URL = "wss://ws.derivws.com/websockets/v3?app_id=99188"
ABUTRE_API_URL = os.getenv("ABUTRE_API_URL", "http://localhost:8000/api/abutre/events")
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN", "paE5sSemx3oANLE")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def check_if_needs_sync():
    """Verifica se o banco está vazio e precisa de sincronização"""
    try:
        response = requests.get(f"{ABUTRE_API_URL}/stats", timeout=10)
        if response.status_code == 200:
            data = response.json()
            total_trades = data.get("data", {}).get("total_trades", 0)

            if total_trades == 0:
                logger.warning("Banco vazio detectado! Iniciando sincronizacao automatica...")
                return True
            else:
                logger.info(f"Banco ja possui {total_trades} trades. Sincronizacao nao necessaria.")
                return False
    except Exception as e:
        logger.error(f"Erro ao verificar status do banco: {e}")
        return True  # Se falhar, tenta sincronizar de qualquer forma


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

            # 3. Enviar trades
            trades_sent = 0
            trades_failed = 0

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

                    response = requests.post(
                        f"{ABUTRE_API_URL}/trade_opened",
                        json=trade_opened,
                        timeout=30
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

                        requests.post(
                            f"{ABUTRE_API_URL}/trade_closed",
                            json=trade_closed,
                            timeout=30
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

    # PASSO 3: Verificar se precisa sincronizar
    logger.info("PASSO 3: Verificando se banco precisa de sincronização...")
    needs_sync = await check_if_needs_sync()

    if needs_sync:
        logger.info("PASSO 4: Sincronizando histórico da Deriv...")
        success = await sync_deriv_history()
        if success:
            logger.info("✅ Sincronização automática completada com sucesso!")
        else:
            logger.error("❌ Falha na sincronização automática!")
    else:
        logger.info("⏭️ Sincronização não necessária, banco já possui dados.")

    logger.info("=" * 60)


def run_sync():
    """Wrapper síncrono para chamar do FastAPI"""
    asyncio.run(auto_sync_on_startup())


if __name__ == "__main__":
    asyncio.run(auto_sync_on_startup())
