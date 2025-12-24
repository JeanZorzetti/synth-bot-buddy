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
DERIV_APP_ID = os.getenv("DERIV_APP_ID")
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN")

if not DERIV_APP_ID or not DERIV_API_TOKEN:
    raise ValueError("❌ DERIV_APP_ID e DERIV_API_TOKEN devem estar configurados nas variáveis de ambiente!")

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

            # 2. Buscar profit table (SEMPRE ASC para inferir Martingale corretamente)
            await ws.send(json.dumps({
                "profit_table": 1,
                "description": 1,
                "limit": 100,
                "sort": "ASC"  # CRÍTICO: Do mais antigo para o novo
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

            # 3. Inferir níveis de Martingale pela sequência de perdas
            logger.info("🔍 Inferindo níveis de Martingale baseado em sequência de perdas...")

            consecutive_losses = 0
            for i, tx in enumerate(transactions):
                # O nível ATUAL é baseado nas perdas ACUMULADAS ANTES deste trade
                current_level = consecutive_losses + 1
                tx['inferred_level'] = current_level

                # Agora calcular resultado para atualizar contador para o PRÓXIMO trade
                profit = float(tx.get("sell_price", 0) - tx.get("buy_price", 0))
                result = "WIN" if profit > 0 else "LOSS"

                if result == "LOSS":
                    consecutive_losses += 1  # Prepara para o próximo
                else:
                    consecutive_losses = 0  # Reseta para o próximo

            logger.info(f"✅ Níveis inferidos para {len(transactions)} trades")

            # 4. Enviar trades (usar async client)
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

                        # Usar nível inferido pela análise de sequência
                        inferred_level = tx.get('inferred_level', 1)

                        # Enviar trade_opened
                        trade_opened = {
                            "timestamp": buy_time.isoformat() + "Z",
                            "trade_id": contract_id,
                            "contract_id": contract_id,
                            "direction": direction,
                            "stake": buy_price,
                            "level": inferred_level
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
                                "max_level_reached": inferred_level
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

            # 2. SYNC INCREMENTAL: Buscar apenas trades novos
            # Consultar último trade no DB
            from database import get_abutre_repository
            repo = get_abutre_repository()

            try:
                conn = repo._get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT MAX(entry_time) as last_sync
                    FROM abutre_trades
                """)
                result = cursor.fetchone()
                last_sync_time = result['last_sync'] if result and result['last_sync'] else None
                cursor.close()
                conn.close()
            except Exception as e:
                logger.warning(f"Erro ao consultar último sync: {e}")
                last_sync_time = None

            # Converter para epoch se houver
            date_from_epoch = None
            if last_sync_time:
                date_from_epoch = int(last_sync_time.timestamp()) + 1  # +1s para não duplicar
                logger.info(f"📅 Último trade no DB: {last_sync_time} (epoch: {date_from_epoch})")
            else:
                logger.info("📅 DB vazio - buscando histórico completo")

            # 3. Paginação incremental (50 por vez para respeitar rate limit)
            batch_size = 50
            offset = 0
            all_transactions = []

            logger.info("🔄 Iniciando sync incremental...")

            while True:
                request_payload = {
                    "profit_table": 1,
                    "description": 1,
                    "limit": batch_size,
                    "offset": offset,
                    "sort": "ASC"  # Do mais antigo para o novo
                }

                # Adicionar filtro de data se houver último sync
                if date_from_epoch:
                    request_payload["date_from"] = date_from_epoch

                await ws.send(json.dumps(request_payload))
                profit_response = json.loads(await ws.recv())

                if "error" in profit_response:
                    logger.error(f"Erro ao buscar histórico: {profit_response['error']['message']}")
                    break

                transactions = profit_response.get("profit_table", {}).get("transactions", [])

                if not transactions:
                    logger.info(f"✅ Fim da paginação (offset={offset})")
                    break

                all_transactions.extend(transactions)
                logger.info(f"📥 Batch {offset//batch_size + 1}: {len(transactions)} trades")

                # Se veio menos que o limite, chegamos ao fim
                if len(transactions) < batch_size:
                    logger.info(f"✅ Última página alcançada")
                    break

                offset += batch_size

                # Rate limit protection
                await asyncio.sleep(0.5)

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

            # 4. Inferir níveis de Martingale pela sequência de perdas
            logger.info("🔍 Inferindo níveis de Martingale baseado em sequência de perdas...")

            consecutive_losses = 0
            for i, tx in enumerate(filtered_transactions):
                # O nível ATUAL é baseado nas perdas ACUMULADAS ANTES deste trade
                current_level = consecutive_losses + 1
                tx['inferred_level'] = current_level

                # Agora calcular resultado para atualizar contador para o PRÓXIMO trade
                profit = float(tx.get("sell_price", 0) - tx.get("buy_price", 0))
                result = "WIN" if profit > 0 else "LOSS"

                if result == "LOSS":
                    consecutive_losses += 1  # Prepara para o próximo
                else:
                    consecutive_losses = 0  # Reseta para o próximo

            logger.info(f"✅ Níveis inferidos para {len(filtered_transactions)} trades")

            # 5. Enviar trades
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

                        # Usar nível inferido pela análise de sequência
                        inferred_level = tx.get('inferred_level', 1)

                        # Enviar trade_opened
                        trade_opened = {
                            "timestamp": buy_time.isoformat() + "Z",
                            "trade_id": contract_id,
                            "contract_id": contract_id,
                            "direction": direction,
                            "stake": buy_price,
                            "level": inferred_level
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
                                "max_level_reached": inferred_level
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
