#!/usr/bin/env python3
"""
SYNC DERIV HISTORY → ABUTRE DASHBOARD
Busca histórico REAL de trades da sua conta Deriv e popula o dashboard
"""
import asyncio
import websockets
import json
import requests
from datetime import datetime
import os

# Configuração
DERIV_WS_URL = "wss://ws.derivws.com/websockets/v3?app_id=1089"
ABUTRE_API_URL = "https://botderivapi.roilabs.com.br/api/abutre/events"

# ⚠️ IMPORTANTE: Configure seu API token aqui
# Criar token em: https://app.deriv.com/account/api-token
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN", "paE5sSemx3oANLE")

if not DERIV_API_TOKEN:
    print("❌ ERRO: Você precisa configurar DERIV_API_TOKEN!")
    print("")
    print("Como criar seu token:")
    print("1. Acesse: https://app.deriv.com/account/api-token")
    print("2. Crie um novo token com permissões:")
    print("   - ✅ Read")
    print("   - ✅ Trade")
    print("   - ✅ Trading information")
    print("3. Copie o token gerado")
    print("")
    print("Como usar:")
    print("  Windows (PowerShell):")
    print('    $env:DERIV_API_TOKEN="seu_token_aqui"')
    print("    python sync_deriv_history.py")
    print("")
    print("  Linux/Mac:")
    print("    export DERIV_API_TOKEN='seu_token_aqui'")
    print("    python sync_deriv_history.py")
    exit(1)


async def fetch_deriv_history():
    """Busca histórico de trades da Deriv"""

    print("=" * 60)
    print("SYNC DERIV HISTORY -> ABUTRE DASHBOARD")
    print("=" * 60)
    print("")

    async with websockets.connect(DERIV_WS_URL) as ws:
        # 1. Authorize com API token
        print("Fazendo login na Deriv...")
        await ws.send(json.dumps({
            "authorize": DERIV_API_TOKEN
        }))

        auth_response = json.loads(await ws.recv())

        if "error" in auth_response:
            print(f"ERRO ao fazer login: {auth_response['error']['message']}")
            return

        print(f"Login bem-sucedido!")
        print(f"   Conta: {auth_response['authorize']['loginid']}")
        print(f"   Balance: ${auth_response['authorize']['balance']:.2f}")
        print(f"   Currency: {auth_response['authorize']['currency']}")
        print("")

        # 2. Buscar profit table (histórico de trades)
        print("Buscando historico de trades...")
        await ws.send(json.dumps({
            "profit_table": 1,
            "description": 1,
            "limit": 100,  # Últimos 100 trades
            "sort": "DESC"
        }))

        profit_response = json.loads(await ws.recv())

        if "error" in profit_response:
            print(f"ERRO ao buscar historico: {profit_response['error']['message']}")
            return

        transactions = profit_response.get("profit_table", {}).get("transactions", [])

        if not transactions:
            print("Nenhum trade encontrado no historico!")
            return

        print(f"{len(transactions)} trades encontrados!")
        print("")

        # 3. Processar e enviar cada trade para API
        print("Enviando trades para dashboard...")
        print("")

        trades_sent = 0
        trades_failed = 0

        for tx in transactions:
            try:
                # Extrair dados do trade
                contract_id = str(tx.get("contract_id", ""))
                buy_time = datetime.fromtimestamp(tx.get("purchase_time", 0))
                sell_time = datetime.fromtimestamp(tx.get("sell_time", 0)) if tx.get("sell_time") else None

                # Determinar direção (CALL/PUT)
                contract_type = tx.get("contract_type", "")
                direction = "CALL" if "CALL" in contract_type.upper() else "PUT"

                # Valores
                buy_price = float(tx.get("buy_price", 0))
                sell_price = float(tx.get("sell_price", 0))
                profit = float(tx.get("sell_price", 0) - tx.get("buy_price", 0))

                # Resultado
                result = "WIN" if profit > 0 else "LOSS"

                # Balance após trade (estimado, pois API não retorna)
                # Vamos usar o balance atual menos o lucro acumulado
                balance_after = float(auth_response['authorize']['balance'])

                # 1. Enviar trade_opened
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
                    timeout=10
                )

                if response.status_code != 201:
                    print(f"  AVISO: Erro ao enviar trade {contract_id}: {response.status_code}")
                    trades_failed += 1
                    continue

                # 2. Se trade já fechou, enviar trade_closed
                if sell_time:
                    trade_closed = {
                        "timestamp": sell_time.isoformat() + "Z",
                        "trade_id": contract_id,
                        "result": result,
                        "profit": profit,
                        "balance": balance_after,
                        "max_level_reached": 1
                    }

                    response = requests.post(
                        f"{ABUTRE_API_URL}/trade_closed",
                        json=trade_closed,
                        timeout=10
                    )

                    if response.status_code != 200:
                        print(f"  AVISO: Erro ao fechar trade {contract_id}: {response.status_code}")

                trades_sent += 1

                # Log
                profit_str = f"+${profit:.2f}" if profit > 0 else f"-${abs(profit):.2f}"
                print(f"  OK {contract_id[:8]}... | {direction:4} | {result:4} | {profit_str:>10}")

            except Exception as e:
                print(f"  ERRO ao processar trade: {e}")
                trades_failed += 1

        print("")
        print("=" * 60)
        print(f"Sincronizacao concluida!")
        print(f"   Trades enviados: {trades_sent}")
        print(f"   Trades com erro: {trades_failed}")
        print("")
        print(f"Acesse: https://botderiv.roilabs.com.br/abutre")
        print("=" * 60)


async def main():
    try:
        await fetch_deriv_history()
    except KeyboardInterrupt:
        print("\n\nCancelado pelo usuario")
    except Exception as e:
        print(f"\nERRO fatal: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
