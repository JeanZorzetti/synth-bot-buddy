#!/usr/bin/env python3
"""
Test script para validar integração com Deriv API oficial
"""
import asyncio
from deriv_api import DerivAPI

async def test_ticks_history():
    """Testa busca de histórico de candles"""

    # Configuração
    app_id = 99188
    token = "paE5sSemx3oANLE"
    symbol = "R_100"

    print(f">> Conectando ao Deriv API...")
    print(f"   App ID: {app_id}")
    print(f"   Token: {token[:10]}...")
    print()

    # Criar API instance
    api = DerivAPI(app_id=app_id)

    print("[OK] DerivAPI criado")
    print(f"   Aguardando conexão...")

    # Aguardar conexão (EasyFuture pode ser awaited diretamente)
    await api.connected
    print("[OK] WebSocket conectado")
    print()

    # Autorizar
    print(f">> Autorizando...")
    auth_response = await api.authorize(token)
    loginid = auth_response.get('authorize', {}).get('loginid', 'unknown')
    balance = auth_response.get('authorize', {}).get('balance', 0)
    currency = auth_response.get('authorize', {}).get('currency', 'USD')
    print(f"[OK] Autenticado: {loginid}")
    print(f"   Saldo: {balance} {currency}")
    print()

    # Buscar candles
    print(f">> Buscando candles de {symbol}...")
    response = await api.ticks_history({
        "ticks_history": symbol,
        "adjust_start_time": 1,
        "count": 100,
        "end": "latest",
        "style": "candles",
        "granularity": 60  # 1 minuto
    })

    if 'error' in response:
        print(f"[ERROR] Erro: {response['error']}")
        return False

    candles = response.get('candles', [])
    print(f"[OK] Recebidos {len(candles)} candles")

    if candles:
        print(f"\n>> Primeiros 3 candles:")
        for i, candle in enumerate(candles[:3]):
            print(f"   {i+1}. Epoch: {candle['epoch']}, "
                  f"O: {candle['open']}, "
                  f"H: {candle['high']}, "
                  f"L: {candle['low']}, "
                  f"C: {candle['close']}")

    print(f"\n[SUCCESS] Dados reais do Deriv API funcionando!")

    # Limpar
    await api.clear()
    return True

if __name__ == "__main__":
    try:
        result = asyncio.run(test_ticks_history())
        if result:
            print(f"\n[OK] Teste completado com sucesso!")
        else:
            print(f"\n[FAIL] Teste falhou")
    except Exception as e:
        print(f"\n[ERROR] Erro no teste: {e}")
        import traceback
        traceback.print_exc()
