#!/usr/bin/env python3
"""
Teste REAL do método get_latest_tick para verificar formato da resposta
"""
import asyncio
import os
import sys
from deriv_api_legacy import DerivAPI

async def test_get_latest_tick():
    """Testa get_latest_tick com token real"""

    # Verificar se token existe
    token = os.getenv("DERIV_API_TOKEN")
    if not token:
        print("❌ ERRO: DERIV_API_TOKEN não configurado")
        print("   Configure: export DERIV_API_TOKEN='seu_token'")
        return

    print(f"✅ Token encontrado: {token[:10]}...")

    # Criar instância da API
    api = DerivAPI()

    try:
        # Conectar
        print("\n📡 Conectando...")
        await api.connect()
        print("✅ Conectado")

        # Autenticar
        print(f"\n🔐 Autenticando...")
        await api.authorize(token)
        print("✅ Autenticado")

        # Testar get_latest_tick
        print(f"\n📊 Chamando get_latest_tick('R_100')...")
        response = await api.get_latest_tick('R_100')

        print("\n📋 RESPOSTA COMPLETA:")
        print("="*60)
        import json
        print(json.dumps(response, indent=2))
        print("="*60)

        # Verificar estrutura
        print("\n🔍 VERIFICAÇÃO DE ESTRUTURA:")

        if 'history' in response:
            print("   ✅ Tem campo 'history'")
            history = response['history']

            if 'prices' in history:
                print(f"   ✅ Tem campo 'prices' ({len(history['prices'])} items)")
                print(f"      Exemplo: {history['prices'][-1] if history['prices'] else 'vazio'}")
            else:
                print("   ❌ NÃO tem campo 'prices'")
                print(f"      Campos disponíveis: {list(history.keys())}")

            if 'times' in history:
                print(f"   ✅ Tem campo 'times' ({len(history['times'])} items)")
                print(f"      Exemplo: {history['times'][-1] if history['times'] else 'vazio'}")
            else:
                print("   ❌ NÃO tem campo 'times'")
        else:
            print("   ❌ NÃO tem campo 'history'")
            print(f"      Campos disponíveis: {list(response.keys())}")

        # Verificar se código do forward_testing funcionaria
        print("\n🧪 TESTE DO CÓDIGO FORWARD_TESTING:")
        if 'history' not in response or not response['history'].get('prices'):
            print("   ❌ FALHA! Condição do forward_testing.py linha 242")
            print("      O código retornaria None (bug confirmado)")
        else:
            print("   ✅ OK! Código forward_testing funcionaria")

            # Simular extração de dados
            history = response['history']
            prices = history['prices']
            times = history['times']

            current_price = float(prices[-1])
            tick_time = int(times[-1])

            print(f"\n   📊 Dados extraídos:")
            print(f"      Price: {current_price}")
            print(f"      Time: {tick_time}")

    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Desconectar
        print("\n🔌 Desconectando...")
        await api.disconnect()
        print("✅ Desconectado")

if __name__ == "__main__":
    asyncio.run(test_get_latest_tick())
