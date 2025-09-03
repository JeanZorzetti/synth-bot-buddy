#!/usr/bin/env python3
"""
Script de teste para a conexão WebSocket com a API Deriv
Usado para validar a implementação sem executar o servidor completo
"""

import asyncio
import os
from dotenv import load_dotenv
from websocket_manager import DerivWebSocketManager, ConnectionState

# Load environment variables
load_dotenv()

async def test_connection():
    """Test WebSocket connection to Deriv API"""
    print("🚀 Iniciando teste de conexão com a API Deriv...")
    
    # Get credentials from environment
    api_token = os.getenv("DERIV_API_TOKEN")
    app_id = os.getenv("DERIV_APP_ID", "1089")
    
    if not api_token or api_token == "your_deriv_api_token_here":
        print("❌ ERRO: Token da API não configurado!")
        print("   1. Copie .env.example para .env")
        print("   2. Configure seu DERIV_API_TOKEN no arquivo .env")
        return False
    
    print(f"📡 Configuração:")
    print(f"   App ID: {app_id}")
    print(f"   Token: {api_token[:10]}...")
    
    # Initialize WebSocket manager
    ws_manager = DerivWebSocketManager(app_id=app_id, api_token=api_token)
    
    # Set up event handlers
    connection_status = {"connected": False, "authenticated": False}
    received_data = {"ticks": 0, "balance": None}
    
    async def handle_tick(tick_data):
        received_data["ticks"] += 1
        print(f"📊 Tick recebido #{received_data['ticks']}: {tick_data['symbol']} = {tick_data['price']}")
        
        # Stop after receiving some ticks
        if received_data["ticks"] >= 5:
            print("✅ Teste de ticks concluído!")
            await ws_manager.disconnect()
    
    async def handle_balance(balance_data):
        received_data["balance"] = balance_data
        print(f"💰 Saldo atualizado: {balance_data['balance']} {balance_data['currency']}")
    
    async def handle_connection(status: ConnectionState):
        print(f"🔄 Status da conexão: {status.value}")
        if status == ConnectionState.CONNECTED:
            connection_status["connected"] = True
        elif status == ConnectionState.AUTHENTICATED:
            connection_status["authenticated"] = True
        elif status == ConnectionState.ERROR:
            print("❌ Erro na conexão!")
    
    # Register handlers
    ws_manager.set_tick_handler(handle_tick)
    ws_manager.set_balance_handler(handle_balance)
    ws_manager.set_connection_handler(handle_connection)
    
    try:
        # Test connection
        print("\n1️⃣  Testando conexão...")
        success = await ws_manager.connect()
        if not success:
            print("❌ Falha na conexão inicial")
            return False
        
        print("✅ Conexão estabelecida!")
        
        # Wait for authentication
        print("\n2️⃣  Aguardando autenticação...")
        max_wait = 10
        wait_time = 0
        while not connection_status["authenticated"] and wait_time < max_wait:
            await asyncio.sleep(0.5)
            wait_time += 0.5
        
        if not connection_status["authenticated"]:
            print("❌ Falha na autenticação")
            return False
        
        print("✅ Autenticação bem-sucedida!")
        
        # Test balance retrieval
        print("\n3️⃣  Obtendo saldo da conta...")
        await ws_manager.get_balance()
        
        # Test tick subscription
        print("\n4️⃣  Testando subscrição de ticks (Volatility 75)...")
        await ws_manager.subscribe_to_ticks("R_75")
        
        # Wait for some ticks
        print("⏳ Aguardando dados de ticks (máximo 30 segundos)...")
        wait_time = 0
        while received_data["ticks"] < 5 and wait_time < 30:
            await asyncio.sleep(1)
            wait_time += 1
        
        print(f"\n📈 Resultados do teste:")
        print(f"   ✅ Conexão: {'OK' if connection_status['connected'] else 'FALHOU'}")
        print(f"   ✅ Autenticação: {'OK' if connection_status['authenticated'] else 'FALHOU'}")
        print(f"   ✅ Saldo obtido: {'OK' if received_data['balance'] else 'FALHOU'}")
        print(f"   ✅ Ticks recebidos: {received_data['ticks']}")
        
        if received_data["balance"]:
            print(f"   💰 Saldo atual: {received_data['balance']['balance']} {received_data['balance']['currency']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro durante o teste: {e}")
        return False
    
    finally:
        print("\n🔚 Encerrando conexão...")
        await ws_manager.disconnect()

if __name__ == "__main__":
    print("=" * 60)
    print("         TESTE DE CONEXÃO - SYNTH BOT BUDDY")
    print("=" * 60)
    
    try:
        result = asyncio.run(test_connection())
        if result:
            print("\n🎉 TESTE CONCLUÍDO COM SUCESSO!")
            print("   Sua conexão com a API Deriv está funcionando.")
        else:
            print("\n⚠️  TESTE FALHOU!")
            print("   Verifique suas credenciais e conexão de internet.")
    except KeyboardInterrupt:
        print("\n🛑 Teste interrompido pelo usuário")
    except Exception as e:
        print(f"\n💥 Erro inesperado: {e}")
    
    print("=" * 60)