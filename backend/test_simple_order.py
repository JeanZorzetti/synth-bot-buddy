#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de teste para executar uma ordem simples na Deriv
Objetivo 1 - Fase 1: Prova de Conceito
"""

import asyncio
import sys
import os

# Fix encoding for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from deriv_api import DerivAPI

async def test_order():
    """Executa uma ordem de teste"""

    # CONFIGURAÇÃO - EDITE AQUI
    TOKEN = os.getenv("DERIV_TOKEN", "nKF6Wo6q9cmVT1d")  # ← Coloque seu token ou use variável de ambiente
    SYMBOL = "R_75"            # Volatility 75 Index
    CONTRACT_TYPE = "CALL"     # CALL (Rise) ou PUT (Fall)
    AMOUNT = 1.0               # Valor em USD
    DURATION = 5               # Duração em minutos

    print("=" * 60)
    print("🚀 TESTE DE EXECUÇÃO DE ORDEM NA DERIV")
    print("=" * 60)
    print(f"\n📋 CONFIGURAÇÃO:")
    print(f"   Token: {'***' + TOKEN[-8:] if len(TOKEN) > 8 else '(não configurado)'}")
    print(f"   Símbolo: {SYMBOL}")
    print(f"   Tipo: {CONTRACT_TYPE}")
    print(f"   Valor: ${AMOUNT}")
    print(f"   Duração: {DURATION} minutos")

    # Verificar se token foi configurado
    if TOKEN == "SEU_TOKEN_AQUI" or len(TOKEN) < 10:
        print("\n" + "=" * 60)
        print("❌ ERRO: Token não configurado!")
        print("\n💡 COMO CONFIGURAR:")
        print("   1. Acesse: https://app.deriv.com/account/api-token")
        print("   2. Crie token com scopes: Read + Trade")
        print("   3. Configure via variável de ambiente:")
        print("      $env:DERIV_TOKEN=\"seu_token\"  (PowerShell)")
        print("      set DERIV_TOKEN=seu_token       (CMD)")
        print("      export DERIV_TOKEN=seu_token    (Linux/Mac)")
        print("   OU")
        print("   4. Edite este arquivo e coloque o token na linha 16")
        print("=" * 60)
        return False

    # Criar cliente
    api = DerivAPI(app_id=1089, demo=True)

    try:
        # 1. CONECTAR
        print("\n1️⃣ Conectando à Deriv API...")
        if not await api.connect():
            print("❌ Falha na conexão")
            print("💡 Verifique sua conexão com a internet")
            return False
        print("✅ Conectado com sucesso")

        # 2. AUTENTICAR
        print(f"\n2️⃣ Autenticando com token...")
        auth_response = await api.authorize(TOKEN)

        if 'error' in auth_response:
            error_msg = auth_response['error'].get('message', 'Erro desconhecido')
            print(f"❌ Erro de autenticação: {error_msg}")
            print("\n💡 POSSÍVEIS CAUSAS:")
            print("   • Token inválido ou expirado")
            print("   • Token não tem scopes Read + Trade")
            print("   • Token foi revogado")
            print("\n💡 SOLUÇÃO:")
            print("   1. Acesse: https://app.deriv.com/account/api-token")
            print("   2. Verifique se o token existe e está ativo")
            print("   3. Crie um novo token se necessário")
            return False

        auth_data = auth_response.get('authorize', {})
        loginid = auth_data.get('loginid')
        balance = auth_data.get('balance')
        currency = auth_data.get('currency')
        account_list = auth_data.get('account_list', [])

        print(f"✅ Autenticado")
        print(f"   LoginID: {loginid}")
        print(f"   Saldo: {balance} {currency}")

        # Mostrar todas as contas disponíveis
        if account_list and len(account_list) > 1:
            print(f"\n📋 Contas disponíveis:")
            for idx, acc in enumerate(account_list, 1):
                acc_type = "Demo" if acc.get('is_virtual') else "Real"
                acc_id = acc.get('loginid')
                acc_balance = acc.get('balance', 'N/A')
                is_current = " [ATUAL]" if acc_id == loginid else ""
                print(f"   {idx}. {acc_id} ({acc_type}) - ${acc_balance}{is_current}")

            # Se a conta atual for Real com saldo zero, instruir usuário
            if loginid.startswith('CR') and balance == 0:
                demo_accounts = [acc for acc in account_list if acc.get('is_virtual')]
                if demo_accounts:
                    demo_loginid = demo_accounts[0].get('loginid')
                    print(f"\n⚠️  CONTA SEM SALDO!")
                    print(f"   Você está logado na conta Real ({loginid}) com $0")
                    print(f"   Conta Demo disponível: {demo_loginid}\n")
                    print(f"📌 COMO USAR CONTA DEMO:")
                    print(f"   1. Acesse: https://app.deriv.com/")
                    print(f"   2. Clique no seletor de conta (canto superior)")
                    print(f"   3. Selecione a conta Demo: {demo_loginid}")
                    print(f"   4. Vá em Settings > API Token")
                    print(f"   5. Gere um NOVO token (agora conectado à Demo)")
                    print(f"   6. Use o novo token neste script\n")
                    print(f"   OU adicione saldo à conta Real {loginid}")
                    return False

        # Verificar saldo suficiente
        if balance < AMOUNT:
            print(f"\n⚠️  AVISO: Saldo insuficiente!")
            print(f"   Saldo atual: {balance} {currency}")
            print(f"   Necessário: {AMOUNT} {currency}")
            print("\n💡 SOLUÇÃO:")
            print("   • Use uma conta Demo (saldo virtual)")
            print("   • Ou reduza o valor da aposta (variável AMOUNT)")
            return False

        # 3. OBTER PROPOSTA
        print(f"\n3️⃣ Obtendo proposta...")
        print(f"   Símbolo: {SYMBOL}")
        print(f"   Tipo: {CONTRACT_TYPE}")
        print(f"   Valor: ${AMOUNT}")
        print(f"   Duração: {DURATION} minutos")

        proposal = await api.get_proposal(
            contract_type=CONTRACT_TYPE,
            symbol=SYMBOL,
            amount=AMOUNT,
            duration=DURATION,
            duration_unit="m",
            basis="stake",
            currency=currency
        )

        if 'error' in proposal:
            error_msg = proposal['error'].get('message', 'Erro desconhecido')
            print(f"❌ Erro na proposta: {error_msg}")
            print("\n💡 POSSÍVEIS CAUSAS:")
            print("   • Mercado fechado")
            print("   • Símbolo inválido")
            print("   • Parâmetros inválidos")
            return False

        # Extrair dados da proposta
        proposal_id = proposal.get('id')
        ask_price = proposal.get('ask_price')
        payout = proposal.get('payout')
        potential_profit = payout - ask_price if (payout and ask_price) else 0

        print(f"✅ Proposta obtida")
        print(f"   ID: {proposal_id}")
        print(f"   Preço: ${ask_price}")
        print(f"   Payout: ${payout}")
        print(f"   Lucro potencial: ${potential_profit:.2f}")

        # 4. CONFIRMAR EXECUÇÃO
        print(f"\n⚠️  ATENÇÃO: Você está prestes a executar uma ordem REAL!")
        print(f"   💰 Custo: ${ask_price}")
        print(f"   🎯 Retorno potencial: ${payout}")
        print(f"   📊 Lucro se ganhar: ${potential_profit:.2f}")
        print(f"   📉 Perda se perder: ${ask_price}")

        # Auto-confirm em modo não-interativo ou pedir confirmação
        if os.getenv("AUTO_CONFIRM", "").lower() == "true":
            print(f"\n🤖 AUTO_CONFIRM ativado - Executando automaticamente...")
            confirm = "sim"
        else:
            confirm = input("\n👉 Deseja continuar? (sim/não): ").lower().strip()

        if confirm not in ['sim', 's', 'yes', 'y']:
            print("❌ Ordem cancelada pelo usuário")
            await api.disconnect()
            return False

        # 5. EXECUTAR COMPRA
        print(f"\n4️⃣ Executando ordem...")

        buy_response = await api.buy(
            contract_type=CONTRACT_TYPE,
            symbol=SYMBOL,
            amount=AMOUNT,
            duration=DURATION,
            duration_unit="m",
            basis="stake",
            currency=currency
        )

        if 'error' in buy_response:
            error_msg = buy_response['error'].get('message', 'Erro desconhecido')
            print(f"❌ Erro na execução: {error_msg}")
            print("\n💡 POSSÍVEIS CAUSAS:")
            print("   • Preço mudou (slippage)")
            print("   • Mercado fechou durante execução")
            print("   • Saldo insuficiente")
            print("   • Limite de trading atingido")
            return False

        # Extrair dados da compra
        buy_data = buy_response.get('buy', {})
        contract_id = buy_data.get('contract_id')
        buy_price = buy_data.get('buy_price')
        longcode = buy_data.get('longcode')
        payout_value = buy_data.get('payout')

        print(f"\n✅ ORDEM EXECUTADA COM SUCESSO!")
        print(f"\n📊 DETALHES DA ORDEM:")
        print(f"   Contract ID: {contract_id}")
        print(f"   Preço pago: ${buy_price}")
        print(f"   Payout: ${payout_value}")
        print(f"   Descrição: {longcode}")
        print(f"\n🔗 Ver contrato na plataforma:")
        print(f"   https://app.deriv.com/contract/{contract_id}")

        # 6. DESCONECTAR
        await api.disconnect()

        print("\n" + "=" * 60)
        print("✅ TESTE CONCLUÍDO COM SUCESSO")
        print("=" * 60)
        print("\n📝 PRÓXIMOS PASSOS:")
        print("   1. Verifique o contrato no link acima")
        print("   2. Aguarde o resultado (5 minutos)")
        print("   3. Continue para Fase 2: Endpoint Backend")
        print("\n" + "=" * 60)

        return True

    except asyncio.TimeoutError:
        print(f"\n❌ TIMEOUT: Operação demorou muito tempo")
        print("💡 Tente novamente ou verifique sua conexão")
        return False

    except Exception as e:
        print(f"\n❌ ERRO DURANTE EXECUÇÃO:")
        print(f"   {str(e)}")
        print("\n🐛 DEBUG INFO:")
        import traceback
        traceback.print_exc()
        print("\n💡 Se o erro persistir:")
        print("   1. Verifique os logs acima")
        print("   2. Consulte GUIA-RAPIDO-IMPLEMENTACAO.md (Troubleshooting)")
        print("   3. Crie issue no GitHub com o erro completo")
        return False

    finally:
        # Garantir desconexão
        if api.websocket:
            try:
                await api.disconnect()
            except:
                pass


def main():
    """Main entry point"""
    print("\n🤖 Synth Bot Buddy - Test Order Script")
    print("📋 Objetivo 1 - Fase 1: Prova de Conceito")
    print("=" * 60)

    try:
        result = asyncio.run(test_order())

        if result:
            print("\n🎉 SUCESSO! Fase 1 completada.")
            print("📖 Próximo passo: Leia GUIA-RAPIDO-IMPLEMENTACAO.md (Fase 2)")
            sys.exit(0)
        else:
            print("\n❌ FALHA! Revise os erros acima.")
            print("📖 Consulte: GUIA-RAPIDO-IMPLEMENTACAO.md (Troubleshooting)")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Operação cancelada pelo usuário (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERRO FATAL: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
