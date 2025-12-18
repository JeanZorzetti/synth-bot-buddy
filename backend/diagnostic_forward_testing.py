"""
Script de Diagnóstico - Forward Testing

Identifica por que o sistema não está fazendo predições.
"""

import asyncio
import logging
from forward_testing import get_forward_testing_engine

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def diagnose():
    """Executa diagnóstico completo do forward testing"""

    print("\n" + "="*60)
    print("DIAGNÓSTICO - FORWARD TESTING")
    print("="*60)

    try:
        # 1. Obter instância do engine
        print("\n[1/6] Obtendo instância do ForwardTestingEngine...")
        engine = get_forward_testing_engine()
        print(f"✅ Engine obtido: {engine}")
        print(f"   - Is Running: {engine.is_running}")
        print(f"   - Symbol: {engine.symbol}")
        print(f"   - Start Time: {engine.start_time}")
        print(f"   - Last Prediction Time: {engine.last_prediction_time}")

        # 2. Verificar modelo ML
        print("\n[2/6] Verificando ML Predictor...")
        print(f"✅ Modelo carregado: {engine.ml_predictor.model_path.name}")
        print(f"   - Threshold: {engine.ml_predictor.threshold}")
        print(f"   - Features: {len(engine.ml_predictor.feature_names)}")

        # 3. Verificar conexão Deriv API
        print("\n[3/6] Verificando conexão Deriv API...")
        print(f"   - Conectado: {engine.deriv_connected}")
        print(f"   - Token configurado: {'SIM' if engine.deriv_api_token else 'NÃO ❌'}")

        if not engine.deriv_api_token:
            print("\n❌ PROBLEMA IDENTIFICADO: Token Deriv API não configurado!")
            print("   Solução: Configurar DERIV_API_TOKEN nas variáveis de ambiente")
            return

        # 4. Testar coleta de dados do mercado
        print("\n[4/6] Testando coleta de market data...")
        try:
            market_data = await engine._fetch_market_data()

            if market_data:
                print(f"✅ Market data coletado com sucesso!")
                print(f"   - Preço atual: {market_data['close']:.5f}")
                print(f"   - Total de candles: {len(market_data.get('all_candles', []))}")
            else:
                print(f"❌ PROBLEMA: Market data retornou None")
                print("   Possível causa: Erro na Deriv API ou token inválido")
                return

        except Exception as e:
            print(f"❌ ERRO ao coletar market data: {e}")
            logger.error(f"Erro detalhado:", exc_info=True)
            return

        # 5. Testar geração de predição
        print("\n[5/6] Testando geração de predição ML...")
        try:
            prediction = await engine._generate_prediction(market_data)

            if prediction:
                print(f"✅ Predição gerada com sucesso!")
                print(f"   - Prediction: {prediction['prediction']}")
                print(f"   - Confidence: {prediction.get('confidence', 0):.4f}")
                print(f"   - Signal Strength: {prediction.get('signal_strength', 'N/A')}")

                # Verificar se confidence atinge threshold
                confidence = prediction.get('confidence', 0)
                if confidence >= engine.confidence_threshold:
                    print(f"✅ Confidence {confidence:.2%} >= Threshold {engine.confidence_threshold:.2%}")
                    print("   Trade DEVERIA ser executado!")
                else:
                    print(f"⚠️ Confidence {confidence:.2%} < Threshold {engine.confidence_threshold:.2%}")
                    print("   Trade NÃO será executado (confidence baixa)")
            else:
                print(f"❌ PROBLEMA: Predição retornou None")
                return

        except Exception as e:
            print(f"❌ ERRO ao gerar predição: {e}")
            logger.error(f"Erro detalhado:", exc_info=True)
            return

        # 6. Verificar frequência de predições
        print("\n[6/6] Verificando frequência de predições...")
        print(f"   - Min time between predictions: {engine.min_time_between_predictions}s")
        print(f"   - Sleep time no loop: 5s (single symbol)")

        # Verificar se está em loop
        print("\n[VERIFICAÇÃO] Estado do trading loop:")
        if not engine.is_running:
            print("❌ PROBLEMA CRÍTICO: engine.is_running = False!")
            print("   O loop de trading NÃO está ativo!")
            print("   Solução: Reiniciar forward testing via API")
        else:
            print("✅ engine.is_running = True")
            print("   Loop de trading deveria estar ativo...")

            # Verificar se há algum bloqueio
            if engine.last_prediction_time is None:
                print("\n⚠️ SUSPEITA: last_prediction_time ainda é None após 1h45m")
                print("   Possíveis causas:")
                print("   1. Loop de trading não está chamando _process_symbol()")
                print("   2. Erro silencioso no trading_loop()")
                print("   3. Deriv API demorando demais para responder")
                print("   4. Exceção sendo capturada e não logada")

        print("\n" + "="*60)
        print("DIAGNÓSTICO COMPLETO")
        print("="*60)
        print("\nRecomendações:")
        print("1. Verificar logs do container Docker para erros não capturados")
        print("2. Verificar se o método start() foi chamado corretamente")
        print("3. Adicionar mais logging no trading_loop()")
        print("4. Verificar se há algum deadlock ou bloqueio assíncrono")

    except Exception as e:
        print(f"\n❌ ERRO CRÍTICO durante diagnóstico: {e}")
        logger.error("Erro detalhado:", exc_info=True)


if __name__ == "__main__":
    asyncio.run(diagnose())
