"""
Simple ML Endpoints Test - Sem dependências externas

Testa endpoints ML usando apenas urllib (built-in)
"""

import sys
import json
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime

# URL base
BASE_URL = "http://localhost:8000"


def test_health():
    """Testa se servidor está online"""
    print("\n" + "="*60)
    print("Verificando servidor...")
    print("="*60)

    try:
        req = urllib.request.Request(f"{BASE_URL}/health")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            print(f"[OK] Servidor online!")
            print(f"  Status: {data.get('status', 'N/A')}")
            print(f"  Version: {data.get('version', 'N/A')}")
            return True
    except urllib.error.URLError as e:
        print(f"[ERRO] Servidor não está rodando!")
        print(f"  Erro: {e}")
        print(f"\nPara iniciar:")
        print(f"  cd backend")
        print(f"  python -m uvicorn main:app --reload")
        return False
    except Exception as e:
        print(f"[ERRO] {e}")
        return False


def test_ml_info():
    """Teste 1: GET /api/ml/info"""
    print("\n" + "="*60)
    print("TESTE 1: GET /api/ml/info")
    print("="*60)

    try:
        req = urllib.request.Request(f"{BASE_URL}/api/ml/info")
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())

            print(f"[OK] Endpoint funcionando!")
            print(f"\n  Modelo:")
            print(f"    Nome: {data.get('model_name', 'N/A')}")
            print(f"    Tipo: {data.get('model_type', 'N/A')}")
            print(f"    Threshold: {data.get('threshold', 'N/A')}")
            print(f"    Features: {data.get('n_features', 'N/A')}")

            perf = data.get('expected_performance', {})
            print(f"\n  Performance Esperada:")
            print(f"    Accuracy: {perf.get('accuracy', 'N/A')}")
            print(f"    Recall: {perf.get('recall', 'N/A')}")
            print(f"    Precision: {perf.get('precision', 'N/A')}")
            print(f"    Profit (6m): {perf.get('profit_6_months', 'N/A')}")
            print(f"    Sharpe: {perf.get('sharpe_ratio', 'N/A')}")
            print(f"    Win Rate: {perf.get('win_rate', 'N/A')}")

            return True

    except urllib.error.HTTPError as e:
        print(f"[ERRO] HTTP {e.code}: {e.reason}")
        print(f"  Response: {e.read().decode()}")
        return False
    except urllib.error.URLError as e:
        print(f"[ERRO] Conexão falhou: {e.reason}")
        return False
    except Exception as e:
        print(f"[ERRO] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ml_predict():
    """Teste 2: GET /api/ml/predict/R_100"""
    print("\n" + "="*60)
    print("TESTE 2: GET /api/ml/predict/R_100")
    print("="*60)

    try:
        url = f"{BASE_URL}/api/ml/predict/R_100?timeframe=1m&count=200"
        print(f"[INFO] Fazendo requisição...")
        print(f"  URL: {url}")

        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=60) as response:
            data = json.loads(response.read().decode())

            print(f"[OK] Previsão recebida!")

            print(f"\n  Previsão:")
            print(f"    Prediction: {data.get('prediction', 'N/A')}")
            conf = data.get('confidence', 0)
            print(f"    Confidence: {conf:.4f} ({conf*100:.2f}%)")
            print(f"    Signal Strength: {data.get('signal_strength', 'N/A')}")
            print(f"    Threshold: {data.get('threshold_used', 'N/A')}")

            print(f"\n  Contexto:")
            print(f"    Symbol: {data.get('symbol', 'N/A')}")
            print(f"    Timeframe: {data.get('timeframe', 'N/A')}")
            print(f"    Data Source: {data.get('data_source', 'N/A')}")
            print(f"    Candles: {data.get('candles_analyzed', 'N/A')}")

            print(f"\n  Interpretação:")
            signal = data.get('signal_strength', '')
            prediction = data.get('prediction', '')

            if signal == 'HIGH' and prediction == 'PRICE_UP':
                print(f"    ✅ SINAL FORTE DE COMPRA")
                print(f"    Recomendação: EXECUTAR TRADE")
            elif signal == 'MEDIUM' and prediction == 'PRICE_UP':
                print(f"    ⚠️  Sinal moderado de compra")
                print(f"    Recomendação: AGUARDAR CONFIRMAÇÃO")
            elif signal == 'LOW':
                print(f"    ❌ Sinal fraco")
                print(f"    Recomendação: NÃO OPERAR")
            else:
                print(f"    ➖ Sem movimento significativo previsto")
                print(f"    Recomendação: AGUARDAR")

            return True

    except urllib.error.HTTPError as e:
        print(f"[ERRO] HTTP {e.code}: {e.reason}")
        error_body = e.read().decode()
        print(f"  Response: {error_body}")
        return False
    except Exception as e:
        print(f"[ERRO] {e}")
        import traceback
        traceback.print_exc()
        return False


def run_tests():
    """Executa todos os testes"""
    print("\n" + "="*80)
    print(" "*25 + "ML ENDPOINTS TEST")
    print("="*80)
    print(f"\nServidor: {BASE_URL}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Verificar servidor
    if not test_health():
        return False

    # Executar testes
    tests = [
        ("GET /api/ml/info", test_ml_info),
        ("GET /api/ml/predict/R_100", test_ml_predict)
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n[ERRO CRÍTICO] {name}: {e}")
            results[name] = False

    # Resumo
    print("\n" + "="*80)
    print(" "*30 + "RESUMO")
    print("="*80)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for name, result in results.items():
        status = "[OK]  " if result else "[ERRO]"
        print(f"{status} {name}")

    print("-" * 80)
    print(f"Total: {passed}/{total} testes passaram ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n✅ SUCESSO! Endpoints ML estão funcionando!")
        print("\nPróximo passo:")
        print("  - Integrar ML com sistema de sinais")
        print("  - Criar lógica de decisão de trading")
        return True
    else:
        print(f"\n⚠️  {total - passed} teste(s) falharam")
        return False


if __name__ == "__main__":
    import time
    print("\nAguardando servidor iniciar (10 segundos)...")
    time.sleep(10)

    success = run_tests()
    sys.exit(0 if success else 1)
