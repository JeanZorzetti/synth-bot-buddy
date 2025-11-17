"""
Test ML Endpoints - Valida endpoints da API ML

Testa os 3 endpoints ML implementados:
1. GET /api/ml/info
2. GET /api/ml/predict/{symbol}
3. POST /api/ml/predict
"""

import sys
import requests
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

# Adicionar backend ao path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from market_data_fetcher import create_sample_dataframe

# URL base da API
BASE_URL = "http://localhost:8000"


def test_ml_info_endpoint():
    """Teste 1: GET /api/ml/info"""
    print("\n" + "="*60)
    print("TESTE 1: GET /api/ml/info")
    print("="*60)

    try:
        response = requests.get(f"{BASE_URL}/api/ml/info")

        if response.status_code == 200:
            data = response.json()
            print(f"[OK] Status: {response.status_code}")
            print(f"  Model: {data.get('model_name', 'N/A')}")
            print(f"  Type: {data.get('model_type', 'N/A')}")
            print(f"  Threshold: {data.get('threshold', 'N/A')}")
            print(f"  Features: {data.get('n_features', 'N/A')}")

            # Verificar performance esperada
            perf = data.get('expected_performance', {})
            print(f"\n  Performance Esperada:")
            print(f"    Accuracy: {perf.get('accuracy', 'N/A')}")
            print(f"    Recall: {perf.get('recall', 'N/A')}")
            print(f"    Profit: {perf.get('profit_6_months', 'N/A')}")
            print(f"    Win Rate: {perf.get('win_rate', 'N/A')}")

            return True
        else:
            print(f"[ERRO] Status: {response.status_code}")
            print(f"  Response: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"[ERRO] Não foi possível conectar ao servidor em {BASE_URL}")
        print(f"  Certifique-se de que o servidor está rodando:")
        print(f"  cd backend && uvicorn main:app --reload")
        return False
    except Exception as e:
        print(f"[ERRO] Exceção: {e}")
        return False


def test_ml_predict_get_endpoint():
    """Teste 2: GET /api/ml/predict/{symbol}"""
    print("\n" + "="*60)
    print("TESTE 2: GET /api/ml/predict/R_100")
    print("="*60)

    try:
        # Testar com símbolo R_100
        params = {
            'timeframe': '1m',
            'count': 200
        }

        print(f"[INFO] Fazendo requisição para R_100...")
        response = requests.get(
            f"{BASE_URL}/api/ml/predict/R_100",
            params=params,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            print(f"[OK] Status: {response.status_code}")
            print(f"\n  Previsão ML:")
            print(f"    Prediction: {data.get('prediction', 'N/A')}")
            print(f"    Confidence: {data.get('confidence', 0):.4f} ({data.get('confidence', 0)*100:.2f}%)")
            print(f"    Signal Strength: {data.get('signal_strength', 'N/A')}")
            print(f"    Threshold Used: {data.get('threshold_used', 'N/A')}")
            print(f"\n  Contexto:")
            print(f"    Symbol: {data.get('symbol', 'N/A')}")
            print(f"    Timeframe: {data.get('timeframe', 'N/A')}")
            print(f"    Data Source: {data.get('data_source', 'N/A')}")
            print(f"    Candles Analyzed: {data.get('candles_analyzed', 'N/A')}")
            print(f"    Timestamp: {data.get('timestamp', 'N/A')}")

            # Interpretar sinal
            print(f"\n  Interpretação:")
            if data.get('signal_strength') == 'HIGH' and data.get('prediction') == 'PRICE_UP':
                print(f"    ✅ SINAL FORTE DE COMPRA!")
            elif data.get('signal_strength') == 'MEDIUM' and data.get('prediction') == 'PRICE_UP':
                print(f"    ⚠️  Sinal moderado de compra")
            elif data.get('signal_strength') == 'LOW':
                print(f"    ❌ Sinal fraco - não trade")
            else:
                print(f"    ➖ Sem movimento previsto")

            return True
        else:
            print(f"[ERRO] Status: {response.status_code}")
            print(f"  Response: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print(f"[ERRO] Timeout - servidor demorou muito para responder")
        print(f"  Isso é normal na primeira requisição (modelo está carregando)")
        return False
    except requests.exceptions.ConnectionError:
        print(f"[ERRO] Não foi possível conectar ao servidor")
        return False
    except Exception as e:
        print(f"[ERRO] Exceção: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ml_predict_post_endpoint():
    """Teste 3: POST /api/ml/predict"""
    print("\n" + "="*60)
    print("TESTE 3: POST /api/ml/predict")
    print("="*60)

    try:
        # Criar dados sintéticos
        df = create_sample_dataframe(bars=250)

        # Converter para formato JSON
        candles = []
        for _, row in df.iterrows():
            candles.append({
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'timestamp': row['timestamp'].isoformat() if hasattr(row['timestamp'], 'isoformat') else str(row['timestamp'])
            })

        payload = {
            'candles': candles,
            'threshold': 0.30
        }

        print(f"[INFO] Enviando {len(candles)} candles para previsão...")
        response = requests.post(
            f"{BASE_URL}/api/ml/predict",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            print(f"[OK] Status: {response.status_code}")
            print(f"\n  Previsão ML:")
            print(f"    Prediction: {data.get('prediction', 'N/A')}")
            print(f"    Confidence: {data.get('confidence', 0):.4f} ({data.get('confidence', 0)*100:.2f}%)")
            print(f"    Signal Strength: {data.get('signal_strength', 'N/A')}")
            print(f"    Threshold Used: {data.get('threshold_used', 'N/A')}")
            print(f"    Model: {data.get('model', 'N/A')}")

            return True
        else:
            print(f"[ERRO] Status: {response.status_code}")
            print(f"  Response: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print(f"[ERRO] Timeout - servidor demorou muito para responder")
        return False
    except requests.exceptions.ConnectionError:
        print(f"[ERRO] Não foi possível conectar ao servidor")
        return False
    except Exception as e:
        print(f"[ERRO] Exceção: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_thresholds():
    """Teste 4: Testar diferentes thresholds"""
    print("\n" + "="*60)
    print("TESTE 4: Diferentes Thresholds")
    print("="*60)

    thresholds = [0.25, 0.30, 0.35, 0.40, 0.50]

    try:
        print(f"\nTestando {len(thresholds)} thresholds:\n")
        print(f"{'Threshold':<12} {'Prediction':<15} {'Confidence':<12} {'Strength':<10}")
        print("-" * 60)

        for threshold in thresholds:
            response = requests.get(
                f"{BASE_URL}/api/ml/predict/R_100",
                params={'timeframe': '1m', 'count': 200, 'threshold': threshold},
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                pred = data.get('prediction', 'N/A')
                conf = data.get('confidence', 0)
                strength = data.get('signal_strength', 'N/A')

                print(f"{threshold:<12.2f} {pred:<15} {conf:<12.4f} {strength:<10}")
            else:
                print(f"{threshold:<12.2f} {'ERROR':<15} {'N/A':<12} {'N/A':<10}")

        print("\n[OK] Todos os thresholds testados!")
        return True

    except Exception as e:
        print(f"[ERRO] Exceção: {e}")
        return False


def run_all_tests():
    """Executa todos os testes"""
    print("\n" + "="*80)
    print(" "*20 + "ML ENDPOINTS TEST SUITE")
    print("="*80)
    print(f"\nServidor: {BASE_URL}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Verificar se servidor está rodando
    print("\n[INFO] Verificando se servidor está rodando...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print(f"[OK] Servidor está online!")
        else:
            print(f"[AVISO] Servidor respondeu com status {response.status_code}")
    except:
        print(f"\n{'='*80}")
        print(f"[ERRO] SERVIDOR NÃO ESTÁ RODANDO!")
        print(f"{'='*80}")
        print(f"\nPara iniciar o servidor, execute:")
        print(f"  cd backend")
        print(f"  uvicorn main:app --reload")
        print(f"\nOu:")
        print(f"  cd backend")
        print(f"  python -m uvicorn main:app --reload")
        print(f"\n{'='*80}\n")
        return False

    # Executar testes
    tests = [
        ("GET /api/ml/info", test_ml_info_endpoint),
        ("GET /api/ml/predict/{symbol}", test_ml_predict_get_endpoint),
        ("POST /api/ml/predict", test_ml_predict_post_endpoint),
        ("Diferentes Thresholds", test_different_thresholds)
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n[ERRO CRÍTICO] Teste '{name}' falhou: {e}")
            results[name] = False

    # Resumo
    print("\n" + "="*80)
    print(" "*30 + "RESUMO DOS TESTES")
    print("="*80)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for name, result in results.items():
        status = "[OK]  " if result else "[ERRO]"
        print(f"{status} {name}")

    print("-" * 80)
    print(f"Total: {passed}/{total} testes passaram ({passed/total*100:.1f}%)")

    if passed == total:
        print("\n[SUCESSO] Todos os endpoints ML estão funcionando corretamente!")
        print("\nPróximo passo: Integrar ML com sistema de sinais")
        return True
    else:
        print(f"\n[ATENÇÃO] {total - passed} teste(s) falharam.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
