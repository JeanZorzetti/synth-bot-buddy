"""
Test ML Integration - Verifica integração do modelo ML com backend

Testa:
1. Carregamento do modelo
2. Cálculo de features
3. Predições
4. Endpoints da API
"""

import sys
import os
from pathlib import Path

# Adicionar backend ao path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

import pandas as pd
from ml_predictor import MLPredictor, get_ml_predictor
from market_data_fetcher import create_sample_dataframe


def test_ml_predictor_initialization():
    """Teste 1: Inicialização do MLPredictor"""
    print("\n" + "="*60)
    print("TESTE 1: Inicialização do MLPredictor")
    print("="*60)

    try:
        predictor = MLPredictor(threshold=0.30)
        print(f"[OK] MLPredictor inicializado")
        print(f"  Modelo: {predictor.model_path.name}")
        print(f"  Threshold: {predictor.threshold}")
        print(f"  Features: {len(predictor.feature_names)}")
        return True
    except Exception as e:
        print(f"[ERRO] Falha na inicialização: {e}")
        return False


def test_feature_calculation():
    """Teste 2: Cálculo de Features"""
    print("\n" + "="*60)
    print("TESTE 2: Cálculo de Features")
    print("="*60)

    try:
        predictor = MLPredictor(threshold=0.30)

        # Criar dados sintéticos
        df = create_sample_dataframe(bars=250)
        print(f"[OK] Dados criados: {len(df)} candles")

        # Calcular features
        features = predictor._calculate_features(df)

        if features is None:
            print(f"[ERRO] Features retornou None")
            return False

        print(f"[OK] Features calculadas: {features.shape}")
        print(f"  Primeiras 5 features: {list(features.columns)[:5]}")
        print(f"  Sample values: {features.iloc[0, :3].to_dict()}")
        return True
    except Exception as e:
        print(f"[ERRO] Falha no cálculo de features: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prediction():
    """Teste 3: Previsão"""
    print("\n" + "="*60)
    print("TESTE 3: Previsão")
    print("="*60)

    try:
        predictor = MLPredictor(threshold=0.30)

        # Criar dados sintéticos
        df = create_sample_dataframe(bars=250)
        print(f"[OK] Dados criados: {len(df)} candles")

        # Fazer previsão
        result = predictor.predict(df, return_confidence=True)

        print(f"[OK] Previsão realizada:")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print(f"  Signal Strength: {result['signal_strength']}")
        print(f"  Threshold Used: {result['threshold_used']}")

        # Validar resultado
        assert 'prediction' in result
        assert 'confidence' in result
        assert 'signal_strength' in result
        assert 0 <= result['confidence'] <= 1

        print(f"[OK] Validações passaram!")
        return True
    except Exception as e:
        print(f"[ERRO] Falha na previsão: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_thresholds():
    """Teste 4: Diferentes Thresholds"""
    print("\n" + "="*60)
    print("TESTE 4: Diferentes Thresholds")
    print("="*60)

    try:
        thresholds = [0.25, 0.30, 0.35, 0.40, 0.50]
        df = create_sample_dataframe(bars=250)

        print(f"\nTestando {len(thresholds)} thresholds com mesmos dados:")
        print(f"{'Threshold':<12} {'Prediction':<15} {'Confidence':<12} {'Strength':<10}")
        print("-" * 60)

        for threshold in thresholds:
            predictor = MLPredictor(threshold=threshold)
            result = predictor.predict(df, return_confidence=False)

            print(f"{threshold:<12.2f} {result['prediction']:<15} {result['confidence']:<12.4f} {result['signal_strength']:<10}")

        print(f"\n[OK] Todos os thresholds testados!")
        return True
    except Exception as e:
        print(f"[ERRO] Falha no teste de thresholds: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_info():
    """Teste 5: Informações do Modelo"""
    print("\n" + "="*60)
    print("TESTE 5: Informações do Modelo")
    print("="*60)

    try:
        predictor = MLPredictor(threshold=0.30)
        info = predictor.get_model_info()

        print(f"[OK] Informações do modelo:")
        print(f"  Model: {info['model_name']}")
        print(f"  Type: {info['model_type']}")
        print(f"  Threshold: {info['threshold']}")
        print(f"  Features: {info['n_features']}")
        print(f"\n  Performance Esperada:")
        for key, value in info['expected_performance'].items():
            print(f"    {key}: {value}")

        return True
    except Exception as e:
        print(f"[ERRO] Falha ao obter info do modelo: {e}")
        return False


def test_singleton_pattern():
    """Teste 6: Singleton Pattern"""
    print("\n" + "="*60)
    print("TESTE 6: Singleton Pattern")
    print("="*60)

    try:
        # Primeira chamada
        predictor1 = get_ml_predictor(threshold=0.30)
        id1 = id(predictor1)
        print(f"[OK] Primeira instância criada: ID={id1}")

        # Segunda chamada (deve retornar mesma instância)
        predictor2 = get_ml_predictor()
        id2 = id(predictor2)
        print(f"[OK] Segunda chamada: ID={id2}")

        if id1 == id2:
            print(f"[OK] Singleton funciona corretamente (mesma instância)")
            return True
        else:
            print(f"[ERRO] Singleton falhou (instâncias diferentes)")
            return False
    except Exception as e:
        print(f"[ERRO] Falha no teste de singleton: {e}")
        return False


def run_all_tests():
    """Executa todos os testes"""
    print("\n" + "="*80)
    print(" "*20 + "ML INTEGRATION TEST SUITE")
    print("="*80)

    tests = [
        ("Inicialização", test_ml_predictor_initialization),
        ("Cálculo de Features", test_feature_calculation),
        ("Previsão", test_prediction),
        ("Diferentes Thresholds", test_different_thresholds),
        ("Info do Modelo", test_model_info),
        ("Singleton Pattern", test_singleton_pattern)
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n[ERRO CRÍTICO] Teste '{name}' falhou com exceção: {e}")
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
        print("\n[SUCESSO] Todos os testes passaram! ML integration está OK!")
        return True
    else:
        print(f"\n[ATENÇÃO] {total - passed} teste(s) falharam. Revisar implementação.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
