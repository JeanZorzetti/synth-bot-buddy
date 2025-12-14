"""
Teste simples para Order Flow Analyzer
Importação direta sem dependências do pacote analysis
"""

import sys
sys.path.insert(0, 'c:\\Users\\jeanz\\OneDrive\\Desktop\\Jizreel\\synth-bot-buddy-main\\backend\\analysis')

from datetime import datetime, timedelta
import importlib.util

# Importar diretamente o arquivo
spec = importlib.util.spec_from_file_location(
    "order_flow_analyzer",
    "c:\\Users\\jeanz\\OneDrive\\Desktop\\Jizreel\\synth-bot-buddy-main\\backend\\analysis\\order_flow_analyzer.py"
)
order_flow_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(order_flow_module)

OrderBookAnalyzer = order_flow_module.OrderBookAnalyzer
AggressiveOrderDetector = order_flow_module.AggressiveOrderDetector
VolumeProfileAnalyzer = order_flow_module.VolumeProfileAnalyzer
TapeReader = order_flow_module.TapeReader
OrderFlowAnalyzer = order_flow_module.OrderFlowAnalyzer


def test_order_book_analyzer():
    """Teste 1: OrderBookAnalyzer"""
    print("\n=== Teste 1: OrderBookAnalyzer ===")
    analyzer = OrderBookAnalyzer()

    order_book = {
        'bids': [[100.0, 1000], [99.9, 500], [99.8, 300]],
        'asks': [[100.1, 400], [100.2, 300], [100.3, 200]]
    }

    result = analyzer.analyze_depth(order_book)

    print(f"Bid Volume: {result['bid_volume']}")
    print(f"Ask Volume: {result['ask_volume']}")
    print(f"Bid Pressure: {result['bid_pressure']}%")
    print(f"Imbalance: {result['imbalance']}")

    assert result['bid_volume'] == 1800
    assert result['ask_volume'] == 900
    assert result['imbalance'] == 'bullish'

    print("✅ PASSED")
    return True


def test_aggressive_order_detector():
    """Teste 2: AggressiveOrderDetector"""
    print("\n=== Teste 2: AggressiveOrderDetector ===")
    detector = AggressiveOrderDetector(size_multiplier=3.0)

    trades = [
        {'price': 100.0, 'size': 100, 'side': 'buy'},
        {'price': 100.1, 'size': 500, 'side': 'buy'},
        {'price': 100.2, 'size': 80, 'side': 'sell'},
        {'price': 100.3, 'size': 90, 'side': 'buy'},
    ]

    result = detector.detect_aggressive_orders(trades)

    print(f"Aggressive Buys: {len(result['aggressive_buys'])}")
    print(f"Delta: {result['delta']}")
    print(f"Sentiment: {result['aggressive_sentiment']}")

    assert len(result['aggressive_buys']) > 0
    assert result['aggressive_sentiment'] == 'bullish'

    print("✅ PASSED")
    return True


def test_volume_profile_analyzer():
    """Teste 3: VolumeProfileAnalyzer"""
    print("\n=== Teste 3: VolumeProfileAnalyzer ===")
    analyzer = VolumeProfileAnalyzer(price_levels=50)

    trades = []
    for i in range(100):
        price = 100.0 + (i % 10) * 0.1
        volume = 100 if abs((i % 10) - 5) < 2 else 50
        trades.append({'price': price, 'volume': volume})

    result = analyzer.calculate_volume_profile(trades)

    print(f"POC: {result['poc']}")
    print(f"VAH: {result['vah']}")
    print(f"VAL: {result['val']}")
    print(f"Total Volume: {result['total_volume']}")

    assert result['poc'] > 0
    assert result['vah'] > result['val']

    print("✅ PASSED")
    return True


def test_tape_reader():
    """Teste 4: TapeReader"""
    print("\n=== Teste 4: TapeReader ===")
    reader = TapeReader(window_size=50)

    trades = []
    for i in range(50):
        trades.append({
            'price': 100.0 + i * 0.01,
            'size': 100,
            'side': 'buy' if i < 35 else 'sell',
            'timestamp': datetime.now() + timedelta(seconds=i)
        })

    result = reader.analyze_tape(trades)

    print(f"Buy Pressure: {result['buy_pressure']}%")
    print(f"Momentum Speed: {result['momentum']['speed']}")

    assert result['buy_pressure'] > 60

    print("✅ PASSED")
    return True


def test_order_flow_analyzer():
    """Teste 5: OrderFlowAnalyzer"""
    print("\n=== Teste 5: OrderFlowAnalyzer (Integração) ===")
    analyzer = OrderFlowAnalyzer()

    order_book = {
        'bids': [[100.0, 1000], [99.9, 500]],
        'asks': [[100.1, 600], [100.2, 400]]
    }

    trade_stream = []
    for i in range(50):
        trade_stream.append({
            'price': 100.0 + i * 0.01,
            'size': 100,
            'side': 'buy' if i % 2 == 0 else 'sell',
            'timestamp': datetime.now()
        })

    result = analyzer.analyze_complete(order_book, trade_stream)

    print(f"Order Book Analyzed: {result['order_book']['bid_volume'] > 0}")
    print(f"Aggressive Orders Analyzed: {result['aggressive_orders']['total_buy_volume'] > 0}")

    assert 'timestamp' in result
    assert result['order_book']['bid_volume'] > 0

    print("✅ PASSED")
    return True


def run_all_tests():
    """Executa todos os testes"""
    print("\n" + "="*60)
    print("TESTES DO ORDER FLOW ANALYZER")
    print("="*60)

    tests = [
        test_order_book_analyzer,
        test_aggressive_order_detector,
        test_volume_profile_analyzer,
        test_tape_reader,
        test_order_flow_analyzer
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ FAILED: {test.__name__}")
            print(f"   Error: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "="*60)
    print(f"RESULTADO: {passed}/{len(tests)} testes passaram")
    print("="*60)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
