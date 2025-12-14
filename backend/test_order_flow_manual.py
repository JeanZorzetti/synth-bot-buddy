"""
Teste manual para Order Flow Analyzer
Validação sem pytest
"""

from datetime import datetime, timedelta
from analysis.order_flow_analyzer import (
    OrderBookAnalyzer,
    AggressiveOrderDetector,
    VolumeProfileAnalyzer,
    TapeReader,
    OrderFlowAnalyzer
)


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
    print(f"Spread: {result['spread']} ({result['spread_pct']}%)")
    print(f"Bid Walls: {len(result['bid_walls'])}")
    print(f"Ask Walls: {len(result['ask_walls'])}")

    assert result['bid_volume'] == 1800, "Bid volume incorreto"
    assert result['ask_volume'] == 900, "Ask volume incorreto"
    assert result['imbalance'] == 'bullish', "Imbalance deveria ser bullish"

    print("✅ PASSED: OrderBookAnalyzer")
    return True


def test_aggressive_order_detector():
    """Teste 2: AggressiveOrderDetector"""
    print("\n=== Teste 2: AggressiveOrderDetector ===")
    detector = AggressiveOrderDetector(size_multiplier=3.0)

    trades = [
        {'price': 100.0, 'size': 100, 'side': 'buy'},
        {'price': 100.1, 'size': 500, 'side': 'buy'},  # Agressiva
        {'price': 100.2, 'size': 80, 'side': 'sell'},
        {'price': 100.3, 'size': 90, 'side': 'buy'},
    ]

    result = detector.detect_aggressive_orders(trades)

    print(f"Aggressive Buys: {len(result['aggressive_buys'])}")
    print(f"Aggressive Sells: {len(result['aggressive_sells'])}")
    print(f"Delta: {result['delta']}")
    print(f"Sentiment: {result['aggressive_sentiment']}")
    print(f"Buy Pressure: {result['buy_pressure']}%")

    assert len(result['aggressive_buys']) > 0, "Deveria detectar compras agressivas"
    assert result['aggressive_sentiment'] == 'bullish', "Sentimento deveria ser bullish"
    assert result['delta'] > 0, "Delta deveria ser positivo"

    print("✅ PASSED: AggressiveOrderDetector")
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
    print(f"Profile Points: {len(result['volume_profile'])}")

    assert result['poc'] > 0, "POC deveria ser > 0"
    assert result['vah'] > result['val'], "VAH deveria ser > VAL"
    assert result['total_volume'] > 0, "Total volume deveria ser > 0"

    print("✅ PASSED: VolumeProfileAnalyzer")
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
            'side': 'buy' if i < 35 else 'sell',  # 70% buys
            'timestamp': datetime.now() + timedelta(seconds=i)
        })

    result = reader.analyze_tape(trades)

    print(f"Buy Pressure: {result['buy_pressure']}%")
    print(f"Sell Pressure: {result['sell_pressure']}%")
    print(f"Buy Volume: {result['buy_volume']}")
    print(f"Sell Volume: {result['sell_volume']}")
    print(f"Momentum Speed: {result['momentum']['speed']}")
    print(f"Interpretation: {result['interpretation']}")

    assert result['buy_pressure'] > 60, "Buy pressure deveria ser > 60%"
    assert result['buy_volume'] > result['sell_volume'], "Buy volume deveria ser maior"

    print("✅ PASSED: TapeReader")
    return True


def test_order_flow_analyzer():
    """Teste 5: OrderFlowAnalyzer (integração completa)"""
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

    print(f"Timestamp: {result['timestamp']}")
    print(f"Order Book Analyzed: {'order_book' in result and result['order_book']}")
    print(f"Aggressive Orders Analyzed: {'aggressive_orders' in result and result['aggressive_orders']}")
    print(f"Volume Profile Analyzed: {'volume_profile' in result and result['volume_profile']}")
    print(f"Tape Analyzed: {'tape' in result and result['tape']}")

    assert 'timestamp' in result, "Deveria ter timestamp"
    assert 'order_book' in result, "Deveria ter order_book"
    assert result['order_book']['bid_volume'] > 0, "Bid volume deveria ser > 0"

    print("✅ PASSED: OrderFlowAnalyzer (Integração)")
    return True


def test_enhance_signal():
    """Teste 6: Enhance Signal com Order Flow"""
    print("\n=== Teste 6: Enhance Signal ===")
    analyzer = OrderFlowAnalyzer()

    technical_signal = {
        'type': 'BUY',
        'confidence': 65,
        'price': 100.5
    }

    order_book = {
        'bids': [[100.0, 2000], [99.9, 1000]],
        'asks': [[100.1, 500], [100.2, 300]]
    }

    trade_stream = []
    for i in range(30):
        trade_stream.append({
            'price': 100.0,
            'size': 100,
            'side': 'buy',
            'timestamp': datetime.now()
        })

    result = analyzer.enhance_signal(technical_signal, order_book, trade_stream)

    print(f"Original Confidence: {technical_signal['confidence']}%")
    print(f"Enhanced Confidence: {result['confidence']}%")
    print(f"Order Flow Score: {result['order_flow_confirmation_score']}")
    print(f"Reasons: {result['order_flow_reasons']}")

    assert 'confidence' in result, "Deveria ter confidence"
    assert 'order_flow_confirmation_score' in result, "Deveria ter confirmation score"

    print("✅ PASSED: Enhance Signal")
    return True


def run_all_tests():
    """Executa todos os testes"""
    print("\n" + "="*60)
    print("INICIANDO TESTES DO ORDER FLOW ANALYZER")
    print("="*60)

    tests = [
        test_order_book_analyzer,
        test_aggressive_order_detector,
        test_volume_profile_analyzer,
        test_tape_reader,
        test_order_flow_analyzer,
        test_enhance_signal
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
            failed += 1

    print("\n" + "="*60)
    print(f"RESULTADO FINAL: {passed}/{len(tests)} testes passaram")
    print("="*60)

    if failed == 0:
        print("✅ TODOS OS TESTES PASSARAM!")
        return True
    else:
        print(f"❌ {failed} TESTES FALHARAM")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
