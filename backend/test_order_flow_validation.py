#!/usr/bin/env python3
"""
Teste de validação rápida do Order Flow Analyzer (sem emojis)
"""
import sys
sys.path.insert(0, '.')

from analysis.order_flow_analyzer import (
    OrderBookAnalyzer,
    AggressiveOrderDetector,
    VolumeProfileAnalyzer,
    TapeReader,
    OrderFlowIntegrator
)

def test_order_book():
    print("\n[TEST] OrderBookAnalyzer")
    analyzer = OrderBookAnalyzer()

    # Format correto: [[price, volume], ...]
    order_book = {
        'bids': [
            [100.0, 500],
            [99.9, 800],
            [99.8, 500]
        ],
        'asks': [
            [100.1, 300],
            [100.2, 400],
            [100.3, 200]
        ]
    }

    result = analyzer.analyze_depth(order_book)

    assert result['bid_volume'] == 1800
    assert result['ask_volume'] == 900
    assert result['imbalance'] == 'bullish'
    assert result['bid_pressure'] > 60

    print("  - Bid Volume: OK")
    print("  - Ask Volume: OK")
    print("  - Imbalance: OK")
    print("  - Bid Pressure: OK")
    print("  PASSED\n")
    return True

def test_aggressive_orders():
    print("[TEST] AggressiveOrderDetector")
    detector = AggressiveOrderDetector()

    trades = [
        {'price': 100.0, 'size': 100, 'side': 'buy', 'timestamp': '2024-01-01T10:00:00'},
        {'price': 100.1, 'size': 500, 'side': 'buy', 'timestamp': '2024-01-01T10:00:01'},
        {'price': 100.0, 'size': 80, 'side': 'sell', 'timestamp': '2024-01-01T10:00:02'},
    ]

    result = detector.detect_aggressive_orders(trades)

    assert 'aggressive_sentiment' in result
    assert result['aggressive_sentiment'] in ['bullish', 'bearish', 'neutral']
    assert 'total_buy_volume' in result
    assert 'total_sell_volume' in result

    print("  - Aggressive Orders: OK")
    print("  - Sentiment: OK")
    print("  - Volumes: OK")
    print("  PASSED\n")
    return True

def test_volume_profile():
    print("[TEST] VolumeProfileAnalyzer")
    analyzer = VolumeProfileAnalyzer(price_levels=10)

    trades = []
    for i in range(100):
        trades.append({
            'price': 100.0 + (i % 10) / 10,
            'volume': 100 + i,
            'timestamp': f'2024-01-01T10:00:{i:02d}'
        })

    result = analyzer.calculate_volume_profile(trades)

    assert 'poc' in result
    assert 'vah' in result
    assert 'val' in result
    assert len(result['volume_profile']) > 0

    print("  - POC: OK")
    print("  - VAH/VAL: OK")
    print("  - Profile: OK")
    print("  PASSED\n")
    return True

def test_tape_reader():
    print("[TEST] TapeReader")
    reader = TapeReader()

    trades = [
        {'price': 100.0, 'volume': 100, 'side': 'buy', 'timestamp': '2024-01-01T10:00:00'},
        {'price': 100.1, 'volume': 200, 'side': 'buy', 'timestamp': '2024-01-01T10:00:01'},
        {'price': 100.0, 'volume': 50, 'side': 'sell', 'timestamp': '2024-01-01T10:00:02'},
    ]

    result = reader.analyze_tape(trades)

    assert 'buy_pressure' in result
    assert 'sell_pressure' in result
    assert 'interpretation' in result
    assert 'absorption' in result
    assert 'momentum' in result

    print("  - Buy/Sell Pressure: OK")
    print("  - Interpretation: OK")
    print("  - Absorption: OK")
    print("  - Momentum: OK")
    print("  PASSED\n")
    return True

def test_integrator():
    print("[TEST] OrderFlowIntegrator")
    integrator = OrderFlowIntegrator()

    signal = {
        'type': 'BUY',
        'confidence': 60,
        'symbol': 'R_100'
    }

    order_flow_data = {
        'order_book': {'imbalance': 'bullish', 'bid_pressure': 70.0},
        'aggressive_orders': {'sentiment': 'bullish', 'buy_volume': 1000},
        'tape_reading': {'interpretation': 'bullish', 'buy_pressure': 0.7}
    }

    result = integrator.confirm_signal_with_order_flow(signal, order_flow_data)

    assert result['type'] == 'BUY'
    assert 'confidence' in result
    assert 'order_flow_confirmation_score' in result
    assert 'order_flow_reasons' in result
    assert result['enhanced_by_order_flow'] == True

    print("  - Signal Confirmation: OK")
    print("  - Confidence Adjusted: OK")
    print("  - Confirmation Score: OK")
    print("  - Enhancement Reasons: OK")
    print("  PASSED\n")
    return True

def main():
    print("=" * 60)
    print("ORDER FLOW ANALYZER - VALIDATION TESTS")
    print("=" * 60)

    tests = [
        test_order_book,
        test_aggressive_orders,
        test_volume_profile,
        test_tape_reader,
        test_integrator
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"  FAILED (Assertion): {e}\n")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  FAILED (Exception): {e}")
            traceback.print_exc()
            print()
            failed += 1

    print("=" * 60)
    print(f"RESULTS: {passed} PASSED, {failed} FAILED")
    print("=" * 60)

    if failed == 0:
        print("\nALL TESTS PASSED - Order Flow Analyzer is ready!")
        return True
    else:
        print(f"\n{failed} tests failed - Please check the implementation")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
