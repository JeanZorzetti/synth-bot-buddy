#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

from analysis.order_flow_analyzer import AggressiveOrderDetector, TapeReader, OrderFlowIntegrator
import json

print("=== TEST AggressiveOrderDetector ===")
detector = AggressiveOrderDetector()

trades = [
    {'price': 100.0, 'size': 100, 'side': 'buy', 'timestamp': '2024-01-01T10:00:00'},
    {'price': 100.1, 'size': 500, 'side': 'buy', 'timestamp': '2024-01-01T10:00:01'},
    {'price': 100.0, 'size': 80, 'side': 'sell', 'timestamp': '2024-01-01T10:00:02'},
]

result = detector.detect_aggressive_orders(trades)
print(json.dumps(result, indent=2))

print("\n=== TEST TapeReader ===")
reader = TapeReader()

trades2 = [
    {'price': 100.0, 'size': 100, 'side': 'buy', 'timestamp': '2024-01-01T10:00:00'},
    {'price': 100.1, 'size': 200, 'side': 'buy', 'timestamp': '2024-01-01T10:00:01'},
    {'price': 100.0, 'size': 50, 'side': 'sell', 'timestamp': '2024-01-01T10:00:02'},
]

result2 = reader.analyze_tape(trades2)
print(json.dumps(result2, indent=2))

print("\n=== TEST OrderFlowIntegrator ===")
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

result3 = integrator.confirm_signal_with_order_flow(signal, order_flow_data)
print(json.dumps(result3, indent=2))
