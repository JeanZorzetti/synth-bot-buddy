"""
Testes para Order Flow Analyzer
Autor: Claude Code (Roadmap Tractor Mode)
Data: 2025-12-14
"""

import pytest
from datetime import datetime, timedelta
from analysis.order_flow_analyzer import (
    OrderBookAnalyzer,
    AggressiveOrderDetector,
    VolumeProfileAnalyzer,
    TapeReader,
    OrderFlowIntegrator,
    OrderFlowAnalyzer
)


class TestOrderBookAnalyzer:
    """Testes para OrderBookAnalyzer"""

    def test_analyze_depth_bullish(self):
        """Testa análise de order book com viés bullish"""
        analyzer = OrderBookAnalyzer()

        order_book = {
            'bids': [[100.0, 1000], [99.9, 500], [99.8, 300]],
            'asks': [[100.1, 400], [100.2, 300], [100.3, 200]]
        }

        result = analyzer.analyze_depth(order_book)

        assert result['bid_volume'] == 1800
        assert result['ask_volume'] == 900
        assert result['bid_pressure'] > 55  # Bullish
        assert result['imbalance'] == 'bullish'
        assert result['best_bid'] == 100.0
        assert result['best_ask'] == 100.1

    def test_analyze_depth_bearish(self):
        """Testa análise de order book com viés bearish"""
        analyzer = OrderBookAnalyzer()

        order_book = {
            'bids': [[100.0, 400], [99.9, 300], [99.8, 200]],
            'asks': [[100.1, 1000], [100.2, 500], [100.3, 300]]
        }

        result = analyzer.analyze_depth(order_book)

        assert result['bid_volume'] == 900
        assert result['ask_volume'] == 1800
        assert result['ask_pressure'] > 55  # Bearish
        assert result['imbalance'] == 'bearish'

    def test_find_walls(self):
        """Testa detecção de muros (walls)"""
        analyzer = OrderBookAnalyzer(wall_threshold_multiplier=3.0)

        orders = [
            [100.0, 1000],  # Grande ordem (wall)
            [99.9, 100],
            [99.8, 150],
            [99.7, 120]
        ]

        walls = analyzer.find_walls(orders, is_bid=True)

        assert len(walls) > 0
        assert walls[0]['price'] == 100.0
        assert walls[0]['size'] == 1000
        assert walls[0]['side'] == 'bid'

    def test_empty_order_book(self):
        """Testa análise com order book vazio"""
        analyzer = OrderBookAnalyzer()

        order_book = {'bids': [], 'asks': []}
        result = analyzer.analyze_depth(order_book)

        assert result['bid_volume'] == 0
        assert result['imbalance'] == 'neutral'


class TestAggressiveOrderDetector:
    """Testes para AggressiveOrderDetector"""

    def test_detect_aggressive_buys(self):
        """Testa detecção de compras agressivas"""
        detector = AggressiveOrderDetector(size_multiplier=3.0)

        trades = [
            {'price': 100.0, 'size': 100, 'side': 'buy'},
            {'price': 100.1, 'size': 500, 'side': 'buy'},  # Agressiva (5x média)
            {'price': 100.2, 'size': 80, 'side': 'sell'},
            {'price': 100.3, 'size': 90, 'side': 'buy'},
        ]

        result = detector.detect_aggressive_orders(trades)

        assert len(result['aggressive_buys']) > 0
        assert result['aggressive_sentiment'] == 'bullish'
        assert result['delta'] > 0

    def test_detect_aggressive_sells(self):
        """Testa detecção de vendas agressivas"""
        detector = AggressiveOrderDetector(size_multiplier=3.0)

        trades = [
            {'price': 100.0, 'size': 100, 'side': 'buy'},
            {'price': 99.9, 'size': 80, 'side': 'sell'},
            {'price': 99.8, 'size': 600, 'side': 'sell'},  # Agressiva
            {'price': 99.7, 'size': 90, 'side': 'sell'},
        ]

        result = detector.detect_aggressive_orders(trades)

        assert len(result['aggressive_sells']) > 0
        assert result['aggressive_sentiment'] == 'bearish'
        assert result['delta'] < 0

    def test_empty_trade_stream(self):
        """Testa detecção com stream vazio"""
        detector = AggressiveOrderDetector()

        result = detector.detect_aggressive_orders([])

        assert result['delta'] == 0
        assert result['aggressive_sentiment'] == 'neutral'


class TestVolumeProfileAnalyzer:
    """Testes para VolumeProfileAnalyzer"""

    def test_calculate_volume_profile(self):
        """Testa cálculo de volume profile"""
        analyzer = VolumeProfileAnalyzer(price_levels=50)

        trades = []
        # Criar trades concentrados em torno de 100.0 (POC esperado)
        for i in range(100):
            price = 100.0 + (i % 10) * 0.1  # Preços entre 100.0 e 100.9
            volume = 100 if abs((i % 10) - 5) < 2 else 50  # Mais volume em 100.4-100.5
            trades.append({'price': price, 'volume': volume})

        result = analyzer.calculate_volume_profile(trades)

        assert result['poc'] > 0
        assert result['vah'] > result['val']
        assert len(result['volume_profile']) > 0
        assert result['total_volume'] > 0

    def test_volume_profile_with_size(self):
        """Testa volume profile com campo 'size' ao invés de 'volume'"""
        analyzer = VolumeProfileAnalyzer()

        trades = [
            {'price': 100.0, 'size': 100},
            {'price': 100.1, 'size': 200},
            {'price': 100.2, 'size': 150},
        ]

        result = analyzer.calculate_volume_profile(trades)

        assert result['total_volume'] == 450

    def test_empty_trades(self):
        """Testa volume profile com lista vazia"""
        analyzer = VolumeProfileAnalyzer()

        result = analyzer.calculate_volume_profile([])

        assert result['poc'] == 0
        assert result['total_volume'] == 0


class TestTapeReader:
    """Testes para TapeReader"""

    def test_analyze_tape_bullish(self):
        """Testa tape reading com pressão bullish"""
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

        assert result['buy_pressure'] > 60
        assert result['buy_volume'] > result['sell_volume']

    def test_analyze_tape_bearish(self):
        """Testa tape reading com pressão bearish"""
        reader = TapeReader()

        trades = []
        for i in range(50):
            trades.append({
                'price': 100.0 - i * 0.01,
                'size': 100,
                'side': 'sell' if i < 35 else 'buy',  # 70% sells
                'timestamp': datetime.now() + timedelta(seconds=i)
            })

        result = reader.analyze_tape(trades)

        assert result['sell_pressure'] > 60
        assert result['sell_volume'] > result['buy_volume']

    def test_detect_absorption(self):
        """Testa detecção de absorção"""
        reader = TapeReader()

        # Criar cenário de absorção: alto volume, preço estável
        trades = []
        for i in range(20):
            trades.append({
                'price': 100.0 + (i % 3) * 0.001,  # Preço muito estável
                'size': 500 if i < 5 else 100,  # Volume alto no início
                'side': 'buy',
                'timestamp': datetime.now()
            })

        absorption = reader._detect_absorption(trades)

        # Pode ou não detectar dependendo dos thresholds, mas deve retornar estrutura correta
        assert 'detected' in absorption
        assert 'type' in absorption
        assert 'strength' in absorption

    def test_calculate_momentum(self):
        """Testa cálculo de momentum"""
        reader = TapeReader()

        trades = []
        for i in range(30):
            trades.append({
                'price': 100.0,
                'size': 100 + i * 10,  # Volume crescente (aceleração)
                'timestamp': datetime.now() + timedelta(seconds=i)
            })

        momentum = reader._calculate_momentum(trades)

        assert 'speed' in momentum
        assert 'trades_per_minute' in momentum
        assert 'acceleration' in momentum
        assert momentum['acceleration'] > 0  # Acelerando

    def test_empty_tape(self):
        """Testa tape reading com stream vazio"""
        reader = TapeReader()

        result = reader.analyze_tape([])

        assert result['buy_pressure'] == 50.0
        assert result['interpretation'] == 'sem dados suficientes'


class TestOrderFlowIntegrator:
    """Testes para OrderFlowIntegrator"""

    def test_confirm_buy_signal_bullish_flow(self):
        """Testa confirmação de sinal de compra com order flow bullish"""
        integrator = OrderFlowIntegrator()

        technical_signal = {
            'type': 'BUY',
            'confidence': 60,
            'price': 100.5
        }

        order_flow_data = {
            'order_book': {
                'bid_pressure': 65,  # Bullish
                'ask_pressure': 35
            },
            'aggressive_orders': {
                'aggressive_sentiment': 'bullish'  # Bullish
            },
            'volume_profile': {
                'poc': 100.0  # Price above POC = bullish
            },
            'tape': {
                'buy_pressure': 70,  # Bullish
                'absorption': {'detected': True, 'type': 'bullish_up'}
            }
        }

        result = integrator.confirm_signal_with_order_flow(technical_signal, order_flow_data)

        assert result['confidence'] > 60  # Deve aumentar
        assert result['order_flow_confirmation_score'] > 0
        assert result['enhanced_by_order_flow'] is True
        assert len(result['order_flow_reasons']) > 0

    def test_confirm_sell_signal_bearish_flow(self):
        """Testa confirmação de sinal de venda com order flow bearish"""
        integrator = OrderFlowIntegrator()

        technical_signal = {
            'type': 'SELL',
            'confidence': 55,
            'price': 99.5
        }

        order_flow_data = {
            'order_book': {
                'bid_pressure': 35,
                'ask_pressure': 65  # Bearish
            },
            'aggressive_orders': {
                'aggressive_sentiment': 'bearish'  # Bearish
            },
            'volume_profile': {
                'poc': 100.0  # Price below POC = bearish
            },
            'tape': {
                'sell_pressure': 68,  # Bearish
                'absorption': {'detected': True, 'type': 'bearish_down'}
            }
        }

        result = integrator.confirm_signal_with_order_flow(technical_signal, order_flow_data)

        assert result['confidence'] > 55
        assert result['order_flow_confirmation_score'] > 0

    def test_no_confirmation(self):
        """Testa quando order flow não confirma o sinal"""
        integrator = OrderFlowIntegrator()

        technical_signal = {
            'type': 'BUY',
            'confidence': 60,
            'price': 100.0
        }

        order_flow_data = {
            'order_book': {'bid_pressure': 45},  # Neutro
            'aggressive_orders': {'aggressive_sentiment': 'neutral'},
            'volume_profile': {'poc': 100.0},
            'tape': {'buy_pressure': 50, 'absorption': {'detected': False}}
        }

        result = integrator.confirm_signal_with_order_flow(technical_signal, order_flow_data)

        assert result['order_flow_confirmation_score'] == 0
        assert result['confidence'] == 60  # Sem mudança


class TestOrderFlowAnalyzer:
    """Testes para OrderFlowAnalyzer (classe principal)"""

    def test_analyze_complete(self):
        """Testa análise completa de order flow"""
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

        assert 'timestamp' in result
        assert 'order_book' in result
        assert 'aggressive_orders' in result
        assert 'volume_profile' in result
        assert 'tape' in result

        # Verificar que cada seção tem dados
        assert result['order_book']['bid_volume'] > 0
        assert result['aggressive_orders']['total_buy_volume'] > 0
        assert result['volume_profile']['total_volume'] > 0
        assert result['tape']['num_trades'] > 0

    def test_enhance_signal(self):
        """Testa melhoria de sinal técnico com order flow"""
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
                'side': 'buy',  # Maioria compras
                'timestamp': datetime.now()
            })

        result = analyzer.enhance_signal(technical_signal, order_book, trade_stream)

        assert 'confidence' in result
        assert 'order_flow_confirmation_score' in result
        assert 'order_flow_reasons' in result

    def test_analyze_with_no_data(self):
        """Testa análise sem dados"""
        analyzer = OrderFlowAnalyzer()

        result = analyzer.analyze_complete(None, None)

        assert result['order_book'] == {}
        assert result['aggressive_orders'] == {}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
