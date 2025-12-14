"""
Order Flow Analysis Module
Análise de livro de ordens, fluxo de ordens e volume profile para identificar intenção institucional.

Autor: Claude Code (Roadmap Tractor Mode)
Data: 2025-12-14
Fase: 5 - Análise de Fluxo de Ordens
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class OrderBookAnalyzer:
    """
    Analisa o livro de ordens (order book) para identificar:
    - Desequilíbrio entre bid/ask
    - Muros de ordens (walls)
    - Pressão compradora/vendedora
    """

    def __init__(self, wall_threshold_multiplier: float = 3.0):
        """
        Inicializa o analisador de order book.

        Args:
            wall_threshold_multiplier: Multiplicador para detectar muros (ex: 3x média)
        """
        self.wall_threshold_multiplier = wall_threshold_multiplier

    def analyze_depth(self, order_book: Dict) -> Dict:
        """
        Analisa a profundidade do mercado e desequilíbrio entre compra e venda.

        Args:
            order_book: Dicionário com 'bids' e 'asks'
                Format: {'bids': [[price, size], ...], 'asks': [[price, size], ...]}

        Returns:
            Dicionário com análise completa do order book
        """
        try:
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])

            if not bids or not asks:
                return self._empty_analysis()

            # Calcular volumes totais
            bid_volume = sum([order[1] for order in bids])
            ask_volume = sum([order[1] for order in asks])
            total_volume = bid_volume + ask_volume

            if total_volume == 0:
                return self._empty_analysis()

            # Calcular pressão (% do volume total)
            bid_pressure = (bid_volume / total_volume) * 100
            ask_pressure = (ask_volume / total_volume) * 100

            # Detectar muros
            bid_walls = self.find_walls(bids, is_bid=True)
            ask_walls = self.find_walls(asks, is_bid=False)

            # Determinar sentimento
            imbalance = self._determine_imbalance(bid_pressure)

            # Calcular spread
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0
            spread = best_ask - best_bid if (best_bid and best_ask) else 0
            spread_pct = (spread / best_ask * 100) if best_ask > 0 else 0

            return {
                'bid_volume': bid_volume,
                'ask_volume': ask_volume,
                'total_volume': total_volume,
                'bid_pressure': round(bid_pressure, 2),
                'ask_pressure': round(ask_pressure, 2),
                'bid_walls': bid_walls,
                'ask_walls': ask_walls,
                'imbalance': imbalance,
                'spread': spread,
                'spread_pct': round(spread_pct, 4),
                'best_bid': best_bid,
                'best_ask': best_ask,
                'depth_ratio': round(bid_volume / ask_volume, 2) if ask_volume > 0 else 0
            }

        except Exception as e:
            logger.error(f"Erro ao analisar order book: {str(e)}")
            return self._empty_analysis()

    def find_walls(self, orders: List[List], is_bid: bool = True) -> List[Dict]:
        """
        Detecta muros (grandes ordens) no livro de ordens.

        Args:
            orders: Lista de ordens [[price, size], ...]
            is_bid: True se são bids, False se são asks

        Returns:
            Lista de muros detectados
        """
        if not orders or len(orders) < 3:
            return []

        sizes = [order[1] for order in orders]
        avg_size = np.mean(sizes)
        wall_threshold = avg_size * self.wall_threshold_multiplier

        walls = []
        for order in orders:
            price, size = order[0], order[1]

            if size >= wall_threshold:
                walls.append({
                    'price': price,
                    'size': size,
                    'side': 'bid' if is_bid else 'ask',
                    'size_vs_avg': round(size / avg_size, 2),
                    'significance': 'high' if size >= wall_threshold * 1.5 else 'medium'
                })

        return walls

    def _determine_imbalance(self, bid_pressure: float) -> str:
        """
        Determina o desequilíbrio do mercado baseado na pressão de compra.

        Args:
            bid_pressure: Percentual de pressão de compra (0-100)

        Returns:
            String indicando o sentimento: 'bullish', 'bearish' ou 'neutral'
        """
        if bid_pressure > 55:
            return 'bullish'
        elif bid_pressure < 45:
            return 'bearish'
        else:
            return 'neutral'

    def _empty_analysis(self) -> Dict:
        """Retorna análise vazia quando não há dados."""
        return {
            'bid_volume': 0,
            'ask_volume': 0,
            'total_volume': 0,
            'bid_pressure': 50.0,
            'ask_pressure': 50.0,
            'bid_walls': [],
            'ask_walls': [],
            'imbalance': 'neutral',
            'spread': 0,
            'spread_pct': 0,
            'best_bid': 0,
            'best_ask': 0,
            'depth_ratio': 1.0
        }


class AggressiveOrderDetector:
    """
    Detecta ordens agressivas (market orders) e calcula delta de volume.
    """

    def __init__(self, size_multiplier: float = 3.0):
        """
        Inicializa o detector de ordens agressivas.

        Args:
            size_multiplier: Multiplicador para detectar ordens grandes (ex: 3x média)
        """
        self.size_multiplier = size_multiplier

    def detect_aggressive_orders(self, trade_stream: List[Dict]) -> Dict:
        """
        Identifica grandes ordens executadas (market orders).

        Args:
            trade_stream: Lista de trades {'price', 'size', 'side', 'timestamp'}

        Returns:
            Análise de ordens agressivas e delta
        """
        if not trade_stream:
            return self._empty_detection()

        try:
            # Calcular tamanho médio dos trades
            sizes = [trade['size'] for trade in trade_stream]
            avg_trade_size = np.mean(sizes)
            aggressive_threshold = avg_trade_size * self.size_multiplier

            aggressive_buys = []
            aggressive_sells = []
            total_buy_volume = 0
            total_sell_volume = 0

            for trade in trade_stream:
                size = trade['size']
                side = trade.get('side', 'buy')  # Default to 'buy' if not specified

                # Acumular volumes totais
                if side == 'buy':
                    total_buy_volume += size
                else:
                    total_sell_volume += size

                # Detectar ordens agressivas (>3x média)
                if size >= aggressive_threshold:
                    trade_info = {
                        'price': trade['price'],
                        'size': size,
                        'timestamp': trade.get('timestamp', datetime.now()),
                        'size_vs_avg': round(size / avg_trade_size, 2)
                    }

                    if side == 'buy':
                        aggressive_buys.append(trade_info)
                    else:
                        aggressive_sells.append(trade_info)

            # Calcular delta (compras - vendas)
            aggressive_buy_volume = sum([t['size'] for t in aggressive_buys])
            aggressive_sell_volume = sum([t['size'] for t in aggressive_sells])
            delta = aggressive_buy_volume - aggressive_sell_volume

            # Sentimento baseado no delta
            aggressive_sentiment = self._determine_sentiment(delta)

            # Intensidade da agressão (0-100)
            total_aggressive = aggressive_buy_volume + aggressive_sell_volume
            total_volume = total_buy_volume + total_sell_volume
            aggression_intensity = (total_aggressive / total_volume * 100) if total_volume > 0 else 0

            return {
                'aggressive_buys': aggressive_buys,
                'aggressive_sells': aggressive_sells,
                'aggressive_buy_volume': aggressive_buy_volume,
                'aggressive_sell_volume': aggressive_sell_volume,
                'delta': delta,
                'aggressive_sentiment': aggressive_sentiment,
                'avg_trade_size': round(avg_trade_size, 2),
                'aggression_intensity': round(aggression_intensity, 2),
                'total_buy_volume': total_buy_volume,
                'total_sell_volume': total_sell_volume,
                'buy_pressure': round((total_buy_volume / total_volume * 100) if total_volume > 0 else 50, 2)
            }

        except Exception as e:
            logger.error(f"Erro ao detectar ordens agressivas: {str(e)}")
            return self._empty_detection()

    def _determine_sentiment(self, delta: float) -> str:
        """
        Determina o sentimento baseado no delta de volume.

        Args:
            delta: Diferença entre compras e vendas agressivas

        Returns:
            'bullish', 'bearish' ou 'neutral'
        """
        if delta > 0:
            return 'bullish'
        elif delta < 0:
            return 'bearish'
        else:
            return 'neutral'

    def _empty_detection(self) -> Dict:
        """Retorna detecção vazia quando não há dados."""
        return {
            'aggressive_buys': [],
            'aggressive_sells': [],
            'aggressive_buy_volume': 0,
            'aggressive_sell_volume': 0,
            'delta': 0,
            'aggressive_sentiment': 'neutral',
            'avg_trade_size': 0,
            'aggression_intensity': 0,
            'total_buy_volume': 0,
            'total_sell_volume': 0,
            'buy_pressure': 50.0
        }


class VolumeProfileAnalyzer:
    """
    Analisa volume profile e identifica POC, VAH, VAL.
    """

    def __init__(self, price_levels: int = 100):
        """
        Inicializa o analisador de volume profile.

        Args:
            price_levels: Número de níveis de preço para discretização
        """
        self.price_levels = price_levels

    def calculate_volume_profile(self, trades: List[Dict]) -> Dict:
        """
        Cria perfil de volume por nível de preço.

        Args:
            trades: Lista de trades {'price', 'volume', 'timestamp'}

        Returns:
            Dicionário com POC, VAH, VAL e volume profile
        """
        if not trades or len(trades) < 10:
            return self._empty_profile()

        try:
            # Extrair preços e volumes
            prices = [t['price'] for t in trades]
            volumes = [t.get('volume', t.get('size', 0)) for t in trades]

            min_price = min(prices)
            max_price = max(prices)

            if min_price == max_price:
                return self._empty_profile()

            # Discretizar preços em níveis
            volume_by_level = defaultdict(float)

            for price, volume in zip(prices, volumes):
                level = self._discretize_price(price, min_price, max_price)
                volume_by_level[level] += volume

            if not volume_by_level:
                return self._empty_profile()

            # POC = nível com maior volume
            poc_level = max(volume_by_level, key=volume_by_level.get)
            poc_volume = volume_by_level[poc_level]

            # Calcular Value Area (70% do volume)
            value_area = self._calculate_value_area(volume_by_level, 0.70)

            # Converter níveis para preços reais
            poc_price = self._level_to_price(poc_level, min_price, max_price)
            vah_price = self._level_to_price(value_area['high'], min_price, max_price)
            val_price = self._level_to_price(value_area['low'], min_price, max_price)

            # Preparar volume profile para visualização
            profile_data = []
            for level in sorted(volume_by_level.keys()):
                price = self._level_to_price(level, min_price, max_price)
                volume = volume_by_level[level]
                profile_data.append({
                    'price': round(price, 5),
                    'volume': round(volume, 2),
                    'level': level
                })

            return {
                'poc': round(poc_price, 5),
                'poc_volume': round(poc_volume, 2),
                'vah': round(vah_price, 5),
                'val': round(val_price, 5),
                'value_area_volume_pct': 70.0,
                'volume_profile': profile_data,
                'total_volume': sum(volumes),
                'price_range': {
                    'min': round(min_price, 5),
                    'max': round(max_price, 5)
                }
            }

        except Exception as e:
            logger.error(f"Erro ao calcular volume profile: {str(e)}")
            return self._empty_profile()

    def _discretize_price(self, price: float, min_price: float, max_price: float) -> int:
        """
        Converte preço em nível discreto (0 a price_levels-1).

        Args:
            price: Preço atual
            min_price: Preço mínimo do período
            max_price: Preço máximo do período

        Returns:
            Nível discreto (int)
        """
        if max_price == min_price:
            return 0

        normalized = (price - min_price) / (max_price - min_price)
        level = int(normalized * (self.price_levels - 1))
        return max(0, min(level, self.price_levels - 1))

    def _level_to_price(self, level: int, min_price: float, max_price: float) -> float:
        """
        Converte nível discreto de volta para preço.

        Args:
            level: Nível discreto
            min_price: Preço mínimo do período
            max_price: Preço máximo do período

        Returns:
            Preço real
        """
        normalized = level / (self.price_levels - 1)
        return min_price + (normalized * (max_price - min_price))

    def _calculate_value_area(self, volume_by_level: Dict[int, float], target_pct: float = 0.70) -> Dict:
        """
        Calcula Value Area (VAH e VAL) que contém target_pct% do volume.

        Args:
            volume_by_level: Dicionário {level: volume}
            target_pct: Percentual do volume para incluir (default 70%)

        Returns:
            Dicionário com 'high' e 'low' levels
        """
        if not volume_by_level:
            return {'high': 0, 'low': 0}

        # Ordenar níveis por volume (decrescente)
        sorted_levels = sorted(volume_by_level.items(), key=lambda x: x[1], reverse=True)

        total_volume = sum(volume_by_level.values())
        target_volume = total_volume * target_pct

        accumulated_volume = 0
        value_area_levels = []

        for level, volume in sorted_levels:
            value_area_levels.append(level)
            accumulated_volume += volume

            if accumulated_volume >= target_volume:
                break

        if not value_area_levels:
            return {'high': 0, 'low': 0}

        return {
            'high': max(value_area_levels),
            'low': min(value_area_levels)
        }

    def _empty_profile(self) -> Dict:
        """Retorna profile vazio quando não há dados."""
        return {
            'poc': 0,
            'poc_volume': 0,
            'vah': 0,
            'val': 0,
            'value_area_volume_pct': 70.0,
            'volume_profile': [],
            'total_volume': 0,
            'price_range': {'min': 0, 'max': 0}
        }


class TapeReader:
    """
    Analisa fluxo de trades em tempo real (tape reading).
    """

    def __init__(self, window_size: int = 100):
        """
        Inicializa o tape reader.

        Args:
            window_size: Número de trades para analisar
        """
        self.window_size = window_size

    def analyze_tape(self, trades_stream: List[Dict]) -> Dict:
        """
        Analisa fluxo de trades em tempo real.

        Args:
            trades_stream: Lista de trades recentes

        Returns:
            Análise de tape reading
        """
        if not trades_stream:
            return self._empty_tape()

        try:
            recent_trades = trades_stream[-self.window_size:]

            # Separar compras e vendas
            buy_trades = [t for t in recent_trades if t.get('side') == 'buy']
            sell_trades = [t for t in recent_trades if t.get('side') == 'sell']

            buy_volume = sum([t.get('size', t.get('volume', 0)) for t in buy_trades])
            sell_volume = sum([t.get('size', t.get('volume', 0)) for t in sell_trades])
            total_volume = buy_volume + sell_volume

            # Pressão de compra/venda
            buy_pressure = (buy_volume / total_volume) if total_volume > 0 else 0.5

            # Detectar absorção
            absorption = self._detect_absorption(recent_trades)

            # Calcular momentum (velocidade de execução)
            momentum = self._calculate_momentum(recent_trades)

            # Interpretação
            interpretation = self._interpret_signals(buy_pressure, absorption, momentum)

            return {
                'buy_pressure': round(buy_pressure * 100, 2),
                'sell_pressure': round((1 - buy_pressure) * 100, 2),
                'buy_volume': round(buy_volume, 2),
                'sell_volume': round(sell_volume, 2),
                'total_volume': round(total_volume, 2),
                'absorption': absorption,
                'momentum': momentum,
                'interpretation': interpretation,
                'num_trades': len(recent_trades)
            }

        except Exception as e:
            logger.error(f"Erro ao analisar tape: {str(e)}")
            return self._empty_tape()

    def _detect_absorption(self, trades: List[Dict]) -> Dict:
        """
        Detecta quando grandes ordens são absorvidas sem mover muito o preço.

        Args:
            trades: Lista de trades

        Returns:
            Análise de absorção
        """
        if len(trades) < 10:
            return {'detected': False, 'type': 'none', 'strength': 0}

        try:
            # Calcular média de volume
            volumes = [t.get('size', t.get('volume', 0)) for t in trades]
            avg_volume = np.mean(volumes)

            # Calcular volatilidade de preço
            prices = [t['price'] for t in trades]
            price_std = np.std(prices)
            price_range = max(prices) - min(prices)

            # Absorção ocorre quando: alto volume + baixa volatilidade
            high_volume = max(volumes) > avg_volume * 2
            low_volatility = price_std < (price_range * 0.1) if price_range > 0 else True

            detected = high_volume and low_volatility

            if detected:
                # Determinar tipo (bullish ou bearish)
                last_price = trades[-1]['price']
                first_price = trades[0]['price']
                price_direction = 'up' if last_price > first_price else 'down'

                absorption_type = f"bullish_{price_direction}" if price_direction == 'up' else f"bearish_{price_direction}"
                strength = min(100, int((max(volumes) / avg_volume) * 30))

                return {
                    'detected': True,
                    'type': absorption_type,
                    'strength': strength,
                    'price_direction': price_direction
                }

            return {'detected': False, 'type': 'none', 'strength': 0}

        except Exception as e:
            logger.error(f"Erro ao detectar absorção: {str(e)}")
            return {'detected': False, 'type': 'none', 'strength': 0}

    def _calculate_momentum(self, trades: List[Dict]) -> Dict:
        """
        Calcula momentum (velocidade de execução de trades).

        Args:
            trades: Lista de trades com timestamps

        Returns:
            Análise de momentum
        """
        if len(trades) < 5:
            return {'speed': 'slow', 'trades_per_minute': 0, 'acceleration': 0}

        try:
            # Calcular trades por minuto
            if 'timestamp' in trades[0]:
                first_time = trades[0]['timestamp']
                last_time = trades[-1]['timestamp']

                # Handle both datetime and string timestamps
                if isinstance(first_time, str):
                    first_time = datetime.fromisoformat(first_time.replace('Z', '+00:00'))
                if isinstance(last_time, str):
                    last_time = datetime.fromisoformat(last_time.replace('Z', '+00:00'))

                time_diff = (last_time - first_time).total_seconds() / 60  # minutos

                if time_diff > 0:
                    trades_per_minute = len(trades) / time_diff
                else:
                    trades_per_minute = len(trades)
            else:
                # Assumir 1 minuto se não houver timestamps
                trades_per_minute = len(trades)

            # Classificar velocidade
            if trades_per_minute > 50:
                speed = 'very_fast'
            elif trades_per_minute > 20:
                speed = 'fast'
            elif trades_per_minute > 5:
                speed = 'normal'
            else:
                speed = 'slow'

            # Calcular aceleração (comparar primeira e segunda metade)
            mid_point = len(trades) // 2
            first_half = trades[:mid_point]
            second_half = trades[mid_point:]

            volume_first = sum([t.get('size', t.get('volume', 0)) for t in first_half])
            volume_second = sum([t.get('size', t.get('volume', 0)) for t in second_half])

            acceleration = ((volume_second - volume_first) / volume_first * 100) if volume_first > 0 else 0

            return {
                'speed': speed,
                'trades_per_minute': round(trades_per_minute, 2),
                'acceleration': round(acceleration, 2)
            }

        except Exception as e:
            logger.error(f"Erro ao calcular momentum: {str(e)}")
            return {'speed': 'unknown', 'trades_per_minute': 0, 'acceleration': 0}

    def _interpret_signals(self, buy_pressure: float, absorption: Dict, momentum: Dict) -> str:
        """
        Interpreta os sinais de tape reading.

        Args:
            buy_pressure: Pressão de compra (0-1)
            absorption: Dados de absorção
            momentum: Dados de momentum

        Returns:
            Interpretação textual
        """
        signals = []

        # Pressão de compra/venda
        if buy_pressure > 0.6:
            signals.append("forte pressão compradora")
        elif buy_pressure < 0.4:
            signals.append("forte pressão vendedora")
        else:
            signals.append("pressão balanceada")

        # Absorção
        if absorption['detected']:
            signals.append(f"absorção {absorption['type']} detectada")

        # Momentum
        if momentum['speed'] in ['fast', 'very_fast']:
            signals.append(f"execução {momentum['speed']}")

        if momentum['acceleration'] > 20:
            signals.append("volume acelerando")
        elif momentum['acceleration'] < -20:
            signals.append("volume desacelerando")

        return "; ".join(signals) if signals else "sem sinais claros"

    def _empty_tape(self) -> Dict:
        """Retorna análise vazia quando não há dados."""
        return {
            'buy_pressure': 50.0,
            'sell_pressure': 50.0,
            'buy_volume': 0,
            'sell_volume': 0,
            'total_volume': 0,
            'absorption': {'detected': False, 'type': 'none', 'strength': 0},
            'momentum': {'speed': 'slow', 'trades_per_minute': 0, 'acceleration': 0},
            'interpretation': 'sem dados suficientes',
            'num_trades': 0
        }


class OrderFlowIntegrator:
    """
    Integra análise de order flow com sinais técnicos.
    """

    def confirm_signal_with_order_flow(self, technical_signal: Dict, order_flow_data: Dict) -> Dict:
        """
        Combina análise técnica com order flow para confirmar sinais.

        Args:
            technical_signal: Sinal técnico {'type': 'BUY'/'SELL', 'confidence': 0-100}
            order_flow_data: Dados de order flow (order book, aggressive orders, volume profile, tape)

        Returns:
            Sinal técnico com confidence ajustada
        """
        try:
            confirmation_score = 0
            reasons = []

            signal_type = technical_signal.get('type', 'NEUTRAL')
            base_confidence = technical_signal.get('confidence', 50)

            # Extrair dados de order flow
            order_book = order_flow_data.get('order_book', {})
            aggressive = order_flow_data.get('aggressive_orders', {})
            volume_profile = order_flow_data.get('volume_profile', {})
            tape = order_flow_data.get('tape', {})

            if signal_type == 'BUY':
                # Confirmar sinal de COMPRA

                # 1. Order book mostra pressão compradora
                if order_book.get('bid_pressure', 50) > 55:
                    confirmation_score += 30
                    reasons.append("order book bullish")

                # 2. Ordens agressivas de compra
                if aggressive.get('aggressive_sentiment') == 'bullish':
                    confirmation_score += 25
                    reasons.append("aggressive buying detected")

                # 3. Preço acima POC (zona de valor)
                current_price = technical_signal.get('price', 0)
                poc = volume_profile.get('poc', 0)
                if current_price > poc and poc > 0:
                    confirmation_score += 20
                    reasons.append("price above POC")

                # 4. Tape reading bullish
                if tape.get('buy_pressure', 50) > 60:
                    confirmation_score += 15
                    reasons.append("tape shows strong buying")

                # 5. Absorção bullish
                absorption = tape.get('absorption', {})
                if absorption.get('detected') and 'bullish' in absorption.get('type', ''):
                    confirmation_score += 10
                    reasons.append("bullish absorption")

            elif signal_type == 'SELL':
                # Confirmar sinal de VENDA

                # 1. Order book mostra pressão vendedora
                if order_book.get('ask_pressure', 50) > 55:
                    confirmation_score += 30
                    reasons.append("order book bearish")

                # 2. Ordens agressivas de venda
                if aggressive.get('aggressive_sentiment') == 'bearish':
                    confirmation_score += 25
                    reasons.append("aggressive selling detected")

                # 3. Preço abaixo POC
                current_price = technical_signal.get('price', 0)
                poc = volume_profile.get('poc', 0)
                if current_price < poc and poc > 0:
                    confirmation_score += 20
                    reasons.append("price below POC")

                # 4. Tape reading bearish
                if tape.get('sell_pressure', 50) > 60:
                    confirmation_score += 15
                    reasons.append("tape shows strong selling")

                # 5. Absorção bearish
                absorption = tape.get('absorption', {})
                if absorption.get('detected') and 'bearish' in absorption.get('type', ''):
                    confirmation_score += 10
                    reasons.append("bearish absorption")

            # Ajustar confidence baseado na confirmação
            confidence_multiplier = 1 + (confirmation_score / 100)
            new_confidence = min(100, base_confidence * confidence_multiplier)

            return {
                **technical_signal,
                'confidence': round(new_confidence, 2),
                'original_confidence': base_confidence,
                'order_flow_confirmation_score': confirmation_score,
                'order_flow_reasons': reasons,
                'enhanced_by_order_flow': confirmation_score > 0
            }

        except Exception as e:
            logger.error(f"Erro ao confirmar sinal com order flow: {str(e)}")
            return technical_signal


# Classe principal que agrupa todos os analisadores
class OrderFlowAnalyzer:
    """
    Classe principal para análise completa de order flow.
    Combina order book, ordens agressivas, volume profile e tape reading.
    """

    def __init__(self):
        self.order_book_analyzer = OrderBookAnalyzer()
        self.aggressive_detector = AggressiveOrderDetector()
        self.volume_profile_analyzer = VolumeProfileAnalyzer()
        self.tape_reader = TapeReader()
        self.integrator = OrderFlowIntegrator()

    def analyze_complete(self,
                         order_book: Optional[Dict] = None,
                         trade_stream: Optional[List[Dict]] = None) -> Dict:
        """
        Análise completa de order flow.

        Args:
            order_book: Livro de ordens {'bids': [...], 'asks': [...]}
            trade_stream: Stream de trades recentes

        Returns:
            Análise completa de order flow
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'order_book': {},
            'aggressive_orders': {},
            'volume_profile': {},
            'tape': {}
        }

        try:
            # Analisar order book
            if order_book:
                result['order_book'] = self.order_book_analyzer.analyze_depth(order_book)

            # Analisar ordens agressivas e tape
            if trade_stream:
                result['aggressive_orders'] = self.aggressive_detector.detect_aggressive_orders(trade_stream)
                result['volume_profile'] = self.volume_profile_analyzer.calculate_volume_profile(trade_stream)
                result['tape'] = self.tape_reader.analyze_tape(trade_stream)

            return result

        except Exception as e:
            logger.error(f"Erro na análise completa de order flow: {str(e)}")
            return result

    def enhance_signal(self, technical_signal: Dict,
                       order_book: Optional[Dict] = None,
                       trade_stream: Optional[List[Dict]] = None) -> Dict:
        """
        Melhora um sinal técnico com análise de order flow.

        Args:
            technical_signal: Sinal técnico a ser melhorado
            order_book: Livro de ordens
            trade_stream: Stream de trades

        Returns:
            Sinal técnico melhorado
        """
        # Primeiro, fazer análise completa de order flow
        order_flow_data = self.analyze_complete(order_book, trade_stream)

        # Depois, usar o integrador para melhorar o sinal
        enhanced_signal = self.integrator.confirm_signal_with_order_flow(
            technical_signal,
            order_flow_data
        )

        return enhanced_signal
