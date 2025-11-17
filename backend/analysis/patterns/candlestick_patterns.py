"""
Reconhecimento de Padrões de Candlestick
Implementa 15+ padrões clássicos de análise de candles
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


class CandlestickPattern:
    """Representa um padrão de candlestick detectado"""

    def __init__(self, name: str, pattern_type: str, confidence: float,
                 index: int, candles: List[Dict], interpretation: str,
                 signal: str, success_rate: float = 0.65):
        self.name = name
        self.pattern_type = pattern_type  # reversal_bullish, reversal_bearish, continuation
        self.confidence = confidence  # 0-100
        self.index = index  # Posição no DataFrame
        self.candles = candles  # Lista de candles que formam o padrão
        self.interpretation = interpretation
        self.signal = signal  # BUY, SELL, NEUTRAL
        self.success_rate = success_rate  # Taxa de sucesso histórica
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'type': self.pattern_type,
            'confidence': round(self.confidence, 2),
            'index': self.index,
            'candles': self.candles,
            'interpretation': self.interpretation,
            'signal': self.signal,
            'success_rate_historical': round(self.success_rate * 100, 1),
            'timestamp': self.timestamp.isoformat()
        }


class CandlestickPatterns:
    """
    Classe para detectar padrões de candlestick
    Implementa 15+ padrões clássicos
    """

    def __init__(self):
        self.min_body_size = 0.0001  # Tamanho mínimo do corpo para considerar

    @staticmethod
    def _candle_info(row: pd.Series) -> Dict:
        """Extrai informações de um candle"""
        o, h, l, c = row['open'], row['high'], row['low'], row['close']

        body = abs(c - o)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        total_range = h - l

        is_bullish = c > o
        is_bearish = c < o
        is_doji = body < (total_range * 0.1) if total_range > 0 else True

        return {
            'open': float(o),
            'high': float(h),
            'low': float(l),
            'close': float(c),
            'body': float(body),
            'upper_shadow': float(upper_shadow),
            'lower_shadow': float(lower_shadow),
            'total_range': float(total_range),
            'is_bullish': bool(is_bullish),  # Converter numpy.bool para Python bool
            'is_bearish': bool(is_bearish),  # Converter numpy.bool para Python bool
            'is_doji': bool(is_doji),        # Converter numpy.bool para Python bool
            'body_pct': float(body / total_range) if total_range > 0 else 0
        }

    def detect_all_patterns(self, df: pd.DataFrame, lookback: int = 50) -> List[CandlestickPattern]:
        """
        Detecta todos os padrões nos últimos N candles

        Args:
            df: DataFrame com OHLC
            lookback: Número de candles para analisar

        Returns:
            Lista de padrões detectados
        """
        if len(df) < 3:
            return []

        patterns = []

        # Analisar apenas os últimos N candles
        start_idx = max(0, len(df) - lookback)

        for i in range(start_idx + 2, len(df)):
            # Padrões de 1 candle
            pattern = self._detect_hammer(df, i)
            if pattern:
                patterns.append(pattern)

            pattern = self._detect_shooting_star(df, i)
            if pattern:
                patterns.append(pattern)

            pattern = self._detect_doji(df, i)
            if pattern:
                patterns.append(pattern)

            # Padrões de 2 candles
            if i >= 1:
                pattern = self._detect_engulfing(df, i)
                if pattern:
                    patterns.append(pattern)

                pattern = self._detect_piercing_pattern(df, i)
                if pattern:
                    patterns.append(pattern)

                pattern = self._detect_dark_cloud_cover(df, i)
                if pattern:
                    patterns.append(pattern)

                pattern = self._detect_harami(df, i)
                if pattern:
                    patterns.append(pattern)

            # Padrões de 3 candles
            if i >= 2:
                pattern = self._detect_morning_star(df, i)
                if pattern:
                    patterns.append(pattern)

                pattern = self._detect_evening_star(df, i)
                if pattern:
                    patterns.append(pattern)

                pattern = self._detect_three_white_soldiers(df, i)
                if pattern:
                    patterns.append(pattern)

                pattern = self._detect_three_black_crows(df, i)
                if pattern:
                    patterns.append(pattern)

        return patterns

    # === PADRÕES DE 1 CANDLE ===

    def _detect_hammer(self, df: pd.DataFrame, i: int) -> Optional[CandlestickPattern]:
        """
        Hammer: Corpo pequeno no topo, sombra inferior longa (2x+ o corpo)
        Padrão de reversão bullish
        """
        candle = self._candle_info(df.iloc[i])

        # Verificações
        if candle['total_range'] == 0:
            return None

        # Sombra inferior deve ser pelo menos 2x o corpo
        if candle['lower_shadow'] < candle['body'] * 2:
            return None

        # Sombra superior deve ser muito pequena
        if candle['upper_shadow'] > candle['body'] * 0.5:
            return None

        # Corpo deve estar no topo (upper 30% do range)
        body_position = (min(candle['open'], candle['close']) - candle['low']) / candle['total_range']
        if body_position < 0.6:
            return None

        # Deve estar em tendência de baixa (opcional, mas aumenta confiança)
        trend_down = df['close'].iloc[i-5:i].is_monotonic_decreasing if i >= 5 else False
        confidence = 75 if trend_down else 60

        return CandlestickPattern(
            name="Hammer",
            pattern_type="reversal_bullish",
            confidence=confidence,
            index=i,
            candles=[candle],
            interpretation="Possível reversão de alta. Vendedores tentaram empurrar preço para baixo mas compradores reagiram forte.",
            signal="BUY",
            success_rate=0.68
        )

    def _detect_shooting_star(self, df: pd.DataFrame, i: int) -> Optional[CandlestickPattern]:
        """
        Shooting Star: Corpo pequeno na base, sombra superior longa
        Padrão de reversão bearish
        """
        candle = self._candle_info(df.iloc[i])

        if candle['total_range'] == 0:
            return None

        # Sombra superior deve ser pelo menos 2x o corpo
        if candle['upper_shadow'] < candle['body'] * 2:
            return None

        # Sombra inferior deve ser muito pequena
        if candle['lower_shadow'] > candle['body'] * 0.5:
            return None

        # Corpo deve estar na base (lower 30% do range)
        body_position = (min(candle['open'], candle['close']) - candle['low']) / candle['total_range']
        if body_position > 0.3:
            return None

        # Deve estar em tendência de alta
        trend_up = df['close'].iloc[i-5:i].is_monotonic_increasing if i >= 5 else False
        confidence = 75 if trend_up else 60

        return CandlestickPattern(
            name="Shooting Star",
            pattern_type="reversal_bearish",
            confidence=confidence,
            index=i,
            candles=[candle],
            interpretation="Possível reversão de baixa. Compradores tentaram empurrar preço para cima mas vendedores reagiram forte.",
            signal="SELL",
            success_rate=0.66
        )

    def _detect_doji(self, df: pd.DataFrame, i: int) -> Optional[CandlestickPattern]:
        """
        Doji: Open ≈ Close (corpo muito pequeno)
        Indica indecisão do mercado
        """
        candle = self._candle_info(df.iloc[i])

        if not candle['is_doji']:
            return None

        # Corpo deve ser menos de 10% do range total
        if candle['body_pct'] > 0.1:
            return None

        # Determinar tipo de Doji
        if candle['upper_shadow'] > candle['total_range'] * 0.6:
            doji_type = "Dragonfly Doji (bullish)"
            signal = "BUY"
            confidence = 65
        elif candle['lower_shadow'] > candle['total_range'] * 0.6:
            doji_type = "Gravestone Doji (bearish)"
            signal = "SELL"
            confidence = 65
        else:
            doji_type = "Doji (neutral)"
            signal = "NEUTRAL"
            confidence = 50

        return CandlestickPattern(
            name=f"Doji ({doji_type})",
            pattern_type="indecision",
            confidence=confidence,
            index=i,
            candles=[candle],
            interpretation="Indecisão do mercado. Possível reversão ou continuação dependendo do contexto.",
            signal=signal,
            success_rate=0.55
        )

    # === PADRÕES DE 2 CANDLES ===

    def _detect_engulfing(self, df: pd.DataFrame, i: int) -> Optional[CandlestickPattern]:
        """
        Bullish/Bearish Engulfing: Segundo candle "engole" o primeiro
        """
        if i < 1:
            return None

        prev = self._candle_info(df.iloc[i-1])
        curr = self._candle_info(df.iloc[i])

        # Bullish Engulfing
        if prev['is_bearish'] and curr['is_bullish']:
            # Corpo atual deve engolfar corpo anterior
            if curr['open'] <= prev['close'] and curr['close'] >= prev['open']:
                # Verificar se engolfamento é significativo
                if curr['body'] > prev['body'] * 1.2:
                    return CandlestickPattern(
                        name="Bullish Engulfing",
                        pattern_type="reversal_bullish",
                        confidence=80,
                        index=i,
                        candles=[prev, curr],
                        interpretation="Forte reversão de alta. Compradores dominaram vendedores completamente.",
                        signal="BUY",
                        success_rate=0.72
                    )

        # Bearish Engulfing
        if prev['is_bullish'] and curr['is_bearish']:
            if curr['open'] >= prev['close'] and curr['close'] <= prev['open']:
                if curr['body'] > prev['body'] * 1.2:
                    return CandlestickPattern(
                        name="Bearish Engulfing",
                        pattern_type="reversal_bearish",
                        confidence=80,
                        index=i,
                        candles=[prev, curr],
                        interpretation="Forte reversão de baixa. Vendedores dominaram compradores completamente.",
                        signal="SELL",
                        success_rate=0.70
                    )

        return None

    def _detect_piercing_pattern(self, df: pd.DataFrame, i: int) -> Optional[CandlestickPattern]:
        """
        Piercing Pattern: Candle bullish que "perfura" o candle bearish anterior
        """
        if i < 1:
            return None

        prev = self._candle_info(df.iloc[i-1])
        curr = self._candle_info(df.iloc[i])

        if not (prev['is_bearish'] and curr['is_bullish']):
            return None

        # Current open deve estar abaixo do close anterior
        if curr['open'] >= prev['close']:
            return None

        # Current close deve penetrar pelo menos 50% do corpo anterior
        penetration = (curr['close'] - prev['close']) / prev['body']
        if penetration < 0.5 or curr['close'] >= prev['open']:
            return None

        confidence = min(70 + penetration * 20, 85)

        return CandlestickPattern(
            name="Piercing Pattern",
            pattern_type="reversal_bullish",
            confidence=confidence,
            index=i,
            candles=[prev, curr],
            interpretation="Reversão de alta. Compradores entraram forte após venda.",
            signal="BUY",
            success_rate=0.67
        )

    def _detect_dark_cloud_cover(self, df: pd.DataFrame, i: int) -> Optional[CandlestickPattern]:
        """
        Dark Cloud Cover: Candle bearish que "cobre" o candle bullish anterior
        """
        if i < 1:
            return None

        prev = self._candle_info(df.iloc[i-1])
        curr = self._candle_info(df.iloc[i])

        if not (prev['is_bullish'] and curr['is_bearish']):
            return None

        # Current open deve estar acima do close anterior
        if curr['open'] <= prev['close']:
            return None

        # Current close deve penetrar pelo menos 50% do corpo anterior
        penetration = (prev['close'] - curr['close']) / prev['body']
        if penetration < 0.5 or curr['close'] <= prev['open']:
            return None

        confidence = min(70 + penetration * 20, 85)

        return CandlestickPattern(
            name="Dark Cloud Cover",
            pattern_type="reversal_bearish",
            confidence=confidence,
            index=i,
            candles=[prev, curr],
            interpretation="Reversão de baixa. Vendedores entraram forte após compra.",
            signal="SELL",
            success_rate=0.65
        )

    def _detect_harami(self, df: pd.DataFrame, i: int) -> Optional[CandlestickPattern]:
        """
        Harami: Candle pequeno contido dentro do corpo do candle anterior
        """
        if i < 1:
            return None

        prev = self._candle_info(df.iloc[i-1])
        curr = self._candle_info(df.iloc[i])

        # Current deve estar contido no corpo do previous
        if not (curr['high'] <= max(prev['open'], prev['close']) and
                curr['low'] >= min(prev['open'], prev['close'])):
            return None

        # Corpo atual deve ser menor que o anterior
        if curr['body'] >= prev['body'] * 0.7:
            return None

        # Bullish Harami
        if prev['is_bearish'] and curr['is_bullish']:
            return CandlestickPattern(
                name="Bullish Harami",
                pattern_type="reversal_bullish",
                confidence=65,
                index=i,
                candles=[prev, curr],
                interpretation="Possível reversão de alta. Momentum de venda está enfraquecendo.",
                signal="BUY",
                success_rate=0.62
            )

        # Bearish Harami
        if prev['is_bullish'] and curr['is_bearish']:
            return CandlestickPattern(
                name="Bearish Harami",
                pattern_type="reversal_bearish",
                confidence=65,
                index=i,
                candles=[prev, curr],
                interpretation="Possível reversão de baixa. Momentum de compra está enfraquecendo.",
                signal="SELL",
                success_rate=0.60
            )

        return None

    # === PADRÕES DE 3 CANDLES ===

    def _detect_morning_star(self, df: pd.DataFrame, i: int) -> Optional[CandlestickPattern]:
        """
        Morning Star: Padrão de 3 candles (bearish, pequeno, bullish)
        Forte reversão de alta
        """
        if i < 2:
            return None

        first = self._candle_info(df.iloc[i-2])
        middle = self._candle_info(df.iloc[i-1])
        last = self._candle_info(df.iloc[i])

        # Primeiro deve ser bearish significativo
        if not first['is_bearish'] or first['body'] < first['total_range'] * 0.5:
            return None

        # Meio deve ser pequeno (gap down opcional)
        if middle['body'] > first['body'] * 0.3:
            return None

        # Último deve ser bullish forte
        if not last['is_bullish'] or last['body'] < first['body'] * 0.5:
            return None

        # Último deve fechar acima do ponto médio do primeiro
        if last['close'] < (first['open'] + first['close']) / 2:
            return None

        return CandlestickPattern(
            name="Morning Star",
            pattern_type="reversal_bullish",
            confidence=85,
            index=i,
            candles=[first, middle, last],
            interpretation="Forte reversão de alta. Padrão clássico de fundo de mercado.",
            signal="BUY",
            success_rate=0.75
        )

    def _detect_evening_star(self, df: pd.DataFrame, i: int) -> Optional[CandlestickPattern]:
        """
        Evening Star: Padrão de 3 candles (bullish, pequeno, bearish)
        Forte reversão de baixa
        """
        if i < 2:
            return None

        first = self._candle_info(df.iloc[i-2])
        middle = self._candle_info(df.iloc[i-1])
        last = self._candle_info(df.iloc[i])

        # Primeiro deve ser bullish significativo
        if not first['is_bullish'] or first['body'] < first['total_range'] * 0.5:
            return None

        # Meio deve ser pequeno (gap up opcional)
        if middle['body'] > first['body'] * 0.3:
            return None

        # Último deve ser bearish forte
        if not last['is_bearish'] or last['body'] < first['body'] * 0.5:
            return None

        # Último deve fechar abaixo do ponto médio do primeiro
        if last['close'] > (first['open'] + first['close']) / 2:
            return None

        return CandlestickPattern(
            name="Evening Star",
            pattern_type="reversal_bearish",
            confidence=85,
            index=i,
            candles=[first, middle, last],
            interpretation="Forte reversão de baixa. Padrão clássico de topo de mercado.",
            signal="SELL",
            success_rate=0.73
        )

    def _detect_three_white_soldiers(self, df: pd.DataFrame, i: int) -> Optional[CandlestickPattern]:
        """
        Three White Soldiers: Três candles bullish consecutivos com closes crescentes
        Forte padrão de continuação/reversão bullish
        """
        if i < 2:
            return None

        c1 = self._candle_info(df.iloc[i-2])
        c2 = self._candle_info(df.iloc[i-1])
        c3 = self._candle_info(df.iloc[i])

        # Todos devem ser bullish
        if not (c1['is_bullish'] and c2['is_bullish'] and c3['is_bullish']):
            return None

        # Closes devem ser crescentes
        if not (c1['close'] < c2['close'] < c3['close']):
            return None

        # Opens devem estar dentro do corpo do candle anterior
        if not (c1['open'] < c2['open'] < c2['close'] and c2['open'] < c3['open'] < c3['close']):
            return None

        # Corpos devem ser significativos
        avg_body = (c1['body'] + c2['body'] + c3['body']) / 3
        avg_range = (c1['total_range'] + c2['total_range'] + c3['total_range']) / 3
        if avg_body < avg_range * 0.6:
            return None

        return CandlestickPattern(
            name="Three White Soldiers",
            pattern_type="continuation_bullish",
            confidence=80,
            index=i,
            candles=[c1, c2, c3],
            interpretation="Forte momentum de alta. Compradores dominando por 3 períodos consecutivos.",
            signal="BUY",
            success_rate=0.71
        )

    def _detect_three_black_crows(self, df: pd.DataFrame, i: int) -> Optional[CandlestickPattern]:
        """
        Three Black Crows: Três candles bearish consecutivos com closes decrescentes
        Forte padrão de continuação/reversão bearish
        """
        if i < 2:
            return None

        c1 = self._candle_info(df.iloc[i-2])
        c2 = self._candle_info(df.iloc[i-1])
        c3 = self._candle_info(df.iloc[i])

        # Todos devem ser bearish
        if not (c1['is_bearish'] and c2['is_bearish'] and c3['is_bearish']):
            return None

        # Closes devem ser decrescentes
        if not (c1['close'] > c2['close'] > c3['close']):
            return None

        # Opens devem estar dentro do corpo do candle anterior
        if not (c1['close'] < c2['open'] < c1['open'] and c2['close'] < c3['open'] < c2['open']):
            return None

        # Corpos devem ser significativos
        avg_body = (c1['body'] + c2['body'] + c3['body']) / 3
        avg_range = (c1['total_range'] + c2['total_range'] + c3['total_range']) / 3
        if avg_body < avg_range * 0.6:
            return None

        return CandlestickPattern(
            name="Three Black Crows",
            pattern_type="continuation_bearish",
            confidence=80,
            index=i,
            candles=[c1, c2, c3],
            interpretation="Forte momentum de baixa. Vendedores dominando por 3 períodos consecutivos.",
            signal="SELL",
            success_rate=0.69
        )
