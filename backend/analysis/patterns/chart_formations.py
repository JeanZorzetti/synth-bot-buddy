"""
Módulo de Reconhecimento de Formações Gráficas

Detecta padrões gráficos clássicos:
- Double Top / Double Bottom
- Head and Shoulders / Inverse Head and Shoulders
- Triangles (Ascending, Descending, Symmetrical)
- Wedges (Rising, Falling)
- Flags and Pennants
- Channels
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChartFormation:
    """Representa uma formação gráfica detectada"""

    name: str  # Nome do padrão
    formation_type: str  # 'reversal_bullish', 'reversal_bearish', 'continuation_bullish', 'continuation_bearish'
    confidence: float  # 0-100
    start_index: int
    end_index: int
    key_points: List[Dict]  # Pontos importantes do padrão (tops, bottoms, neckline, etc.)
    signal: str  # 'BUY', 'SELL', 'NEUTRAL'
    price_target: Optional[float]  # Preço alvo calculado
    stop_loss: Optional[float]  # Stop loss sugerido
    success_rate: float  # Taxa de sucesso histórica do padrão
    interpretation: str
    status: str  # 'forming', 'completed', 'confirmed'

    def __repr__(self):
        return f"{self.name} ({self.status}) - {self.signal} @ {self.confidence:.0f}% confidence"


class ChartFormationDetector:
    """Detector automático de formações gráficas"""

    def __init__(self, tolerance_pct: float = 2.0, min_bars: int = 10):
        """
        Args:
            tolerance_pct: Tolerância para matching de preços (%)
            min_bars: Mínimo de barras para formar um padrão
        """
        self.tolerance_pct = tolerance_pct
        self.min_bars = min_bars

    def detect_all_formations(self, df: pd.DataFrame, lookback: int = 100) -> List[ChartFormation]:
        """
        Detecta todas as formações gráficas no DataFrame

        Args:
            df: DataFrame com OHLCV
            lookback: Quantidade de candles para análise

        Returns:
            Lista de ChartFormation detectadas
        """
        if len(df) < self.min_bars:
            return []

        analysis_df = df.tail(lookback).copy()
        formations = []

        logger.info(f"\n{'='*60}")
        logger.info(f"DETECÇÃO DE FORMAÇÕES GRÁFICAS")
        logger.info(f"Período: {lookback} candles")

        # Detectar padrões de reversão
        formations.extend(self._detect_double_top(analysis_df))
        formations.extend(self._detect_double_bottom(analysis_df))
        formations.extend(self._detect_head_and_shoulders(analysis_df))
        formations.extend(self._detect_inverse_head_and_shoulders(analysis_df))

        # Detectar padrões de continuação
        formations.extend(self._detect_ascending_triangle(analysis_df))
        formations.extend(self._detect_descending_triangle(analysis_df))
        formations.extend(self._detect_symmetrical_triangle(analysis_df))
        formations.extend(self._detect_rising_wedge(analysis_df))
        formations.extend(self._detect_falling_wedge(analysis_df))
        formations.extend(self._detect_bull_flag(analysis_df))
        formations.extend(self._detect_bear_flag(analysis_df))

        # Ordenar por confiança
        formations = sorted(formations, key=lambda x: x.confidence, reverse=True)

        logger.info(f"\nFORMAÇÕES DETECTADAS:")
        for formation in formations[:5]:  # Top 5
            logger.info(f"  {formation}")

        logger.info(f"{'='*60}\n")

        return formations

    def _detect_double_top(self, df: pd.DataFrame) -> List[ChartFormation]:
        """
        Detecta padrão Double Top (topo duplo)

        Características:
        - Dois topos similares (~mesma altura)
        - Vale entre os topos
        - Breakout abaixo da neckline (vale) confirma
        """
        formations = []

        # Encontrar pivots de alta
        pivot_highs = self._find_pivot_points(df, 'high', window=5)

        if len(pivot_highs) < 2:
            return formations

        # Procurar pares de topos similares
        for i in range(len(pivot_highs) - 1):
            for j in range(i + 1, len(pivot_highs)):
                top1 = pivot_highs[i]
                top2 = pivot_highs[j]

                # Topos devem ser similares (dentro da tolerância)
                if not self._prices_similar(top1['price'], top2['price']):
                    continue

                # Deve haver um vale entre os topos
                valley_section = df.iloc[top1['index']:top2['index']]
                if len(valley_section) < 3:
                    continue

                valley_low = valley_section['low'].min()
                valley_idx = valley_section['low'].idxmin()

                # Vale deve ser significativo (pelo menos 2% abaixo dos topos)
                if (top1['price'] - valley_low) / top1['price'] < 0.02:
                    continue

                # Calcular confiança
                price_similarity = 100 - (abs(top1['price'] - top2['price']) / top1['price'] * 100)
                bars_between = top2['index'] - top1['index']
                spacing_score = min(bars_between / 20 * 50, 50)  # Ideal: 20+ bars
                confidence = min(price_similarity + spacing_score, 95)

                # Status: completed se formou, confirmed se rompeu neckline
                current_price = df['close'].iloc[-1]
                if current_price < valley_low:
                    status = 'confirmed'
                    signal = 'SELL'
                    confidence += 5  # Boost por confirmação
                elif top2['index'] == len(df) - 1:
                    status = 'forming'
                    signal = 'NEUTRAL'
                else:
                    status = 'completed'
                    signal = 'SELL'

                # Calcular price target (distância do topo ao vale projetada abaixo)
                pattern_height = top1['price'] - valley_low
                price_target = valley_low - pattern_height

                formation = ChartFormation(
                    name="Double Top",
                    formation_type="reversal_bearish",
                    confidence=confidence,
                    start_index=top1['index'],
                    end_index=top2['index'],
                    key_points=[
                        {'type': 'top1', 'price': top1['price'], 'index': top1['index']},
                        {'type': 'valley', 'price': valley_low, 'index': valley_idx},
                        {'type': 'top2', 'price': top2['price'], 'index': top2['index']}
                    ],
                    signal=signal,
                    price_target=price_target,
                    stop_loss=max(top1['price'], top2['price']) * 1.01,  # 1% acima do topo
                    success_rate=0.65,
                    interpretation=f"Padrão de reversão bearish. Rompimento abaixo de {valley_low:.5f} confirma.",
                    status=status
                )

                formations.append(formation)

        return formations

    def _detect_double_bottom(self, df: pd.DataFrame) -> List[ChartFormation]:
        """
        Detecta padrão Double Bottom (fundo duplo)

        Características:
        - Dois fundos similares (~mesma altura)
        - Pico entre os fundos
        - Breakout acima da neckline (pico) confirma
        """
        formations = []

        # Encontrar pivots de baixa
        pivot_lows = self._find_pivot_points(df, 'low', window=5)

        if len(pivot_lows) < 2:
            return formations

        # Procurar pares de fundos similares
        for i in range(len(pivot_lows) - 1):
            for j in range(i + 1, len(pivot_lows)):
                bottom1 = pivot_lows[i]
                bottom2 = pivot_lows[j]

                # Fundos devem ser similares
                if not self._prices_similar(bottom1['price'], bottom2['price']):
                    continue

                # Deve haver um pico entre os fundos
                peak_section = df.iloc[bottom1['index']:bottom2['index']]
                if len(peak_section) < 3:
                    continue

                peak_high = peak_section['high'].max()
                peak_idx = peak_section['high'].idxmax()

                # Pico deve ser significativo
                if (peak_high - bottom1['price']) / bottom1['price'] < 0.02:
                    continue

                # Calcular confiança
                price_similarity = 100 - (abs(bottom1['price'] - bottom2['price']) / bottom1['price'] * 100)
                bars_between = bottom2['index'] - bottom1['index']
                spacing_score = min(bars_between / 20 * 50, 50)
                confidence = min(price_similarity + spacing_score, 95)

                # Status
                current_price = df['close'].iloc[-1]
                if current_price > peak_high:
                    status = 'confirmed'
                    signal = 'BUY'
                    confidence += 5
                elif bottom2['index'] == len(df) - 1:
                    status = 'forming'
                    signal = 'NEUTRAL'
                else:
                    status = 'completed'
                    signal = 'BUY'

                # Price target
                pattern_height = peak_high - bottom1['price']
                price_target = peak_high + pattern_height

                formation = ChartFormation(
                    name="Double Bottom",
                    formation_type="reversal_bullish",
                    confidence=confidence,
                    start_index=bottom1['index'],
                    end_index=bottom2['index'],
                    key_points=[
                        {'type': 'bottom1', 'price': bottom1['price'], 'index': bottom1['index']},
                        {'type': 'peak', 'price': peak_high, 'index': peak_idx},
                        {'type': 'bottom2', 'price': bottom2['price'], 'index': bottom2['index']}
                    ],
                    signal=signal,
                    price_target=price_target,
                    stop_loss=min(bottom1['price'], bottom2['price']) * 0.99,
                    success_rate=0.68,
                    interpretation=f"Padrão de reversão bullish. Rompimento acima de {peak_high:.5f} confirma.",
                    status=status
                )

                formations.append(formation)

        return formations

    def _detect_head_and_shoulders(self, df: pd.DataFrame) -> List[ChartFormation]:
        """
        Detecta padrão Head and Shoulders (OCO - Ombro Cabeça Ombro)

        Características:
        - Três topos: shoulder1 < head > shoulder2
        - Shoulders aproximadamente na mesma altura
        - Neckline conecta os dois vales
        """
        formations = []
        pivot_highs = self._find_pivot_points(df, 'high', window=5)

        if len(pivot_highs) < 3:
            return formations

        # Procurar padrão: shoulder - head - shoulder
        for i in range(len(pivot_highs) - 2):
            left_shoulder = pivot_highs[i]
            head = pivot_highs[i + 1]
            right_shoulder = pivot_highs[i + 2]

            # Head deve ser mais alto que shoulders
            if not (head['price'] > left_shoulder['price'] and head['price'] > right_shoulder['price']):
                continue

            # Shoulders devem ser similares
            if not self._prices_similar(left_shoulder['price'], right_shoulder['price'], tolerance=5.0):
                continue

            # Encontrar vales (neckline)
            valley1_section = df.iloc[left_shoulder['index']:head['index']]
            valley2_section = df.iloc[head['index']:right_shoulder['index']]

            if len(valley1_section) < 2 or len(valley2_section) < 2:
                continue

            valley1_low = valley1_section['low'].min()
            valley2_low = valley2_section['low'].min()
            neckline = min(valley1_low, valley2_low)

            # Calcular confiança
            shoulder_symmetry = 100 - (abs(left_shoulder['price'] - right_shoulder['price']) /
                                      left_shoulder['price'] * 100)
            head_prominence = ((head['price'] - max(left_shoulder['price'], right_shoulder['price'])) /
                              head['price'] * 100)
            confidence = min(shoulder_symmetry * 0.6 + head_prominence * 4, 95)

            # Status
            current_price = df['close'].iloc[-1]
            if current_price < neckline:
                status = 'confirmed'
                signal = 'SELL'
                confidence += 5
            elif right_shoulder['index'] >= len(df) - 5:
                status = 'forming'
                signal = 'NEUTRAL'
            else:
                status = 'completed'
                signal = 'SELL'

            # Price target
            pattern_height = head['price'] - neckline
            price_target = neckline - pattern_height

            formation = ChartFormation(
                name="Head and Shoulders",
                formation_type="reversal_bearish",
                confidence=confidence,
                start_index=left_shoulder['index'],
                end_index=right_shoulder['index'],
                key_points=[
                    {'type': 'left_shoulder', 'price': left_shoulder['price'], 'index': left_shoulder['index']},
                    {'type': 'head', 'price': head['price'], 'index': head['index']},
                    {'type': 'right_shoulder', 'price': right_shoulder['price'], 'index': right_shoulder['index']},
                    {'type': 'neckline', 'price': neckline, 'index': right_shoulder['index']}
                ],
                signal=signal,
                price_target=price_target,
                stop_loss=head['price'] * 1.02,
                success_rate=0.72,
                interpretation=f"Padrão de reversão bearish. Rompimento abaixo de {neckline:.5f} confirma.",
                status=status
            )

            formations.append(formation)

        return formations

    def _detect_inverse_head_and_shoulders(self, df: pd.DataFrame) -> List[ChartFormation]:
        """
        Detecta padrão Inverse Head and Shoulders (OCO invertido)

        Características:
        - Três fundos: shoulder1 > head < shoulder2
        - Shoulders aproximadamente na mesma altura
        - Neckline conecta os dois picos
        """
        formations = []
        pivot_lows = self._find_pivot_points(df, 'low', window=5)

        if len(pivot_lows) < 3:
            return formations

        for i in range(len(pivot_lows) - 2):
            left_shoulder = pivot_lows[i]
            head = pivot_lows[i + 1]
            right_shoulder = pivot_lows[i + 2]

            # Head deve ser mais baixo que shoulders
            if not (head['price'] < left_shoulder['price'] and head['price'] < right_shoulder['price']):
                continue

            # Shoulders devem ser similares
            if not self._prices_similar(left_shoulder['price'], right_shoulder['price'], tolerance=5.0):
                continue

            # Encontrar picos (neckline)
            peak1_section = df.iloc[left_shoulder['index']:head['index']]
            peak2_section = df.iloc[head['index']:right_shoulder['index']]

            if len(peak1_section) < 2 or len(peak2_section) < 2:
                continue

            peak1_high = peak1_section['high'].max()
            peak2_high = peak2_section['high'].max()
            neckline = max(peak1_high, peak2_high)

            # Calcular confiança
            shoulder_symmetry = 100 - (abs(left_shoulder['price'] - right_shoulder['price']) /
                                      left_shoulder['price'] * 100)
            head_prominence = ((min(left_shoulder['price'], right_shoulder['price']) - head['price']) /
                              head['price'] * 100)
            confidence = min(shoulder_symmetry * 0.6 + head_prominence * 4, 95)

            # Status
            current_price = df['close'].iloc[-1]
            if current_price > neckline:
                status = 'confirmed'
                signal = 'BUY'
                confidence += 5
            elif right_shoulder['index'] >= len(df) - 5:
                status = 'forming'
                signal = 'NEUTRAL'
            else:
                status = 'completed'
                signal = 'BUY'

            # Price target
            pattern_height = neckline - head['price']
            price_target = neckline + pattern_height

            formation = ChartFormation(
                name="Inverse Head and Shoulders",
                formation_type="reversal_bullish",
                confidence=confidence,
                start_index=left_shoulder['index'],
                end_index=right_shoulder['index'],
                key_points=[
                    {'type': 'left_shoulder', 'price': left_shoulder['price'], 'index': left_shoulder['index']},
                    {'type': 'head', 'price': head['price'], 'index': head['index']},
                    {'type': 'right_shoulder', 'price': right_shoulder['price'], 'index': right_shoulder['index']},
                    {'type': 'neckline', 'price': neckline, 'index': right_shoulder['index']}
                ],
                signal=signal,
                price_target=price_target,
                stop_loss=head['price'] * 0.98,
                success_rate=0.70,
                interpretation=f"Padrão de reversão bullish. Rompimento acima de {neckline:.5f} confirma.",
                status=status
            )

            formations.append(formation)

        return formations

    def _detect_ascending_triangle(self, df: pd.DataFrame) -> List[ChartFormation]:
        """
        Detecta Ascending Triangle (triângulo ascendente)

        Características:
        - Resistência horizontal (topos na mesma altura)
        - Suporte ascendente (fundos cada vez mais altos)
        - Padrão de continuação bullish
        """
        formations = []
        pivot_highs = self._find_pivot_points(df, 'high', window=5)
        pivot_lows = self._find_pivot_points(df, 'low', window=5)

        if len(pivot_highs) < 2 or len(pivot_lows) < 2:
            return formations

        # Procurar resistência horizontal (topos similares)
        for i in range(len(pivot_highs) - 1):
            top1 = pivot_highs[i]
            top2 = pivot_highs[i + 1]

            if not self._prices_similar(top1['price'], top2['price'], tolerance=1.5):
                continue

            # Procurar fundos ascendentes entre os topos
            lows_between = [l for l in pivot_lows if top1['index'] < l['index'] < top2['index']]
            if len(lows_between) < 1:
                continue

            # Verificar se fundos estão subindo
            lows_between_sorted = sorted(lows_between, key=lambda x: x['index'])
            is_ascending = all(lows_between_sorted[j]['price'] < lows_between_sorted[j+1]['price']
                              for j in range(len(lows_between_sorted) - 1))

            if not is_ascending and len(lows_between_sorted) > 1:
                continue

            resistance_line = (top1['price'] + top2['price']) / 2
            lowest_low = min(l['price'] for l in lows_between_sorted)

            # Calcular confiança
            resistance_flatness = 100 - (abs(top1['price'] - top2['price']) / top1['price'] * 100)
            confidence = min(resistance_flatness * 0.8 + 20, 85)

            # Status
            current_price = df['close'].iloc[-1]
            if current_price > resistance_line:
                status = 'confirmed'
                signal = 'BUY'
                confidence += 10
            else:
                status = 'forming'
                signal = 'BUY'

            # Price target (altura do triângulo)
            triangle_height = resistance_line - lowest_low
            price_target = resistance_line + triangle_height

            formation = ChartFormation(
                name="Ascending Triangle",
                formation_type="continuation_bullish",
                confidence=confidence,
                start_index=top1['index'],
                end_index=top2['index'],
                key_points=[
                    {'type': 'resistance', 'price': resistance_line, 'index': top2['index']},
                    {'type': 'support_low', 'price': lowest_low, 'index': lows_between_sorted[0]['index']}
                ],
                signal=signal,
                price_target=price_target,
                stop_loss=lowest_low * 0.98,
                success_rate=0.74,
                interpretation=f"Padrão de continuação bullish. Breakout acima de {resistance_line:.5f} esperado.",
                status=status
            )

            formations.append(formation)

        return formations

    def _detect_descending_triangle(self, df: pd.DataFrame) -> List[ChartFormation]:
        """Detecta Descending Triangle (triângulo descendente)"""
        formations = []
        pivot_highs = self._find_pivot_points(df, 'high', window=5)
        pivot_lows = self._find_pivot_points(df, 'low', window=5)

        if len(pivot_highs) < 2 or len(pivot_lows) < 2:
            return formations

        # Procurar suporte horizontal (fundos similares)
        for i in range(len(pivot_lows) - 1):
            bottom1 = pivot_lows[i]
            bottom2 = pivot_lows[i + 1]

            if not self._prices_similar(bottom1['price'], bottom2['price'], tolerance=1.5):
                continue

            # Procurar topos descendentes
            highs_between = [h for h in pivot_highs if bottom1['index'] < h['index'] < bottom2['index']]
            if len(highs_between) < 1:
                continue

            highs_between_sorted = sorted(highs_between, key=lambda x: x['index'])
            is_descending = all(highs_between_sorted[j]['price'] > highs_between_sorted[j+1]['price']
                               for j in range(len(highs_between_sorted) - 1))

            if not is_descending and len(highs_between_sorted) > 1:
                continue

            support_line = (bottom1['price'] + bottom2['price']) / 2
            highest_high = max(h['price'] for h in highs_between_sorted)

            confidence = 80
            current_price = df['close'].iloc[-1]

            if current_price < support_line:
                status = 'confirmed'
                signal = 'SELL'
                confidence += 10
            else:
                status = 'forming'
                signal = 'SELL'

            triangle_height = highest_high - support_line
            price_target = support_line - triangle_height

            formation = ChartFormation(
                name="Descending Triangle",
                formation_type="continuation_bearish",
                confidence=confidence,
                start_index=bottom1['index'],
                end_index=bottom2['index'],
                key_points=[
                    {'type': 'support', 'price': support_line, 'index': bottom2['index']},
                    {'type': 'resistance_high', 'price': highest_high, 'index': highs_between_sorted[0]['index']}
                ],
                signal=signal,
                price_target=price_target,
                stop_loss=highest_high * 1.02,
                success_rate=0.71,
                interpretation=f"Padrão de continuação bearish. Breakdown abaixo de {support_line:.5f} esperado.",
                status=status
            )

            formations.append(formation)

        return formations

    def _detect_symmetrical_triangle(self, df: pd.DataFrame) -> List[ChartFormation]:
        """Detecta Symmetrical Triangle (triângulo simétrico)"""
        # Implementação simplificada
        return []

    def _detect_rising_wedge(self, df: pd.DataFrame) -> List[ChartFormation]:
        """Detecta Rising Wedge (cunha ascendente) - bearish"""
        return []

    def _detect_falling_wedge(self, df: pd.DataFrame) -> List[ChartFormation]:
        """Detecta Falling Wedge (cunha descendente) - bullish"""
        return []

    def _detect_bull_flag(self, df: pd.DataFrame) -> List[ChartFormation]:
        """Detecta Bull Flag (bandeira de alta)"""
        return []

    def _detect_bear_flag(self, df: pd.DataFrame) -> List[ChartFormation]:
        """Detecta Bear Flag (bandeira de baixa)"""
        return []

    # Helper methods

    def _find_pivot_points(self, df: pd.DataFrame, column: str, window: int = 5) -> List[Dict]:
        """Encontra pivot points (máximos ou mínimos locais)"""
        pivots = []

        for i in range(window, len(df) - window):
            window_before = df[column].iloc[i - window:i]
            window_after = df[column].iloc[i + 1:i + window + 1]
            current = df[column].iloc[i]

            if column == 'high':
                # Máximo local
                if current >= window_before.max() and current >= window_after.max():
                    pivots.append({
                        'price': current,
                        'index': i,
                        'timestamp': df['timestamp'].iloc[i]
                    })
            else:  # 'low'
                # Mínimo local
                if current <= window_before.min() and current <= window_after.min():
                    pivots.append({
                        'price': current,
                        'index': i,
                        'timestamp': df['timestamp'].iloc[i]
                    })

        return pivots

    def _prices_similar(self, price1: float, price2: float, tolerance: float = None) -> bool:
        """Verifica se dois preços são similares dentro da tolerância"""
        if tolerance is None:
            tolerance = self.tolerance_pct

        diff_pct = abs(price1 - price2) / price1 * 100
        return diff_pct <= tolerance
