"""
Módulo de Detecção de Suporte e Resistência Dinâmica

Identifica automaticamente níveis de suporte e resistência usando:
- Pivot points (máximos e mínimos locais)
- Volume profile
- Análise de força dos níveis
- Detecção de breakouts e bounces
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class SupportResistanceLevel:
    """Representa um nível de suporte ou resistência identificado"""

    level_type: str  # 'support' ou 'resistance'
    price: float
    strength: float  # 0-100, baseado em touches, volume, idade
    touches: int  # Quantidade de vezes que o preço testou este nível
    first_touch: datetime
    last_touch: datetime
    volume_at_level: float  # Volume médio quando preço está neste nível
    is_active: bool  # Se o nível ainda é relevante
    distance_from_current: float  # Distância percentual do preço atual
    zone_range: Tuple[float, float]  # (lower_bound, upper_bound) da zona

    def __repr__(self):
        return f"{self.level_type.upper()} @ {self.price:.5f} (strength: {self.strength:.0f}, touches: {self.touches})"


class SupportResistanceDetector:
    """Detector automático de níveis de suporte e resistência"""

    def __init__(self, window: int = 20, min_touches: int = 2, zone_width_pct: float = 0.1):
        """
        Args:
            window: Janela para detecção de pivot points
            min_touches: Mínimo de touches para considerar um nível válido
            zone_width_pct: Largura da zona de S/R como % do preço (0.1 = 0.1%)
        """
        self.window = window
        self.min_touches = min_touches
        self.zone_width_pct = zone_width_pct

    def detect_levels(self, df: pd.DataFrame, lookback: int = 100) -> List[SupportResistanceLevel]:
        """
        Detecta todos os níveis de suporte e resistência

        Args:
            df: DataFrame com OHLCV data
            lookback: Quantidade de candles para análise

        Returns:
            Lista de SupportResistanceLevel ordenada por strength
        """
        if len(df) < self.window * 2:
            logger.warning(f"Dataframe muito pequeno para análise S/R: {len(df)} candles")
            return []

        # Usar apenas lookback candles mais recentes
        analysis_df = df.tail(lookback).copy()
        current_price = analysis_df['close'].iloc[-1]

        logger.info(f"\n{'='*60}")
        logger.info(f"DETECÇÃO DE SUPORTE E RESISTÊNCIA")
        logger.info(f"Período: {lookback} candles | Preço atual: {current_price:.5f}")
        logger.info(f"Parâmetros: window={self.window}, min_touches={self.min_touches}")

        # 1. Encontrar pivot points
        resistance_pivots = self._find_pivot_highs(analysis_df)
        support_pivots = self._find_pivot_lows(analysis_df)

        logger.info(f"Pivots encontrados: {len(resistance_pivots)} resistências, {len(support_pivots)} suportes")

        # 2. Agrupar pivots próximos em níveis (clustering)
        resistance_levels = self._cluster_pivots(analysis_df, resistance_pivots, 'resistance', current_price)
        support_levels = self._cluster_pivots(analysis_df, support_pivots, 'support', current_price)

        # 3. Combinar e ordenar por força
        all_levels = resistance_levels + support_levels
        all_levels = sorted(all_levels, key=lambda x: x.strength, reverse=True)

        # 4. Filtrar níveis fracos e muito distantes
        filtered_levels = self._filter_levels(all_levels, current_price)

        logger.info(f"\nNÍVEIS DETECTADOS (após filtragem):")
        for level in filtered_levels[:10]:  # Top 10
            logger.info(f"  {level}")

        logger.info(f"{'='*60}\n")

        return filtered_levels

    def _find_pivot_highs(self, df: pd.DataFrame) -> List[Dict]:
        """Encontra máximos locais (resistências em potencial)"""
        pivots = []

        for i in range(self.window, len(df) - self.window):
            # Verificar se é máximo local
            window_before = df['high'].iloc[i - self.window:i]
            window_after = df['high'].iloc[i + 1:i + self.window + 1]
            current_high = df['high'].iloc[i]

            if current_high >= window_before.max() and current_high >= window_after.max():
                pivots.append({
                    'price': current_high,
                    'index': i,
                    'timestamp': df['timestamp'].iloc[i],
                    'volume': df['volume'].iloc[i]
                })

        return pivots

    def _find_pivot_lows(self, df: pd.DataFrame) -> List[Dict]:
        """Encontra mínimos locais (suportes em potencial)"""
        pivots = []

        for i in range(self.window, len(df) - self.window):
            # Verificar se é mínimo local
            window_before = df['low'].iloc[i - self.window:i]
            window_after = df['low'].iloc[i + 1:i + self.window + 1]
            current_low = df['low'].iloc[i]

            if current_low <= window_before.min() and current_low <= window_after.min():
                pivots.append({
                    'price': current_low,
                    'index': i,
                    'timestamp': df['timestamp'].iloc[i],
                    'volume': df['volume'].iloc[i]
                })

        return pivots

    def _cluster_pivots(self, df: pd.DataFrame, pivots: List[Dict],
                       level_type: str, current_price: float) -> List[SupportResistanceLevel]:
        """
        Agrupa pivots próximos em níveis de S/R

        Pivots que estão dentro de zone_width_pct são considerados o mesmo nível
        """
        if not pivots:
            return []

        levels = []
        zone_width = current_price * (self.zone_width_pct / 100)

        # Ordenar pivots por preço
        sorted_pivots = sorted(pivots, key=lambda x: x['price'])

        # Agrupar pivots próximos
        clusters = []
        current_cluster = [sorted_pivots[0]]

        for pivot in sorted_pivots[1:]:
            # Se pivot está próximo do cluster atual, adicionar
            cluster_avg = np.mean([p['price'] for p in current_cluster])
            if abs(pivot['price'] - cluster_avg) <= zone_width:
                current_cluster.append(pivot)
            else:
                # Cluster completo, começar novo
                clusters.append(current_cluster)
                current_cluster = [pivot]

        # Adicionar último cluster
        if current_cluster:
            clusters.append(current_cluster)

        # Converter clusters em níveis
        for cluster in clusters:
            if len(cluster) < self.min_touches:
                continue

            # Calcular métricas do nível
            prices = [p['price'] for p in cluster]
            volumes = [p['volume'] for p in cluster]
            timestamps = [p['timestamp'] for p in cluster]

            avg_price = np.mean(prices)
            avg_volume = np.mean(volumes)
            touches = len(cluster)

            # Calcular força do nível (0-100)
            strength = self._calculate_level_strength(
                touches=touches,
                volume=avg_volume,
                first_touch=min(timestamps),
                last_touch=max(timestamps),
                current_time=df['timestamp'].iloc[-1]
            )

            # Calcular distância do preço atual
            distance_pct = ((avg_price - current_price) / current_price) * 100

            # Definir zona de S/R (upper/lower bounds)
            zone_lower = min(prices) - zone_width / 2
            zone_upper = max(prices) + zone_width / 2

            # Determinar se nível está ativo (testado recentemente ou próximo do preço)
            days_since_last = (df['timestamp'].iloc[-1] - max(timestamps)).total_seconds() / 86400
            is_active = days_since_last <= 30 or abs(distance_pct) <= 2.0  # 30 dias ou 2% de distância

            level = SupportResistanceLevel(
                level_type=level_type,
                price=avg_price,
                strength=strength,
                touches=touches,
                first_touch=min(timestamps),
                last_touch=max(timestamps),
                volume_at_level=avg_volume,
                is_active=is_active,
                distance_from_current=distance_pct,
                zone_range=(zone_lower, zone_upper)
            )

            levels.append(level)

        return levels

    def _calculate_level_strength(self, touches: int, volume: float,
                                  first_touch: datetime, last_touch: datetime,
                                  current_time: datetime) -> float:
        """
        Calcula a força de um nível de S/R (0-100)

        Fatores:
        - Número de touches (mais touches = mais forte)
        - Volume nos touches (mais volume = mais forte)
        - Idade do nível (níveis antigos ainda relevantes são mais fortes)
        - Recência do último touch (touches recentes = mais relevante)
        """
        # 1. Score por touches (max 40 pontos)
        touch_score = min(touches * 8, 40)  # 5 touches = 40 pontos

        # 2. Score por volume (max 30 pontos)
        # Normalizar volume (assumindo volume médio = 1.0)
        volume_score = min(np.log1p(volume) * 10, 30)

        # 3. Score por idade (max 15 pontos)
        # Níveis que existem há muito tempo e ainda são relevantes são fortes
        age_days = (last_touch - first_touch).total_seconds() / 86400
        age_score = min(age_days / 2, 15)  # 30 dias = 15 pontos

        # 4. Score por recência (max 15 pontos)
        # Touches recentes são mais relevantes
        days_since_last = (current_time - last_touch).total_seconds() / 86400
        if days_since_last <= 7:
            recency_score = 15
        elif days_since_last <= 30:
            recency_score = 10
        else:
            recency_score = 5

        total_strength = touch_score + volume_score + age_score + recency_score
        return min(total_strength, 100)

    def _filter_levels(self, levels: List[SupportResistanceLevel],
                      current_price: float, max_distance_pct: float = 10.0) -> List[SupportResistanceLevel]:
        """
        Filtra níveis fracos ou muito distantes

        Args:
            levels: Lista de níveis detectados
            current_price: Preço atual
            max_distance_pct: Distância máxima em % para considerar um nível

        Returns:
            Lista filtrada de níveis relevantes
        """
        filtered = []

        for level in levels:
            # Remover níveis muito fracos (strength < 30)
            if level.strength < 30:
                continue

            # Remover níveis muito distantes (>10% do preço atual)
            if abs(level.distance_from_current) > max_distance_pct:
                continue

            # Manter apenas níveis ativos
            if not level.is_active:
                continue

            filtered.append(level)

        return filtered

    def find_nearest_levels(self, df: pd.DataFrame, side: str = 'both') -> Dict:
        """
        Encontra os níveis de S/R mais próximos do preço atual

        Args:
            df: DataFrame com OHLCV
            side: 'support', 'resistance', ou 'both'

        Returns:
            Dict com nearest_support e/ou nearest_resistance
        """
        levels = self.detect_levels(df)
        current_price = df['close'].iloc[-1]

        result = {}

        if side in ['support', 'both']:
            # Encontrar suporte mais próximo abaixo do preço
            supports_below = [l for l in levels
                            if l.level_type == 'support' and l.price < current_price]
            if supports_below:
                nearest_support = max(supports_below, key=lambda x: x.price)
                result['nearest_support'] = {
                    'price': nearest_support.price,
                    'distance_pct': nearest_support.distance_from_current,
                    'strength': nearest_support.strength,
                    'touches': nearest_support.touches
                }

        if side in ['resistance', 'both']:
            # Encontrar resistência mais próxima acima do preço
            resistances_above = [l for l in levels
                               if l.level_type == 'resistance' and l.price > current_price]
            if resistances_above:
                nearest_resistance = min(resistances_above, key=lambda x: x.price)
                result['nearest_resistance'] = {
                    'price': nearest_resistance.price,
                    'distance_pct': nearest_resistance.distance_from_current,
                    'strength': nearest_resistance.strength,
                    'touches': nearest_resistance.touches
                }

        return result

    def detect_breakout(self, df: pd.DataFrame, lookback: int = 50) -> Optional[Dict]:
        """
        Detecta se houve breakout de um nível de S/R

        Returns:
            Dict com informações do breakout ou None se não houver
        """
        levels = self.detect_levels(df, lookback=lookback)
        current_price = df['close'].iloc[-1]
        previous_price = df['close'].iloc[-2]

        # Verificar se rompeu resistência (breakout bullish)
        for level in levels:
            if level.level_type == 'resistance':
                # Preço anterior abaixo, preço atual acima
                if previous_price < level.zone_range[1] and current_price > level.zone_range[1]:
                    return {
                        'type': 'bullish_breakout',
                        'level_price': level.price,
                        'level_strength': level.strength,
                        'breakout_candle_close': current_price,
                        'breakout_percentage': ((current_price - level.price) / level.price) * 100,
                        'interpretation': f"Rompimento de resistência em {level.price:.5f} (força {level.strength:.0f})"
                    }

            # Verificar se rompeu suporte (breakdown bearish)
            elif level.level_type == 'support':
                # Preço anterior acima, preço atual abaixo
                if previous_price > level.zone_range[0] and current_price < level.zone_range[0]:
                    return {
                        'type': 'bearish_breakdown',
                        'level_price': level.price,
                        'level_strength': level.strength,
                        'breakout_candle_close': current_price,
                        'breakout_percentage': ((level.price - current_price) / level.price) * 100,
                        'interpretation': f"Rompimento de suporte em {level.price:.5f} (força {level.strength:.0f})"
                    }

        return None

    def detect_bounce(self, df: pd.DataFrame, lookback: int = 50) -> Optional[Dict]:
        """
        Detecta se houve bounce (rejeição) de um nível de S/R

        Returns:
            Dict com informações do bounce ou None se não houver
        """
        levels = self.detect_levels(df, lookback=lookback)
        current_candle = df.iloc[-1]

        # Verificar se rejeitou suporte (bounce bullish)
        for level in levels:
            if level.level_type == 'support':
                # Low do candle testou a zona, mas fechou acima
                if (current_candle['low'] <= level.zone_range[1] and
                    current_candle['close'] > level.zone_range[1]):

                    wick_size = current_candle['close'] - current_candle['low']
                    body_size = abs(current_candle['close'] - current_candle['open'])

                    # Bounce válido se wick >= 1.5x body
                    if wick_size >= body_size * 1.5:
                        return {
                            'type': 'bullish_bounce',
                            'level_price': level.price,
                            'level_strength': level.strength,
                            'bounce_candle_low': current_candle['low'],
                            'bounce_candle_close': current_candle['close'],
                            'wick_to_body_ratio': wick_size / body_size if body_size > 0 else 999,
                            'interpretation': f"Rejeição de suporte em {level.price:.5f} (força {level.strength:.0f})"
                        }

            # Verificar se rejeitou resistência (bounce bearish)
            elif level.level_type == 'resistance':
                # High do candle testou a zona, mas fechou abaixo
                if (current_candle['high'] >= level.zone_range[0] and
                    current_candle['close'] < level.zone_range[0]):

                    wick_size = current_candle['high'] - current_candle['close']
                    body_size = abs(current_candle['close'] - current_candle['open'])

                    # Bounce válido se wick >= 1.5x body
                    if wick_size >= body_size * 1.5:
                        return {
                            'type': 'bearish_bounce',
                            'level_price': level.price,
                            'level_strength': level.strength,
                            'bounce_candle_high': current_candle['high'],
                            'bounce_candle_close': current_candle['close'],
                            'wick_to_body_ratio': wick_size / body_size if body_size > 0 else 999,
                            'interpretation': f"Rejeição de resistência em {level.price:.5f} (força {level.strength:.0f})"
                        }

        return None

    def get_analysis_summary(self, df: pd.DataFrame) -> Dict:
        """
        Retorna análise completa de S/R para o DataFrame

        Returns:
            Dict com todos os níveis, nearest levels, breakouts e bounces
        """
        levels = self.detect_levels(df)
        nearest = self.find_nearest_levels(df)
        breakout = self.detect_breakout(df)
        bounce = self.detect_bounce(df)

        current_price = df['close'].iloc[-1]

        # Separar por tipo
        supports = [l for l in levels if l.level_type == 'support']
        resistances = [l for l in levels if l.level_type == 'resistance']

        return {
            'current_price': current_price,
            'total_levels': len(levels),
            'support_levels': len(supports),
            'resistance_levels': len(resistances),
            'nearest_support': nearest.get('nearest_support'),
            'nearest_resistance': nearest.get('nearest_resistance'),
            'breakout_detected': breakout,
            'bounce_detected': bounce,
            'all_levels': [
                {
                    'type': l.level_type,
                    'price': l.price,
                    'strength': l.strength,
                    'touches': l.touches,
                    'distance_pct': l.distance_from_current,
                    'is_active': l.is_active
                }
                for l in levels[:20]  # Top 20 níveis
            ]
        }
