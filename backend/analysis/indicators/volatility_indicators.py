"""
Indicadores Técnicos de Volatilidade
Implementa Bollinger Bands, ATR para medir volatilidade do mercado
"""

import pandas as pd
import numpy as np
from typing import Dict


class VolatilityIndicators:
    """
    Classe para calcular indicadores de volatilidade
    """

    @staticmethod
    def bollinger_bands(prices: pd.Series, period: int = 20,
                       std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        Bollinger Bands
        Bandas de volatilidade baseadas em desvio padrão

        Args:
            prices: Série de preços (close)
            period: Período da média móvel (padrão: 20)
            std_dev: Número de desvios padrão (padrão: 2.0)

        Returns:
            Dicionário com upper, middle, lower bands e width
        """
        if len(prices) < period:
            raise ValueError(f"Não há dados suficientes para Bollinger Bands")

        # Banda do meio (SMA)
        middle_band = prices.rolling(window=period).mean()

        # Desvio padrão
        rolling_std = prices.rolling(window=period).std()

        # Bandas superior e inferior
        upper_band = middle_band + (rolling_std * std_dev)
        lower_band = middle_band - (rolling_std * std_dev)

        # Largura das bandas (indicador de volatilidade)
        band_width = (upper_band - lower_band) / middle_band

        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band,
            'width': band_width
        }

    @staticmethod
    def interpret_bollinger(price: float, upper: float, middle: float,
                           lower: float, width: float) -> Dict[str, any]:
        """
        Interpreta posição do preço em relação às Bollinger Bands

        Args:
            price: Preço atual
            upper: Banda superior
            middle: Banda do meio
            lower: Banda inferior
            width: Largura das bandas

        Returns:
            Dicionário com interpretação
        """
        # Squeeze (compressão) - baixa volatilidade
        if width < 0.02:  # 2% de largura
            return {
                'signal': 'NEUTRAL',
                'condition': 'squeeze',
                'strength': 0,
                'description': 'Bollinger Squeeze detectado - esperar breakout',
                'volatility': 'low'
            }

        # Preço toca ou ultrapassa banda superior
        if price >= upper:
            return {
                'signal': 'SELL',
                'condition': 'touching_upper',
                'strength': min((price - upper) / upper * 1000, 100),
                'description': 'Preço na banda superior - possível reversão de baixa',
                'volatility': 'high' if width > 0.05 else 'normal'
            }

        # Preço toca ou ultrapassa banda inferior
        if price <= lower:
            return {
                'signal': 'BUY',
                'condition': 'touching_lower',
                'strength': min((lower - price) / lower * 1000, 100),
                'description': 'Preço na banda inferior - possível reversão de alta',
                'volatility': 'high' if width > 0.05 else 'normal'
            }

        # Preço próximo da banda superior (zona de sobrecompra)
        if price > middle and (upper - price) / (upper - middle) < 0.3:
            return {
                'signal': 'SELL',
                'condition': 'near_upper',
                'strength': 50,
                'description': 'Preço próximo da banda superior',
                'volatility': 'normal'
            }

        # Preço próximo da banda inferior (zona de sobrevenda)
        if price < middle and (price - lower) / (middle - lower) < 0.3:
            return {
                'signal': 'BUY',
                'condition': 'near_lower',
                'strength': 50,
                'description': 'Preço próximo da banda inferior',
                'volatility': 'normal'
            }

        # Preço na zona central
        return {
            'signal': 'NEUTRAL',
            'condition': 'middle_zone',
            'strength': 0,
            'description': 'Preço na zona central das bandas',
            'volatility': 'normal'
        }

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series,
           period: int = 14) -> pd.Series:
        """
        Average True Range (ATR)
        Mede volatilidade baseado no range verdadeiro

        Args:
            high: Série de preços máximos
            low: Série de preços mínimos
            close: Série de preços de fechamento
            period: Período do ATR (padrão: 14)

        Returns:
            Série com valores do ATR
        """
        if len(close) < period + 1:
            raise ValueError(f"Não há dados suficientes para ATR")

        # True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR = EMA do True Range
        atr = true_range.ewm(span=period, adjust=False).mean()

        return atr

    @staticmethod
    def interpret_atr(atr_value: float, atr_sma: float, price: float) -> Dict[str, any]:
        """
        Interpreta valor do ATR

        Args:
            atr_value: Valor atual do ATR
            atr_sma: Média do ATR (últimos 50 períodos)
            price: Preço atual

        Returns:
            Dicionário com interpretação
        """
        # ATR como % do preço
        atr_percent = (atr_value / price) * 100

        # Comparar ATR atual com sua média
        atr_ratio = atr_value / atr_sma if atr_sma > 0 else 1.0

        if atr_ratio > 1.5:
            return {
                'volatility': 'very_high',
                'atr_percent': atr_percent,
                'description': f'Volatilidade muito alta (ATR: {atr_percent:.2f}%)',
                'recommendation': 'Aumentar stop loss, reduzir position size'
            }
        elif atr_ratio > 1.2:
            return {
                'volatility': 'high',
                'atr_percent': atr_percent,
                'description': f'Volatilidade alta (ATR: {atr_percent:.2f}%)',
                'recommendation': 'Ajustar stop loss para ATR expandido'
            }
        elif atr_ratio < 0.7:
            return {
                'volatility': 'low',
                'atr_percent': atr_percent,
                'description': f'Volatilidade baixa (ATR: {atr_percent:.2f}%)',
                'recommendation': 'Possível breakout iminente'
            }
        else:
            return {
                'volatility': 'normal',
                'atr_percent': atr_percent,
                'description': f'Volatilidade normal (ATR: {atr_percent:.2f}%)',
                'recommendation': 'Condições normais de mercado'
            }

    @staticmethod
    def calculate_atr_stop_loss(entry_price: float, atr_value: float,
                               is_long: bool, multiplier: float = 2.0) -> float:
        """
        Calcula stop loss baseado no ATR

        Args:
            entry_price: Preço de entrada
            atr_value: Valor do ATR
            is_long: True se posição long, False se short
            multiplier: Multiplicador do ATR (padrão: 2.0)

        Returns:
            Preço do stop loss
        """
        atr_distance = atr_value * multiplier

        if is_long:
            # Stop loss abaixo do entry para posição long
            stop_loss = entry_price - atr_distance
        else:
            # Stop loss acima do entry para posição short
            stop_loss = entry_price + atr_distance

        return stop_loss

    @staticmethod
    def volatility_regime(atr_series: pd.Series, lookback: int = 50) -> str:
        """
        Identifica regime de volatilidade atual

        Args:
            atr_series: Série de valores do ATR
            lookback: Períodos para análise

        Returns:
            Regime de volatilidade (low/normal/high/extreme)
        """
        if len(atr_series) < lookback:
            return 'unknown'

        recent_atr = atr_series.tail(lookback)
        current_atr = atr_series.iloc[-1]

        percentile_25 = recent_atr.quantile(0.25)
        percentile_50 = recent_atr.quantile(0.50)
        percentile_75 = recent_atr.quantile(0.75)
        percentile_90 = recent_atr.quantile(0.90)

        if current_atr > percentile_90:
            return 'extreme'
        elif current_atr > percentile_75:
            return 'high'
        elif current_atr < percentile_25:
            return 'low'
        else:
            return 'normal'
