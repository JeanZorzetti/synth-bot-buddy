"""
Indicadores Técnicos de Tendência
Implementa SMA, EMA e outros indicadores para identificar direção do mercado
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class TrendIndicators:
    """
    Classe para calcular indicadores de tendência
    """

    @staticmethod
    def sma(prices: pd.Series, period: int) -> pd.Series:
        """
        Simple Moving Average (SMA)

        Args:
            prices: Série de preços (close)
            period: Período da média (ex: 20, 50, 200)

        Returns:
            Série com valores da SMA
        """
        if len(prices) < period:
            raise ValueError(f"Não há dados suficientes. Necessário {period}, disponível {len(prices)}")

        return prices.rolling(window=period).mean()

    @staticmethod
    def ema(prices: pd.Series, period: int) -> pd.Series:
        """
        Exponential Moving Average (EMA)
        Mais responsiva a mudanças recentes de preço

        Args:
            prices: Série de preços (close)
            period: Período da média (ex: 9, 21, 55)

        Returns:
            Série com valores da EMA
        """
        if len(prices) < period:
            raise ValueError(f"Não há dados suficientes. Necessário {period}, disponível {len(prices)}")

        return prices.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_all_moving_averages(prices: pd.Series) -> Dict[str, pd.Series]:
        """
        Calcula todas as médias móveis relevantes

        Args:
            prices: Série de preços (close)

        Returns:
            Dicionário com todas as médias calculadas
        """
        result = {}

        # SMAs
        sma_periods = [20, 50, 100, 200]
        for period in sma_periods:
            if len(prices) >= period:
                result[f'sma_{period}'] = TrendIndicators.sma(prices, period)

        # EMAs
        ema_periods = [9, 21, 55]
        for period in ema_periods:
            if len(prices) >= period:
                result[f'ema_{period}'] = TrendIndicators.ema(prices, period)

        return result

    @staticmethod
    def detect_crossover(fast_ma: pd.Series, slow_ma: pd.Series) -> Dict[str, any]:
        """
        Detecta cruzamento de médias móveis (crossover)

        Args:
            fast_ma: Média móvel rápida (ex: EMA 9)
            slow_ma: Média móvel lenta (ex: EMA 21)

        Returns:
            Dicionário com informação do cruzamento
        """
        # Últimos 2 valores
        fast_prev, fast_curr = fast_ma.iloc[-2], fast_ma.iloc[-1]
        slow_prev, slow_curr = slow_ma.iloc[-2], slow_ma.iloc[-1]

        # Bullish crossover (média rápida cruza acima da lenta)
        if fast_prev <= slow_prev and fast_curr > slow_curr:
            return {
                'type': 'bullish',
                'signal': 'BUY',
                'strength': abs((fast_curr - slow_curr) / slow_curr * 100),
                'description': 'Média rápida cruzou acima da média lenta'
            }

        # Bearish crossover (média rápida cruza abaixo da lenta)
        if fast_prev >= slow_prev and fast_curr < slow_curr:
            return {
                'type': 'bearish',
                'signal': 'SELL',
                'strength': abs((fast_curr - slow_curr) / slow_curr * 100),
                'description': 'Média rápida cruzou abaixo da média lenta'
            }

        # Sem cruzamento
        return {
            'type': 'none',
            'signal': 'NEUTRAL',
            'strength': 0,
            'description': 'Sem cruzamento detectado'
        }

    @staticmethod
    def identify_trend(price: float, sma_20: float, sma_50: float, sma_200: float) -> Dict[str, any]:
        """
        Identifica tendência do mercado baseado nas médias móveis

        Args:
            price: Preço atual
            sma_20: SMA de 20 períodos
            sma_50: SMA de 50 períodos
            sma_200: SMA de 200 períodos

        Returns:
            Dicionário com informação da tendência
        """
        # Tendência de alta forte: Preço > SMA20 > SMA50 > SMA200
        if price > sma_20 > sma_50 > sma_200:
            return {
                'trend': 'strong_uptrend',
                'direction': 'UP',
                'strength': 100,
                'description': 'Tendência de alta forte confirmada'
            }

        # Tendência de alta: Preço > SMA50 > SMA200
        if price > sma_50 > sma_200:
            return {
                'trend': 'uptrend',
                'direction': 'UP',
                'strength': 75,
                'description': 'Tendência de alta'
            }

        # Tendência de baixa forte: Preço < SMA20 < SMA50 < SMA200
        if price < sma_20 < sma_50 < sma_200:
            return {
                'trend': 'strong_downtrend',
                'direction': 'DOWN',
                'strength': 100,
                'description': 'Tendência de baixa forte confirmada'
            }

        # Tendência de baixa: Preço < SMA50 < SMA200
        if price < sma_50 < sma_200:
            return {
                'trend': 'downtrend',
                'direction': 'DOWN',
                'strength': 75,
                'description': 'Tendência de baixa'
            }

        # Mercado lateral/indefinido
        return {
            'trend': 'sideways',
            'direction': 'NEUTRAL',
            'strength': 0,
            'description': 'Mercado sem tendência clara (lateral)'
        }

    @staticmethod
    def calculate_trend_strength(prices: pd.Series, period: int = 14) -> float:
        """
        Calcula força da tendência usando R-squared da regressão linear

        Args:
            prices: Série de preços
            period: Período para análise

        Returns:
            Valor entre 0 e 1 indicando força da tendência
        """
        if len(prices) < period:
            return 0.0

        # Últimos N períodos
        recent_prices = prices.tail(period).values

        # Regressão linear simples
        x = np.arange(len(recent_prices))
        y = recent_prices

        # Coeficiente de correlação ao quadrado (R²)
        correlation = np.corrcoef(x, y)[0, 1]
        r_squared = correlation ** 2

        return float(r_squared)
