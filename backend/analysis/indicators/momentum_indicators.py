"""
Indicadores Técnicos de Momentum
Implementa RSI, MACD, Stochastic para medir força e velocidade de movimento
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


class MomentumIndicators:
    """
    Classe para calcular indicadores de momentum
    """

    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index (RSI)
        Oscilador entre 0-100 que mede velocidade e magnitude das mudanças de preço

        Args:
            prices: Série de preços (close)
            period: Período do RSI (padrão: 14)

        Returns:
            Série com valores do RSI
        """
        if len(prices) < period + 1:
            raise ValueError(f"Não há dados suficientes para RSI. Necessário {period + 1}, disponível {len(prices)}")

        # Calcular variações de preço
        delta = prices.diff()

        # Separar ganhos e perdas
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Médias móveis exponenciais dos ganhos e perdas
        avg_gains = gains.ewm(span=period, adjust=False).mean()
        avg_losses = losses.ewm(span=period, adjust=False).mean()

        # Relative Strength
        rs = avg_gains / avg_losses

        # RSI
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def interpret_rsi(rsi_value: float) -> Dict[str, any]:
        """
        Interpreta valor do RSI

        Args:
            rsi_value: Valor atual do RSI

        Returns:
            Dicionário com interpretação
        """
        if rsi_value >= 70:
            return {
                'condition': 'overbought',
                'signal': 'SELL',
                'strength': (rsi_value - 70) * 3.33,  # 0-100 scale
                'description': f'RSI em sobrecompra ({rsi_value:.1f}), possível reversão de baixa'
            }
        elif rsi_value <= 30:
            return {
                'condition': 'oversold',
                'signal': 'BUY',
                'strength': (30 - rsi_value) * 3.33,  # 0-100 scale
                'description': f'RSI em sobrevenda ({rsi_value:.1f}), possível reversão de alta'
            }
        elif 40 <= rsi_value <= 60:
            return {
                'condition': 'neutral',
                'signal': 'NEUTRAL',
                'strength': 0,
                'description': f'RSI neutro ({rsi_value:.1f}), sem sinal claro'
            }
        else:
            return {
                'condition': 'normal',
                'signal': 'NEUTRAL',
                'strength': 0,
                'description': f'RSI em zona normal ({rsi_value:.1f})'
            }

    @staticmethod
    def macd(prices: pd.Series,
             fast_period: int = 12,
             slow_period: int = 26,
             signal_period: int = 9) -> Dict[str, pd.Series]:
        """
        Moving Average Convergence Divergence (MACD)
        Indicador de tendência que mostra relação entre duas médias móveis

        Args:
            prices: Série de preços (close)
            fast_period: Período da EMA rápida (padrão: 12)
            slow_period: Período da EMA lenta (padrão: 26)
            signal_period: Período da linha de sinal (padrão: 9)

        Returns:
            Dicionário com MACD line, Signal line e Histogram
        """
        if len(prices) < slow_period + signal_period:
            raise ValueError(f"Não há dados suficientes para MACD")

        # EMAs
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()

        # MACD Line
        macd_line = ema_fast - ema_slow

        # Signal Line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        # Histogram
        histogram = macd_line - signal_line

        return {
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram
        }

    @staticmethod
    def interpret_macd(macd_line: float, signal_line: float, histogram: float,
                       prev_histogram: float) -> Dict[str, any]:
        """
        Interpreta sinais do MACD

        Args:
            macd_line: Valor atual da MACD line
            signal_line: Valor atual da Signal line
            histogram: Valor atual do histogram
            prev_histogram: Valor anterior do histogram

        Returns:
            Dicionário com interpretação
        """
        # Cruzamento bullish (MACD cruza acima da Signal)
        if prev_histogram <= 0 and histogram > 0:
            return {
                'signal': 'BUY',
                'type': 'bullish_crossover',
                'strength': min(abs(histogram) * 100, 100),
                'description': 'MACD cruzou acima da linha de sinal (bullish)'
            }

        # Cruzamento bearish (MACD cruza abaixo da Signal)
        if prev_histogram >= 0 and histogram < 0:
            return {
                'signal': 'SELL',
                'type': 'bearish_crossover',
                'strength': min(abs(histogram) * 100, 100),
                'description': 'MACD cruzou abaixo da linha de sinal (bearish)'
            }

        # Momentum bullish (histogram positivo e crescente)
        if histogram > 0 and histogram > prev_histogram:
            return {
                'signal': 'BUY',
                'type': 'bullish_momentum',
                'strength': min(abs(histogram) * 50, 100),
                'description': 'Momentum de alta crescente'
            }

        # Momentum bearish (histogram negativo e decrescente)
        if histogram < 0 and histogram < prev_histogram:
            return {
                'signal': 'SELL',
                'type': 'bearish_momentum',
                'strength': min(abs(histogram) * 50, 100),
                'description': 'Momentum de baixa crescente'
            }

        return {
            'signal': 'NEUTRAL',
            'type': 'no_signal',
            'strength': 0,
            'description': 'Sem sinal claro no MACD'
        }

    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                   k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """
        Stochastic Oscillator
        Compara preço de fechamento com range de preços em determinado período

        Args:
            high: Série de preços máximos
            low: Série de preços mínimos
            close: Série de preços de fechamento
            k_period: Período do %K (padrão: 14)
            d_period: Período do %D (padrão: 3)

        Returns:
            Dicionário com %K e %D
        """
        if len(close) < k_period:
            raise ValueError(f"Não há dados suficientes para Stochastic")

        # %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)

        # %D = SMA de %K
        d_percent = k_percent.rolling(window=d_period).mean()

        return {
            'k_percent': k_percent,
            'd_percent': d_percent
        }

    @staticmethod
    def interpret_stochastic(k_value: float, d_value: float,
                            prev_k: float, prev_d: float) -> Dict[str, any]:
        """
        Interpreta sinais do Stochastic

        Args:
            k_value: Valor atual do %K
            d_value: Valor atual do %D
            prev_k: Valor anterior do %K
            prev_d: Valor anterior do %D

        Returns:
            Dicionário com interpretação
        """
        # Sobrecompra (> 80)
        if k_value > 80 and d_value > 80:
            # Cruzamento bearish em zona de sobrecompra
            if prev_k >= prev_d and k_value < d_value:
                return {
                    'signal': 'SELL',
                    'condition': 'overbought_crossover',
                    'strength': min((k_value - 80) * 5, 100),
                    'description': 'Cruzamento bearish em zona de sobrecompra'
                }
            return {
                'signal': 'SELL',
                'condition': 'overbought',
                'strength': min((k_value - 80) * 3, 100),
                'description': f'Stochastic em sobrecompra ({k_value:.1f})'
            }

        # Sobrevenda (< 20)
        if k_value < 20 and d_value < 20:
            # Cruzamento bullish em zona de sobrevenda
            if prev_k <= prev_d and k_value > d_value:
                return {
                    'signal': 'BUY',
                    'condition': 'oversold_crossover',
                    'strength': min((20 - k_value) * 5, 100),
                    'description': 'Cruzamento bullish em zona de sobrevenda'
                }
            return {
                'signal': 'BUY',
                'condition': 'oversold',
                'strength': min((20 - k_value) * 3, 100),
                'description': f'Stochastic em sobrevenda ({k_value:.1f})'
            }

        # Cruzamento bullish em zona neutra
        if prev_k <= prev_d and k_value > d_value:
            return {
                'signal': 'BUY',
                'condition': 'bullish_crossover',
                'strength': 50,
                'description': 'Cruzamento bullish do Stochastic'
            }

        # Cruzamento bearish em zona neutra
        if prev_k >= prev_d and k_value < d_value:
            return {
                'signal': 'SELL',
                'condition': 'bearish_crossover',
                'strength': 50,
                'description': 'Cruzamento bearish do Stochastic'
            }

        return {
            'signal': 'NEUTRAL',
            'condition': 'normal',
            'strength': 0,
            'description': 'Stochastic em zona neutra sem sinal'
        }
