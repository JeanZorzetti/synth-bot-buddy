"""
Módulo de Reconhecimento de Padrões
Fase 2 do Roadmap: Padrões de Candlestick e Formações Gráficas
"""

from .candlestick_patterns import CandlestickPatterns, CandlestickPattern
from .support_resistance import SupportResistanceDetector, SupportResistanceLevel
from .chart_formations import ChartFormationDetector, ChartFormation

__all__ = [
    'CandlestickPatterns',
    'CandlestickPattern',
    'SupportResistanceDetector',
    'SupportResistanceLevel',
    'ChartFormationDetector',
    'ChartFormation'
]
