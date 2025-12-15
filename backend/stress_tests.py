"""
Stress Tests para Paper Trading Engine

Este módulo implementa cenários de teste extremos para validar
o comportamento do sistema em condições adversas de mercado.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class StressTestScenario:
    """Base class para cenários de stress test"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def generate_data(self, n_candles: int = 1000, initial_price: float = 1000.0) -> pd.DataFrame:
        """
        Gera dados de mercado para o cenário

        Args:
            n_candles: Número de candles a gerar
            initial_price: Preço inicial

        Returns:
            DataFrame com colunas: open, high, low, close, timestamp
        """
        raise NotImplementedError("Subclasses devem implementar generate_data()")


class HighVolatilityScenario(StressTestScenario):
    """
    Cenário 1: Alta Volatilidade
    Simula spikes de 5%+ em curtos períodos
    """

    def __init__(self):
        super().__init__(
            name="Alta Volatilidade",
            description="Mercado com spikes frequentes de 5%+ (similar a criptomoedas)"
        )

    def generate_data(self, n_candles: int = 1000, initial_price: float = 1000.0) -> pd.DataFrame:
        """Gera dados com alta volatilidade"""
        prices = [initial_price]
        timestamps = []
        start_time = datetime.now() - timedelta(minutes=n_candles)

        for i in range(n_candles):
            # 10% de chance de spike grande (5-10%)
            if np.random.random() < 0.10:
                spike = np.random.choice([-1, 1]) * np.random.uniform(0.05, 0.10)
            else:
                # Movimento normal com volatilidade aumentada
                spike = np.random.normal(0, 0.02)  # Desvio padrão de 2%

            new_price = prices[-1] * (1 + spike)
            prices.append(new_price)
            timestamps.append(start_time + timedelta(minutes=i))

        # Criar OHLC
        data = []
        for i in range(1, len(prices)):
            open_price = prices[i - 1]
            close_price = prices[i]

            # High/Low com volatilidade intra-candle
            intra_volatility = abs(close_price - open_price) * 1.5
            high_price = max(open_price, close_price) + np.random.uniform(0, intra_volatility)
            low_price = min(open_price, close_price) - np.random.uniform(0, intra_volatility)

            data.append({
                'timestamp': timestamps[i].isoformat(),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
            })

        return pd.DataFrame(data)


class LowVolumeScenario(StressTestScenario):
    """
    Cenário 2: Baixo Volume
    Mercado ilíquido com movimentos erráticos
    """

    def __init__(self):
        super().__init__(
            name="Baixo Volume",
            description="Mercado ilíquido com spreads largos e movimentos erráticos"
        )

    def generate_data(self, n_candles: int = 1000, initial_price: float = 1000.0) -> pd.DataFrame:
        """Gera dados com baixo volume"""
        prices = [initial_price]
        timestamps = []
        start_time = datetime.now() - timedelta(minutes=n_candles)

        for i in range(n_candles):
            # Movimentos pequenos e erráticos
            if np.random.random() < 0.30:
                # 30% de chance de não haver movimento
                change = 0
            else:
                # Movimentos pequenos mas erráticos
                change = np.random.choice([-1, 1]) * np.random.uniform(0.001, 0.005)

            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
            timestamps.append(start_time + timedelta(minutes=i))

        # Criar OHLC com spreads largos
        data = []
        for i in range(1, len(prices)):
            open_price = prices[i - 1]
            close_price = prices[i]

            # Spreads largos (1-2%)
            spread = open_price * np.random.uniform(0.01, 0.02)
            high_price = max(open_price, close_price) + spread / 2
            low_price = min(open_price, close_price) - spread / 2

            data.append({
                'timestamp': timestamps[i].isoformat(),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
            })

        return pd.DataFrame(data)


class FlashCrashScenario(StressTestScenario):
    """
    Cenário 3: Flash Crash
    Queda súbita de 10% seguida de recuperação
    """

    def __init__(self):
        super().__init__(
            name="Flash Crash",
            description="Queda súbita de 10% em minutos, seguida de recuperação parcial"
        )

    def generate_data(self, n_candles: int = 1000, initial_price: float = 1000.0) -> pd.DataFrame:
        """Gera dados com flash crash"""
        prices = [initial_price]
        timestamps = []
        start_time = datetime.now() - timedelta(minutes=n_candles)

        # Crash acontece em 40% do caminho
        crash_start = int(n_candles * 0.40)
        crash_duration = 50  # 50 candles para crash completo
        crash_end = crash_start + crash_duration

        for i in range(n_candles):
            if crash_start <= i < crash_end:
                # Durante o crash: queda rápida de 10%
                progress = (i - crash_start) / crash_duration
                crash_magnitude = -0.10 * progress
                new_price = initial_price * (1 + crash_magnitude)
            elif crash_end <= i < crash_end + 100:
                # Recuperação parcial (50% da queda)
                recovery_progress = (i - crash_end) / 100
                recovered_amount = 0.05 * recovery_progress  # Recupera 5% dos 10% perdidos
                new_price = (initial_price * 0.90) * (1 + recovered_amount)
            else:
                # Antes e depois: movimento normal
                change = np.random.normal(0, 0.005)
                new_price = prices[-1] * (1 + change)

            prices.append(new_price)
            timestamps.append(start_time + timedelta(minutes=i))

        # Criar OHLC
        data = []
        for i in range(1, len(prices)):
            open_price = prices[i - 1]
            close_price = prices[i]

            intra_volatility = abs(close_price - open_price) * 1.2
            high_price = max(open_price, close_price) + np.random.uniform(0, intra_volatility)
            low_price = min(open_price, close_price) - np.random.uniform(0, intra_volatility)

            data.append({
                'timestamp': timestamps[i].isoformat(),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
            })

        return pd.DataFrame(data)


class StrongTrendScenario(StressTestScenario):
    """
    Cenário 4: Tendência Forte
    Bull market prolongado com alta consistente
    """

    def __init__(self):
        super().__init__(
            name="Tendência Forte",
            description="Bull market prolongado com alta consistente de 2-3% ao dia"
        )

    def generate_data(self, n_candles: int = 1000, initial_price: float = 1000.0) -> pd.DataFrame:
        """Gera dados com tendência forte de alta"""
        prices = [initial_price]
        timestamps = []
        start_time = datetime.now() - timedelta(minutes=n_candles)

        # Tendência diária de 2-3% (assumindo 1440 candles/dia)
        daily_trend = 0.025  # 2.5% ao dia
        candle_trend = daily_trend / 1440  # Por candle (assumindo 1min)

        for i in range(n_candles):
            # Tendência de alta com pequenas correções
            if np.random.random() < 0.15:
                # 15% de chance de pequena correção
                change = -np.random.uniform(0.005, 0.015)
            else:
                # Movimento de alta com a tendência
                change = candle_trend + np.random.normal(0, 0.003)

            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
            timestamps.append(start_time + timedelta(minutes=i))

        # Criar OHLC
        data = []
        for i in range(1, len(prices)):
            open_price = prices[i - 1]
            close_price = prices[i]

            intra_volatility = abs(close_price - open_price) * 0.8
            high_price = max(open_price, close_price) + np.random.uniform(0, intra_volatility)
            low_price = min(open_price, close_price) - np.random.uniform(0, intra_volatility)

            data.append({
                'timestamp': timestamps[i].isoformat(),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
            })

        return pd.DataFrame(data)


class RangeBoundScenario(StressTestScenario):
    """
    Cenário 5: Mercado Lateral (Range-Bound)
    Preço oscila em range estreito sem direção clara
    """

    def __init__(self):
        super().__init__(
            name="Mercado Lateral",
            description="Preço oscila em range de ±2% sem direção clara"
        )

    def generate_data(self, n_candles: int = 1000, initial_price: float = 1000.0) -> pd.DataFrame:
        """Gera dados com mercado lateral"""
        prices = [initial_price]
        timestamps = []
        start_time = datetime.now() - timedelta(minutes=n_candles)

        # Range de ±2%
        range_upper = initial_price * 1.02
        range_lower = initial_price * 0.98
        range_center = initial_price

        for i in range(n_candles):
            current_price = prices[-1]

            # Mean reversion: puxar de volta para o centro
            distance_from_center = (current_price - range_center) / range_center

            # Se longe do centro, maior chance de reverter
            if abs(distance_from_center) > 0.015:
                # Reverter para o centro
                change = -distance_from_center * np.random.uniform(0.1, 0.3)
            else:
                # Movimento aleatório dentro do range
                change = np.random.normal(0, 0.005)

            new_price = current_price * (1 + change)

            # Forçar dentro do range
            new_price = max(range_lower, min(range_upper, new_price))

            prices.append(new_price)
            timestamps.append(start_time + timedelta(minutes=i))

        # Criar OHLC
        data = []
        for i in range(1, len(prices)):
            open_price = prices[i - 1]
            close_price = prices[i]

            intra_volatility = abs(close_price - open_price) * 1.0
            high_price = max(open_price, close_price) + np.random.uniform(0, intra_volatility)
            low_price = min(open_price, close_price) - np.random.uniform(0, intra_volatility)

            data.append({
                'timestamp': timestamps[i].isoformat(),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
            })

        return pd.DataFrame(data)


# Registry de cenários
STRESS_TEST_SCENARIOS = {
    'high_volatility': HighVolatilityScenario(),
    'low_volume': LowVolumeScenario(),
    'flash_crash': FlashCrashScenario(),
    'strong_trend': StrongTrendScenario(),
    'range_bound': RangeBoundScenario(),
}


def get_scenario(scenario_name: str) -> StressTestScenario:
    """
    Retorna um cenário de stress test pelo nome

    Args:
        scenario_name: Nome do cenário

    Returns:
        Instância do cenário

    Raises:
        ValueError: Se o cenário não existir
    """
    if scenario_name not in STRESS_TEST_SCENARIOS:
        available = ', '.join(STRESS_TEST_SCENARIOS.keys())
        raise ValueError(f"Cenário '{scenario_name}' não encontrado. Disponíveis: {available}")

    return STRESS_TEST_SCENARIOS[scenario_name]


def list_scenarios() -> List[Dict[str, str]]:
    """
    Lista todos os cenários disponíveis

    Returns:
        Lista de dicts com nome e descrição de cada cenário
    """
    return [
        {
            'name': name,
            'title': scenario.name,
            'description': scenario.description
        }
        for name, scenario in STRESS_TEST_SCENARIOS.items()
    ]


def run_stress_test(scenario_name: str, n_candles: int = 1000, initial_price: float = 1000.0) -> pd.DataFrame:
    """
    Executa um stress test e retorna os dados gerados

    Args:
        scenario_name: Nome do cenário
        n_candles: Número de candles a gerar
        initial_price: Preço inicial

    Returns:
        DataFrame com dados OHLC
    """
    scenario = get_scenario(scenario_name)
    logger.info(f"Executando stress test: {scenario.name}")
    logger.info(f"Descrição: {scenario.description}")

    data = scenario.generate_data(n_candles=n_candles, initial_price=initial_price)

    logger.info(f"Gerados {len(data)} candles")
    logger.info(f"Preço inicial: ${initial_price:.2f}")
    logger.info(f"Preço final: ${data.iloc[-1]['close']:.2f}")
    logger.info(f"Variação: {((data.iloc[-1]['close'] - initial_price) / initial_price * 100):+.2f}%")

    return data
