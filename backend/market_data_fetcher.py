"""
Market Data Fetcher
Busca dados históricos de mercado do Deriv API
"""

import asyncio
import json
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class MarketDataFetcher:
    """
    Classe para buscar dados históricos de mercado
    """

    def __init__(self, deriv_api=None):
        """
        Args:
            deriv_api: Instância do DerivAPI para fazer requisições
        """
        self.deriv_api = deriv_api

    async def fetch_candles(self, symbol: str, timeframe: str = '1m',
                           count: int = 500) -> pd.DataFrame:
        """
        Busca candles (OHLCV) históricos do Deriv

        Args:
            symbol: Símbolo do ativo (ex: 1HZ75V, R_100, BOOM1000)
            timeframe: Timeframe dos candles (1m, 5m, 15m, 1h, 4h, 1d)
            count: Número de candles para buscar (máximo recomendado: 5000)

        Returns:
            DataFrame com colunas: timestamp, open, high, low, close, volume
        """
        if not self.deriv_api:
            raise ValueError("DerivAPI não inicializado")

        try:
            # Mapear timeframe para granularity em segundos
            timeframe_map = {
                '1m': 60,
                '5m': 300,
                '15m': 900,
                '30m': 1800,
                '1h': 3600,
                '4h': 14400,
                '1d': 86400
            }

            granularity = timeframe_map.get(timeframe, 60)

            # Calcular timestamps
            end_time = int(datetime.now().timestamp())
            start_time = end_time - (granularity * count)

            # Requisitar candles do Deriv
            request = {
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": count,
                "end": "latest",
                "start": start_time,
                "style": "candles",
                "granularity": granularity
            }

            logger.info(f"Buscando {count} candles de {symbol} com granularidade {timeframe}")

            response = await self.deriv_api.send(request)

            if 'error' in response:
                raise Exception(f"Erro do Deriv API: {response['error']['message']}")

            # Processar resposta
            candles = response.get('candles', [])

            if not candles:
                raise Exception(f"Nenhum candle retornado para {symbol}")

            # Converter para DataFrame
            df = pd.DataFrame(candles)

            # Renomear colunas para padrão
            df = df.rename(columns={
                'epoch': 'timestamp'
            })

            # Converter timestamp para datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

            # Garantir tipos corretos
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)

            # Volume pode não estar disponível para synthetic indices
            if 'volume' not in df.columns:
                df['volume'] = 0

            logger.info(f"Sucesso: {len(df)} candles carregados para {symbol}")

            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            logger.error(f"Erro ao buscar candles de {symbol}: {e}")
            raise

    async def fetch_ticks(self, symbol: str, count: int = 1000) -> pd.DataFrame:
        """
        Busca ticks (preços individuais) do Deriv

        Args:
            symbol: Símbolo do ativo
            count: Número de ticks para buscar

        Returns:
            DataFrame com colunas: timestamp, price
        """
        if not self.deriv_api:
            raise ValueError("DerivAPI não inicializado")

        try:
            request = {
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": count,
                "end": "latest",
                "style": "ticks"
            }

            logger.info(f"Buscando {count} ticks de {symbol}")

            response = await self.deriv_api.send(request)

            if 'error' in response:
                raise Exception(f"Erro do Deriv API: {response['error']['message']}")

            # Processar ticks
            history = response.get('history', {})
            times = history.get('times', [])
            prices = history.get('prices', [])

            if not times or not prices:
                raise Exception(f"Nenhum tick retornado para {symbol}")

            # Criar DataFrame
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(times, unit='s'),
                'price': [float(p) for p in prices]
            })

            logger.info(f"Sucesso: {len(df)} ticks carregados para {symbol}")

            return df

        except Exception as e:
            logger.error(f"Erro ao buscar ticks de {symbol}: {e}")
            raise

    def candles_to_ohlc(self, ticks: pd.DataFrame, timeframe: str = '1m') -> pd.DataFrame:
        """
        Converte ticks em candles OHLC

        Args:
            ticks: DataFrame com ticks
            timeframe: Timeframe desejado (1m, 5m, 15m, etc.)

        Returns:
            DataFrame com candles OHLC
        """
        # Mapear timeframe para pandas resample frequency
        freq_map = {
            '1m': '1T',
            '5m': '5T',
            '15m': '15T',
            '30m': '30T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }

        freq = freq_map.get(timeframe, '1T')

        # Criar índice de tempo
        ticks = ticks.set_index('timestamp')

        # Resample para OHLC
        ohlc = ticks['price'].resample(freq).ohlc()
        ohlc['volume'] = ticks['price'].resample(freq).count()

        # Reset index
        ohlc = ohlc.reset_index()

        # Remover linhas sem dados
        ohlc = ohlc.dropna()

        return ohlc

    async def get_symbol_info(self, symbol: str) -> Dict:
        """
        Busca informações sobre um símbolo

        Args:
            symbol: Símbolo do ativo

        Returns:
            Dicionário com informações do símbolo
        """
        if not self.deriv_api:
            raise ValueError("DerivAPI não inicializado")

        try:
            request = {
                "active_symbols": "brief",
                "product_type": "basic"
            }

            response = await self.deriv_api.send(request)

            if 'error' in response:
                raise Exception(f"Erro do Deriv API: {response['error']['message']}")

            # Procurar símbolo
            symbols = response.get('active_symbols', [])

            for s in symbols:
                if s.get('symbol') == symbol:
                    return s

            raise Exception(f"Símbolo {symbol} não encontrado")

        except Exception as e:
            logger.error(f"Erro ao buscar info de {symbol}: {e}")
            raise


# Função auxiliar para criar DataFrame de exemplo (para testes sem API)
def create_sample_dataframe(bars: int = 500) -> pd.DataFrame:
    """
    Cria DataFrame de exemplo para testes

    Args:
        bars: Número de barras

    Returns:
        DataFrame com dados sintéticos
    """
    import numpy as np

    # Gerar dados sintéticos
    timestamps = pd.date_range(end=datetime.now(), periods=bars, freq='1min')

    # Preço base
    base_price = 100.0

    # Gerar movimento browniano
    returns = np.random.normal(0, 0.001, bars)
    prices = base_price * (1 + returns).cumprod()

    # Criar OHLC
    data = {
        'timestamp': timestamps,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.002, bars))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.002, bars))),
        'close': prices * (1 + np.random.normal(0, 0.001, bars)),
        'volume': np.random.randint(100, 1000, bars)
    }

    df = pd.DataFrame(data)

    # Ajustar high/low para incluir open/close
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    return df
