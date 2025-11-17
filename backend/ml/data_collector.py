"""
Data Collector para Machine Learning
Coleta dados históricos do Deriv API para treinamento de modelos
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import json
from pathlib import Path
from deriv_api import DerivAPI

logger = logging.getLogger(__name__)


class HistoricalDataCollector:
    """
    Coleta e armazena dados históricos do Deriv API
    para treinamento de modelos de Machine Learning
    """

    def __init__(self, app_id: str = "99188", api_token: Optional[str] = None):
        """
        Args:
            app_id: ID da aplicação Deriv
            api_token: Token de API (opcional, mas recomendado)
        """
        self.app_id = app_id
        self.api_token = api_token
        self.data_dir = Path(__file__).parent / "data"
        self.data_dir.mkdir(exist_ok=True)

    async def collect_historical_candles(
        self,
        symbol: str,
        timeframe: str = "1m",
        months_back: int = 6,
        batch_size: int = 5000
    ) -> pd.DataFrame:
        """
        Coleta dados históricos de candles do Deriv API

        Args:
            symbol: Símbolo do ativo (ex: R_100, R_75, BOOM1000)
            timeframe: Timeframe dos candles (1m, 5m, 15m, 1h, 4h, 1d)
            months_back: Quantos meses de histórico buscar
            batch_size: Máximo de candles por requisição (recomendado: 5000)

        Returns:
            DataFrame com dados históricos completos
        """
        logger.info(f"Iniciando coleta de dados históricos para {symbol}")
        logger.info(f"Timeframe: {timeframe} | Período: {months_back} meses")

        # Mapear timeframe para granularidade em segundos
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
        start_time = end_time - (months_back * 30 * 24 * 3600)  # Aproximadamente

        logger.info(f"Período: {datetime.fromtimestamp(start_time)} até {datetime.fromtimestamp(end_time)}")

        # Calcular número total de candles esperados
        total_candles_expected = (end_time - start_time) // granularity
        logger.info(f"Candles esperados: ~{total_candles_expected:,}")

        # Criar instância do Deriv API
        api = DerivAPI(app_id=int(self.app_id))

        try:
            # Conectar
            await api.connected
            logger.info("[OK] Conectado ao Deriv WebSocket")

            # Autorizar se token disponível
            if self.api_token:
                await api.authorize(self.api_token)
                logger.info("[OK] Autenticado com sucesso")

            # Coletar dados em batches
            all_candles = []
            current_end = end_time
            batch_number = 1

            while current_end > start_time:
                logger.info(f"\n>> Batch {batch_number}: Buscando até {datetime.fromtimestamp(current_end)}")

                # Requisitar batch
                response = await api.ticks_history({
                    "ticks_history": symbol,
                    "adjust_start_time": 1,
                    "count": batch_size,
                    "end": current_end,
                    "style": "candles",
                    "granularity": granularity
                })

                if 'error' in response:
                    logger.error(f"Erro do Deriv API: {response['error']}")
                    break

                candles = response.get('candles', [])

                if not candles:
                    logger.warning("Nenhum candle retornado, finalizando coleta")
                    break

                logger.info(f"   Recebidos: {len(candles)} candles")

                # Adicionar à lista
                all_candles.extend(candles)

                # Próximo batch: usar timestamp do primeiro candle como novo end
                current_end = candles[0]['epoch'] - 1
                batch_number += 1

                # Pequeno delay para não sobrecarregar API
                await asyncio.sleep(0.5)

                # Verificar se atingimos o período desejado
                if candles[0]['epoch'] <= start_time:
                    logger.info("Período completo coletado!")
                    break

            # Limpar conexão
            await api.clear()

            logger.info(f"\n[SUCESSO] Total de candles coletados: {len(all_candles):,}")

            # Converter para DataFrame
            df = pd.DataFrame(all_candles)

            # Renomear e processar colunas
            df = df.rename(columns={'epoch': 'timestamp'})
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)

            # Volume (pode não estar disponível para synthetic indices)
            if 'volume' not in df.columns:
                df['volume'] = 0

            # Ordenar por timestamp (do mais antigo para o mais recente)
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Remover duplicatas
            df = df.drop_duplicates(subset=['timestamp'])

            # Adicionar metadados
            df['symbol'] = symbol
            df['timeframe'] = timeframe

            logger.info(f"DataFrame final: {len(df)} candles")
            logger.info(f"Período: {df['timestamp'].min()} até {df['timestamp'].max()}")

            return df

        except Exception as e:
            logger.error(f"Erro ao coletar dados: {e}")
            raise

    async def collect_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = "1m",
        months_back: int = 6
    ) -> Dict[str, pd.DataFrame]:
        """
        Coleta dados históricos para múltiplos símbolos

        Args:
            symbols: Lista de símbolos (ex: ['R_100', 'R_75', 'BOOM1000'])
            timeframe: Timeframe dos candles
            months_back: Meses de histórico

        Returns:
            Dicionário {symbol: DataFrame}
        """
        results = {}

        for symbol in symbols:
            logger.info(f"\n{'='*60}")
            logger.info(f"Coletando dados para {symbol}")
            logger.info(f"{'='*60}")

            try:
                df = await self.collect_historical_candles(
                    symbol=symbol,
                    timeframe=timeframe,
                    months_back=months_back
                )
                results[symbol] = df

                # Salvar em arquivo
                self.save_to_file(df, symbol, timeframe)

            except Exception as e:
                logger.error(f"Erro ao coletar {symbol}: {e}")
                continue

            # Delay entre símbolos
            await asyncio.sleep(2)

        logger.info(f"\n[CONCLUÍDO] Coletados dados de {len(results)} símbolos")
        return results

    def save_to_file(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """
        Salva DataFrame em arquivo CSV e pickle

        Args:
            df: DataFrame com dados
            symbol: Símbolo do ativo
            timeframe: Timeframe
        """
        # Criar nome do arquivo
        filename = f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d')}"

        # Salvar CSV (human-readable)
        csv_path = self.data_dir / f"{filename}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"[SALVO] CSV: {csv_path}")

        # Salvar pickle (mais rápido para carregar)
        pkl_path = self.data_dir / f"{filename}.pkl"
        df.to_pickle(pkl_path)
        logger.info(f"[SALVO] Pickle: {pkl_path}")

        # Salvar metadados
        metadata = {
            'symbol': symbol,
            'timeframe': timeframe,
            'total_candles': len(df),
            'start_date': df['timestamp'].min().isoformat(),
            'end_date': df['timestamp'].max().isoformat(),
            'collected_at': datetime.now().isoformat()
        }

        meta_path = self.data_dir / f"{filename}_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"[SALVO] Metadata: {meta_path}")

    def load_from_file(self, symbol: str, timeframe: str, date: Optional[str] = None) -> pd.DataFrame:
        """
        Carrega dados de arquivo

        Args:
            symbol: Símbolo do ativo
            timeframe: Timeframe
            date: Data da coleta (YYYYMMDD), se None usa mais recente

        Returns:
            DataFrame com dados históricos
        """
        if date:
            filename = f"{symbol}_{timeframe}_{date}"
        else:
            # Encontrar arquivo mais recente
            pattern = f"{symbol}_{timeframe}_*.pkl"
            files = list(self.data_dir.glob(pattern))

            if not files:
                raise FileNotFoundError(f"Nenhum arquivo encontrado para {symbol} {timeframe}")

            # Pegar mais recente
            latest_file = max(files, key=lambda p: p.stat().st_mtime)
            filename = latest_file.stem

        pkl_path = self.data_dir / f"{filename}.pkl"

        if not pkl_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {pkl_path}")

        logger.info(f"[CARREGANDO] {pkl_path}")
        df = pd.read_pickle(pkl_path)
        logger.info(f"[OK] Carregados {len(df)} candles")

        return df

    def get_available_datasets(self) -> List[Dict]:
        """
        Lista todos os datasets disponíveis

        Returns:
            Lista de dicionários com metadados
        """
        datasets = []

        for meta_file in self.data_dir.glob("*_metadata.json"):
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
                datasets.append(metadata)

        return datasets


async def main():
    """
    Exemplo de uso
    """
    # Configuração
    collector = HistoricalDataCollector(
        app_id="99188",
        api_token="paE5sSemx3oANLE"  # Seu token
    )

    # Lista de símbolos para coletar
    symbols = [
        'R_100',   # Volatility 100 Index
        'R_75',    # Volatility 75 Index
        'R_50',    # Volatility 50 Index
        'BOOM1000',  # Boom 1000 Index
        'CRASH1000'  # Crash 1000 Index
    ]

    # Coletar 6 meses de dados (1 minuto)
    results = await collector.collect_multiple_symbols(
        symbols=symbols,
        timeframe='1m',
        months_back=6
    )

    # Mostrar resumo
    print("\n" + "="*60)
    print("RESUMO DA COLETA")
    print("="*60)

    for symbol, df in results.items():
        print(f"\n{symbol}:")
        print(f"  Candles: {len(df):,}")
        print(f"  Período: {df['timestamp'].min()} até {df['timestamp'].max()}")
        print(f"  Duração: {(df['timestamp'].max() - df['timestamp'].min()).days} dias")

    # Listar datasets salvos
    print("\n" + "="*60)
    print("DATASETS DISPONÍVEIS")
    print("="*60)

    datasets = collector.get_available_datasets()
    for ds in datasets:
        print(f"\n{ds['symbol']} ({ds['timeframe']}):")
        print(f"  Candles: {ds['total_candles']:,}")
        print(f"  Período: {ds['start_date'][:10]} até {ds['end_date'][:10]}")
        print(f"  Coletado: {ds['collected_at'][:10]}")


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s: %(message)s'
    )

    # Executar coleta
    asyncio.run(main())
