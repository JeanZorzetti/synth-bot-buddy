"""
Async Market Analyzer
Permite análise de múltiplos símbolos simultaneamente usando asyncio
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd

from analysis import TechnicalAnalysis
from ml_predictor import get_ml_predictor
from metrics import get_metrics_manager

logger = logging.getLogger(__name__)


class AsyncMarketAnalyzer:
    """
    Analisador de mercado assíncrono
    Processa múltiplos símbolos em paralelo
    """

    def __init__(self):
        self.ta = TechnicalAnalysis()
        self.ml_predictor = get_ml_predictor()
        self.metrics = get_metrics_manager()

    async def analyze_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
        use_ml: bool = True
    ) -> Dict:
        """
        Analisa um símbolo de forma assíncrona

        Args:
            symbol: Símbolo do ativo
            df: DataFrame com OHLC data
            use_ml: Se deve usar ML predictor

        Returns:
            Dict com análise completa
        """
        start_time = time.time()

        try:
            # Executar análise técnica em thread pool (para não bloquear event loop)
            loop = asyncio.get_event_loop()

            # Análise técnica
            ta_signal = await loop.run_in_executor(
                None,
                self.ta.generate_signal,
                df,
                symbol
            )

            result = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'technical_analysis': ta_signal.to_dict(),
                'ml_prediction': None
            }

            # ML prediction (se habilitado)
            if use_ml and self.ml_predictor:
                ml_result = await loop.run_in_executor(
                    None,
                    self.ml_predictor.predict,
                    df
                )
                result['ml_prediction'] = ml_result

            # Combinar sinais (TA + ML)
            if use_ml and ml_result:
                # Pegar sinal ML
                ml_signal = ml_result['signal']
                ml_confidence = ml_result['confidence']

                # Combinar com TA (dar mais peso ao ML se confiança alta)
                if ml_confidence >= 70:
                    result['final_signal'] = ml_signal
                    result['final_confidence'] = ml_confidence
                    result['signal_source'] = 'ml_high_confidence'
                else:
                    # Usar TA se ML tem baixa confiança
                    result['final_signal'] = ta_signal.signal_type
                    result['final_confidence'] = ta_signal.confidence
                    result['signal_source'] = 'technical_analysis'
            else:
                result['final_signal'] = ta_signal.signal_type
                result['final_confidence'] = ta_signal.confidence
                result['signal_source'] = 'technical_analysis_only'

            # Calcular latência
            latency_ms = (time.time() - start_time) * 1000
            result['latency_ms'] = round(latency_ms, 2)

            # Registrar métricas
            self.metrics.record_signal(
                signal_type=result['final_signal'],
                confidence=result['final_confidence'],
                latency_ms=latency_ms
            )

            logger.info(f"[{symbol}] Análise completa: {result['final_signal']} ({result['final_confidence']:.2f}%) em {latency_ms:.2f}ms")

            return result

        except Exception as e:
            logger.error(f"[{symbol}] Erro na análise: {e}")
            self.metrics.record_error('analysis', 'error')

            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'final_signal': 'NEUTRAL',
                'final_confidence': 0,
                'signal_source': 'error'
            }

    async def analyze_multiple_symbols(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        use_ml: bool = True,
        max_concurrent: int = 10
    ) -> Dict[str, Dict]:
        """
        Analisa múltiplos símbolos simultaneamente

        Args:
            symbols_data: Dict com {symbol: DataFrame}
            use_ml: Se deve usar ML predictor
            max_concurrent: Número máximo de análises simultâneas

        Returns:
            Dict com {symbol: resultado_análise}
        """
        logger.info(f"Iniciando análise assíncrona de {len(symbols_data)} símbolos...")
        start_time = time.time()

        # Criar semáforo para limitar concorrência
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_with_semaphore(symbol: str, df: pd.DataFrame):
            """Wrapper para análise com semáforo"""
            async with semaphore:
                return symbol, await self.analyze_symbol(symbol, df, use_ml)

        # Criar tasks para todos os símbolos
        tasks = [
            analyze_with_semaphore(symbol, df)
            for symbol, df in symbols_data.items()
        ]

        # Executar todas as análises em paralelo
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Processar resultados
        analysis_results = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Erro em análise: {result}")
                continue

            symbol, analysis = result
            analysis_results[symbol] = analysis

        # Calcular estatísticas
        elapsed = time.time() - start_time
        successful = len([r for r in analysis_results.values() if 'error' not in r])
        failed = len(analysis_results) - successful

        logger.info(f"Análise assíncrona completa:")
        logger.info(f"  Total: {len(symbols_data)} símbolos")
        logger.info(f"  Sucesso: {successful}")
        logger.info(f"  Falhas: {failed}")
        logger.info(f"  Tempo total: {elapsed:.2f}s")
        logger.info(f"  Tempo médio: {elapsed/len(symbols_data):.2f}s por símbolo")
        logger.info(f"  Throughput: {len(symbols_data)/elapsed:.2f} símbolos/s")

        return analysis_results

    async def analyze_symbols_batch(
        self,
        symbols: List[str],
        fetch_data_fn,
        use_ml: bool = True,
        batch_size: int = 10
    ) -> Dict[str, Dict]:
        """
        Analisa símbolos em batches com fetching assíncrono de dados

        Args:
            symbols: Lista de símbolos
            fetch_data_fn: Função async para buscar dados (symbol) -> DataFrame
            use_ml: Se deve usar ML
            batch_size: Tamanho do batch

        Returns:
            Dict com resultados
        """
        logger.info(f"Iniciando análise em batches de {len(symbols)} símbolos...")

        all_results = {}

        # Processar em batches
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            logger.info(f"Processando batch {i//batch_size + 1}: {batch}")

            # Fetch dados em paralelo
            fetch_tasks = [fetch_data_fn(symbol) for symbol in batch]
            dataframes = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            # Criar dict de símbolos com dados válidos
            batch_data = {}
            for symbol, df in zip(batch, dataframes):
                if not isinstance(df, Exception) and df is not None:
                    batch_data[symbol] = df
                else:
                    logger.error(f"[{symbol}] Erro ao buscar dados: {df}")

            # Analisar batch
            if batch_data:
                batch_results = await self.analyze_multiple_symbols(
                    batch_data,
                    use_ml=use_ml,
                    max_concurrent=batch_size
                )
                all_results.update(batch_results)

        return all_results


# Instância global
_async_analyzer: Optional[AsyncMarketAnalyzer] = None


def get_async_analyzer() -> AsyncMarketAnalyzer:
    """
    Retorna instância global do async analyzer (singleton)

    Returns:
        AsyncMarketAnalyzer instance
    """
    global _async_analyzer
    if _async_analyzer is None:
        _async_analyzer = AsyncMarketAnalyzer()
    return _async_analyzer


if __name__ == "__main__":
    # Teste do async analyzer
    import numpy as np

    logging.basicConfig(level=logging.INFO)

    # Gerar dados sintéticos para teste
    def generate_test_data(symbol: str, n_candles: int = 300) -> pd.DataFrame:
        """Gera dados sintéticos para teste"""
        np.random.seed(hash(symbol) % 2**32)
        prices = 100 * (1 + np.random.randn(n_candles) * 0.02).cumprod()

        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_candles, freq='5min'),
            'open': prices + np.random.randn(n_candles) * 0.5,
            'high': prices + np.abs(np.random.randn(n_candles)) * 1.0,
            'low': prices - np.abs(np.random.randn(n_candles)) * 1.0,
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_candles)
        })

    async def test():
        """Teste assíncrono"""
        analyzer = AsyncMarketAnalyzer()

        # Testar análise de múltiplos símbolos
        symbols = ['R_100', 'R_75', 'R_50', 'R_25', 'VOLATILITY_100']
        symbols_data = {
            symbol: generate_test_data(symbol)
            for symbol in symbols
        }

        results = await analyzer.analyze_multiple_symbols(
            symbols_data,
            use_ml=False,  # Desabilitar ML para teste rápido
            max_concurrent=5
        )

        print("\n" + "="*60)
        print("RESULTADOS DA ANÁLISE ASSÍNCRONA")
        print("="*60)

        for symbol, result in results.items():
            print(f"\n{symbol}:")
            print(f"  Sinal: {result['final_signal']}")
            print(f"  Confiança: {result['final_confidence']:.2f}%")
            print(f"  Latência: {result['latency_ms']:.2f}ms")
            print(f"  Fonte: {result['signal_source']}")

        print("\n[OK] Teste concluído!")

    # Executar teste
    asyncio.run(test())
