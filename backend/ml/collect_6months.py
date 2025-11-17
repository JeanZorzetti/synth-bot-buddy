"""
Script para coletar 6 meses de dados históricos
"""

import asyncio
import logging
import sys
from pathlib import Path

# Adicionar path
sys.path.append(str(Path(__file__).parent))

from data_collector import HistoricalDataCollector
from training.feature_engineering import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s: %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """
    Coleta 6 meses de dados e prepara dataset ML
    """
    print("\n" + "="*60)
    print("COLETANDO 6 MESES DE DADOS HISTÓRICOS")
    print("="*60)
    print("Símbolo: R_100")
    print("Timeframe: 1 minuto")
    print("Período: 6 meses (~260,000 candles)")
    print("="*60 + "\n")

    # 1. Coletar dados
    collector = HistoricalDataCollector(
        app_id="99188",
        api_token="paE5sSemx3oANLE"
    )

    logger.info("Iniciando coleta de 6 meses...")
    df = await collector.collect_historical_candles(
        symbol='R_100',
        timeframe='1m',
        months_back=6,
        batch_size=5000
    )

    # Salvar dados brutos
    collector.save_to_file(df, 'R_100', '1m')

    print("\n" + "="*60)
    print("DADOS COLETADOS")
    print("="*60)
    print(f"Total de candles: {len(df):,}")
    print(f"Período: {df['timestamp'].min()} até {df['timestamp'].max()}")
    print(f"Duração: {(df['timestamp'].max() - df['timestamp'].min()).days} dias")
    print("="*60 + "\n")

    # 2. Preparar dataset ML
    logger.info("Preparando features para ML...")
    engineer = FeatureEngineer()

    ml_df = engineer.prepare_ml_dataset(
        df,
        prediction_horizon=15,
        price_threshold=0.3
    )

    # Salvar dataset ML
    data_dir = Path(__file__).parent / "data"
    output_file = data_dir / "ml_dataset_R100_1m_6months.pkl"
    ml_df.to_pickle(output_file)

    print("\n" + "="*60)
    print("DATASET ML PREPARADO")
    print("="*60)
    print(f"Total de amostras: {len(ml_df):,}")
    print(f"Features: {len(ml_df.columns)}")
    print(f"Target distribution:")
    print(ml_df['target'].value_counts())
    print(f"\nSalvo em: {output_file}")
    print("="*60)

    return ml_df


if __name__ == "__main__":
    asyncio.run(main())
