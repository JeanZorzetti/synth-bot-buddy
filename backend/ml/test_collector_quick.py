"""
Teste rápido do coletor de dados
Coleta apenas 1 mês de R_100 para validar funcionamento
"""

import asyncio
import logging
from data_collector import HistoricalDataCollector

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s: %(message)s'
)

async def quick_test():
    """
    Teste rápido - 1 mês de R_100
    """
    print("\n" + "="*60)
    print("TESTE RÁPIDO - Coletor de Dados Históricos")
    print("="*60)
    print("Símbolo: R_100")
    print("Timeframe: 1m")
    print("Período: 1 mês")
    print("="*60 + "\n")

    # Criar coletor
    collector = HistoricalDataCollector(
        app_id="99188",
        api_token="paE5sSemx3oANLE"
    )

    try:
        # Coletar 1 mês de dados
        df = await collector.collect_historical_candles(
            symbol='R_100',
            timeframe='1m',
            months_back=1  # Apenas 1 mês para teste rápido
        )

        # Salvar
        collector.save_to_file(df, 'R_100', '1m')

        # Mostrar resumo
        print("\n" + "="*60)
        print("RESULTADO DO TESTE")
        print("="*60)
        print(f"Total de candles: {len(df):,}")
        print(f"Início: {df['timestamp'].min()}")
        print(f"Fim: {df['timestamp'].max()}")
        print(f"Duração: {(df['timestamp'].max() - df['timestamp'].min()).days} dias")
        print("\nPrimeiras 5 linhas:")
        print(df.head())
        print("\nÚltimas 5 linhas:")
        print(df.tail())
        print("\nEstatísticas:")
        print(df[['open', 'high', 'low', 'close']].describe())

        print("\n[SUCESSO] Teste completado!")

    except Exception as e:
        print(f"\n[ERRO] Falha no teste: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(quick_test())
