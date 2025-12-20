"""
Download de dados históricos de ativos sintéticos da Deriv

Ativos suportados:
- CRASH500: Sobe gradualmente e crasha a cada ~500 ticks
- CRASH1000: Sobe gradualmente e crasha a cada ~1000 ticks (mais lento)
- BOOM500: Desce gradualmente e explode a cada ~500 ticks
- BOOM1000: Desce gradualmente e explode a cada ~1000 ticks (mais lento)

Timeframes: M1 (60s), M5 (300s), M15 (900s), H1 (3600s)
"""
import asyncio
from deriv_api import DerivAPI
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Mapeamento de símbolos e granularidades
SYMBOLS = {
    'CRASH500': 'CRASH500',
    'CRASH1000': 'CRASH1000',
    'BOOM500': 'BOOM500',
    'BOOM1000': 'BOOM1000',
}

TIMEFRAMES = {
    'M1': 60,
    'M5': 300,
    'M15': 900,
    'H1': 3600,
}

async def download_asset_data(symbol, timeframe='M5', days=180):
    """
    Baixa dados históricos de um ativo sintético

    Args:
        symbol: Nome do ativo (CRASH500, CRASH1000, BOOM500, BOOM1000)
        timeframe: Timeframe (M1, M5, M15, H1)
        days: Número de dias históricos (padrão 180)

    Returns:
        DataFrame com OHLC + timestamp
    """
    if symbol not in SYMBOLS:
        raise ValueError(f"Símbolo inválido: {symbol}. Use: {list(SYMBOLS.keys())}")

    if timeframe not in TIMEFRAMES:
        raise ValueError(f"Timeframe inválido: {timeframe}. Use: {list(TIMEFRAMES.keys())}")

    granularity = TIMEFRAMES[timeframe]

    print("="*70)
    print(f"DOWNLOAD: {symbol} - {days} dias {timeframe}")
    print("="*70)

    # Conectar à API
    api = DerivAPI(app_id=1089)

    # Calcular timestamps
    end_time = int(datetime.now().timestamp())
    start_time = int((datetime.now() - timedelta(days=days)).timestamp())

    print(f"\nPeríodo:")
    print(f"  Início: {datetime.fromtimestamp(start_time)}")
    print(f"  Fim: {datetime.fromtimestamp(end_time)}")

    # Baixar dados em chunks (API limita a 5000 candles por request)
    all_candles = []
    current_time = start_time
    chunk_size = 5000

    print(f"\n[DOWNLOAD] Baixando em chunks de {chunk_size} candles...")

    chunk_count = 0
    while current_time < end_time:
        chunk_count += 1

        try:
            # Request de candles
            response = await api.ticks_history({
                "ticks_history": SYMBOLS[symbol],
                "adjust_start_time": 1,
                "count": chunk_size,
                "end": "latest" if current_time + (chunk_size * granularity) > end_time else current_time + (chunk_size * granularity),
                "start": current_time,
                "style": "candles",
                "granularity": granularity
            })

            if 'candles' in response:
                candles = response['candles']
                all_candles.extend(candles)

                # Atualizar timestamp
                if candles:
                    current_time = candles[-1]['epoch'] + granularity
                    print(f"  Chunk {chunk_count}: {len(candles)} candles (Total: {len(all_candles)})")
                else:
                    break
            else:
                print(f"  Erro no chunk {chunk_count}: {response}")
                break

        except Exception as e:
            print(f"  Erro: {e}")
            break

        # Delay para não sobrecarregar API
        await asyncio.sleep(0.5)

    # Converter para DataFrame
    print(f"\n[PROCESS] Convertendo para DataFrame...")
    df = pd.DataFrame(all_candles)

    if df.empty:
        print(f"\n[ERRO] Nenhum dado foi baixado!")
        await api.clear()
        return None

    # Renomear colunas
    df = df.rename(columns={'epoch': 'timestamp'})

    # Ordenar por timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Salvar
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{symbol}_{timeframe}_{days}days.csv"

    df.to_csv(output_file, index=False)

    print(f"\n[OK] Dataset salvo!")
    print(f"  Arquivo: {output_file}")
    print(f"  Candles: {len(df):,}")
    print(f"  Período: {datetime.fromtimestamp(df['timestamp'].min())} -> {datetime.fromtimestamp(df['timestamp'].max())}")
    print(f"  Colunas: {list(df.columns)}")

    # Estatísticas
    print(f"\n[STATS] Estatísticas:")
    print(f"  Preço médio: {df['close'].mean():.2f}")
    print(f"  Range: {df['close'].min():.2f} - {df['close'].max():.2f}")

    # Movimento médio por candle
    df['pct_change'] = df['close'].pct_change() * 100
    avg_move = df['pct_change'].abs().mean()
    print(f"  Movimento médio/candle: {avg_move:.4f}%")

    # Volatilidade (std diária)
    candles_per_day = (24 * 3600) / granularity
    daily_vol = df['pct_change'].std() * (candles_per_day ** 0.5)
    print(f"  Volatilidade diária: {daily_vol:.2f}%")

    await api.clear()

    return df


async def main():
    """
    Download múltiplos ativos para comparação
    """
    # Lista de ativos a baixar
    assets_to_download = [
        ('BOOM1000', 'M5', 180),
        ('CRASH1000', 'M5', 180),
    ]

    results = []

    for symbol, timeframe, days in assets_to_download:
        try:
            print(f"\n\n")
            df = await download_asset_data(symbol, timeframe, days)

            if df is not None:
                results.append({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'candles': len(df),
                    'avg_move_pct': df['close'].pct_change().abs().mean() * 100,
                    'daily_vol': df['close'].pct_change().std() * ((24*3600)/TIMEFRAMES[timeframe])**0.5 * 100,
                })

        except Exception as e:
            print(f"\nERRO ao baixar {symbol}: {e}")

    # Relatório final
    if results:
        print(f"\n\n{'='*70}")
        print("RESUMO - COMPARAÇÃO DE ATIVOS")
        print(f"{'='*70}\n")

        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))

        print(f"\n\nMELHOR PARA SCALPING:")
        best = results_df.loc[results_df['avg_move_pct'].idxmax()]
        print(f"  {best['symbol']} - Movimento médio: {best['avg_move_pct']:.4f}%/candle")

    print(f"\n{'='*70}")
    print("DOWNLOAD COMPLETO!")
    print(f"{'='*70}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Uso: python download_synthetic_assets.py CRASH1000 M5 180
        symbol = sys.argv[1]
        timeframe = sys.argv[2] if len(sys.argv) > 2 else 'M5'
        days = int(sys.argv[3]) if len(sys.argv) > 3 else 180

        asyncio.run(download_asset_data(symbol, timeframe, days))
    else:
        # Download múltiplos ativos
        asyncio.run(main())
