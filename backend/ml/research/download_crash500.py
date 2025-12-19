"""
Download de dados históricos CRASH 500 da Deriv

CRASH 500: Índice programado para subir gradualmente e crashar a cada ~500 ticks
Perfeito para Survival Analysis (prever tempo até spike)
"""
import asyncio
from deriv_api import DerivAPI
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json

async def download_crash500_data():
    """
    Baixa 180 dias de dados M5 do CRASH 500
    """
    print("="*70)
    print("DOWNLOAD: CRASH 500 - 180 dias M5")
    print("="*70)

    # Conectar à API
    api = DerivAPI(app_id=1089)

    # Calcular timestamps
    end_time = int(datetime.now().timestamp())
    start_time = int((datetime.now() - timedelta(days=180)).timestamp())

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
                "ticks_history": "CRASH500",
                "adjust_start_time": 1,
                "count": chunk_size,
                "end": "latest" if current_time + (chunk_size * 300) > end_time else current_time + (chunk_size * 300),
                "start": current_time,
                "style": "candles",
                "granularity": 300  # 300s = 5min
            })

            if 'candles' in response:
                candles = response['candles']
                all_candles.extend(candles)

                # Atualizar timestamp
                if candles:
                    current_time = candles[-1]['epoch'] + 300
                    print(f"  Chunk {chunk_count}: {len(candles)} candles (Total: {len(all_candles)})")
                else:
                    break
            else:
                print(f"  Erro no chunk {chunk_count}: {response}")
                break

        except Exception as e:
            print(f"  Erro: {e}")
            break

    # Converter para DataFrame
    print(f"\n[PROCESS] Convertendo para DataFrame...")
    df = pd.DataFrame(all_candles)

    # Renomear colunas
    df = df.rename(columns={'epoch': 'timestamp'})

    # Ordenar por timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Salvar
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "CRASH500_5min_180days.csv"

    df.to_csv(output_file, index=False)

    print(f"\n[OK] Dataset salvo!")
    print(f"  Arquivo: {output_file}")
    print(f"  Candles: {len(df):,}")
    print(f"  Período: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Colunas: {list(df.columns)}")

    # Estatísticas
    print(f"\n[STATS] Estatísticas:")
    print(f"  Preço médio: {df['close'].mean():.2f}")
    print(f"  Range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"  Volatilidade diária: {df['close'].pct_change().std() * 100 * (24*12)**0.5:.2f}%")

    await api.clear()

    return df

async def main():
    try:
        df = await download_crash500_data()
        print(f"\n{'='*70}")
        print("DOWNLOAD COMPLETO!")
        print(f"{'='*70}")
    except Exception as e:
        print(f"\nERRO: {e}")
        print("\nNOTA: Se falhar, baixe manualmente de:")
        print("https://api.deriv.com/api-explorer#ticks_history")

if __name__ == "__main__":
    asyncio.run(main())
