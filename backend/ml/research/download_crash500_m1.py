"""
Download de dados históricos CRASH 500 M1 (1 minuto) da Deriv

OBJETIVO: Testar se M1 tem melhor performance que M5 para scalping

DIFERENÇA M1 vs M5:
- M1: 20 candles = 20 minutos (menos timeout)
- M5: 20 candles = 100 minutos (muito timeout)

EXPECTATIVA:
- Win rate natural maior em M1 (mais chances de atingir TP)
- Modelo consegue aprender padrões
"""
import asyncio
from deriv_api import DerivAPI
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json

async def download_crash500_m1_data(days=90):
    """
    Baixa dados M1 do CRASH 500

    Args:
        days: Quantidade de dias (default 90, pois M1 gera 5x mais dados que M5)
    """
    print("="*70)
    print(f"DOWNLOAD: CRASH 500 - {days} dias M1 (1 minuto)")
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
    print(f"  (M1 gera ~1440 candles/dia, esperamos ~{days * 1440:,} candles)")

    chunk_count = 0
    while current_time < end_time:
        chunk_count += 1

        try:
            # Request de candles
            response = await api.ticks_history({
                "ticks_history": "CRASH500",
                "adjust_start_time": 1,
                "count": chunk_size,
                "end": "latest" if current_time + (chunk_size * 60) > end_time else current_time + (chunk_size * 60),
                "start": current_time,
                "style": "candles",
                "granularity": 60  # 60s = 1min
            })

            if 'candles' in response:
                candles = response['candles']
                all_candles.extend(candles)

                # Atualizar timestamp
                if candles:
                    current_time = candles[-1]['epoch'] + 60
                    print(f"  Chunk {chunk_count}: {len(candles)} candles (Total: {len(all_candles):,})")
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
    output_file = output_dir / f"CRASH500_1min_{days}days.csv"

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
    df['return'] = df['close'].pct_change()
    avg_move = df['return'].abs().mean() * 100
    print(f"  Movimento médio por candle: {avg_move:.3f}%")
    print(f"  Candles para atingir TP 2%: ~{2.0 / avg_move:.0f} candles")

    # Volatilidade
    vol_daily = df['return'].std() * 100 * (24*60)**0.5
    print(f"  Volatilidade diária: {vol_daily:.2f}%")

    await api.clear()

    return df

async def main():
    try:
        df = await download_crash500_m1_data(days=90)

        print(f"\n{'='*70}")
        print("DOWNLOAD COMPLETO!")
        print(f"{'='*70}")

        print(f"\n[COMPARAÇÃO M1 vs M5]")
        print(f"  M1 (1 min):")
        print(f"    - Timeout 20 candles = 20 minutos")
        print(f"    - Movimento mais rápido")
        print(f"    - Mais chances de atingir TP")

        print(f"\n  M5 (5 min):")
        print(f"    - Timeout 20 candles = 100 minutos")
        print(f"    - Movimento lento")
        print(f"    - 95% dos trades fecham por timeout")

    except Exception as e:
        print(f"\nERRO: {e}")
        print("\nNOTA: Se falhar, baixe manualmente de:")
        print("https://api.deriv.com/api-explorer#ticks_history")

if __name__ == "__main__":
    asyncio.run(main())
