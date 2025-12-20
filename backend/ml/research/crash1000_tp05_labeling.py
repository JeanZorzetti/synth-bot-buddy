"""
CRASH1000 M5 - Labeling com TP 0.5% / SL 0.3%

MOTIVACAO:
- TP 2% / SL 1% e MUITO ALTO para CRASH1000 M5 (movimento 0.046%/candle)
- Features tem correlacao ~0.02 (quase zero) com TP 2%
- TP 2% leva 44 candles (220 min = 3.7 horas!)

NOVA ABORDAGEM:
- TP: 0.5% (mais realista, ~11 candles = 55 min)
- SL: 0.3% (R/R ainda 1.67:1)
- Timeout: 50 candles (250 min = 4.2 horas)

EXPECTATIVA:
- Win rate natural deve subir para ~55-60%
- Features terao maior correlacao com movimento menor
- Modelo LSTM tera mais chance de aprender padroes
"""
import sys
from pathlib import Path

# Reutilizar codigo do labeler original
from generate_labels_multi_assets import UniversalTPBeforeSLLabeler

def main():
    print("="*70)
    print("CRASH1000 M5 - TP 0.5% / SL 0.3% LABELING")
    print("="*70)

    # Carregar dados
    data_dir = Path(__file__).parent / "data"
    input_file = data_dir / "CRASH1000_M5_180days.csv"

    print(f"\n[LOAD] {input_file.name}")

    import pandas as pd
    df = pd.read_csv(input_file)
    print(f"  Candles: {len(df):,}")

    # Criar labeler com TP 0.5% / SL 0.3%
    print(f"\n[LABELING] TP 0.5% | SL 0.3% | Timeout 50 candles")

    labeler = UniversalTPBeforeSLLabeler(
        df,
        tp_pct=0.5,           # TP 0.5% (vs 2.0% anterior)
        sl_pct=0.3,           # SL 0.3% (vs 1.0% anterior)
        max_hold_candles=50,  # Timeout 50 candles (vs 20 anterior)
        slippage_pct=0.1,
        latency_candles=1,
    )

    # Gerar labels
    df_labeled = labeler.generate_labels()

    # Adicionar features
    print(f"\n[FEATURES] Adicionando features...")
    df_labeled = labeler.add_features()

    # Estatisticas
    wins = (df_labeled['tp_before_sl'] == 1).sum()
    losses = (df_labeled['tp_before_sl'] == 0).sum()
    total = wins + losses
    win_rate = wins / total * 100 if total > 0 else 0

    print(f"\n[RESULTADOS]")
    print(f"  Total labels: {total:,}")
    print(f"  WINS (TP hit): {wins:,}")
    print(f"  LOSSES (SL/timeout): {losses:,}")
    print(f"  WIN RATE NATURAL: {win_rate:.2f}%")

    # Comparacao com TP 2%
    print(f"\n[COMPARACAO]")
    print(f"  TP 2.0% / SL 1.0% / Timeout 20:")
    print(f"    Win Rate: 40.12%")
    print(f"    Tempo para TP: ~44 candles (220 min)")
    print(f"\n  TP 0.5% / SL 0.3% / Timeout 50:")
    print(f"    Win Rate: {win_rate:.2f}%")
    print(f"    Tempo para TP: ~11 candles (55 min)")

    if win_rate >= 55:
        print(f"\n  APROVADO! Win rate >= 55% permite ML aprender padroes")
    elif win_rate >= 50:
        print(f"\n  MARGINAL. Win rate ~50% e dificil para ML, mas melhor que antes")
    else:
        print(f"\n  REPROVADO. Win rate < 50% indica que mercado nao favorece scalping")

    # Salvar
    output_file = data_dir / "CRASH1000_M5_tp05_labeled.csv"
    df_labeled.to_csv(output_file, index=False)
    print(f"\n[SAVE] {output_file.name}")

    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()
