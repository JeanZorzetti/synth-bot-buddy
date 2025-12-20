"""
Analise de Feature Importance para CRASH1000

OBJETIVO: Descobrir por que modelo LSTM falha mesmo com undersampling

Hipotese: Features (OHLC, RSI, ATR, vol) NAO conseguem separar WIN de LOSS
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

def analyze_feature_importance():
    print("="*70)
    print("CRASH1000 - ANALISE DE FEATURE IMPORTANCE")
    print("="*70)

    # Carregar dados
    data_dir = Path(__file__).parent / "data"
    df = pd.read_csv(data_dir / "CRASH1000_M5_tp_before_sl_labeled.csv")

    print(f"\n[DATA] Total: {len(df):,} candles")

    # Separar WIN vs LOSS
    df_win = df[df['tp_before_sl'] == 1]
    df_loss = df[df['tp_before_sl'] == 0]

    print(f"\n  WIN: {len(df_win):,} ({len(df_win)/len(df)*100:.1f}%)")
    print(f"  LOSS: {len(df_loss):,} ({len(df_loss)/len(df)*100:.1f}%)")

    # Features a analisar
    features = ['open', 'high', 'low', 'close', 'realized_vol', 'rsi', 'atr', 'return']

    # Estatisticas descritivas
    print(f"\n{'='*70}")
    print(f"ESTATISTICAS DESCRITIVAS (WIN vs LOSS)")
    print(f"{'='*70}\n")

    results = []

    for feature in features:
        win_mean = df_win[feature].mean()
        loss_mean = df_loss[feature].mean()
        win_std = df_win[feature].std()
        loss_std = df_loss[feature].std()

        # T-test para verificar se as medias sao estatisticamente diferentes
        t_stat, p_value = ttest_ind(df_win[feature].dropna(), df_loss[feature].dropna())

        # Diferenca percentual
        diff_pct = abs(win_mean - loss_mean) / loss_mean * 100

        results.append({
            'feature': feature,
            'win_mean': win_mean,
            'loss_mean': loss_mean,
            'diff_pct': diff_pct,
            'p_value': p_value,
            'significant': 'SIM' if p_value < 0.05 else 'NAO'
        })

        print(f"{feature.upper()}")
        print(f"  WIN:  mean={win_mean:.6f}, std={win_std:.6f}")
        print(f"  LOSS: mean={loss_mean:.6f}, std={loss_std:.6f}")
        print(f"  Diff: {diff_pct:.4f}%")
        print(f"  P-value: {p_value:.6f} ({'SIGNIFICANTE' if p_value < 0.05 else 'NAO SIGNIFICANTE'})")
        print()

    # Correlacao com label
    print(f"\n{'='*70}")
    print(f"CORRELACAO COM LABEL (tp_before_sl)")
    print(f"{'='*70}\n")

    correlations = []
    for feature in features:
        corr = df[[feature, 'tp_before_sl']].corr().iloc[0, 1]
        correlations.append({
            'feature': feature,
            'correlation': corr,
            'abs_correlation': abs(corr)
        })
        print(f"  {feature}: {corr:.6f}")

    correlations_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)

    print(f"\n\nFEATURES ORDENADAS POR CORRELACAO (abs):")
    print(correlations_df.to_string(index=False))

    # Visualizacao
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()

    for idx, feature in enumerate(features):
        ax = axes[idx]

        # Boxplot WIN vs LOSS
        data_to_plot = [
            df_win[feature].dropna(),
            df_loss[feature].dropna()
        ]

        bp = ax.boxplot(data_to_plot, labels=['WIN', 'LOSS'], patch_artist=True)

        # Colorir
        bp['boxes'][0].set_facecolor('green')
        bp['boxes'][0].set_alpha(0.5)
        bp['boxes'][1].set_facecolor('red')
        bp['boxes'][1].set_alpha(0.5)

        ax.set_title(f'{feature.upper()}')
        ax.set_ylabel('Value')
        ax.grid(alpha=0.3)

        # Adicionar p-value
        p_val = [r for r in results if r['feature'] == feature][0]['p_value']
        ax.text(0.5, 0.95, f'p={p_val:.4f}', transform=ax.transAxes,
                ha='center', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='yellow' if p_val < 0.05 else 'lightgray', alpha=0.5))

    # Remover subplot extra
    fig.delaxes(axes[-1])

    plt.tight_layout()

    # Salvar
    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    plot_path = reports_dir / "crash1000_feature_importance.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n[PLOT] Salvo em: {plot_path}")

    # Conclusao
    print(f"\n{'='*70}")
    print(f"CONCLUSAO")
    print(f"{'='*70}\n")

    # Contar features significantes
    n_significant = sum(1 for r in results if r['significant'] == 'SIM')
    max_corr = correlations_df.iloc[0]['abs_correlation']

    print(f"Features estatisticamente significantes (p < 0.05): {n_significant}/{len(features)}")
    print(f"Maior correlacao (abs): {max_corr:.6f} ({correlations_df.iloc[0]['feature']})")

    if n_significant < 3 or max_corr < 0.05:
        print(f"\n  DIAGNOSTICO: FEATURES NAO TEM PODER PREDITIVO")
        print(f"  Motivos:")
        print(f"    1. Poucas features significantes (< 3)")
        print(f"    2. Correlacao muito baixa (< 0.05)")
        print(f"    3. CRASH1000 e muito aleatorio para scalping com TP 2%")
        print(f"\n  RECOMENDACAO:")
        print(f"    - Mudar parametros (TP 0.5% / SL 0.3%)")
        print(f"    - Mudar timeframe (M5 -> M1)")
        print(f"    - Desistir de ativos sinteticos (testar Forex/Indices)")
    else:
        print(f"\n  DIAGNOSTICO: FEATURES TEM ALGUM PODER PREDITIVO")
        print(f"  Modelo LSTM pode aprender padroes se:")
        print(f"    - Arquitetura for adequada")
        print(f"    - Hiperparametros forem otimizados")
        print(f"    - Treinamento for longo o suficiente")

    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    analyze_feature_importance()
