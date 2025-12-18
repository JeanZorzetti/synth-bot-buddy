"""
FASE 0.3 - ANÁLISE DE PREDIÇÕES DO MODELO ML

Este script coleta e analisa predições do modelo para entender:
1. Distribuição de confidence (percentis, histograma)
2. Predições por classe (PRICE_UP vs NO_MOVE vs PRICE_DOWN)
3. Taxa de acerto por faixa de confidence
4. Calibration curve (modelo calibrado?)

Autor: Sistema Autônomo de Análise ML
Data: 2025-12-18
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import asyncio

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ml_predictor import MLPredictor

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

# Configurar estilo dos gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_data() -> pd.DataFrame:
    """
    Carrega dados históricos para análise

    Returns:
        DataFrame com candles OHLC
    """
    logger.info("📦 Carregando dados históricos...")

    data_path = Path(__file__).parent / "output" / "fase0_eda" / "r100_30days_1min.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {data_path}\n"
            "Execute fase0_eda.py primeiro para coletar dados."
        )

    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    logger.info(f"✅ Dataset carregado: {len(df)} candles")
    logger.info(f"   Período: {df.index[0]} a {df.index[-1]}")

    return df


def collect_predictions(predictor: MLPredictor, df: pd.DataFrame, n_samples: int = 1000) -> List[Dict]:
    """
    Coleta N predições do modelo em diferentes pontos temporais

    Args:
        predictor: Modelo ML
        df: Dataset com candles
        n_samples: Número de predições a coletar

    Returns:
        Lista de predições
    """
    logger.info(f"🔮 Coletando {n_samples} predições...")

    predictions = []

    # Garantir pelo menos 200 candles para features
    min_idx = 200
    max_idx = len(df) - 1

    # Calcular índices espaçados uniformemente
    step = (max_idx - min_idx) // n_samples

    for i, idx in enumerate(range(min_idx, max_idx, step)):
        if len(predictions) >= n_samples:
            break

        # Pegar slice até o ponto atual
        df_slice = df.iloc[:idx+1].reset_index()

        try:
            # Fazer predição
            result = predictor.predict(df_slice, return_confidence=True)

            # Calcular movimento real (próximos 5 candles)
            future_idx = min(idx + 5, len(df) - 1)
            current_close = df.iloc[idx]['close']
            future_close = df.iloc[future_idx]['close']
            actual_return = (future_close - current_close) / current_close

            # Classificar movimento real
            if actual_return > 0.001:  # >0.1%
                actual_label = "PRICE_UP"
            elif actual_return < -0.001:  # <-0.1%
                actual_label = "PRICE_DOWN"
            else:
                actual_label = "NO_MOVE"

            # Armazenar predição + ground truth
            predictions.append({
                'timestamp': df.index[idx],
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'signal_strength': result['signal_strength'],
                'actual_label': actual_label,
                'actual_return': actual_return,
                'is_correct': result['prediction'] == actual_label
            })

        except Exception as e:
            logger.warning(f"Erro ao processar índice {idx}: {e}")
            continue

        # Log progresso a cada 200 predições
        if (i + 1) % 200 == 0:
            logger.info(f"   Coletadas: {len(predictions)}/{n_samples}")

    logger.info(f"✅ {len(predictions)} predições coletadas")

    return predictions


def analyze_confidence_distribution(predictions: List[Dict]) -> Dict:
    """
    Analisa distribuição de confidence

    Args:
        predictions: Lista de predições

    Returns:
        Dict com estatísticas
    """
    logger.info("📊 Analisando distribuição de confidence...")

    confidences = [p['confidence'] for p in predictions]

    stats = {
        'mean': np.mean(confidences),
        'median': np.median(confidences),
        'std': np.std(confidences),
        'min': np.min(confidences),
        'max': np.max(confidences),
        'p25': np.percentile(confidences, 25),
        'p75': np.percentile(confidences, 75),
        'p95': np.percentile(confidences, 95)
    }

    logger.info(f"✅ Confidence - Média: {stats['mean']:.3f}, Mediana: {stats['median']:.3f}")
    logger.info(f"   P25: {stats['p25']:.3f}, P75: {stats['p75']:.3f}, P95: {stats['p95']:.3f}")

    return stats


def analyze_predictions_by_class(predictions: List[Dict]) -> Dict:
    """
    Analisa predições por classe

    Args:
        predictions: Lista de predições

    Returns:
        Dict com contagens e estatísticas por classe
    """
    logger.info("🎯 Analisando predições por classe...")

    df = pd.DataFrame(predictions)

    # Contar predições por classe
    pred_counts = df['prediction'].value_counts().to_dict()

    # Confidence média por classe
    conf_by_class = df.groupby('prediction')['confidence'].mean().to_dict()

    # Taxa de acerto por classe
    accuracy_by_class = df.groupby('prediction')['is_correct'].mean().to_dict()

    logger.info(f"✅ Distribuição de predições:")
    for pred_class, count in pred_counts.items():
        pct = (count / len(predictions)) * 100
        conf_avg = conf_by_class.get(pred_class, 0)
        acc = accuracy_by_class.get(pred_class, 0) * 100
        logger.info(f"   {pred_class}: {count} ({pct:.1f}%) | Conf: {conf_avg:.3f} | Acc: {acc:.1f}%")

    return {
        'counts': pred_counts,
        'confidence_by_class': conf_by_class,
        'accuracy_by_class': accuracy_by_class
    }


def analyze_accuracy_by_confidence(predictions: List[Dict]) -> pd.DataFrame:
    """
    Analisa taxa de acerto por faixa de confidence

    Args:
        predictions: Lista de predições

    Returns:
        DataFrame com estatísticas por faixa
    """
    logger.info("📈 Analisando acurácia por faixa de confidence...")

    df = pd.DataFrame(predictions)

    # Definir faixas de confidence
    bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
    labels = ['<30%', '30-40%', '40-50%', '50-60%', '60-70%', '>70%']

    df['confidence_bin'] = pd.cut(df['confidence'], bins=bins, labels=labels)

    # Calcular estatísticas por faixa
    stats_by_bin = df.groupby('confidence_bin', observed=False).agg({
        'is_correct': ['count', 'mean'],
        'confidence': 'mean'
    }).round(3)

    stats_by_bin.columns = ['count', 'accuracy', 'avg_confidence']
    stats_by_bin['accuracy_pct'] = (stats_by_bin['accuracy'] * 100).round(1)

    logger.info(f"✅ Acurácia por faixa de confidence:")
    for idx, row in stats_by_bin.iterrows():
        logger.info(f"   {idx}: {row['count']:.0f} predições | Acc: {row['accuracy_pct']:.1f}% | Conf: {row['avg_confidence']:.3f}")

    return stats_by_bin


def create_calibration_data(predictions: List[Dict], n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cria dados para calibration curve

    Args:
        predictions: Lista de predições
        n_bins: Número de bins

    Returns:
        Tuple (prob_true, prob_pred)
    """
    logger.info("🎯 Criando calibration curve...")

    df = pd.DataFrame(predictions)

    # Converter predições para binário (PRICE_UP = 1, outros = 0)
    y_true = (df['actual_label'] == 'PRICE_UP').astype(int).values
    y_prob = df['confidence'].values

    # Calcular calibration curve
    prob_true = []
    prob_pred = []

    # Dividir em bins
    for i in range(n_bins):
        lower = i / n_bins
        upper = (i + 1) / n_bins

        # Selecionar predições neste bin
        mask = (y_prob >= lower) & (y_prob < upper)

        if mask.sum() > 0:
            prob_pred.append(y_prob[mask].mean())
            prob_true.append(y_true[mask].mean())

    logger.info(f"✅ Calibration curve criada ({len(prob_true)} bins)")

    return np.array(prob_true), np.array(prob_pred)


def generate_plots(
    predictions: List[Dict],
    conf_stats: Dict,
    class_stats: Dict,
    accuracy_by_conf: pd.DataFrame,
    calibration_data: Tuple,
    output_dir: Path
):
    """
    Gera gráficos de análise

    Args:
        predictions: Lista de predições
        conf_stats: Estatísticas de confidence
        class_stats: Estatísticas por classe
        accuracy_by_conf: Acurácia por faixa
        calibration_data: Dados de calibração
        output_dir: Diretório de saída
    """
    logger.info("📊 Gerando gráficos...")

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(predictions)

    # 1. Histograma de Confidence
    plt.figure(figsize=(12, 6))
    plt.hist(df['confidence'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(conf_stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Média: {conf_stats['mean']:.3f}")
    plt.axvline(conf_stats['median'], color='orange', linestyle='--', linewidth=2, label=f"Mediana: {conf_stats['median']:.3f}")
    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('Frequência', fontsize=12)
    plt.title('Distribuição de Confidence', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "01_confidence_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Predições por Classe
    plt.figure(figsize=(10, 6))
    counts = pd.Series(class_stats['counts'])
    bars = plt.bar(counts.index, counts.values, color=['green', 'gray', 'red'])
    plt.xlabel('Classe Predita', fontsize=12)
    plt.ylabel('Quantidade', fontsize=12)
    plt.title('Distribuição de Predições por Classe', fontsize=16, fontweight='bold')

    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(predictions)*100:.1f}%)',
                ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(plots_dir / "02_predictions_by_class.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Acurácia por Faixa de Confidence
    plt.figure(figsize=(12, 6))
    x = range(len(accuracy_by_conf))
    plt.bar(x, accuracy_by_conf['accuracy_pct'], color='steelblue', alpha=0.7)
    plt.plot(x, accuracy_by_conf['avg_confidence'] * 100, color='red', marker='o', linewidth=2, label='Confidence Média')
    plt.xticks(x, accuracy_by_conf.index, rotation=0)
    plt.xlabel('Faixa de Confidence', fontsize=12)
    plt.ylabel('Acurácia (%)', fontsize=12)
    plt.title('Acurácia por Faixa de Confidence', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "03_accuracy_by_confidence.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Calibration Curve
    prob_true, prob_pred = calibration_data
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfeitamente Calibrado')
    plt.plot(prob_pred, prob_true, 'o-', color='steelblue', linewidth=2, markersize=8, label='Modelo')
    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('True Probability', fontsize=12)
    plt.title('Calibration Curve', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(plots_dir / "04_calibration_curve.png", dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"   ✅ 4 gráficos salvos em {plots_dir}")


def generate_report(
    predictions: List[Dict],
    conf_stats: Dict,
    class_stats: Dict,
    accuracy_by_conf: pd.DataFrame,
    output_dir: Path
):
    """
    Gera relatório markdown com resultados

    Args:
        predictions: Lista de predições
        conf_stats: Estatísticas de confidence
        class_stats: Estatísticas por classe
        accuracy_by_conf: Acurácia por faixa
        output_dir: Diretório de saída
    """
    logger.info("📝 Gerando relatório markdown...")

    report_path = output_dir / "PREDICTION_ANALYSIS_REPORT.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Relatório - Análise de Predições do Modelo ML\n\n")

        # Metadata
        f.write("## Metadados\n\n")
        f.write(f"- **Data da Análise**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Total de Predições**: {len(predictions):,}\n")
        f.write(f"- **Features Utilizadas**: 65 (todas implementadas)\n")
        f.write("\n---\n\n")

        # Distribuição de Confidence
        f.write("## 📊 Distribuição de Confidence\n\n")
        f.write("| Estatística | Valor |\n")
        f.write("|-------------|-------|\n")
        f.write(f"| Média | {conf_stats['mean']:.3f} |\n")
        f.write(f"| Mediana | {conf_stats['median']:.3f} |\n")
        f.write(f"| Desvio Padrão | {conf_stats['std']:.3f} |\n")
        f.write(f"| Mínimo | {conf_stats['min']:.3f} |\n")
        f.write(f"| Máximo | {conf_stats['max']:.3f} |\n")
        f.write(f"| P25 | {conf_stats['p25']:.3f} |\n")
        f.write(f"| P75 | {conf_stats['p75']:.3f} |\n")
        f.write(f"| P95 | {conf_stats['p95']:.3f} |\n")
        f.write("\n**Interpretação**:\n")
        f.write(f"- Confidence média de **{conf_stats['mean']:.1%}** indica {'alta' if conf_stats['mean'] > 0.5 else 'baixa' if conf_stats['mean'] < 0.4 else 'moderada'} confiança\n")
        f.write(f"- 95% das predições têm confidence < {conf_stats['p95']:.1%}\n")
        f.write("\n---\n\n")

        # Predições por Classe
        f.write("## 🎯 Predições por Classe\n\n")
        f.write("| Classe | Quantidade | % Total | Confidence Média | Acurácia |\n")
        f.write("|--------|-----------|---------|------------------|----------|\n")

        total = len(predictions)
        for pred_class in ['PRICE_UP', 'NO_MOVE', 'PRICE_DOWN']:
            count = class_stats['counts'].get(pred_class, 0)
            pct = (count / total) * 100 if total > 0 else 0
            conf = class_stats['confidence_by_class'].get(pred_class, 0)
            acc = class_stats['accuracy_by_class'].get(pred_class, 0) * 100
            f.write(f"| {pred_class} | {count} | {pct:.1f}% | {conf:.3f} | {acc:.1f}% |\n")

        f.write("\n**Descobertas**:\n")

        # Analisar se modelo está desbalanceado
        most_common = max(class_stats['counts'].items(), key=lambda x: x[1])
        if most_common[1] / total > 0.7:
            f.write(f"- ⚠️ **Modelo desbalanceado**: {most_common[1]/total*100:.1f}% das predições são {most_common[0]}\n")
        else:
            f.write(f"- ✅ **Predições balanceadas**: Classes bem distribuídas\n")

        # Analisar se prevê PRICE_DOWN
        down_count = class_stats['counts'].get('PRICE_DOWN', 0)
        if down_count == 0:
            f.write(f"- ❌ **Modelo não prevê PRICE_DOWN**: Nunca identifica quedas!\n")
        elif down_count / total < 0.1:
            f.write(f"- ⚠️ **Poucas predições PRICE_DOWN**: Apenas {down_count/total*100:.1f}%\n")

        f.write("\n---\n\n")

        # Acurácia por Faixa
        f.write("## 📈 Acurácia por Faixa de Confidence\n\n")
        f.write("| Faixa | Predições | Acurácia | Confidence Média |\n")
        f.write("|-------|-----------|----------|------------------|\n")

        for idx, row in accuracy_by_conf.iterrows():
            f.write(f"| {idx} | {row['count']:.0f} | {row['accuracy_pct']:.1f}% | {row['avg_confidence']:.3f} |\n")

        f.write("\n**Análise de Calibração**:\n")

        # Verificar se modelo está calibrado
        is_calibrated = True
        for idx, row in accuracy_by_conf.iterrows():
            if row['count'] > 10:  # Apenas bins com dados suficientes
                diff = abs(row['avg_confidence'] - row['accuracy'])
                if diff > 0.15:  # >15% de diferença
                    is_calibrated = False
                    f.write(f"- ⚠️ Faixa {idx}: Confidence {row['avg_confidence']:.1%} mas Acurácia {row['accuracy']:.1%} (diff: {diff:.1%})\n")

        if is_calibrated:
            f.write("- ✅ **Modelo bem calibrado**: Confidence reflete acurácia real\n")
        else:
            f.write("- ❌ **Modelo descalibrado**: Confidence não reflete acurácia\n")

        f.write("\n---\n\n")

        # Conclusões
        f.write("## 🎯 Conclusões e Recomendações\n\n")

        # Calcular acurácia geral
        overall_acc = np.mean([p['is_correct'] for p in predictions]) * 100

        f.write(f"### Performance Geral\n\n")
        f.write(f"- **Acurácia Geral**: {overall_acc:.1f}%\n")
        f.write(f"- **Confidence Média**: {conf_stats['mean']:.1%}\n")
        f.write(f"- **Total de Predições**: {len(predictions):,}\n")
        f.write("\n### Ações Recomendadas\n\n")

        if overall_acc < 45:
            f.write("1. ❌ **Performance Baixa**: Modelo precisa re-treino urgente\n")
        elif overall_acc < 55:
            f.write("1. ⚠️ **Performance Moderada**: Otimizar threshold e features\n")
        else:
            f.write("1. ✅ **Performance Boa**: Focar em melhorias incrementais\n")

        if not is_calibrated:
            f.write("2. ❌ **Calibrar Modelo**: Usar Platt scaling ou isotonic regression\n")

        if down_count == 0:
            f.write("3. ❌ **Implementar Predição SHORT**: Modelo só prevê LONG\n")

        f.write("\n---\n\n")
        f.write(f"**Gerado em**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    logger.info(f"📄 Relatório salvo em: {report_path}")


async def main():
    """Função principal"""
    logger.info("=" * 80)
    logger.info("FASE 0.3 - ANÁLISE DE PREDIÇÕES DO MODELO ML")
    logger.info("=" * 80)

    # Diretório de saída
    output_dir = Path(__file__).parent / "output" / "fase0_prediction_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Carregar dados
        df = load_data()

        # 2. Criar predictor
        predictor = MLPredictor()

        # 3. Coletar predições
        predictions = collect_predictions(predictor, df, n_samples=1000)

        # 4. Analisar confidence
        conf_stats = analyze_confidence_distribution(predictions)

        # 5. Analisar por classe
        class_stats = analyze_predictions_by_class(predictions)

        # 6. Analisar acurácia por faixa
        accuracy_by_conf = analyze_accuracy_by_confidence(predictions)

        # 7. Criar calibration data
        calibration_data = create_calibration_data(predictions)

        # 8. Gerar gráficos
        generate_plots(predictions, conf_stats, class_stats, accuracy_by_conf, calibration_data, output_dir)

        # 9. Gerar relatório
        generate_report(predictions, conf_stats, class_stats, accuracy_by_conf, output_dir)

        logger.info("=" * 80)
        logger.info("✅ FASE 0.3 CONCLUÍDA!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ Erro na execução: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
