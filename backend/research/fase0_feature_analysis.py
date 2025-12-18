"""
FASE 0.2 - ANÁLISE DE FEATURES DO MODELO ML

Este script analisa as 65 features do modelo XGBoost para identificar:
1. Top 20 features mais importantes (SHAP values)
2. Features redundantes (correlação > 0.9)
3. Features com missing values
4. Distribuição de cada feature

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
import shap
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


async def load_model_and_data() -> Tuple[MLPredictor, pd.DataFrame]:
    """
    Carrega modelo XGBoost e prepara dataset de teste

    Returns:
        Tuple[MLPredictor, pd.DataFrame]: Modelo e dataset
    """
    logger.info("📦 Carregando modelo XGBoost...")

    predictor = MLPredictor()

    # Carregar dados históricos coletados na Fase 0.1
    data_path = Path(__file__).parent / "output" / "fase0_eda" / "r100_30days_1min.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {data_path}\n"
            "Execute fase0_eda.py primeiro para coletar dados."
        )

    logger.info(f"📂 Carregando dataset: {data_path}")
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    logger.info(f"✅ Dataset carregado: {len(df)} candles")
    logger.info(f"   Período: {df.index[0]} a {df.index[-1]}")

    return predictor, df


def calculate_features_for_analysis(predictor: MLPredictor, df: pd.DataFrame, sample_size: int = 5000) -> pd.DataFrame:
    """
    Calcula features do modelo para análise

    Args:
        predictor: Modelo ML
        df: Dataset com candles OHLC
        sample_size: Número de amostras para análise (default: 5000)

    Returns:
        DataFrame com features calculadas
    """
    logger.info(f"🔧 Calculando features para {sample_size} amostras...")

    # Pegar últimas N amostras (mais recentes)
    df_sample = df.tail(sample_size + 200).copy()  # +200 para ter histórico suficiente

    # Lista para armazenar features
    features_list = []

    # Calcular features para cada candle
    for i in range(200, len(df_sample)):  # Começar após 200 candles de warm-up
        candle_slice = df_sample.iloc[:i+1]

        # Usar método interno do predictor para calcular features
        try:
            # Chamar _calculate_features do predictor
            features = predictor._calculate_features(candle_slice)

            if features is not None and not features.empty:
                # Pegar última linha (features do candle atual)
                feature_row = features.iloc[-1].to_dict()
                feature_row['timestamp'] = df_sample.index[i]
                features_list.append(feature_row)
        except Exception as e:
            logger.warning(f"Erro ao calcular features no índice {i}: {e}")
            continue

        # Log progresso a cada 1000 candles
        if (i - 200) % 1000 == 0:
            logger.info(f"   Processados: {i-200}/{sample_size}")

    # Converter para DataFrame
    features_df = pd.DataFrame(features_list)
    features_df = features_df.set_index('timestamp')

    logger.info(f"✅ Features calculadas: {len(features_df)} amostras, {len(features_df.columns)} features")

    return features_df


def analyze_shap_importance(predictor: MLPredictor, features_df: pd.DataFrame, top_n: int = 20) -> Dict:
    """
    Calcula SHAP values para identificar features mais importantes

    Args:
        predictor: Modelo ML
        features_df: DataFrame com features
        top_n: Número de top features (default: 20)

    Returns:
        Dict com análise de importância
    """
    logger.info(f"📊 Calculando SHAP values (top {top_n})...")

    # Remover colunas não-feature (target, etc)
    X = features_df.copy()

    # Lidar com missing values
    X = X.fillna(0)

    # Criar explainer SHAP
    logger.info("   Criando TreeExplainer...")
    explainer = shap.TreeExplainer(predictor.model)

    # Calcular SHAP values (usar sample se dataset muito grande)
    sample_size = min(1000, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)

    logger.info(f"   Calculando SHAP values para {sample_size} amostras...")
    shap_values = explainer.shap_values(X_sample)

    # Se modelo retorna múltiplas classes, pegar classe PRICE_UP (classe 1)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Classe PRICE_UP

    # Calcular importância absoluta média
    feature_importance = np.abs(shap_values).mean(axis=0)

    # Criar DataFrame de importância
    importance_df = pd.DataFrame({
        'feature': X_sample.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    # Top N features
    top_features = importance_df.head(top_n)

    logger.info(f"✅ Top {top_n} features mais importantes:")
    for idx, row in top_features.iterrows():
        logger.info(f"   {row['feature']}: {row['importance']:.6f}")

    return {
        'importance_df': importance_df,
        'top_features': top_features,
        'shap_values': shap_values,
        'X_sample': X_sample
    }


def analyze_feature_correlation(features_df: pd.DataFrame, threshold: float = 0.9) -> Dict:
    """
    Identifica features redundantes com correlação > threshold

    Args:
        features_df: DataFrame com features
        threshold: Limiar de correlação (default: 0.9)

    Returns:
        Dict com análise de correlação
    """
    logger.info(f"🔗 Analisando correlação entre features (threshold: {threshold})...")

    # Remover missing values
    X = features_df.fillna(0)

    # Calcular matriz de correlação
    corr_matrix = X.corr()

    # Encontrar pares com correlação > threshold
    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = abs(corr_matrix.iloc[i, j])

            if corr_value > threshold:
                high_corr_pairs.append({
                    'feature_1': corr_matrix.columns[i],
                    'feature_2': corr_matrix.columns[j],
                    'correlation': corr_value
                })

    # Ordenar por correlação
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', ascending=False)

    logger.info(f"✅ Encontrados {len(high_corr_df)} pares com correlação > {threshold}")

    if len(high_corr_df) > 0:
        logger.info("   Top 10 pares mais correlacionados:")
        for idx, row in high_corr_df.head(10).iterrows():
            logger.info(f"   {row['feature_1']} <-> {row['feature_2']}: {row['correlation']:.4f}")

    return {
        'corr_matrix': corr_matrix,
        'high_corr_pairs': high_corr_df,
        'redundant_features': high_corr_df['feature_2'].unique().tolist() if len(high_corr_df) > 0 else []
    }


def analyze_missing_values(features_df: pd.DataFrame) -> Dict:
    """
    Analisa features com missing values (NaN)

    Args:
        features_df: DataFrame com features

    Returns:
        Dict com análise de missing values
    """
    logger.info("❓ Analisando missing values...")

    # Calcular % de missing por feature
    missing_pct = (features_df.isnull().sum() / len(features_df) * 100).sort_values(ascending=False)

    # Filtrar apenas features com missing > 0%
    features_with_missing = missing_pct[missing_pct > 0]

    logger.info(f"✅ {len(features_with_missing)} features com missing values:")

    if len(features_with_missing) > 0:
        for feature, pct in features_with_missing.items():
            logger.info(f"   {feature}: {pct:.2f}%")
    else:
        logger.info("   ✅ Nenhuma feature com missing values!")

    return {
        'missing_pct': missing_pct,
        'features_with_missing': features_with_missing
    }


def generate_plots(shap_analysis: Dict, corr_analysis: Dict, output_dir: Path):
    """
    Gera gráficos de análise

    Args:
        shap_analysis: Resultados da análise SHAP
        corr_analysis: Resultados da análise de correlação
        output_dir: Diretório de saída
    """
    logger.info("📊 Gerando gráficos...")

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. SHAP Summary Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_analysis['shap_values'],
        shap_analysis['X_sample'],
        max_display=20,
        show=False
    )
    plt.title('Top 20 Features - SHAP Importance', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / "01_shap_importance.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Feature Importance Bar Chart
    plt.figure(figsize=(12, 8))
    top_20 = shap_analysis['top_features'].head(20)
    plt.barh(range(len(top_20)), top_20['importance'], color='steelblue')
    plt.yticks(range(len(top_20)), top_20['feature'])
    plt.xlabel('SHAP Importance (mean |SHAP value|)', fontsize=12)
    plt.title('Top 20 Features - Importance Ranking', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(plots_dir / "02_importance_ranking.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Correlation Heatmap (Top 30 features)
    plt.figure(figsize=(14, 12))
    top_30_features = shap_analysis['top_features'].head(30)['feature'].tolist()
    corr_subset = corr_analysis['corr_matrix'].loc[top_30_features, top_30_features]

    sns.heatmap(
        corr_subset,
        annot=False,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Correlation'}
    )
    plt.title('Correlation Heatmap - Top 30 Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir / "03_correlation_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"   ✅ 3 gráficos salvos em {plots_dir}")


def generate_report(
    shap_analysis: Dict,
    corr_analysis: Dict,
    missing_analysis: Dict,
    features_df: pd.DataFrame,
    output_dir: Path
):
    """
    Gera relatório markdown com resultados

    Args:
        shap_analysis: Resultados SHAP
        corr_analysis: Resultados correlação
        missing_analysis: Resultados missing values
        features_df: DataFrame com features
        output_dir: Diretório de saída
    """
    logger.info("📝 Gerando relatório markdown...")

    report_path = output_dir / "FEATURE_ANALYSIS_REPORT.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Relatório - Análise de Features do Modelo ML\n\n")

        # Metadata
        f.write("## Metadados\n\n")
        f.write(f"- **Data da Análise**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Total de Features**: {len(features_df.columns)}\n")
        f.write(f"- **Amostras Analisadas**: {len(features_df):,}\n")
        f.write("\n---\n\n")

        # Top 20 Features
        f.write("## 📊 Top 20 Features Mais Importantes (SHAP)\n\n")
        f.write("| Rank | Feature | SHAP Importance |\n")
        f.write("|------|---------|----------------|\n")

        for idx, (_, row) in enumerate(shap_analysis['top_features'].head(20).iterrows(), 1):
            f.write(f"| {idx} | {row['feature']} | {row['importance']:.6f} |\n")

        f.write("\n**Interpretação**:\n")
        f.write("- Features no topo têm maior impacto nas predições do modelo\n")
        f.write("- SHAP value mede contribuição média de cada feature\n")
        f.write("\n---\n\n")

        # Features Redundantes
        f.write("## 🔗 Features Redundantes (Correlação > 0.9)\n\n")

        if len(corr_analysis['high_corr_pairs']) > 0:
            f.write(f"**Total de Pares**: {len(corr_analysis['high_corr_pairs'])}\n\n")
            f.write("| Feature 1 | Feature 2 | Correlação |\n")
            f.write("|-----------|-----------|------------|\n")

            for _, row in corr_analysis['high_corr_pairs'].head(20).iterrows():
                f.write(f"| {row['feature_1']} | {row['feature_2']} | {row['correlation']:.4f} |\n")

            f.write("\n**Recomendação**:\n")
            f.write(f"- Remover {len(corr_analysis['redundant_features'])} features redundantes:\n")
            for feat in corr_analysis['redundant_features'][:10]:
                f.write(f"  - `{feat}`\n")
        else:
            f.write("✅ **Nenhuma feature redundante encontrada!**\n")

        f.write("\n---\n\n")

        # Missing Values
        f.write("## ❓ Features com Missing Values\n\n")

        if len(missing_analysis['features_with_missing']) > 0:
            f.write(f"**Total**: {len(missing_analysis['features_with_missing'])} features\n\n")
            f.write("| Feature | Missing (%) |\n")
            f.write("|---------|-------------|\n")

            for feature, pct in missing_analysis['features_with_missing'].items():
                f.write(f"| {feature} | {pct:.2f}% |\n")

            f.write("\n**Recomendação**:\n")
            f.write("- Features com >10% missing: remover ou imputar\n")
            f.write("- Features com <10% missing: forward fill ou mean imputation\n")
        else:
            f.write("✅ **Nenhuma feature com missing values!**\n")

        f.write("\n---\n\n")

        # Conclusões
        f.write("## 🎯 Conclusões e Recomendações\n\n")

        f.write("### Ações Imediatas\n\n")
        f.write("1. **Remover Features Redundantes**:\n")
        if len(corr_analysis['redundant_features']) > 0:
            f.write(f"   - {len(corr_analysis['redundant_features'])} features com correlação > 0.9\n")
        else:
            f.write("   - ✅ Nenhuma remoção necessária\n")

        f.write("\n2. **Tratar Missing Values**:\n")
        if len(missing_analysis['features_with_missing']) > 0:
            f.write(f"   - {len(missing_analysis['features_with_missing'])} features precisam tratamento\n")
        else:
            f.write("   - ✅ Nenhum tratamento necessário\n")

        f.write("\n3. **Focar nas Top Features**:\n")
        top_5 = shap_analysis['top_features'].head(5)['feature'].tolist()
        f.write("   - Otimizar hiperparâmetros das top 5:\n")
        for feat in top_5:
            f.write(f"     - `{feat}`\n")

        f.write("\n---\n\n")
        f.write(f"**Gerado em**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    logger.info(f"📄 Relatório salvo em: {report_path}")


async def main():
    """Função principal"""
    logger.info("=" * 80)
    logger.info("FASE 0.2 - ANÁLISE DE FEATURES DO MODELO ML")
    logger.info("=" * 80)

    # Diretório de saída
    output_dir = Path(__file__).parent / "output" / "fase0_feature_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Carregar modelo e dados
        predictor, df = await load_model_and_data()

        # 2. Calcular features
        features_df = calculate_features_for_analysis(predictor, df, sample_size=5000)

        # 3. Análise SHAP
        shap_analysis = analyze_shap_importance(predictor, features_df, top_n=20)

        # 4. Análise de correlação
        corr_analysis = analyze_feature_correlation(features_df, threshold=0.9)

        # 5. Análise de missing values
        missing_analysis = analyze_missing_values(features_df)

        # 6. Gerar gráficos
        generate_plots(shap_analysis, corr_analysis, output_dir)

        # 7. Gerar relatório
        generate_report(shap_analysis, corr_analysis, missing_analysis, features_df, output_dir)

        logger.info("=" * 80)
        logger.info("✅ FASE 0.2 CONCLUÍDA!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ Erro na execução: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
