#!/usr/bin/env python3
"""
FASE 0.1 - Análise Exploratória de Dados (EDA)

Coleta 30 dias de dados históricos de R_100 e realiza análise estatística completa
para entender o comportamento do mercado.
"""

import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import logging
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import sys

# Adicionar path do backend
sys.path.insert(0, str(Path(__file__).parent.parent))

from deriv_api_legacy import DerivAPI
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar estilo de plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


async def collect_historical_data(symbol: str = 'R_100', days: int = 30, granularity: int = 60):
    """
    Coleta dados históricos da Deriv API

    Args:
        symbol: Símbolo a coletar
        days: Número de dias de histórico
        granularity: Timeframe em segundos (60 = 1min)

    Returns:
        DataFrame com candles OHLC
    """
    logger.info(f"🔍 Coletando {days} dias de dados de {symbol} (timeframe: {granularity}s)")

    # Conectar à Deriv API
    api = DerivAPI()
    await api.connect()

    # Calcular número de candles necessários
    candles_per_day = (24 * 60 * 60) // granularity
    total_candles = candles_per_day * days

    logger.info(f"   Candles necessários: {total_candles} ({candles_per_day}/dia)")

    # Deriv API limita a 5000 candles por request
    # Vamos coletar em chunks de 5000
    all_candles = []
    chunks = (total_candles // 5000) + 1

    for i in range(chunks):
        count = min(5000, total_candles - (i * 5000))
        if count <= 0:
            break

        logger.info(f"   Chunk {i+1}/{chunks}: {count} candles")

        response = await api.get_candles(
            symbol=symbol,
            count=count,
            granularity=granularity
        )

        if 'candles' in response:
            all_candles.extend(response['candles'])

        # Aguardar 1s entre requests para não sobrecarregar API
        await asyncio.sleep(1)

    await api.disconnect()

    logger.info(f"✅ {len(all_candles)} candles coletados")

    # Converter para DataFrame
    data = []
    for candle in all_candles:
        data.append({
            'timestamp': datetime.fromtimestamp(candle['epoch']),
            'open': float(candle['open']),
            'high': float(candle['high']),
            'low': float(candle['low']),
            'close': float(candle['close']),
        })

    df = pd.DataFrame(data)
    df = df.set_index('timestamp')
    df = df.sort_index()

    return df


def analyze_distribution(df: pd.DataFrame) -> dict:
    """
    Analisa distribuição de preços e retornos

    Returns:
        Dict com estatísticas descritivas
    """
    logger.info("📊 Analisando distribuição de preços...")

    # Calcular retornos
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    returns = df['returns'].dropna()

    stats_dict = {
        'mean': returns.mean(),
        'std': returns.std(),
        'skewness': returns.skew(),
        'kurtosis': returns.kurtosis(),
        'min': returns.min(),
        'max': returns.max(),
        'median': returns.median(),
        'q25': returns.quantile(0.25),
        'q75': returns.quantile(0.75),
    }

    # Teste de normalidade
    stat, p_value = stats.normaltest(returns)
    stats_dict['normality_pvalue'] = p_value
    stats_dict['is_normal'] = p_value > 0.05

    # Teste de estacionariedade (ADF)
    adf_result = adfuller(df['close'].dropna())
    stats_dict['adf_statistic'] = adf_result[0]
    stats_dict['adf_pvalue'] = adf_result[1]
    stats_dict['is_stationary'] = adf_result[1] < 0.05

    logger.info(f"   Mean: {stats_dict['mean']:.6f}")
    logger.info(f"   Std: {stats_dict['std']:.6f}")
    logger.info(f"   Skewness: {stats_dict['skewness']:.4f}")
    logger.info(f"   Kurtosis: {stats_dict['kurtosis']:.4f}")
    logger.info(f"   Normalidade (p-value): {stats_dict['normality_pvalue']:.4f} {'✅ Normal' if stats_dict['is_normal'] else '❌ Não-normal'}")
    logger.info(f"   Estacionariedade (ADF p-value): {stats_dict['adf_pvalue']:.4f} {'✅ Estacionário' if stats_dict['is_stationary'] else '❌ Não-estacionário'}")

    return stats_dict


def calculate_volatility_metrics(df: pd.DataFrame) -> dict:
    """
    Calcula métricas de volatilidade

    Returns:
        Dict com métricas de volatilidade
    """
    logger.info("📈 Calculando métricas de volatilidade...")

    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()

    df['atr'] = atr
    df['atr_pct'] = (atr / df['close']) * 100

    # Range médio
    df['range'] = df['high'] - df['low']
    df['range_pct'] = (df['range'] / df['close']) * 100

    volatility_dict = {
        'atr_mean': df['atr'].mean(),
        'atr_std': df['atr'].std(),
        'atr_pct_mean': df['atr_pct'].mean(),
        'atr_pct_std': df['atr_pct'].std(),
        'range_mean': df['range'].mean(),
        'range_pct_mean': df['range_pct'].mean(),
    }

    logger.info(f"   ATR médio: {volatility_dict['atr_mean']:.5f} ({volatility_dict['atr_pct_mean']:.3f}%)")
    logger.info(f"   Range médio: {volatility_dict['range_mean']:.5f} ({volatility_dict['range_pct_mean']:.3f}%)")

    return volatility_dict


def analyze_temporal_patterns(df: pd.DataFrame) -> dict:
    """
    Identifica padrões temporais (hora do dia, dia da semana)

    Returns:
        Dict com padrões temporais
    """
    logger.info("⏰ Analisando padrões temporais...")

    # Adicionar colunas de tempo
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_name'] = df.index.day_name()

    # Volatilidade por hora
    hourly_volatility = df.groupby('hour')['atr_pct'].mean()
    peak_hour = hourly_volatility.idxmax()

    # Volatilidade por dia da semana
    daily_volatility = df.groupby('day_name')['atr_pct'].mean()
    peak_day = daily_volatility.idxmax()

    # Retornos por hora
    hourly_returns = df.groupby('hour')['returns'].mean()

    # Retornos por dia
    daily_returns = df.groupby('day_name')['returns'].mean()

    temporal_dict = {
        'peak_volatility_hour': int(peak_hour),
        'peak_volatility_day': peak_day,
        'hourly_volatility': hourly_volatility.to_dict(),
        'daily_volatility': daily_volatility.to_dict(),
        'hourly_returns': hourly_returns.to_dict(),
        'daily_returns': daily_returns.to_dict(),
    }

    logger.info(f"   Hora com maior volatilidade: {peak_hour}h")
    logger.info(f"   Dia com maior volatilidade: {peak_day}")

    return temporal_dict


def calculate_movement_time(df: pd.DataFrame, threshold_pct: float = 0.5) -> dict:
    """
    Calcula tempo médio para movimento de X%

    Args:
        threshold_pct: Percentual de movimento (0.5 = 0.5%)

    Returns:
        Dict com métricas de tempo de movimento
    """
    logger.info(f"⏱️ Calculando tempo para movimento de {threshold_pct}%...")

    moves = []
    current_ref = df['close'].iloc[0]
    candles_since_move = 0

    for i in range(1, len(df)):
        price = df['close'].iloc[i]
        move_pct = abs((price - current_ref) / current_ref) * 100

        candles_since_move += 1

        if move_pct >= threshold_pct:
            moves.append(candles_since_move)
            current_ref = price
            candles_since_move = 0

    if moves:
        avg_candles = np.mean(moves)
        median_candles = np.median(moves)

        movement_dict = {
            'threshold_pct': threshold_pct,
            'avg_candles': avg_candles,
            'median_candles': median_candles,
            'min_candles': min(moves),
            'max_candles': max(moves),
            'total_moves': len(moves),
        }

        logger.info(f"   Movimento {threshold_pct}% ocorre a cada {avg_candles:.1f} candles (mediana: {median_candles:.1f})")
        logger.info(f"   Total de movimentos: {len(moves)}")

        return movement_dict
    else:
        logger.warning(f"   Nenhum movimento de {threshold_pct}% detectado!")
        return {'threshold_pct': threshold_pct, 'total_moves': 0}


def generate_plots(df: pd.DataFrame, output_dir: Path):
    """
    Gera gráficos para análise visual
    """
    logger.info("📊 Gerando gráficos...")

    output_dir.mkdir(exist_ok=True, parents=True)

    # 1. Série temporal de preços
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['close'], linewidth=0.8)
    plt.title('R_100 - Série Temporal de Preços (30 dias)')
    plt.xlabel('Data/Hora')
    plt.ylabel('Preço')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / '01_price_series.png', dpi=150)
    plt.close()

    # 2. Histograma de retornos
    plt.figure(figsize=(12, 6))
    returns = df['returns'].dropna()
    plt.hist(returns, bins=100, edgecolor='black', alpha=0.7)
    plt.axvline(returns.mean(), color='red', linestyle='--', label=f'Média: {returns.mean():.6f}')
    plt.axvline(returns.median(), color='green', linestyle='--', label=f'Mediana: {returns.median():.6f}')
    plt.title('Distribuição de Retornos')
    plt.xlabel('Retorno')
    plt.ylabel('Frequência')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / '02_returns_histogram.png', dpi=150)
    plt.close()

    # 3. Q-Q Plot (normalidade)
    plt.figure(figsize=(8, 8))
    stats.probplot(returns, dist="norm", plot=plt)
    plt.title('Q-Q Plot - Teste de Normalidade')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / '03_qq_plot.png', dpi=150)
    plt.close()

    # 4. ATR ao longo do tempo
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['atr_pct'], linewidth=0.8, color='orange')
    plt.axhline(df['atr_pct'].mean(), color='red', linestyle='--', label=f'Média: {df["atr_pct"].mean():.3f}%')
    plt.title('Average True Range (ATR) - Volatilidade ao Longo do Tempo')
    plt.xlabel('Data/Hora')
    plt.ylabel('ATR (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / '04_atr_series.png', dpi=150)
    plt.close()

    # 5. Volatilidade por hora do dia
    plt.figure(figsize=(12, 6))
    hourly_atr = df.groupby('hour')['atr_pct'].mean()
    plt.bar(hourly_atr.index, hourly_atr.values, color='steelblue', edgecolor='black')
    plt.title('Volatilidade Média por Hora do Dia')
    plt.xlabel('Hora')
    plt.ylabel('ATR Médio (%)')
    plt.xticks(range(24))
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / '05_hourly_volatility.png', dpi=150)
    plt.close()

    # 6. Volatilidade por dia da semana
    plt.figure(figsize=(10, 6))
    daily_atr = df.groupby('day_name')['atr_pct'].mean().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    )
    plt.bar(range(len(daily_atr)), daily_atr.values, color='coral', edgecolor='black')
    plt.title('Volatilidade Média por Dia da Semana')
    plt.xlabel('Dia')
    plt.ylabel('ATR Médio (%)')
    plt.xticks(range(len(daily_atr)), daily_atr.index, rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / '06_daily_volatility.png', dpi=150)
    plt.close()

    logger.info(f"   ✅ 6 gráficos salvos em {output_dir}")


async def main():
    """Função principal"""
    logger.info("="*80)
    logger.info("FASE 0.1 - ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
    logger.info("="*80)

    # Criar diretório de output
    output_dir = Path(__file__).parent / "output" / "fase0_eda"
    output_dir.mkdir(exist_ok=True, parents=True)

    # 1. Coletar dados
    df = await collect_historical_data(symbol='R_100', days=30, granularity=60)

    # Salvar dados brutos
    csv_path = output_dir / "r100_30days_1min.csv"
    df.to_csv(csv_path)
    logger.info(f"💾 Dados salvos em: {csv_path}")

    # 2. Análise de distribuição
    distribution_stats = analyze_distribution(df)

    # 3. Métricas de volatilidade
    volatility_stats = calculate_volatility_metrics(df)

    # 4. Padrões temporais
    temporal_stats = analyze_temporal_patterns(df)

    # 5. Tempo de movimento
    movement_stats_05 = calculate_movement_time(df, threshold_pct=0.5)
    movement_stats_10 = calculate_movement_time(df, threshold_pct=1.0)
    movement_stats_15 = calculate_movement_time(df, threshold_pct=1.5)

    # 6. Gerar gráficos
    generate_plots(df, output_dir / "plots")

    # 7. Salvar relatório JSON
    report = {
        'metadata': {
            'symbol': 'R_100',
            'timeframe': '1min',
            'days_collected': 30,
            'total_candles': len(df),
            'start_date': df.index.min().isoformat(),
            'end_date': df.index.max().isoformat(),
        },
        'distribution': distribution_stats,
        'volatility': volatility_stats,
        'temporal_patterns': temporal_stats,
        'movement_time': {
            '0.5_pct': movement_stats_05,
            '1.0_pct': movement_stats_10,
            '1.5_pct': movement_stats_15,
        }
    }

    import json
    report_path = output_dir / "eda_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"📄 Relatório JSON salvo em: {report_path}")

    # 8. Gerar relatório Markdown
    md_report = generate_markdown_report(report, df)
    md_path = output_dir / "EDA_REPORT.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)

    logger.info(f"📝 Relatório Markdown salvo em: {md_path}")

    logger.info("="*80)
    logger.info("✅ FASE 0.1 CONCLUÍDA!")
    logger.info("="*80)


def generate_markdown_report(report: dict, df: pd.DataFrame) -> str:
    """Gera relatório em Markdown"""

    dist = report['distribution']
    vol = report['volatility']
    temp = report['temporal_patterns']
    mov_05 = report['movement_time']['0.5_pct']
    mov_10 = report['movement_time']['1.0_pct']
    mov_15 = report['movement_time']['1.5_pct']

    md = f"""# Relatório EDA - R_100 (30 dias)

## Metadados

- **Símbolo**: {report['metadata']['symbol']}
- **Timeframe**: {report['metadata']['timeframe']}
- **Período**: {report['metadata']['start_date']} a {report['metadata']['end_date']}
- **Total de Candles**: {report['metadata']['total_candles']:,}

---

## 📊 Distribuição de Preços e Retornos

### Estatísticas Descritivas

| Métrica | Valor |
|---------|-------|
| Média | {dist['mean']:.6f} |
| Desvio Padrão | {dist['std']:.6f} |
| Mínimo | {dist['min']:.6f} |
| Q25 | {dist['q25']:.6f} |
| Mediana | {dist['median']:.6f} |
| Q75 | {dist['q75']:.6f} |
| Máximo | {dist['max']:.6f} |
| **Skewness** | {dist['skewness']:.4f} |
| **Kurtosis** | {dist['kurtosis']:.4f} |

### Testes Estatísticos

- **Normalidade** (Teste D'Agostino-Pearson):
  - p-value: {dist['normality_pvalue']:.4f}
  - Conclusão: {'✅ Distribuição normal' if dist['is_normal'] else '❌ Distribuição NÃO normal'}

- **Estacionariedade** (Teste ADF):
  - ADF Statistic: {dist['adf_statistic']:.4f}
  - p-value: {dist['adf_pvalue']:.4f}
  - Conclusão: {'✅ Série estacionária' if dist['is_stationary'] else '❌ Série NÃO estacionária'}

**Interpretação**:
- Skewness {'positivo' if dist['skewness'] > 0 else 'negativo'} indica {'cauda direita mais longa' if dist['skewness'] > 0 else 'cauda esquerda mais longa'}
- Kurtosis {dist['kurtosis']:.2f} {'> 3' if dist['kurtosis'] > 3 else '< 3'} indica {'caudas pesadas (mais outliers)' if dist['kurtosis'] > 3 else 'caudas leves (menos outliers)'}

---

## 📈 Volatilidade

| Métrica | Valor Absoluto | Percentual |
|---------|----------------|------------|
| **ATR Médio** | {vol['atr_mean']:.5f} | {vol['atr_pct_mean']:.3f}% |
| ATR Desvio Padrão | {vol['atr_std']:.5f} | {vol['atr_pct_std']:.3f}% |
| **Range Médio** | {vol['range_mean']:.5f} | {vol['range_pct_mean']:.3f}% |

**Recomendações de SL/TP baseadas em ATR**:
- **Stop Loss recomendado**: {vol['atr_pct_mean'] * 1.5:.3f}% (1.5x ATR)
- **Take Profit recomendado**: {vol['atr_pct_mean'] * 2.5:.3f}% (2.5x ATR)

---

## ⏰ Padrões Temporais

### Hora do Dia com Maior Volatilidade

**Pico**: {temp['peak_volatility_hour']}h

### Dia da Semana com Maior Volatilidade

**Pico**: {temp['peak_volatility_day']}

---

## ⏱️ Tempo de Movimento

### Movimento de 0.5%

- **Média**: {mov_05.get('avg_candles', 0):.1f} candles
- **Mediana**: {mov_05.get('median_candles', 0):.1f} candles
- **Total de movimentos**: {mov_05.get('total_moves', 0)}

### Movimento de 1.0%

- **Média**: {mov_10.get('avg_candles', 0):.1f} candles
- **Mediana**: {mov_10.get('median_candles', 0):.1f} candles
- **Total de movimentos**: {mov_10.get('total_moves', 0)}

### Movimento de 1.5%

- **Média**: {mov_15.get('avg_candles', 0):.1f} candles
- **Mediana**: {mov_15.get('median_candles', 0):.1f} candles
- **Total de movimentos**: {mov_15.get('total_moves', 0)}

**Timeout Recomendado**:
- Para TP de 0.75%: ~{mov_05.get('avg_candles', 0) * 1.5:.0f} minutos
- Para TP de 1.5%: ~{mov_10.get('avg_candles', 0) * 1.5:.0f} minutos

---

## 🎯 Conclusões e Próximos Passos

1. **Normalidade**: Retornos {'SÃO' if dist['is_normal'] else 'NÃO SÃO'} normalmente distribuídos
   - {'Podemos usar estatísticas paramétricas' if dist['is_normal'] else 'Devemos usar estatísticas não-paramétricas'}

2. **Estacionariedade**: Série {'É' if dist['is_stationary'] else 'NÃO É'} estacionária
   - {'Modelo pode usar features diretas' if dist['is_stationary'] else 'Modelo precisa usar diferenças/retornos'}

3. **Volatilidade**: ATR médio de {vol['atr_pct_mean']:.3f}%
   - SL atual (0.5%) está {'ABAIXO' if 0.5 < vol['atr_pct_mean'] * 1.5 else 'ADEQUADO'}
   - TP atual (0.75%) está {'ABAIXO' if 0.75 < vol['atr_pct_mean'] * 2.5 else 'ADEQUADO'}

4. **Timeout**: Movimento de 0.5% leva ~{mov_05.get('avg_candles', 0):.0f} candles (1min)
   - Timeout de 3min pode ser {'CURTO' if mov_05.get('avg_candles', 0) > 3 else 'ADEQUADO'}

---

**Gerado em**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

    return md


if __name__ == "__main__":
    asyncio.run(main())
