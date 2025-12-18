# Relatório EDA - R_100 (30 dias)

## Metadados

- **Símbolo**: R_100
- **Timeframe**: 1min
- **Período**: 2025-12-14T23:00:00 a 2025-12-18T10:19:00
- **Total de Candles**: 43,200

---

## 📊 Distribuição de Preços e Retornos

### Estatísticas Descritivas

| Métrica | Valor |
|---------|-------|
| Média | 0.000003 |
| Desvio Padrão | 0.000459 |
| Mínimo | -0.004165 |
| Q25 | 0.000000 |
| Mediana | 0.000000 |
| Q75 | 0.000000 |
| Máximo | 0.005680 |
| **Skewness** | 0.1828 |
| **Kurtosis** | 22.4411 |

### Testes Estatísticos

- **Normalidade** (Teste D'Agostino-Pearson):
  - p-value: 0.0000
  - Conclusão: ❌ Distribuição NÃO normal

- **Estacionariedade** (Teste ADF):
  - ADF Statistic: -1.6173
  - p-value: 0.4741
  - Conclusão: ❌ Série NÃO estacionária

**Interpretação**:
- Skewness positivo indica cauda direita mais longa
- Kurtosis 22.44 > 3 indica caudas pesadas (mais outliers)

---

## 📈 Volatilidade

| Métrica | Valor Absoluto | Percentual |
|---------|----------------|------------|
| **ATR Médio** | 1.15056 | 0.188% |
| ATR Desvio Padrão | 0.27792 | 0.044% |
| **Range Médio** | 1.14790 | 0.188% |

**Recomendações de SL/TP baseadas em ATR**:
- **Stop Loss recomendado**: 0.283% (1.5x ATR)
- **Take Profit recomendado**: 0.471% (2.5x ATR)

---

## ⏰ Padrões Temporais

### Hora do Dia com Maior Volatilidade

**Pico**: 14h

### Dia da Semana com Maior Volatilidade

**Pico**: Sunday

---

## ⏱️ Tempo de Movimento

### Movimento de 0.5%

- **Média**: 153.4 candles
- **Mediana**: 117.0 candles
- **Total de movimentos**: 281

### Movimento de 1.0%

- **Média**: 582.1 candles
- **Mediana**: 441.0 candles
- **Total de movimentos**: 74

### Movimento de 1.5%

- **Média**: 1391.1 candles
- **Mediana**: 1113.0 candles
- **Total de movimentos**: 30

**Timeout Recomendado**:
- Para TP de 0.75%: ~230 minutos
- Para TP de 1.5%: ~873 minutos

---

## 🎯 Conclusões e Próximos Passos

1. **Normalidade**: Retornos NÃO SÃO normalmente distribuídos
   - Devemos usar estatísticas não-paramétricas

2. **Estacionariedade**: Série NÃO É estacionária
   - Modelo precisa usar diferenças/retornos

3. **Volatilidade**: ATR médio de 0.188%
   - SL atual (0.5%) está ADEQUADO
   - TP atual (0.75%) está ADEQUADO

4. **Timeout**: Movimento de 0.5% leva ~153 candles (1min)
   - Timeout de 3min pode ser CURTO

---

**Gerado em**: 2025-12-18 10:19:51
