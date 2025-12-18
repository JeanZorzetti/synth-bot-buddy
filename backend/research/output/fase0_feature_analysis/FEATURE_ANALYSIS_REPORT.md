# Relatório - Análise de Features do Modelo ML

## Metadados

- **Data da Análise**: 2025-12-18 11:12:24
- **Total de Features**: 65
- **Amostras Analisadas**: 5,000

---

## 📊 Top 20 Features Mais Importantes (SHAP)

| Rank | Feature | SHAP Importance |
|------|---------|----------------|
| 1 | ema_21 | 0.162353 |
| 2 | bb_upper | 0.134261 |
| 3 | day_of_month | 0.106513 |
| 4 | sma_20 | 0.099884 |
| 5 | sma_50 | 0.089911 |
| 6 | volatility_20 | 0.078142 |
| 7 | atr | 0.061658 |
| 8 | hour | 0.060001 |
| 9 | ema_9 | 0.059446 |
| 10 | hour_sin | 0.039858 |
| 11 | rsi | 0.031105 |
| 12 | bb_width | 0.023269 |
| 13 | bb_lower | 0.020602 |
| 14 | upper_shadow | 0.010012 |
| 15 | bb_middle | 0.009284 |
| 16 | bb_position | 0.008105 |
| 17 | hour_cos | 0.007363 |
| 18 | macd_signal | 0.007146 |
| 19 | price_to_sma20 | 0.006965 |
| 20 | lower_shadow | 0.005651 |

**Interpretação**:
- Features no topo têm maior impacto nas predições do modelo
- SHAP value mede contribuição média de cada feature

---

## 🔗 Features Redundantes (Correlação > 0.9)

**Total de Pares**: 22

| Feature 1 | Feature 2 | Correlação |
|-----------|-----------|------------|
| sma_20 | bb_middle | 1.0000 |
| sma_20 | ema_21 | 0.9997 |
| ema_21 | bb_middle | 0.9997 |
| ema_9 | ema_21 | 0.9989 |
| sma_20 | ema_9 | 0.9982 |
| ema_9 | bb_middle | 0.9982 |
| sma_20 | bb_upper | 0.9971 |
| bb_upper | bb_middle | 0.9971 |
| bb_middle | bb_lower | 0.9970 |
| sma_20 | bb_lower | 0.9970 |
| ema_21 | bb_lower | 0.9969 |
| ema_21 | bb_upper | 0.9966 |
| ema_9 | bb_lower | 0.9957 |
| sma_50 | ema_21 | 0.9953 |
| ema_9 | bb_upper | 0.9948 |
| sma_20 | sma_50 | 0.9946 |
| sma_50 | bb_middle | 0.9946 |
| sma_50 | bb_upper | 0.9918 |
| sma_50 | bb_lower | 0.9916 |
| sma_50 | ema_9 | 0.9904 |

**Recomendação**:
- Remover 7 features redundantes:
  - `bb_middle`
  - `ema_21`
  - `ema_9`
  - `bb_upper`
  - `bb_lower`
  - `sma_50`
  - `stoch_d`

---

## ❓ Features com Missing Values

✅ **Nenhuma feature com missing values!**

---

## 🎯 Conclusões e Recomendações

### Ações Imediatas

1. **Remover Features Redundantes**:
   - 7 features com correlação > 0.9

2. **Tratar Missing Values**:
   - ✅ Nenhum tratamento necessário

3. **Focar nas Top Features**:
   - Otimizar hiperparâmetros das top 5:
     - `ema_21`
     - `bb_upper`
     - `day_of_month`
     - `sma_20`
     - `sma_50`

---

**Gerado em**: 2025-12-18 11:12:24
