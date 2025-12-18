# Relatório - Análise de Predições do Modelo ML

## Metadados

- **Data da Análise**: 2025-12-18 11:30:11
- **Total de Predições**: 1,000
- **Features Utilizadas**: 65 (todas implementadas)

---

## 📊 Distribuição de Confidence

| Estatística | Valor |
|-------------|-------|
| Média | 0.438 |
| Mediana | 0.445 |
| Desvio Padrão | 0.044 |
| Mínimo | 0.309 |
| Máximo | 0.545 |
| P25 | 0.410 |
| P75 | 0.470 |
| P95 | 0.499 |

**Interpretação**:
- Confidence média de **43.8%** indica moderada confiança
- 95% das predições têm confidence < 49.9%

---

## 🎯 Predições por Classe

| Classe | Quantidade | % Total | Confidence Média | Acurácia |
|--------|-----------|---------|------------------|----------|
| PRICE_UP | 1000 | 100.0% | 0.438 | 15.3% |
| NO_MOVE | 0 | 0.0% | 0.000 | 0.0% |
| PRICE_DOWN | 0 | 0.0% | 0.000 | 0.0% |

**Descobertas**:
- ⚠️ **Modelo desbalanceado**: 100.0% das predições são PRICE_UP
- ❌ **Modelo não prevê PRICE_DOWN**: Nunca identifica quedas!

---

## 📈 Acurácia por Faixa de Confidence

| Faixa | Predições | Acurácia | Confidence Média |
|-------|-----------|----------|------------------|
| <30% | 0 | nan% | nan |
| 30-40% | 203 | 14.8% | 0.369 |
| 40-50% | 750 | 15.2% | 0.452 |
| 50-60% | 47 | 19.1% | 0.512 |
| 60-70% | 0 | nan% | nan |
| >70% | 0 | nan% | nan |

**Análise de Calibração**:
- ⚠️ Faixa 30-40%: Confidence 36.9% mas Acurácia 14.8% (diff: 22.1%)
- ⚠️ Faixa 40-50%: Confidence 45.2% mas Acurácia 15.2% (diff: 30.0%)
- ⚠️ Faixa 50-60%: Confidence 51.2% mas Acurácia 19.1% (diff: 32.1%)
- ❌ **Modelo descalibrado**: Confidence não reflete acurácia

---

## 🎯 Conclusões e Recomendações

### Performance Geral

- **Acurácia Geral**: 15.3%
- **Confidence Média**: 43.8%
- **Total de Predições**: 1,000

### Ações Recomendadas

1. ❌ **Performance Baixa**: Modelo precisa re-treino urgente
2. ❌ **Calibrar Modelo**: Usar Platt scaling ou isotonic regression
3. ❌ **Implementar Predição SHORT**: Modelo só prevê LONG

---

**Gerado em**: 2025-12-18 11:30:11
