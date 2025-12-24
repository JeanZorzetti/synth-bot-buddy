# 📊 ANÁLISE QUANTITATIVA - VOLATILITY 100 (1s) INDEX

## O que esse script faz?

Este script analisa **5.000 candles históricos** do `1HZ100V` (Volatility 100 de 1 segundo) para:

1. ✅ **Calcular a distribuição de streaks** (sequências consecutivas de velas da mesma cor)
2. ✅ **Recomendar o Delay ideal** baseado em análise probabilística (P99, P95, Mediana)
3. ✅ **Fazer backtest** com diferentes configurações de Martingale
4. ✅ **Identificar a melhor combinação** de Delay + Multiplicador Martingale

---

## Como executar?

### 1. Ativar ambiente virtual

```bash
cd backend
source ../.venv/bin/activate  # Linux/Mac
# OU
../.venv/Scripts/activate      # Windows
```

### 2. Instalar dependências (se necessário)

```bash
pip install websockets pandas numpy python-dotenv
```

### 3. Configurar token da Deriv

Certifique-se de que o `.env` contém:

```env
DERIV_API_TOKEN=seu_token_aqui
DERIV_APP_ID=99188
```

### 4. Executar análise

```bash
python quant_analysis_1hz100v.py
```

---

## O que esperar?

### Output Exemplo:

```
============================================================
ANÁLISE QUANTITATIVA - VOLATILITY 100 (1s) INDEX
Estratégia: Color Streak Martingale
============================================================

📊 Buscando 5000 candles do 1HZ100V...
✅ 5000 candles baixados!
📅 Período: 2025-12-10 até 2025-12-24

🔍 ANÁLISE DE DISTRIBUIÇÃO DE STREAKS
------------------------------------------------------------
Total de sequências: 1234

VERMELHAS:
  - Máximo: 18
  - Média: 3.45
  - Mediana: 3.00
  - P95: 8.00
  - P99: 12.00

VERDES:
  - Máximo: 16
  - Média: 3.42
  - Mediana: 3.00
  - P95: 7.00
  - P99: 11.00

💡 RECOMENDAÇÕES DE DELAY
------------------------------------------------------------
Conservador (P99 * 0.5): 5 velas
Moderado (P95 * 0.6):    4 velas
Agressivo (Median * 0.8): 2 velas

📈 BACKTESTS - OTIMIZAÇÃO DE PARÂMETROS
============================================================

🧪 Testando: Conservador (Delay alto, Mart 2x)
   Delay: 5 | Multiplicador: 2.0
   ✅ Trades: 342 | Win Rate: 48.25%
   💰 Profit: $1234.56 (12.35%)
   📉 Max Drawdown: $45.32
   ⚡ Profit Factor: 1.45
   🎯 Max Level: 4

🧪 Testando: Moderado (Delay médio, Mart 2x)
   Delay: 4 | Multiplicador: 2.0
   ✅ Trades: 489 | Win Rate: 47.89%
   💰 Profit: $1567.23 (15.67%)
   📉 Max Drawdown: $67.89
   ⚡ Profit Factor: 1.52
   🎯 Max Level: 5

============================================================
🏆 CONFIGURAÇÃO RECOMENDADA
============================================================
Estratégia: Moderado (Delay médio, Mart 2x)
Delay: 4 velas
Multiplicador Martingale: 2.0x
Win Rate: 47.89%
ROI: 15.67%
Profit Factor: 1.52
Max Drawdown: $67.89

💡 NOTA: No V100 (1s), a volatilidade é MAIOR que no V100 padrão.
   Considere usar Delay mais alto ou multiplicador mais baixo (1.5x).
============================================================
```

---

## Interpretação dos Resultados

### 📊 Distribuição de Streaks

- **P99** (Percentil 99): 99% das streaks são **menores** que esse valor
  - Se P99 = 12, significa que 99% das vezes, a sequência não passa de 12 velas

- **P95** (Percentil 95): 95% das streaks são **menores** que esse valor
  - Mais comum que P99, mas ainda muito seguro

- **Mediana**: 50% das streaks são menores que esse valor
  - Mais agressivo, gera mais sinais

### 💡 Recomendações de Delay

- **Conservador**: Delay = 50% do P99
  - ✅ Máxima segurança
  - ❌ Menos sinais de entrada

- **Moderado**: Delay = 60% do P95
  - ✅ Equilíbrio entre segurança e frequência
  - **Recomendado para iniciantes**

- **Agressivo**: Delay = 80% da Mediana
  - ✅ Mais sinais de entrada
  - ❌ Maior risco de "Death Sequence"

### 📈 Métricas de Backtest

- **Win Rate**: % de trades vencedores (ideal: > 45%)
- **Profit Factor**: Lucro bruto / Perda bruta (ideal: > 1.5)
- **Max Drawdown**: Maior perda consecutiva (ideal: < $100)
- **Max Level**: Maior nível de Martingale usado (ideal: < 6)

---

## ⚠️ Diferenças V100 vs V100 (1s)

| Característica | V100 Padrão | V100 (1s) |
|----------------|-------------|-----------|
| Volatilidade | Moderada | **ALTA** |
| Delay seguro | 8 velas | **4-6 velas** |
| Streaks máx | ~15 | **~20** |
| Martingale | 2.0x | **1.5x recomendado** |

---

## 🎯 Próximos Passos

1. ✅ Rodar análise
2. ✅ Anotar Delay recomendado
3. ✅ Atualizar XML do bot com novo Delay
4. ✅ Testar em conta demo
5. ✅ Ir para real com capital pequeno

---

## 📝 Notas Importantes

- **Backtest ≠ Futuro**: Resultados passados não garantem lucros futuros
- **Volatilidade 1s é BRUTAL**: Use capital de risco apenas
- **Martingale é arriscado**: Sempre configure Stop Loss
- **Teste em DEMO primeiro**: No mínimo 1 semana

---

**Desenvolvido por Sistema Abutre - Quant Research** 🦅
