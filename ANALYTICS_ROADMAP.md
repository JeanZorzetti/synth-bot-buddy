# 📊 ABUTRE ANALYTICS - Roadmap de Implementação

## ✅ FASE 1: Backend Analytics (COMPLETO)

### Endpoints Criados

#### 1. GET `/api/abutre/analytics/survival`
**Análise de Sobrevivência**
```json
{
  "max_level_reached": 9,
  "max_level_frequency": 3,
  "death_sequences": [
    {
      "trade_id": "123",
      "level": 9,
      "stake": 89.60,
      "time": "2025-12-23T14:30:00Z",
      "result": "WIN"
    }
  ],
  "recovery_factor": 0.48,
  "critical_hours": [14, 15, 16]
}
```

#### 2. GET `/api/abutre/analytics/performance`
**KPIs de Performance**
```json
{
  "total_trades": 777,
  "win_rate": 43.2,
  "profit_factor": 1.15,
  "total_profit": 87.50,
  "max_drawdown": -45.20,
  "avg_win": 3.50,
  "avg_loss": -2.80,
  "max_win_streak": 5,
  "max_loss_streak": 8,
  "sharpe_ratio": 1.23
}
```

#### 3. GET `/api/abutre/analytics/hourly`
**Análise por Horário**
```json
[
  {
    "hour": 14,
    "trades": 50,
    "win_rate": 30.0,
    "avg_profit": -1.20,
    "risk_score": 8.5
  },
  {
    "hour": 10,
    "trades": 40,
    "win_rate": 55.0,
    "avg_profit": 2.30,
    "risk_score": 3.2
  }
]
```

#### 4. GET `/api/abutre/analytics/equity-curve`
**Curva de Equity**
```json
{
  "status": "success",
  "data": [
    {
      "timestamp": "2025-12-23T10:00:00Z",
      "balance": 10127.39,
      "cumulative_profit": 87.50,
      "trade_id": "123"
    }
  ],
  "summary": {
    "initial_balance": 10000.00,
    "final_balance": 10127.39,
    "total_profit": 127.39,
    "peak_balance": 10200.00,
    "lowest_balance": 9950.00
  }
}
```

---

## 🚧 FASE 2: Frontend Dashboard (TODO)

### Componentes a Criar

#### 1. **SurvivalCard.tsx**
Card mostrando métricas de sobrevivência:
- Badge com nível máximo atingido
- Alerta visual se chegou em nível crítico (>= 7)
- Lista de "quase mortes" (sequências perigosas)
- Fator de recuperação com barra de progresso

#### 2. **PerformanceMetrics.tsx**
Grid de KPIs principais:
- Win Rate (gauge circular)
- Profit Factor
- Total Profit
- Max Drawdown
- Streaks (win/loss)

#### 3. **HourlyHeatmap.tsx**
Heatmap mostrando:
- Eixo X: Horas do dia (0-23)
- Eixo Y: Intensidade de risco
- Cores: Verde (seguro) -> Amarelo (médio) -> Vermelho (perigoso)

#### 4. **EquityCurveChart.tsx**
Gráfico de linha mostrando:
- Evolução do saldo ao longo do tempo
- Zonas de drawdown destacadas
- Picos e vales anotados

### Nova Página: `/abutre/analytics`

**Layout sugerido:**

```
┌─────────────────────────────────────────┐
│  📊 Abutre Analytics                     │
│  [Seletor de Período]                    │
└─────────────────────────────────────────┘

┌──────────────┬──────────────┬──────────────┐
│  Win Rate    │ Profit Factor│ Total Profit │
│    43.2%     │     1.15     │   +$127.39   │
└──────────────┴──────────────┴──────────────┘

┌─────────────────────────────────────────┐
│  ⚠️ Análise de Sobrevivência             │
│                                          │
│  Nível Máximo: 9 (3x)                   │
│  Fator Recuperação: 0.48 (BAIXO!)      │
│  Horários Críticos: 14h, 15h, 16h       │
│                                          │
│  🔴 Sequências Perigosas:                │
│  • 14:30 - Nível 9 ($89.60) - WIN       │
│  • 15:15 - Nível 8 ($44.80) - LOSS      │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  🔥 Heatmap de Risco por Horário         │
│                                          │
│  [Heatmap visual aqui]                   │
│                                          │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  📈 Curva de Equity                      │
│                                          │
│  [Gráfico de linha aqui]                 │
│                                          │
└─────────────────────────────────────────┘
```

---

## 📦 Bibliotecas Recomendadas (Frontend)

### Para Gráficos:
```bash
npm install recharts
```
OU
```bash
npm install chart.js react-chartjs-2
```

### Para Visualizações:
```bash
npm install @tremor/react  # UI components for dashboards
```

---

## 🎯 Próximos Passos

### Curto Prazo (Essencial):
1. ✅ Criar hook `useAnalytics` para consumir endpoints
2. ✅ Criar componente `PerformanceMetrics`
3. ✅ Adicionar gráfico de equity curve
4. ✅ Criar página `/abutre/analytics`

### Médio Prazo (Desejável):
- Heatmap de horários
- Exportação de relatórios em PDF
- Comparação entre períodos
- Alertas de risco em tempo real

### Longo Prazo (Avançado):
- Machine Learning para prever horários de risco
- Otimização automática de horários de operação
- Backtesting visual interativo

---

## 🔗 Integração com Página Atual

A página `/abutre/history` atual pode ter um botão:

```tsx
<Link href="/abutre/analytics">
  <button className="...">
    📊 Ver Análise Detalhada
  </button>
</Link>
```

Ou podemos adicionar cards de resumo direto na página de histórico.

---

## 📝 Exemplo de Uso

```typescript
// useAnalytics.ts
import { useState, useCallback } from 'react'

const API_URL = process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:8000'

export function useAnalytics() {
  const [isLoading, setIsLoading] = useState(false)
  
  const getSurvivalMetrics = useCallback(async (dateFrom?: string, dateTo?: string) => {
    setIsLoading(true)
    try {
      const params = new URLSearchParams()
      if (dateFrom) params.append('date_from', dateFrom)
      if (dateTo) params.append('date_to', dateTo)
      
      const response = await fetch(`${API_URL}/api/abutre/analytics/survival?${params}`)
      const data = await response.json()
      return data
    } finally {
      setIsLoading(false)
    }
  }, [])
  
  // ... outras funções
  
  return { getSurvivalMetrics, isLoading }
}
```

---

**Status Atual**: Backend pronto, frontend pendente
**Estimativa**: 2-3 horas para MVP do frontend
