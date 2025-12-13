# 📋 Plano de Migração: ML Monitoring → Dashboard

**Data:** 2025-12-13
**Objetivo:** Consolidar página ML Monitoring no Dashboard principal
**Tempo Estimado:** 5-7 horas

---

## 🔍 Análise do MLMonitoring.tsx

### Componentes Identificados (925 linhas)

#### 1. **Interfaces** (linhas 51-82)
```typescript
interface MLModelInfo {
  model_path: string;
  model_name: string;
  threshold: number;
  confidence_threshold: number;
  n_features: number;
  feature_names: string[];
  model_type: string;
  optimization: string;
  expected_performance: {...}
}

interface MLPrediction {
  prediction: string;
  confidence: number;
  signal_strength: string;
  threshold_used: number;
  model: string;
  symbol?: string;
  timeframe?: string;
  data_source?: string;
  candles_analyzed?: number;
  timestamp?: string;
  actual_result?: string;
}
```

**✅ Ação:** Copiar ambas interfaces para Dashboard.tsx

---

#### 2. **Estados** (linhas 85-106)
```typescript
const [modelInfo, setModelInfo] = useState<MLModelInfo | null>(null);
const [lastPrediction, setLastPrediction] = useState<MLPrediction | null>(null);
const [predictionHistory, setPredictionHistory] = useState<MLPrediction[]>([]);
const [isLoading, setIsLoading] = useState(true);
const [error, setError] = useState<string | null>(null);
const [isRefreshing, setIsRefreshing] = useState(false);

// Trading execution
const [showConfirmDialog, setShowConfirmDialog] = useState(false);
const [showSettingsPanel, setShowSettingsPanel] = useState(false);
const [isExecuting, setIsExecuting] = useState(false);
const [executionResult, setExecutionResult] = useState<string | null>(null);

// Trading settings
const [tradeSettings, setTradeSettings] = useState({
  symbol: 'R_100',
  amount: 10,
  stopLossPercent: 5,
  takeProfitPercent: 10,
  paperTrading: true,
  autoTrade: false,
});
```

**✅ Ação:** Adicionar todos os estados ao Dashboard.tsx

---

#### 3. **Funções API** (linhas 119-186)

##### 3.1 `loadModelInfo()` - Carrega info do modelo XGBoost
```typescript
const loadModelInfo = async () => {
  const response = await fetch(`${apiUrl}/api/ml/info`);
  const data = await response.json();
  setModelInfo(data);
};
```

##### 3.2 `loadLastPrediction()` - Carrega última previsão
```typescript
const loadLastPrediction = async () => {
  const response = await fetch(`${apiUrl}/api/ml/predict/R_100?timeframe=1m&count=200`, {
    headers: { 'X-API-Token': token }
  });
  const data = await response.json();
  setLastPrediction({...data, timestamp: new Date().toISOString()});
  setPredictionHistory(prev => [data, ...prev].slice(0, 20));
};
```

##### 3.3 `handleRefresh()` - Atualiza dados
```typescript
const handleRefresh = async () => {
  setIsRefreshing(true);
  await Promise.all([loadModelInfo(), loadLastPrediction()]);
  setIsRefreshing(false);
};
```

##### 3.4 `handleExecuteTrade()` - Executa trade (linhas 189-234)
```typescript
const handleExecuteTrade = async () => {
  const response = await fetch(`${apiUrl}/api/ml/execute`, {
    method: 'POST',
    body: JSON.stringify({
      prediction: lastPrediction.prediction,
      confidence: lastPrediction.confidence,
      symbol: tradeSettings.symbol,
      amount: tradeSettings.amount,
      stop_loss_percent: tradeSettings.stopLossPercent,
      take_profit_percent: tradeSettings.takeProfitPercent,
      paper_trading: tradeSettings.paperTrading,
    }),
  });
  const result = await response.json();
  setExecutionResult(result.message);
};
```

**✅ Ação:** Copiar todas as 4 funções para Dashboard.tsx

---

#### 4. **Helper Functions** (linhas 267-290)

```typescript
const getSignalBadgeColor = (strength: string) => {
  switch (strength) {
    case 'HIGH': return 'bg-green-500 hover:bg-green-600';
    case 'MEDIUM': return 'bg-yellow-500 hover:bg-yellow-600';
    case 'LOW': return 'bg-gray-500 hover:bg-gray-600';
    default: return 'bg-gray-400';
  }
};

const getDataSourceBadge = (source?: string) => {
  const isReal = source?.includes('real');
  return (
    <Badge variant={isReal ? 'default' : 'secondary'} className="gap-1">
      {isReal ? <CheckCircle className="h-3 w-3" /> : <AlertTriangle className="h-3 w-3" />}
      {isReal ? 'Dados Reais' : 'Dados Sintéticos'}
    </Badge>
  );
};
```

**✅ Ação:** Copiar ambas funções para Dashboard.tsx

---

#### 5. **Auto-refresh Effect** (linhas 250-265)

```typescript
useEffect(() => {
  const initialize = async () => {
    setIsLoading(true);
    await Promise.all([loadModelInfo(), loadLastPrediction()]);
    setIsLoading(false);
  };

  initialize();

  // Auto-refresh a cada 30 segundos
  const interval = setInterval(() => {
    loadLastPrediction();
  }, 30000);

  return () => clearInterval(interval);
}, []);
```

**✅ Ação:** Copiar useEffect para Dashboard.tsx

---

## 🎨 Componentes Visuais a Migrar

### Card 1: **Model Info** (linhas 332-367)
- Tipo de modelo
- Threshold
- Features
- Confidence Min

**📍 Destino:** Aba "ML XGBoost" no Dashboard

---

### Card 2: **Expected Performance** (linhas 370-421)
- Accuracy: 62.58%
- Recall: 54.03%
- Precision: 71.39%
- Win Rate: 43%
- Sharpe Ratio: 3.05
- Profit (6m): +5832%

**📍 Destino:** Aba "ML XGBoost" no Dashboard

---

### Card 3: **Latest Prediction** (linhas 425-505)
- Direção (PRICE_UP / NO_MOVE)
- Confidence (barra de progresso)
- Signal Strength (badge colorido)
- Threshold Usado
- Metadata (symbol, timeframe, candles, data source)

**📍 Destino:** Aba "ML XGBoost" no Dashboard

---

### Card 4: **Quick Actions** (linhas 508-716)
- Botão "Execute Paper Trade"
- Botão "Run Backtest" (placeholder)
- Painel de configurações (collapsible)
  - Symbol selection
  - Amount
  - Stop Loss %
  - Take Profit %
  - Paper Trading switch
  - Auto-Trade switch
- Confirmation Dialog

**⚠️ Ação:**
- Manter "Execute Paper Trade" (funcional)
- **Substituir "Run Backtest" placeholder** → Implementar Backtesting Visual (FASE 7)

---

### Card 5: **Statistics Cards** (linhas 797-859)
- Total Previsões
- Sinais HIGH
- Sinais MEDIUM
- Confidence Média

**📍 Destino:** Aba "ML XGBoost" no Dashboard

---

### Card 6: **Prediction History** (linhas 862-909)
- Lista das últimas 20 previsões
- Cada item mostra: direção, timestamp, confidence, signal strength

**📍 Destino:** Aba "ML XGBoost" no Dashboard

---

### Alert: **Info Box** (linhas 912-920)
- Explicação do sistema
- Info sobre auto-refresh (30s)
- Info sobre token Deriv API

**📍 Destino:** Aba "ML XGBoost" no Dashboard

---

## 📊 Estrutura Final do Dashboard

### Nova Arquitetura (Tabs)

```tsx
<Tabs defaultValue="overview">
  <TabsList>
    <TabsTrigger value="overview">Overview</TabsTrigger>
    <TabsTrigger value="ml-xgboost">ML XGBoost</TabsTrigger>
    <TabsTrigger value="performance">Performance</TabsTrigger>
    <TabsTrigger value="backtest">Backtesting</TabsTrigger>
  </TabsList>

  {/* TAB 1: OVERVIEW (atual Dashboard) */}
  <TabsContent value="overview">
    {/* Cards de resumo AI, Trading, System */}
    {/* Gráfico Equity Curve */}
    {/* Log em tempo real */}
  </TabsContent>

  {/* TAB 2: ML XGBOOST (migrado de MLMonitoring) */}
  <TabsContent value="ml-xgboost">
    {/* Model Info */}
    {/* Expected Performance */}
    {/* Latest Prediction */}
    {/* Quick Actions */}
    {/* Statistics Cards */}
    {/* Prediction History */}
    {/* Info Box */}
  </TabsContent>

  {/* TAB 3: PERFORMANCE (novo) */}
  <TabsContent value="performance">
    {/* Confusion Matrix (a implementar) */}
    {/* ROC Curve (a implementar) */}
    {/* Feature Importance (já existe no RiskManagement) */}
  </TabsContent>

  {/* TAB 4: BACKTESTING (novo - FASE 7) */}
  <TabsContent value="backtest">
    {/* Dialog com parâmetros (período, threshold, capital) */}
    {/* Conectar ao backend /api/ml/backtest/walkforward */}
    {/* Visualizar resultados (equity curve, métricas) */}
    {/* Download CSV/JSON */}
  </TabsContent>
</Tabs>
```

---

## 🔧 Passos de Implementação

### ✅ Fase 1: Preparação (30 min)
1. Ler Dashboard.tsx atual
2. Adicionar imports necessários (Tabs, Dialog, Switch, Select)
3. Copiar interfaces de MLMonitoring

### ✅ Fase 2: Estados e Funções (1h)
4. Adicionar estados ML XGBoost
5. Copiar funções API
6. Copiar helper functions
7. Copiar useEffect de auto-refresh

### ✅ Fase 3: Tab ML XGBoost (2h)
8. Criar estrutura de Tabs
9. Migrar Card "Model Info"
10. Migrar Card "Expected Performance"
11. Migrar Card "Latest Prediction"
12. Migrar Card "Quick Actions" (sem placeholder backtest)
13. Migrar Statistics Cards
14. Migrar Prediction History
15. Migrar Info Box

### ✅ Fase 4: Tab Performance (1h)
16. Criar Tab "Performance"
17. Adicionar Confusion Matrix (buscar implementação existente)
18. Adicionar ROC Curve (buscar implementação existente)
19. Adicionar Feature Importance (copiar de RiskManagement)

### ✅ Fase 5: Tab Backtesting (2h)
20. Criar Tab "Backtesting"
21. Dialog com parâmetros:
    - Período (start_date, end_date)
    - Threshold (0.1 a 0.5)
    - Capital inicial ($1000)
    - Symbol (R_100, R_75, etc.)
22. Função `runBacktest()`:
    ```typescript
    const runBacktest = async (params) => {
      const response = await fetch(`${apiUrl}/api/ml/backtest/walkforward`, {
        method: 'POST',
        body: JSON.stringify(params)
      });
      const result = await response.json();
      setBacktestResult(result);
    };
    ```
23. Visualização de resultados:
    - Equity Curve (LineChart)
    - Métricas (accuracy, win_rate, sharpe_ratio, profit)
    - Tabela de trades
24. Botão de download (CSV/JSON)

### ✅ Fase 6: Cleanup (30 min)
25. Remover MLMonitoring.tsx
26. Atualizar rotas (App.tsx ou Router)
27. Remover link "ML Monitoring" do menu
28. Build e teste

### ✅ Fase 7: Documentação (30 min)
29. Atualizar ANALISE_REDUNDANCIAS_FRONTEND.md
30. Atualizar roadmap (FASE 7 completa)
31. Criar CHANGELOG.md

---

## 📦 Dependências Necessárias

Já instaladas (verificar package.json):
- `recharts` (gráficos)
- `lucide-react` (ícones)
- `@radix-ui/react-tabs` (tabs)
- `@radix-ui/react-dialog` (dialog)
- `@radix-ui/react-switch` (switch)
- `@radix-ui/react-select` (select)

---

## 🚨 Atenções Especiais

### 1. **Auto-refresh**
- Manter intervalo de 30s no useEffect
- Cleanup no return para evitar memory leak

### 2. **Paper Trading Mode**
- Manter switch de Paper Trading sempre visível
- Alert vermelho quando em modo REAL

### 3. **Confirmation Dialog**
- Sempre mostrar confirmação antes de executar trade
- Revisar todos os parâmetros

### 4. **Error Handling**
- Try/catch em todas as chamadas API
- Toast notifications para feedback visual

### 5. **Backtesting Endpoint**
- Endpoint: `/api/ml/backtest/walkforward`
- Método: POST
- Body: `{ start_date, end_date, threshold, initial_capital, symbol }`
- Response: `{ equity_curve: [...], metrics: {...}, trades: [...] }`

---

## 📈 Componentes a Implementar do Zero

### 1. Confusion Matrix
```typescript
interface ConfusionMatrix {
  true_positive: number;
  false_positive: number;
  true_negative: number;
  false_negative: number;
}

// Visualizar como heatmap (recharts)
```

### 2. ROC Curve
```typescript
interface ROCPoint {
  fpr: number; // False Positive Rate
  tpr: number; // True Positive Rate
  threshold: number;
}

// Visualizar como LineChart (recharts)
```

### 3. Backtesting Dialog
```typescript
const BacktestDialog = () => (
  <Dialog>
    <DialogContent>
      <DialogHeader>
        <DialogTitle>Run Backtesting</DialogTitle>
        <DialogDescription>
          Walk-forward analysis com múltiplas janelas
        </DialogDescription>
      </DialogHeader>

      <div className="space-y-4">
        <Label>Período</Label>
        <DateRangePicker />

        <Label>Threshold</Label>
        <Slider min={0.1} max={0.5} step={0.01} />

        <Label>Capital Inicial</Label>
        <Input type="number" defaultValue={1000} />

        <Label>Symbol</Label>
        <Select>...</Select>
      </div>

      <DialogFooter>
        <Button onClick={runBacktest}>Run Backtest</Button>
      </DialogFooter>
    </DialogContent>
  </Dialog>
);
```

---

## ✅ Checklist Final

### Antes do Deploy
- [ ] Build do frontend sem erros
- [ ] Testes visuais de todas as tabs
- [ ] Verificar auto-refresh funciona
- [ ] Testar Paper Trade execution
- [ ] Testar Backtesting Visual
- [ ] Verificar responsividade mobile
- [ ] Remover arquivo MLMonitoring.tsx
- [ ] Atualizar rotas
- [ ] Atualizar menu lateral

### Após Deploy
- [ ] Testar em produção
- [ ] Verificar endpoints ML funcionam
- [ ] Monitorar console de erros
- [ ] Coletar feedback de UX

---

## 📊 Comparação de Código

| Aspecto | Antes (ML Monitoring) | Depois (Dashboard) |
|---------|----------------------|-------------------|
| **Linhas** | 925 linhas | +500 linhas no Dashboard |
| **Páginas** | 5 páginas | 4 páginas |
| **Redundância** | ML duplicado em 2 lugares | ML centralizado |
| **Backtesting** | Placeholder | Funcional (FASE 7) |
| **Confusion Matrix** | Ausente | Implementado |
| **ROC Curve** | Ausente | Implementado |

---

## 🎯 Resultado Esperado

**Dashboard Final:**
- 4 tabs claras e especializadas
- ML XGBoost centralizado em 1 lugar
- Backtesting Visual funcional
- Performance metrics completo (Confusion Matrix + ROC)
- Menos redundância, mais profissional

**Tempo Total:** 5-7 horas de trabalho

---

**Status Atual:** ✅ ANÁLISE COMPLETA - PRONTO PARA IMPLEMENTAÇÃO

---

**Assinatura Digital:**
🤖 Generated with [Claude Code](https://claude.com/claude-code)
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
