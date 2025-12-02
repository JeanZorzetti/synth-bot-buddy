/**
 * ML Monitoring Page - Monitoramento detalhado do modelo XGBoost em produção
 * Fase 3: Machine Learning - Dashboard de acompanhamento
 */

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Brain,
  TrendingUp,
  TrendingDown,
  Target,
  Activity,
  CheckCircle,
  XCircle,
  AlertTriangle,
  RefreshCw,
  BarChart3,
  LineChart,
  PieChart
} from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

interface MLModelInfo {
  model_path: string;
  model_name: string;
  threshold: number;
  confidence_threshold: number;
  n_features: number;
  feature_names: string[];
  model_type: string;
  optimization: string;
  expected_performance: {
    accuracy: string;
    recall: string;
    precision: string;
    profit_6_months: string;
    sharpe_ratio: number;
    win_rate: string;
  };
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
  actual_result?: string; // Para tracking depois de 15min
}

const MLMonitoring = () => {
  const [modelInfo, setModelInfo] = useState<MLModelInfo | null>(null);
  const [lastPrediction, setLastPrediction] = useState<MLPrediction | null>(null);
  const [predictionHistory, setPredictionHistory] = useState<MLPrediction[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Calcular estatísticas do histórico
  const stats = {
    total: predictionHistory.length,
    high: predictionHistory.filter(p => p.signal_strength === 'HIGH').length,
    medium: predictionHistory.filter(p => p.signal_strength === 'MEDIUM').length,
    low: predictionHistory.filter(p => p.signal_strength === 'LOW').length,
    avgConfidence: predictionHistory.length > 0
      ? predictionHistory.reduce((sum, p) => sum + p.confidence, 0) / predictionHistory.length
      : 0,
  };

  const loadModelInfo = async () => {
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'https://botderivapi.roilabs.com.br';
      const response = await fetch(`${apiUrl}/api/ml/info`);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      setModelInfo(data);
      setError(null);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Erro desconhecido';
      setError(`Erro ao carregar info do modelo: ${errorMessage}`);
      console.error('Error loading ML model info:', err);
    }
  };

  const loadLastPrediction = async () => {
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'https://botderivapi.roilabs.com.br';

      // Fazer previsão para R_100 (padrão)
      const response = await fetch(`${apiUrl}/api/ml/predict/R_100?timeframe=1m&count=200`);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();

      // Adicionar timestamp
      const predictionWithTimestamp = {
        ...data,
        timestamp: new Date().toISOString()
      };

      setLastPrediction(predictionWithTimestamp);

      // Adicionar ao histórico (manter últimas 20)
      setPredictionHistory(prev => [predictionWithTimestamp, ...prev].slice(0, 20));

      setError(null);
    } catch (err) {
      console.error('Error loading prediction:', err);
      // Não mostrar erro aqui para não poluir a UI
    }
  };

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await Promise.all([loadModelInfo(), loadLastPrediction()]);
    setIsRefreshing(false);
  };

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

  const getSignalBadgeColor = (strength: string) => {
    switch (strength) {
      case 'HIGH':
        return 'bg-green-500 hover:bg-green-600';
      case 'MEDIUM':
        return 'bg-yellow-500 hover:bg-yellow-600';
      case 'LOW':
        return 'bg-gray-500 hover:bg-gray-600';
      default:
        return 'bg-gray-400';
    }
  };

  const getDataSourceBadge = (source?: string) => {
    if (!source) return null;

    const isReal = source.includes('real');
    return (
      <Badge variant={isReal ? 'default' : 'secondary'} className="gap-1">
        {isReal ? <CheckCircle className="h-3 w-3" /> : <AlertTriangle className="h-3 w-3" />}
        {isReal ? 'Dados Reais' : 'Dados Sintéticos'}
      </Badge>
    );
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <Brain className="h-12 w-12 mx-auto mb-4 animate-pulse text-purple-500" />
          <p className="text-lg font-medium">Carregando modelo ML...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Brain className="h-8 w-8 text-purple-500" />
            ML Monitoring - XGBoost
          </h1>
          <p className="text-muted-foreground mt-1">
            Acompanhamento em tempo real do modelo de Machine Learning em produção
          </p>
        </div>
        <Button onClick={handleRefresh} disabled={isRefreshing} variant="outline" className="gap-2">
          <RefreshCw className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
          Atualizar
        </Button>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Erro de Conexão</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Model Info Section */}
      {modelInfo && (
        <div className="grid gap-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5" />
                Informações do Modelo
              </CardTitle>
              <CardDescription>
                {modelInfo.model_name}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-sm text-muted-foreground">Tipo</p>
                  <p className="text-lg font-semibold">{modelInfo.model_type}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Threshold</p>
                  <p className="text-lg font-semibold">{modelInfo.threshold}</p>
                  <Badge variant="outline" className="mt-1 text-xs">
                    {modelInfo.optimization}
                  </Badge>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Features</p>
                  <p className="text-lg font-semibold">{modelInfo.n_features}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Confidence Min</p>
                  <p className="text-lg font-semibold">{modelInfo.confidence_threshold}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Expected Performance */}
          <Card className="border-purple-200 bg-purple-50/50">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-purple-600" />
                Performance Esperada (Backtesting)
              </CardTitle>
              <CardDescription>
                Métricas obtidas em walk-forward validation (14 janelas, 6 meses)
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                <div className="text-center">
                  <p className="text-sm text-muted-foreground">Accuracy</p>
                  <p className="text-2xl font-bold text-blue-600">
                    {modelInfo.expected_performance.accuracy}
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-muted-foreground">Recall</p>
                  <p className="text-2xl font-bold text-green-600">
                    {modelInfo.expected_performance.recall}
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-muted-foreground">Precision</p>
                  <p className="text-2xl font-bold text-orange-600">
                    {modelInfo.expected_performance.precision}
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-muted-foreground">Win Rate</p>
                  <p className="text-2xl font-bold text-purple-600">
                    {modelInfo.expected_performance.win_rate}
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-muted-foreground">Sharpe Ratio</p>
                  <p className="text-2xl font-bold text-indigo-600">
                    {modelInfo.expected_performance.sharpe_ratio.toFixed(2)}
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-sm text-muted-foreground">Profit (6m)</p>
                  <p className="text-2xl font-bold text-emerald-600">
                    {modelInfo.expected_performance.profit_6_months}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Latest Prediction */}
      {lastPrediction && (
        <Card className="border-blue-200">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              Última Previsão
            </CardTitle>
            <CardDescription>
              {lastPrediction.timestamp && new Date(lastPrediction.timestamp).toLocaleString('pt-BR')}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-2 gap-6">
              {/* Prediction Details */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Direção:</span>
                  <Badge
                    variant={lastPrediction.prediction === 'PRICE_UP' ? 'default' : 'secondary'}
                    className="text-lg px-4 py-1"
                  >
                    {lastPrediction.prediction === 'PRICE_UP' ? (
                      <><TrendingUp className="h-4 w-4 mr-1" /> PRICE_UP</>
                    ) : (
                      <><TrendingDown className="h-4 w-4 mr-1" /> NO_MOVE</>
                    )}
                  </Badge>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Confidence:</span>
                  <div className="text-right">
                    <span className="text-2xl font-bold">{(lastPrediction.confidence * 100).toFixed(1)}%</span>
                    <div className="w-32 bg-muted rounded-full h-2 mt-1">
                      <div
                        className="bg-blue-500 h-2 rounded-full transition-all"
                        style={{ width: `${lastPrediction.confidence * 100}%` }}
                      />
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Signal Strength:</span>
                  <Badge className={getSignalBadgeColor(lastPrediction.signal_strength)}>
                    {lastPrediction.signal_strength}
                  </Badge>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Threshold Usado:</span>
                  <span className="font-medium">{lastPrediction.threshold_used}</span>
                </div>
              </div>

              {/* Metadata */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Símbolo:</span>
                  <Badge variant="outline">{lastPrediction.symbol || 'R_100'}</Badge>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Timeframe:</span>
                  <Badge variant="outline">{lastPrediction.timeframe || '1m'}</Badge>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Candles Analisados:</span>
                  <span className="font-medium">{lastPrediction.candles_analyzed || 200}</span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Data Source:</span>
                  {getDataSourceBadge(lastPrediction.data_source)}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Statistics from History */}
      <div className="grid md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <PieChart className="h-4 w-4" />
              Total Previsões
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{stats.total}</div>
            <p className="text-xs text-muted-foreground mt-1">
              Últimas 20 registradas
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <CheckCircle className="h-4 w-4 text-green-500" />
              Sinais HIGH
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-green-600">{stats.high}</div>
            <p className="text-xs text-muted-foreground mt-1">
              {stats.total > 0 ? ((stats.high / stats.total) * 100).toFixed(1) : 0}% do total
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Activity className="h-4 w-4 text-yellow-500" />
              Sinais MEDIUM
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-yellow-600">{stats.medium}</div>
            <p className="text-xs text-muted-foreground mt-1">
              {stats.total > 0 ? ((stats.medium / stats.total) * 100).toFixed(1) : 0}% do total
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <LineChart className="h-4 w-4 text-blue-500" />
              Confidence Média
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-blue-600">
              {(stats.avgConfidence * 100).toFixed(1)}%
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Últimas {stats.total} previsões
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Prediction History */}
      <Card>
        <CardHeader>
          <CardTitle>Histórico de Previsões</CardTitle>
          <CardDescription>
            Últimas 20 previsões realizadas pelo modelo
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {predictionHistory.length === 0 ? (
              <p className="text-center text-muted-foreground py-8">
                Nenhuma previsão no histórico ainda
              </p>
            ) : (
              predictionHistory.map((pred, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between p-3 rounded-lg border bg-card hover:bg-accent/50 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    {pred.prediction === 'PRICE_UP' ? (
                      <TrendingUp className="h-5 w-5 text-green-500" />
                    ) : (
                      <TrendingDown className="h-5 w-5 text-gray-500" />
                    )}
                    <div>
                      <p className="font-medium">{pred.prediction}</p>
                      <p className="text-xs text-muted-foreground">
                        {pred.timestamp && new Date(pred.timestamp).toLocaleTimeString('pt-BR')}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-3">
                    <div className="text-right">
                      <p className="font-semibold">{(pred.confidence * 100).toFixed(1)}%</p>
                      <p className="text-xs text-muted-foreground">confidence</p>
                    </div>
                    <Badge className={getSignalBadgeColor(pred.signal_strength)} variant="secondary">
                      {pred.signal_strength}
                    </Badge>
                  </div>
                </div>
              ))
            )}
          </div>
        </CardContent>
      </Card>

      {/* Info Box */}
      <Alert>
        <Brain className="h-4 w-4" />
        <AlertTitle>Sistema de Monitoramento ML</AlertTitle>
        <AlertDescription>
          Este dashboard monitora o modelo XGBoost em produção com threshold otimizado (0.30).
          As previsões são atualizadas automaticamente a cada 30 segundos.
          Para ativar dados reais, configure o token Deriv API no backend.
        </AlertDescription>
      </Alert>
    </div>
  );
};

export default MLMonitoring;
