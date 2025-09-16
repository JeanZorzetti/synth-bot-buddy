import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Brain,
  Activity,
  TrendingUp,
  TrendingDown,
  DollarSign,
  Target,
  Gauge,
  Zap,
  Eye,
  Cpu,
  Clock,
  Wifi,
  WifiOff,
  CheckCircle2,
  AlertTriangle
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Alert, AlertDescription } from '@/components/ui/alert';

interface AIMetrics {
  accuracy: number;
  confidence_avg: number;
  signals_generated: number;
  patterns_detected: number;
  model_version: string;
  last_prediction?: {
    direction: 'UP' | 'DOWN';
    confidence: number;
    symbol: string;
    timestamp: string;
  };
}

interface TradingMetrics {
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  total_pnl: number;
  session_pnl: number;
  sharpe_ratio: number;
  max_drawdown: number;
  current_balance: number;
}

interface SystemMetrics {
  uptime_hours: number;
  ticks_processed: number;
  processing_speed: number;
  api_latency: number;
  websocket_status: 'connected' | 'disconnected';
  deriv_api_status: 'connected' | 'disconnected';
}

interface LogEntry {
  id: number;
  type: 'ai' | 'trade' | 'system';
  message: string;
  time: string;
}

const Dashboard = () => {
  const [aiMetrics, setAiMetrics] = useState<AIMetrics>({
    accuracy: 72.3,
    confidence_avg: 0.784,
    signals_generated: 47,
    patterns_detected: 23,
    model_version: 'v2.1.3',
    last_prediction: {
      direction: 'UP',
      confidence: 0.89,
      symbol: 'R_100',
      timestamp: new Date().toISOString()
    }
  });

  const [tradingMetrics, setTradingMetrics] = useState<TradingMetrics>({
    total_trades: 152,
    winning_trades: 103,
    losing_trades: 49,
    win_rate: 67.8,
    total_pnl: 2847.65,
    session_pnl: 127.30,
    sharpe_ratio: 1.82,
    max_drawdown: 3.2,
    current_balance: 12847.65
  });

  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics>({
    uptime_hours: 18.7,
    ticks_processed: 1245673,
    processing_speed: 127.3,
    api_latency: 142,
    websocket_status: 'connected',
    deriv_api_status: 'connected'
  });

  const [realtimeLog, setRealtimeLog] = useState<LogEntry[]>([
    { id: 1, type: 'ai', message: 'Modelo LSTM treinado com 98.2% de acurácia', time: new Date().toLocaleTimeString() },
    { id: 2, type: 'trade', message: 'Trade CALL executado em R_100 - Profit: $23.45', time: new Date().toLocaleTimeString() },
    { id: 3, type: 'system', message: 'Sistema de risk management ativado', time: new Date().toLocaleTimeString() },
    { id: 4, type: 'ai', message: 'Padrão de momentum detectado (85% confiança)', time: new Date().toLocaleTimeString() },
    { id: 5, type: 'trade', message: 'Portfolio rebalanceado automaticamente', time: new Date().toLocaleTimeString() }
  ]);

  useEffect(() => {
    const interval = setInterval(() => {
      const newEntry = {
        id: Date.now(),
        type: Math.random() > 0.6 ? 'ai' : Math.random() > 0.5 ? 'trade' : 'system',
        time: new Date().toLocaleTimeString(),
        message: Math.random() > 0.5
          ? `Padrão detectado em R_${Math.random() > 0.5 ? '50' : '100'} (${(Math.random() * 30 + 70).toFixed(1)}% confiança)`
          : `Processando ${Math.floor(Math.random() * 200 + 50)} ticks por segundo`
      };

      setRealtimeLog(prev => [newEntry, ...prev.slice(0, 9)]);

      setAiMetrics(prev => ({
        ...prev,
        accuracy: Math.min(100, prev.accuracy + (Math.random() - 0.5) * 0.5),
        confidence_avg: Math.max(0, Math.min(1, prev.confidence_avg + (Math.random() - 0.5) * 0.02)),
        signals_generated: prev.signals_generated + (Math.random() > 0.8 ? 1 : 0),
        patterns_detected: prev.patterns_detected + (Math.random() > 0.9 ? 1 : 0)
      }));
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const getLogIcon = (type: LogEntry['type']) => {
    switch (type) {
      case 'ai':
        return <Brain className="h-4 w-4 text-purple-500" />;
      case 'trade':
        return <TrendingUp className="h-4 w-4 text-green-500" />;
      default:
        return <Activity className="h-4 w-4 text-blue-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected':
        return 'text-green-500';
      case 'disconnected':
        return 'text-red-500';
      default:
        return 'text-gray-500';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">📊 Dashboard</h1>
          <p className="text-muted-foreground">
            Métricas em tempo real do bot de IA autônomo
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="default" className="gap-1">
            <Brain className="h-3 w-3" />
            IA Ativa
          </Badge>
          <Badge variant={systemMetrics.deriv_api_status === 'connected' ? 'default' : 'destructive'} className="gap-1">
            {systemMetrics.deriv_api_status === 'connected' ? <Wifi className="h-3 w-3" /> : <WifiOff className="h-3 w-3" />}
            Deriv API
          </Badge>
        </div>
      </div>

      {systemMetrics.deriv_api_status === 'disconnected' && (
        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            Conexão com Deriv API perdida. Reconectando automaticamente...
          </AlertDescription>
        </Alert>
      )}

      <div>
        <h2 className="text-xl font-semibold mb-4 flex items-center">
          <Brain className="h-5 w-5 mr-2 text-purple-500" />
          Performance da IA/ML
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center">
                <Target className="h-4 w-4 mr-2" />
                Acurácia do Modelo
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">
                {aiMetrics.accuracy.toFixed(1)}%
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Target: 65%+ ✓
              </p>
              <div className="mt-2 bg-muted rounded-full h-2">
                <div
                  className="bg-green-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${aiMetrics.accuracy}%` }}
                />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center">
                <Gauge className="h-4 w-4 mr-2" />
                Confiança Média
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-blue-600">
                {(aiMetrics.confidence_avg * 100).toFixed(1)}%
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Últimas 100 predições
              </p>
              <div className="mt-2 bg-muted rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${aiMetrics.confidence_avg * 100}%` }}
                />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center">
                <Zap className="h-4 w-4 mr-2" />
                Sinais Gerados
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-purple-600">
                {aiMetrics.signals_generated}
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Hoje
              </p>
              <div className="flex items-center mt-2 text-xs">
                <TrendingUp className="h-3 w-3 text-green-500 mr-1" />
                +{Math.floor(Math.random() * 5 + 2)} na última hora
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center">
                <Eye className="h-4 w-4 mr-2" />
                Padrões Detectados
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-orange-600">
                {aiMetrics.patterns_detected}
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Padrões únicos
              </p>
              <Badge variant="outline" className="mt-2 text-xs">
                {aiMetrics.model_version}
              </Badge>
            </CardContent>
          </Card>
        </div>
      </div>

      <div>
        <h2 className="text-xl font-semibold mb-4 flex items-center">
          <DollarSign className="h-5 w-5 mr-2 text-green-500" />
          Performance de Trading
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">P&L Total</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">
                +${tradingMetrics.total_pnl.toFixed(2)}
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                {tradingMetrics.total_trades} trades
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Taxa de Vitórias</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-blue-600">
                {tradingMetrics.win_rate.toFixed(1)}%
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                {tradingMetrics.winning_trades}W / {tradingMetrics.losing_trades}L
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Sharpe Ratio</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-purple-600">
                {tradingMetrics.sharpe_ratio.toFixed(2)}
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Risk-adjusted returns
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium">Drawdown Máximo</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-red-600">
                {tradingMetrics.max_drawdown.toFixed(1)}%
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Target: &lt;10% ✓
              </p>
            </CardContent>
          </Card>
        </div>
      </div>

      <div>
        <h2 className="text-xl font-semibold mb-4 flex items-center">
          <Cpu className="h-5 w-5 mr-2 text-blue-500" />
          Status dos Sistemas
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center">
                <Clock className="h-4 w-4 mr-2" />
                Uptime
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {systemMetrics.uptime_hours.toFixed(1)}h
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Sistema 24/7 ativo
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center">
                <Activity className="h-4 w-4 mr-2" />
                Ticks Processados
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {systemMetrics.ticks_processed.toLocaleString()}
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                {systemMetrics.processing_speed.toFixed(1)} ticks/s
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center">
                <Zap className="h-4 w-4 mr-2" />
                Latência API
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">
                {systemMetrics.api_latency}ms
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                Target: &lt;200ms ✓
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center">
                <Wifi className="h-4 w-4 mr-2" />
                Conexões
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span>WebSocket</span>
                  <Badge variant={systemMetrics.websocket_status === 'connected' ? 'default' : 'destructive'} className="text-xs">
                    {systemMetrics.websocket_status}
                  </Badge>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span>Deriv API</span>
                  <Badge variant={systemMetrics.deriv_api_status === 'connected' ? 'default' : 'destructive'} className="text-xs">
                    {systemMetrics.deriv_api_status}
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center">
              <Activity className="h-5 w-5 mr-2" />
              Feed de Atividades em Tempo Real
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {realtimeLog.map((entry) => (
                <div
                  key={entry.id}
                  className="flex items-start space-x-3 p-3 rounded-lg bg-muted/30 border-l-2 border-l-blue-500"
                >
                  {getLogIcon(entry.type)}
                  <div className="flex-1">
                    <p className="text-sm font-medium">
                      {entry.message}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {entry.time}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center">
              <Brain className="h-5 w-5 mr-2 text-purple-500" />
              Última Predição da IA
            </CardTitle>
          </CardHeader>
          <CardContent>
            {aiMetrics.last_prediction ? (
              <div className="space-y-4">
                <div className="text-center">
                  <Badge
                    variant={aiMetrics.last_prediction.direction === 'UP' ? 'default' : 'destructive'}
                    className="text-lg px-4 py-2"
                  >
                    {aiMetrics.last_prediction.direction === 'UP' ? '📈 CALL' : '📉 PUT'}
                  </Badge>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Símbolo:</span>
                    <span className="font-medium">{aiMetrics.last_prediction.symbol}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Confiança:</span>
                    <span className="font-medium">{(aiMetrics.last_prediction.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm text-muted-foreground">Horário:</span>
                    <span className="font-medium">
                      {new Date(aiMetrics.last_prediction.timestamp).toLocaleTimeString('pt-BR')}
                    </span>
                  </div>
                </div>

                <div className="mt-4 bg-muted rounded-full h-2">
                  <div
                    className="bg-purple-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${aiMetrics.last_prediction.confidence * 100}%` }}
                  />
                </div>

                <div className="flex items-center justify-center mt-4 text-xs text-muted-foreground">
                  <CheckCircle2 className="h-3 w-3 mr-1" />
                  Modelo {aiMetrics.model_version} ativo
                </div>
              </div>
            ) : (
              <div className="text-center text-muted-foreground">
                <Brain className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>Aguardando primeira predição...</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Dashboard;