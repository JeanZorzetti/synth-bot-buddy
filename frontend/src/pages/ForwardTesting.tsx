import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Play,
  StopCircle,
  FileText,
  RefreshCw,
  TrendingUp,
  AlertCircle,
  CheckCircle2,
  Brain,
  Target,
  BarChart3,
  Bug,
  Clock,
  Download,
  FileDown,
  Settings,
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { EquityCurveChart } from '@/components/EquityCurveChart';
import { LiveMetricsGrid } from '@/components/LiveMetricsGrid';
import { AlertNotifications } from '@/components/AlertNotifications';
import { TradeHistoryTable } from '@/components/TradeHistoryTable';
import { ModeComparison } from '@/components/ModeComparison';

interface ForwardTestingStatus {
  is_running: boolean;
  start_time: string | null;
  duration_seconds: number;
  duration_hours: number;
  duration_days: number;
  symbol: string;
  total_predictions: number;
  total_trades: number;
  total_bugs: number;
  last_prediction_time: string | null;
  paper_trading_metrics: {
    status: string;
    current_capital: number;
    total_pnl: number;
    total_pnl_pct: number;
    win_rate_pct: number;
    sharpe_ratio: number;
    max_drawdown_pct: number;
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    profit_factor: number;
  };
}

interface Prediction {
  timestamp: string;
  prediction: string;
  confidence: number;
  price: number;
}

interface Bug {
  timestamp: string;
  type: string;
  description: string;
  severity: string;
}

interface LogFile {
  filename: string;
  size_bytes: number;
  size_mb: number;
  modified_at: string;
  download_url: string;
}

const DERIV_SYMBOLS = [
  { value: 'R_100', label: 'R_100 - Random Index 100', volatility: 'Baixa', description: 'Índice aleatório estável' },
  { value: '1HZ10V', label: 'V10 (1s) - Volatility 10', volatility: 'Baixa', description: '10% volatilidade, tick 1s' },
  { value: '1HZ25V', label: 'V25 (1s) - Volatility 25', volatility: 'Média', description: '25% volatilidade, tick 1s' },
  { value: '1HZ50V', label: 'V50 (1s) - Volatility 50', volatility: 'Média-Alta', description: '50% volatilidade, tick 1s' },
  { value: '1HZ75V', label: 'V75 (1s) - Volatility 75 ⚡', volatility: 'Alta', description: '75% volatilidade, tick 1s - RECOMENDADO' },
  { value: '1HZ100V', label: 'V100 (1s) - Volatility 100 🔥', volatility: 'Muito Alta', description: '100% volatilidade, tick 1s - EXTREMAMENTE RÁPIDO' },
  { value: 'BOOM300N', label: 'Boom 300', volatility: 'Alta', description: 'Spikes a cada ~300 ticks' },
  { value: 'CRASH300N', label: 'Crash 300', volatility: 'Alta', description: 'Crashes a cada ~300 ticks' },
];

const TRADING_MODES = [
  {
    id: 'scalping_aggressive',
    name: 'Scalping Agressivo 🔥',
    description: 'Entradas/saídas ultra-rápidas',
    stopLoss: 0.5,
    takeProfit: 0.75,
    timeout: 3,
    riskReward: '1:1.5',
    avgDuration: '1-3 min',
    tradesPerDay: '20-50',
    recommended: ['1HZ100V', '1HZ75V'],
  },
  {
    id: 'scalping_moderate',
    name: 'Scalping Moderado ⚡',
    description: 'Equilíbrio entre velocidade e segurança',
    stopLoss: 1.0,
    takeProfit: 1.5,
    timeout: 5,
    riskReward: '1:1.5',
    avgDuration: '3-8 min',
    tradesPerDay: '10-30',
    recommended: ['1HZ75V', '1HZ50V', 'BOOM300N'],
  },
  {
    id: 'swing',
    name: 'Swing Trading 📈',
    description: 'Posições de médio prazo',
    stopLoss: 2.0,
    takeProfit: 4.0,
    timeout: 30,
    riskReward: '1:2',
    avgDuration: '30-120 min',
    tradesPerDay: '3-10',
    recommended: ['1HZ50V', '1HZ25V', 'R_100'],
  },
];

export default function ForwardTesting() {
  const { toast } = useToast();
  const [status, setStatus] = useState<ForwardTestingStatus | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [bugs, setBugs] = useState<Bug[]>([]);
  const [logs, setLogs] = useState<LogFile[]>([]);
  const [liveMetrics, setLiveMetrics] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isStarting, setIsStarting] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [isLoadingLogs, setIsLoadingLogs] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState('1HZ75V');
  const [selectedMode, setSelectedMode] = useState('scalping_moderate');

  const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://botderivapi.roilabs.com.br';

  useEffect(() => {
    loadStatus();
    loadLogs(); // Carregar logs ao montar
    const interval = setInterval(loadStatus, 5000); // Atualizar a cada 5 segundos
    return () => clearInterval(interval);
  }, []);

  // Polling de live metrics quando sistema está rodando
  useEffect(() => {
    if (!status?.is_running) {
      setLiveMetrics(null);
      return;
    }

    const loadLiveMetrics = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/forward-testing/live-metrics`);
        const data = await response.json();

        if (data.status === 'success') {
          setLiveMetrics(data.data);
        }
      } catch (error) {
        console.error('Erro ao carregar live metrics:', error);
      }
    };

    loadLiveMetrics(); // Carregar imediatamente
    const interval = setInterval(loadLiveMetrics, 5000); // Atualizar a cada 5 segundos
    return () => clearInterval(interval);
  }, [status?.is_running]);

  const loadStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/forward-testing/status`);
      const data = await response.json();

      if (data.status === 'success') {
        setStatus(data.data);

        // Carregar previsões e bugs se estiver rodando
        if (data.data.is_running) {
          loadPredictions();
          loadBugs();
        }
      }
    } catch (error) {
      console.error('Erro ao carregar status:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const loadPredictions = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/forward-testing/predictions?limit=20`);
      const data = await response.json();

      if (data.status === 'success') {
        setPredictions(data.data);
      }
    } catch (error) {
      console.error('Erro ao carregar previsões:', error);
    }
  };

  const loadBugs = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/forward-testing/bugs`);
      const data = await response.json();

      if (data.status === 'success') {
        setBugs(data.data);
      }
    } catch (error) {
      console.error('Erro ao carregar bugs:', error);
    }
  };

  const handleStart = async () => {
    try {
      setIsStarting(true);

      const mode = TRADING_MODES.find(m => m.id === selectedMode);
      if (!mode) throw new Error('Modo inválido');

      const response = await fetch(`${API_BASE_URL}/api/forward-testing/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symbol: selectedSymbol,
          mode: selectedMode,
          stop_loss_pct: mode.stopLoss,
          take_profit_pct: mode.takeProfit,
          position_timeout_minutes: mode.timeout,
        }),
      });

      const data = await response.json();

      if (data.status === 'success') {
        const symbolLabel = DERIV_SYMBOLS.find(s => s.value === selectedSymbol)?.label || selectedSymbol;
        toast({
          title: 'Forward Testing Iniciado',
          description: `${mode.name} com ${symbolLabel}`,
        });

        // Recarregar status
        setTimeout(loadStatus, 1000);
      } else {
        throw new Error(data.detail || 'Erro ao iniciar');
      }
    } catch (error: any) {
      toast({
        title: 'Erro ao Iniciar',
        description: error.message,
        variant: 'destructive',
      });
    } finally {
      setIsStarting(false);
    }
  };

  const handleStop = async () => {
    try {
      setIsStopping(true);

      const response = await fetch(`${API_BASE_URL}/api/forward-testing/stop`, {
        method: 'POST',
      });

      const data = await response.json();

      if (data.status === 'success') {
        toast({
          title: 'Forward Testing Parado',
          description: `Relatório gerado: ${data.report_path}`,
        });

        // Recarregar status
        setTimeout(loadStatus, 1000);
      } else {
        throw new Error(data.detail || 'Erro ao parar');
      }
    } catch (error: any) {
      toast({
        title: 'Erro ao Parar',
        description: error.message,
        variant: 'destructive',
      });
    } finally {
      setIsStopping(false);
    }
  };

  const handleGenerateReport = async () => {
    try {
      setIsGeneratingReport(true);

      const response = await fetch(`${API_BASE_URL}/api/forward-testing/report`, {
        method: 'POST',
      });

      const data = await response.json();

      if (data.status === 'success') {
        toast({
          title: 'Relatório Gerado',
          description: data.report_path,
        });
      } else {
        throw new Error(data.detail || 'Erro ao gerar relatório');
      }
    } catch (error: any) {
      toast({
        title: 'Erro ao Gerar Relatório',
        description: error.message,
        variant: 'destructive',
      });
    } finally {
      setIsGeneratingReport(false);
    }
  };

  const loadLogs = async () => {
    try {
      setIsLoadingLogs(true);
      const response = await fetch(`${API_BASE_URL}/api/forward-testing/logs`);
      const data = await response.json();

      if (data.status === 'success') {
        setLogs(data.data.logs || []);
      }
    } catch (error) {
      console.error('Erro ao carregar logs:', error);
    } finally {
      setIsLoadingLogs(false);
    }
  };

  const handleDownloadLog = (filename: string) => {
    const downloadUrl = `${API_BASE_URL}/api/forward-testing/logs/${filename}`;
    window.open(downloadUrl, '_blank');

    toast({
      title: 'Download Iniciado',
      description: `Baixando ${filename}`,
    });
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const formatDate = (isoString: string) => {
    const date = new Date(isoString);
    return date.toLocaleString('pt-BR', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const formatDuration = (seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const mins = Math.floor((seconds % 3600) / 60);

    if (days > 0) return `${days}d ${hours}h ${mins}m`;
    if (hours > 0) return `${hours}h ${mins}m`;
    return `${mins}m`;
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-4 text-primary" />
          <p className="text-muted-foreground">Carregando Forward Testing...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Forward Testing ML</h1>
          <p className="text-muted-foreground mt-1">
            Teste automatizado com ML Predictor + Paper Trading em tempo real
          </p>
        </div>

        <div className="flex gap-2">
          {status?.is_running ? (
            <>
              <Button onClick={handleStop} disabled={isStopping} variant="destructive">
                <StopCircle className={`h-4 w-4 mr-2 ${isStopping ? 'animate-spin' : ''}`} />
                {isStopping ? 'Parando...' : 'Parar'}
              </Button>
              <Button onClick={handleGenerateReport} disabled={isGeneratingReport} variant="outline">
                <FileText className={`h-4 w-4 mr-2 ${isGeneratingReport ? 'animate-spin' : ''}`} />
                Gerar Relatório
              </Button>
            </>
          ) : (
            <Button onClick={handleStart} disabled={isStarting}>
              <Play className={`h-4 w-4 mr-2 ${isStarting ? 'animate-spin' : ''}`} />
              {isStarting ? 'Iniciando...' : 'Iniciar Forward Testing'}
            </Button>
          )}
        </div>
      </div>

      {/* Asset Selector */}
      {!status?.is_running && (
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Settings className="h-5 w-5" />
              <CardTitle>Configuração do Ativo</CardTitle>
            </div>
            <CardDescription>
              Escolha o ativo para Forward Testing. Ativos mais voláteis geram mais oportunidades de trade.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
              {DERIV_SYMBOLS.map((symbol) => (
                <button
                  key={symbol.value}
                  onClick={() => setSelectedSymbol(symbol.value)}
                  className={`p-4 border-2 rounded-lg text-left transition-all hover:border-primary ${
                    selectedSymbol === symbol.value
                      ? 'border-primary bg-primary/5'
                      : 'border-gray-200'
                  }`}
                >
                  <div className="flex items-start justify-between mb-2">
                    <h3 className="font-semibold text-sm">{symbol.label}</h3>
                    {selectedSymbol === symbol.value && (
                      <CheckCircle2 className="h-4 w-4 text-primary" />
                    )}
                  </div>
                  <Badge
                    variant={
                      symbol.volatility === 'Muito Alta' ? 'destructive' :
                      symbol.volatility === 'Alta' ? 'default' :
                      symbol.volatility === 'Média-Alta' ? 'secondary' :
                      'outline'
                    }
                    className="mb-2"
                  >
                    {symbol.volatility}
                  </Badge>
                  <p className="text-xs text-muted-foreground">{symbol.description}</p>
                </button>
              ))}
            </div>
            <Alert className="mt-4 border-blue-200 bg-blue-50">
              <AlertCircle className="h-4 w-4 text-blue-600" />
              <AlertDescription className="text-blue-900">
                <strong>Selecionado:</strong> {DERIV_SYMBOLS.find(s => s.value === selectedSymbol)?.label} -
                Volatilidade {DERIV_SYMBOLS.find(s => s.value === selectedSymbol)?.volatility}
              </AlertDescription>
            </Alert>
          </CardContent>
        </Card>
      )}

      {/* Trading Mode Selector */}
      {!status?.is_running && (
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Target className="h-5 w-5" />
              <CardTitle>Modo de Trading</CardTitle>
            </div>
            <CardDescription>
              Escolha a estratégia de entrada/saída. Scalping = rápido, Swing = posições longas.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {TRADING_MODES.map((mode) => {
                const isRecommended = mode.recommended.includes(selectedSymbol);
                return (
                  <button
                    key={mode.id}
                    onClick={() => setSelectedMode(mode.id)}
                    className={`p-5 border-2 rounded-lg text-left transition-all hover:border-primary relative ${
                      selectedMode === mode.id
                        ? 'border-primary bg-primary/5'
                        : 'border-gray-200'
                    }`}
                  >
                    {isRecommended && (
                      <Badge className="absolute top-2 right-2 bg-green-500">
                        Recomendado
                      </Badge>
                    )}
                    <div className="flex items-start justify-between mb-3">
                      <h3 className="font-bold text-lg">{mode.name}</h3>
                      {selectedMode === mode.id && (
                        <CheckCircle2 className="h-5 w-5 text-primary mt-1" />
                      )}
                    </div>
                    <p className="text-sm text-muted-foreground mb-4">{mode.description}</p>

                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Stop Loss:</span>
                        <span className="font-semibold">{mode.stopLoss}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Take Profit:</span>
                        <span className="font-semibold">{mode.takeProfit}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Risk:Reward:</span>
                        <span className="font-semibold">{mode.riskReward}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Timeout:</span>
                        <span className="font-semibold">{mode.timeout} min</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Duração Média:</span>
                        <span className="font-semibold">{mode.avgDuration}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Trades/Dia:</span>
                        <span className="font-semibold">{mode.tradesPerDay}</span>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
            <Alert className="mt-4 border-green-200 bg-green-50">
              <CheckCircle2 className="h-4 w-4 text-green-600" />
              <AlertDescription className="text-green-900">
                <strong>Configuração:</strong> {TRADING_MODES.find(m => m.id === selectedMode)?.name} |
                SL: {TRADING_MODES.find(m => m.id === selectedMode)?.stopLoss}% |
                TP: {TRADING_MODES.find(m => m.id === selectedMode)?.takeProfit}% |
                Timeout: {TRADING_MODES.find(m => m.id === selectedMode)?.timeout} min
              </AlertDescription>
            </Alert>
          </CardContent>
        </Card>
      )}

      {/* Status Banner */}
      {status && (
        <Alert className={status.is_running ? 'border-green-500 bg-green-50' : 'border-gray-300'}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {status.is_running ? (
                <CheckCircle2 className="h-5 w-5 text-green-600" />
              ) : (
                <AlertCircle className="h-5 w-5 text-gray-600" />
              )}
              <div>
                <AlertDescription className="font-semibold text-lg">
                  {status.is_running ? '🟢 Sistema Rodando' : '⏸️ Sistema Parado'}
                </AlertDescription>
                {status.is_running && status.start_time && (
                  <p className="text-sm text-muted-foreground mt-1">
                    Iniciado em {new Date(status.start_time).toLocaleString('pt-BR')} •
                    Duração: {formatDuration(status.duration_seconds)}
                  </p>
                )}
              </div>
            </div>

            {status.is_running && (
              <div className="flex gap-4">
                <div className="text-center">
                  <p className="text-2xl font-bold text-green-600">{status.total_predictions}</p>
                  <p className="text-xs text-muted-foreground">Previsões</p>
                </div>
                <div className="text-center">
                  <p className="text-2xl font-bold text-blue-600">{status.total_trades}</p>
                  <p className="text-xs text-muted-foreground">Trades</p>
                </div>
                <div className="text-center">
                  <p className="text-2xl font-bold text-red-600">{status.total_bugs}</p>
                  <p className="text-xs text-muted-foreground">Bugs</p>
                </div>
              </div>
            )}
          </div>
        </Alert>
      )}

      {/* Metrics Cards */}
      {status?.is_running && status.paper_trading_metrics && (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Capital Atual</CardTitle>
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                ${(status.paper_trading_metrics.current_capital ?? 0).toLocaleString()}
              </div>
              <p className={`text-xs ${(status.paper_trading_metrics.total_pnl ?? 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {(status.paper_trading_metrics.total_pnl ?? 0) >= 0 ? '+' : ''}
                ${(status.paper_trading_metrics.total_pnl ?? 0).toFixed(2)}
                ({(status.paper_trading_metrics.total_pnl_pct ?? 0) >= 0 ? '+' : ''}
                {(status.paper_trading_metrics.total_pnl_pct ?? 0).toFixed(2)}%)
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Win Rate</CardTitle>
              <Target className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {(status.paper_trading_metrics.win_rate_pct ?? 0).toFixed(1)}%
              </div>
              <p className="text-xs text-muted-foreground">
                {status.paper_trading_metrics.winning_trades ?? 0}W /
                {status.paper_trading_metrics.losing_trades ?? 0}L
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Sharpe Ratio</CardTitle>
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">
                {(status.paper_trading_metrics.sharpe_ratio ?? 0).toFixed(2)}
              </div>
              <p className="text-xs text-muted-foreground">
                Target: &gt; 1.5
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Max Drawdown</CardTitle>
              <AlertCircle className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-red-600">
                {(status.paper_trading_metrics.max_drawdown_pct ?? 0).toFixed(2)}%
              </div>
              <p className="text-xs text-muted-foreground">
                Target: &lt; 15%
              </p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Dashboard de Métricas em Tempo Real */}
      {status?.is_running && liveMetrics && (
        <>
          {/* Equity Curve */}
          <EquityCurveChart
            data={liveMetrics.equity_curve}
            initialCapital={liveMetrics.capital.initial}
          />

          {/* Live Metrics Grid */}
          <LiveMetricsGrid metrics={liveMetrics} />
        </>
      )}

      {/* Alert Notifications */}
      <AlertNotifications
        apiBaseUrl={API_BASE_URL}
        isRunning={status?.is_running || false}
        pollInterval={10000}
      />

      {/* Trade History Table */}
      <TradeHistoryTable
        apiBaseUrl={API_BASE_URL}
        isRunning={status?.is_running || false}
        pollInterval={30000}
      />

      {/* Mode Comparison */}
      <ModeComparison
        apiBaseUrl={API_BASE_URL}
        isRunning={status?.is_running || false}
      />

      {/* Recent Predictions */}
      {predictions.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5" />
              Previsões ML Recentes
            </CardTitle>
            <CardDescription>
              Últimas {predictions.length} previsões geradas pelo modelo
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Timestamp</TableHead>
                  <TableHead>Previsão</TableHead>
                  <TableHead>Confidence</TableHead>
                  <TableHead>Preço</TableHead>
                  <TableHead>Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {predictions.slice(0, 10).map((pred, idx) => (
                  <TableRow key={idx}>
                    <TableCell className="text-sm">
                      {new Date(pred.timestamp).toLocaleTimeString('pt-BR')}
                    </TableCell>
                    <TableCell>
                      <Badge variant={pred.prediction === 'UP' ? 'default' : 'destructive'}>
                        {pred.prediction}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <Badge variant={pred.confidence >= 0.60 ? 'default' : 'secondary'}>
                        {(pred.confidence * 100).toFixed(1)}%
                      </Badge>
                    </TableCell>
                    <TableCell className="font-mono">
                      ${pred.price.toFixed(4)}
                    </TableCell>
                    <TableCell>
                      {pred.confidence >= 0.60 ? (
                        <CheckCircle2 className="h-4 w-4 text-green-500" />
                      ) : (
                        <Clock className="h-4 w-4 text-gray-400" />
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}

      {/* Bugs Log */}
      {bugs.length > 0 && (
        <Card className="border-red-200">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-red-600">
              <Bug className="h-5 w-5" />
              Bugs Registrados ({bugs.length})
            </CardTitle>
            <CardDescription>
              Problemas encontrados durante a execução
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Timestamp</TableHead>
                  <TableHead>Tipo</TableHead>
                  <TableHead>Severidade</TableHead>
                  <TableHead>Descrição</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {bugs.slice(0, 10).map((bug, idx) => (
                  <TableRow key={idx}>
                    <TableCell className="text-sm">
                      {new Date(bug.timestamp).toLocaleString('pt-BR')}
                    </TableCell>
                    <TableCell className="font-mono text-sm">
                      {bug.type}
                    </TableCell>
                    <TableCell>
                      <Badge variant={
                        bug.severity === 'CRITICAL' ? 'destructive' :
                        bug.severity === 'ERROR' ? 'default' : 'secondary'
                      }>
                        {bug.severity}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-sm max-w-md truncate">
                      {bug.description}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}

      {/* Logs Download */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <FileDown className="h-5 w-5" />
                Logs e Relatórios
              </CardTitle>
              <CardDescription>
                Download de logs de execução (.log) e relatórios de validação (.md)
              </CardDescription>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={loadLogs}
              disabled={isLoadingLogs}
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${isLoadingLogs ? 'animate-spin' : ''}`} />
              Atualizar
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {isLoadingLogs ? (
            <div className="text-center py-8 text-muted-foreground">
              <RefreshCw className="h-6 w-6 animate-spin mx-auto mb-2" />
              Carregando logs...
            </div>
          ) : logs.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <FileText className="h-12 w-12 mx-auto mb-3 opacity-30" />
              <p>Nenhum log disponível</p>
              <p className="text-sm">Inicie o Forward Testing para gerar logs</p>
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Nome do Arquivo</TableHead>
                  <TableHead>Tamanho</TableHead>
                  <TableHead>Última Modificação</TableHead>
                  <TableHead className="text-right">Ações</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {logs.map((log) => (
                  <TableRow key={log.filename}>
                    <TableCell className="font-mono text-sm">
                      <div className="flex items-center gap-2">
                        {log.filename}
                        {log.filename.endsWith('.md') && (
                          <Badge variant="secondary" className="text-xs">
                            Relatório
                          </Badge>
                        )}
                      </div>
                    </TableCell>
                    <TableCell className="text-sm">
                      {formatFileSize(log.size_bytes)}
                    </TableCell>
                    <TableCell className="text-sm">
                      {formatDate(log.modified_at)}
                    </TableCell>
                    <TableCell className="text-right">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleDownloadLog(log.filename)}
                      >
                        <Download className="h-4 w-4 mr-2" />
                        Baixar
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      {/* Info Card */}
      <Card className="border-blue-200 bg-blue-50">
        <CardContent className="pt-6">
          <div className="flex gap-3">
            <Brain className="h-5 w-5 text-blue-600 flex-shrink-0 mt-0.5" />
            <div className="space-y-2">
              <p className="text-sm font-medium text-blue-900">
                Sobre o Forward Testing
              </p>
              <p className="text-sm text-blue-700">
                Este sistema integra o ML Predictor com o Paper Trading Engine para validar
                a estratégia em condições reais de mercado. O sistema coleta dados do Deriv API,
                gera previsões ML e executa trades automaticamente quando a confidence é alta (&gt;60%).
              </p>
              <p className="text-sm text-blue-700">
                <strong>Critérios de Aprovação:</strong> Win Rate &gt; 60%, Sharpe Ratio &gt; 1.5,
                Max Drawdown &lt; 15%, Profit Factor &gt; 1.5
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
