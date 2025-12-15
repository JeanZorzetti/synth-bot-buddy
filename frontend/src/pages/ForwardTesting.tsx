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
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

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

export default function ForwardTesting() {
  const { toast } = useToast();
  const [status, setStatus] = useState<ForwardTestingStatus | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [bugs, setBugs] = useState<Bug[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isStarting, setIsStarting] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);

  const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://botderivapi.roilabs.com.br';

  useEffect(() => {
    loadStatus();
    const interval = setInterval(loadStatus, 5000); // Atualizar a cada 5 segundos
    return () => clearInterval(interval);
  }, []);

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

      const response = await fetch(`${API_BASE_URL}/api/forward-testing/start`, {
        method: 'POST',
      });

      const data = await response.json();

      if (data.status === 'success') {
        toast({
          title: 'Forward Testing Iniciado',
          description: 'Sistema rodando com ML Predictor + Paper Trading',
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
