import { useState, useEffect } from 'react';
import { Layout } from '@/components/Layout';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  Activity,
  TrendingUp,
  TrendingDown,
  Target,
  DollarSign,
  BarChart3,
  AlertTriangle,
  RefreshCw,
  Download,
  Loader2
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { apiService } from '@/services/api';
import { useToast } from '@/hooks/use-toast';

interface PerformanceData {
  session_stats: {
    session_pnl: number;
    total_trades: number;
    active_trades: number;
    wins: number;
    losses: number;
    win_rate: number;
    total_invested: number;
    total_returned: number;
    net_profit: number;
  };
  risk_metrics: {
    current_risk_level: string;
    is_in_loss_sequence: boolean;
    next_trade_amount: number;
    accumulated_profit: number;
  };
  trading_engine: {
    is_running: boolean;
    active_since: number | null;
  };
}

export default function Performance() {
  const { toast } = useToast();
  const [performanceData, setPerformanceData] = useState<PerformanceData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const loadPerformanceData = async () => {
    try {
      const data = await apiService.get<PerformanceData>('/trading/performance');
      setPerformanceData(data);
      setLastUpdated(new Date());
    } catch (error: any) {
      console.error('Failed to load performance data:', error);
      toast({
        title: "Erro ao carregar dados",
        description: "Não foi possível carregar os dados de performance. Verifique se o trading engine está ativo.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  };

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await loadPerformanceData();
  };

  const handleResetSession = async () => {
    try {
      await apiService.post('/trading/reset-session');
      toast({
        title: "Sessão Resetada",
        description: "A sessão de trading foi resetada com sucesso.",
      });
      await loadPerformanceData();
    } catch (error: any) {
      toast({
        title: "Erro ao resetar sessão",
        description: error.message || "Não foi possível resetar a sessão.",
        variant: "destructive",
      });
    }
  };

  useEffect(() => {
    loadPerformanceData();
    
    // Auto-refresh every 10 seconds
    const interval = setInterval(loadPerformanceData, 10000);
    return () => clearInterval(interval);
  }, []);

  if (isLoading) {
    return (
      <Layout>
        <div className="flex items-center justify-center h-64">
          <div className="flex flex-col items-center space-y-4">
            <Loader2 className="h-8 w-8 animate-spin" />
            <p className="text-muted-foreground">Carregando dados de performance...</p>
          </div>
        </div>
      </Layout>
    );
  }

  if (!performanceData) {
    return (
      <Layout>
        <div className="flex items-center justify-center h-64">
          <div className="flex flex-col items-center space-y-4">
            <AlertTriangle className="h-12 w-12 text-muted-foreground" />
            <p className="text-muted-foreground">Dados de performance não disponíveis</p>
            <Button onClick={handleRefresh}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Tentar Novamente
            </Button>
          </div>
        </div>
      </Layout>
    );
  }

  const { session_stats, risk_metrics, trading_engine } = performanceData;
  
  // Calculate additional metrics
  const profitMargin = session_stats.total_invested > 0 
    ? (session_stats.net_profit / session_stats.total_invested) * 100 
    : 0;

  const avgWinSize = session_stats.wins > 0 
    ? (session_stats.total_returned - session_stats.total_invested + (session_stats.losses * session_stats.total_invested / session_stats.total_trades)) / session_stats.wins
    : 0;

  const avgLossSize = session_stats.losses > 0 
    ? session_stats.total_invested / session_stats.total_trades
    : 0;

  const getRiskLevelColor = (level: string) => {
    switch (level.toLowerCase()) {
      case 'low': return 'text-success';
      case 'medium': return 'text-warning';
      case 'high': return 'text-danger';
      default: return 'text-muted-foreground';
    }
  };

  return (
    <Layout>
      <div className="space-y-6">
        {/* Page Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Relatório de Performance</h1>
            <p className="text-muted-foreground">
              Análise detalhada do desempenho do trading automatizado
            </p>
            {lastUpdated && (
              <p className="text-xs text-muted-foreground mt-1">
                Última atualização: {lastUpdated.toLocaleTimeString('pt-BR')}
              </p>
            )}
          </div>
          <div className="flex gap-2">
            <Button variant="outline" onClick={handleRefresh} disabled={isRefreshing}>
              <RefreshCw className={cn("h-4 w-4 mr-2", isRefreshing && "animate-spin")} />
              Atualizar
            </Button>
            <Button variant="outline" onClick={handleResetSession}>
              <AlertTriangle className="h-4 w-4 mr-2" />
              Resetar Sessão
            </Button>
            <Button variant="outline">
              <Download className="h-4 w-4 mr-2" />
              Exportar
            </Button>
          </div>
        </div>

        {/* Trading Engine Status */}
        <Card className="trading-card">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div className={cn(
                  "status-dot",
                  trading_engine.is_running ? "status-running" : "status-stopped"
                )} />
                <div>
                  <h3 className="font-semibold">Status do Trading Engine</h3>
                  <p className="text-sm text-muted-foreground">
                    {trading_engine.is_running ? 'Ativo e operando' : 'Inativo'}
                  </p>
                </div>
              </div>
              <Badge variant={trading_engine.is_running ? "default" : "secondary"}>
                {trading_engine.is_running ? 'RODANDO' : 'PARADO'}
              </Badge>
            </div>
          </CardContent>
        </Card>

        {/* Performance Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {/* Session P/L */}
          <Card className="trading-card">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">P/L da Sessão</p>
                  <p className={cn(
                    "text-2xl font-bold",
                    session_stats.session_pnl >= 0 ? "text-success" : "text-danger"
                  )}>
                    {session_stats.session_pnl >= 0 ? '+' : ''}${session_stats.session_pnl.toFixed(2)}
                  </p>
                </div>
                <DollarSign className={cn(
                  "h-8 w-8",
                  session_stats.session_pnl >= 0 ? "text-success" : "text-danger"
                )} />
              </div>
            </CardContent>
          </Card>

          {/* Win Rate */}
          <Card className="trading-card">
            <CardContent className="p-6">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <p className="text-sm text-muted-foreground">Taxa de Vitórias</p>
                  <Target className="h-5 w-5 text-muted-foreground" />
                </div>
                <p className="text-2xl font-bold">{session_stats.win_rate.toFixed(1)}%</p>
                <Progress value={session_stats.win_rate} className="h-2" />
                <p className="text-xs text-muted-foreground">
                  {session_stats.wins}W / {session_stats.losses}L de {session_stats.total_trades} trades
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Net Profit */}
          <Card className="trading-card">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Lucro Líquido</p>
                  <p className={cn(
                    "text-2xl font-bold",
                    session_stats.net_profit >= 0 ? "text-success" : "text-danger"
                  )}>
                    {session_stats.net_profit >= 0 ? '+' : ''}${session_stats.net_profit.toFixed(2)}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Margem: {profitMargin.toFixed(1)}%
                  </p>
                </div>
                {session_stats.net_profit >= 0 ? (
                  <TrendingUp className="h-8 w-8 text-success" />
                ) : (
                  <TrendingDown className="h-8 w-8 text-danger" />
                )}
              </div>
            </CardContent>
          </Card>

          {/* Risk Level */}
          <Card className="trading-card">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Nível de Risco</p>
                  <p className={cn("text-2xl font-bold", getRiskLevelColor(risk_metrics.current_risk_level))}>
                    {risk_metrics.current_risk_level}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Próximo trade: ${risk_metrics.next_trade_amount.toFixed(2)}
                  </p>
                </div>
                <AlertTriangle className={cn("h-8 w-8", getRiskLevelColor(risk_metrics.current_risk_level))} />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Detailed Analytics */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Trading Statistics */}
          <Card className="trading-card">
            <CardHeader>
              <CardTitle className="flex items-center">
                <BarChart3 className="h-5 w-5 mr-2" />
                Estatísticas de Trading
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Total de Trades</span>
                <span className="font-medium">{session_stats.total_trades}</span>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Trades Ativos</span>
                <span className="font-medium">{session_stats.active_trades}</span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Total Investido</span>
                <span className="font-medium">${session_stats.total_invested.toFixed(2)}</span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Total Retornado</span>
                <span className="font-medium">${session_stats.total_returned.toFixed(2)}</span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Média de Lucro por Trade</span>
                <span className={cn(
                  "font-medium",
                  avgWinSize >= 0 ? "text-success" : "text-danger"
                )}>
                  ${avgWinSize.toFixed(2)}
                </span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Média de Perda por Trade</span>
                <span className="font-medium text-danger">
                  -${avgLossSize.toFixed(2)}
                </span>
              </div>
            </CardContent>
          </Card>

          {/* Risk Management */}
          <Card className="trading-card">
            <CardHeader>
              <CardTitle className="flex items-center">
                <AlertTriangle className="h-5 w-5 mr-2" />
                Gestão de Risco
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Nível de Risco Atual</span>
                <Badge variant={
                  risk_metrics.current_risk_level.toLowerCase() === 'low' ? 'default' :
                  risk_metrics.current_risk_level.toLowerCase() === 'medium' ? 'secondary' : 'destructive'
                }>
                  {risk_metrics.current_risk_level}
                </Badge>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Em Sequência de Perda</span>
                <Badge variant={risk_metrics.is_in_loss_sequence ? "destructive" : "default"}>
                  {risk_metrics.is_in_loss_sequence ? 'SIM' : 'NÃO'}
                </Badge>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Próximo Valor de Trade</span>
                <span className="font-medium">${risk_metrics.next_trade_amount.toFixed(2)}</span>
              </div>

              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Lucro Acumulado</span>
                <span className={cn(
                  "font-medium",
                  risk_metrics.accumulated_profit >= 0 ? "text-success" : "text-danger"
                )}>
                  {risk_metrics.accumulated_profit >= 0 ? '+' : ''}${risk_metrics.accumulated_profit.toFixed(2)}
                </span>
              </div>

              {/* Risk Level Indicator */}
              <div className="pt-4">
                <div className="flex justify-between text-sm mb-2">
                  <span>Indicador de Risco</span>
                  <span>{risk_metrics.current_risk_level}</span>
                </div>
                <Progress 
                  value={
                    risk_metrics.current_risk_level.toLowerCase() === 'low' ? 25 :
                    risk_metrics.current_risk_level.toLowerCase() === 'medium' ? 65 : 90
                  }
                  className={cn(
                    "h-3",
                    risk_metrics.current_risk_level.toLowerCase() === 'high' && "bg-red-200"
                  )}
                />
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Performance Alerts */}
        {(risk_metrics.current_risk_level === 'HIGH' || risk_metrics.is_in_loss_sequence) && (
          <Card className="trading-card border-warning bg-warning/5">
            <CardContent className="p-6">
              <div className="flex items-start space-x-4">
                <AlertTriangle className="h-5 w-5 text-warning mt-0.5" />
                <div>
                  <h3 className="font-semibold text-warning">Alerta de Risco</h3>
                  <div className="text-sm text-muted-foreground mt-1 space-y-1">
                    {risk_metrics.current_risk_level === 'HIGH' && (
                      <p>• Nível de risco alto detectado. Considere revisar a estratégia.</p>
                    )}
                    {risk_metrics.is_in_loss_sequence && (
                      <p>• Bot em sequência de perdas. Monitoramento ativo recomendado.</p>
                    )}
                    <p>• Próximo trade será executado com ${risk_metrics.next_trade_amount.toFixed(2)}.</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </Layout>
  );
}