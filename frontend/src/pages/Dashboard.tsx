import { useState, useEffect } from 'react';
import { Layout } from '@/components/Layout';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import DerivStatus from '@/components/DerivStatus';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  Play, 
  Pause, 
  TrendingUp, 
  TrendingDown, 
  Activity,
  DollarSign,
  Target,
  BarChart3,
  AlertCircle,
  CheckCircle2,
  Settings,
  Wifi,
  WifiOff,
  Loader2
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { useBot } from '@/hooks/useBot';
import { Alert, AlertDescription } from '@/components/ui/alert';

interface TradeData {
  time: string;
  balance: number;
}

interface ActivityLog {
  id: string;
  timestamp: string;
  message: string;
  type: 'info' | 'success' | 'error';
}

export default function Dashboard() {
  const {
    botStatus,
    isLoading,
    isApiAvailable,
    startBot,
    stopBot,
    connectToApi,
    isStarting,
    isStopping,
    isConnecting,
    buyContract
  } = useBot();
  
  const [uptime, setUptime] = useState(0);
  
  const [activityLog, setActivityLog] = useState<ActivityLog[]>([
    {
      id: '1',
      timestamp: '14:23:15',
      message: 'Bot inicializado com sucesso',
      type: 'success'
    },
    {
      id: '2', 
      timestamp: '14:23:45',
      message: 'Analisando Volatility 75 Index',
      type: 'info'
    },
    {
      id: '3',
      timestamp: '14:24:10',
      message: 'Ordem CALL executada - $10.00',
      type: 'success'
    },
    {
      id: '4',
      timestamp: '14:24:35',
      message: 'Trade finalizado - Lucro: $8.50',
      type: 'success'
    }
  ]);

  // Get real data from backend
  const botRunning = botStatus?.is_running || false;
  const balance = botStatus?.balance || 0;
  const sessionPnL = botStatus?.session_pnl || 0;
  const totalTrades = botStatus?.trades_count || 0;
  const lastTick = botStatus?.last_tick;
  const connectionStatus = botStatus?.connection_status || 'disconnected';
  
  // Calculate win rate (placeholder calculation)
  const winRate = totalTrades > 0 ? Math.max(0, (sessionPnL / (totalTrades * 10)) * 100 + 50) : 0;

  // Update activity log based on real data
  useEffect(() => {
    if (lastTick && lastTick.timestamp) {
      const newActivity: ActivityLog = {
        id: `tick-${lastTick.timestamp}`,
        timestamp: new Date(lastTick.timestamp * 1000).toLocaleTimeString('pt-BR', { hour12: false }),
        message: `Tick recebido: ${lastTick.symbol} = ${lastTick.price}`,
        type: 'info'
      };
      
      setActivityLog(prev => {
        // Avoid duplicates
        if (prev[0]?.id === newActivity.id) return prev;
        return [newActivity, ...prev.slice(0, 9)];
      });
    }
  }, [lastTick]);

  // Update uptime counter
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (botRunning) {
      interval = setInterval(() => {
        setUptime(prev => prev + 1);
      }, 1000);
    } else {
      setUptime(0);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [botRunning]);

  const toggleBot = () => {
    if (botRunning) {
      stopBot();
    } else {
      startBot();
    }
  };

  const handleConnect = () => {
    connectToApi();
  };

  const handleBuyCall = () => {
    buyContract({
      contract_type: 'CALL',
      amount: 10,
      duration: 5,
      symbol: 'R_75'
    });
  };

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'success':
        return <CheckCircle2 className="h-4 w-4 text-success" />;
      case 'error':
        return <AlertCircle className="h-4 w-4 text-danger" />;
      default:
        return <Activity className="h-4 w-4 text-primary" />;
    }
  };

  return (
    <Layout>
      <div className="space-y-6">
        {/* Page Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Dashboard</h1>
            <p className="text-muted-foreground">
              Monitore o desempenho do seu bot de trading em tempo real
            </p>
          </div>
          <div className="flex items-center gap-2">
            {isApiAvailable ? (
              <Badge variant="default" className="gap-1">
                <Wifi className="h-3 w-3" />
                Backend Online
              </Badge>
            ) : (
              <Badge variant="destructive" className="gap-1">
                <WifiOff className="h-3 w-3" />
                Backend Offline
              </Badge>
            )}
          </div>
        </div>

        {/* Connection Status Alert */}
        {!isApiAvailable && (
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              Backend não está disponível. Para usar o sistema:
              <br />
              1. Execute <code className="bg-muted px-1 rounded">python start.py</code> na pasta backend
              <br />
              2. Configure seu token Deriv no arquivo .env
            </AlertDescription>
          </Alert>
        )}

        {isApiAvailable && connectionStatus === 'disconnected' && (
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription className="flex items-center justify-between">
              <span>Desconectado da API Deriv. Clique para conectar.</span>
              <Button 
                onClick={handleConnect} 
                size="sm" 
                disabled={isConnecting}
                className="ml-2"
              >
                {isConnecting ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Conectando...
                  </>
                ) : (
                  <>
                    <Wifi className="mr-2 h-4 w-4" />
                    Conectar
                  </>
                )}
              </Button>
            </AlertDescription>
          </Alert>
        )}

        {isLoading && (
          <Alert>
            <Loader2 className="h-4 w-4 animate-spin" />
            <AlertDescription>
              Carregando status do bot...
            </AlertDescription>
          </Alert>
        )}

        {/* Main Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {/* Bot Status Card */}
          <Card className="trading-card">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center justify-between text-base">
                Status do Bot
                <div className={cn(
                  "status-dot",
                  botRunning ? "status-running" : "status-stopped"
                )} />
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="text-center">
                <Badge 
                  variant={botRunning ? "default" : "secondary"}
                  className={cn(
                    "text-sm px-3 py-1",
                    botRunning && "bg-success text-success-foreground"
                  )}
                >
                  {botRunning ? 'Rodando' : 'Parado'}
                </Badge>
                <div className="mt-2 text-sm text-muted-foreground">
                  Uptime: {formatUptime(uptime)}
                </div>
              </div>
              <Button 
                onClick={toggleBot}
                disabled={!isApiAvailable || connectionStatus === 'disconnected' || isStarting || isStopping}
                className={cn(
                  "w-full",
                  botRunning 
                    ? "bg-danger hover:bg-danger/90 text-danger-foreground" 
                    : "success-gradient hover:opacity-90"
                )}
              >
                {isStarting ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Iniciando...
                  </>
                ) : isStopping ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Parando...
                  </>
                ) : botRunning ? (
                  <>
                    <Pause className="h-4 w-4 mr-2" />
                    Parar Bot
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Iniciar Bot
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {/* Session P/L Card */}
          <Card className="trading-card">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center text-base">
                <DollarSign className="h-5 w-5 mr-2" />
                P/L da Sessão
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center">
                <div className={cn(
                  "text-2xl font-bold",
                  sessionPnL >= 0 ? "text-success" : "text-danger"
                )}>
                  {sessionPnL >= 0 ? '+' : ''}${sessionPnL.toFixed(2)}
                </div>
                <div className="flex items-center justify-center mt-2 text-sm">
                  {sessionPnL >= 0 ? (
                    <TrendingUp className="h-4 w-4 text-success mr-1" />
                  ) : (
                    <TrendingDown className="h-4 w-4 text-danger mr-1" />
                  )}
                  <span className={sessionPnL >= 0 ? "text-success" : "text-danger"}>
                    {Math.abs((sessionPnL / balance) * 100).toFixed(2)}%
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Win Rate Card */}
          <Card className="trading-card">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center text-base">
                <Target className="h-5 w-5 mr-2" />
                Taxa de Vitórias
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center">
                <div className="text-2xl font-bold">
                  {winRate.toFixed(1)}%
                </div>
                <div className="mt-2 text-sm text-muted-foreground">
                  {totalTrades} trades realizados
                </div>
                <div className="mt-3 bg-muted rounded-full h-2">
                  <div 
                    className="bg-success h-2 rounded-full transition-all duration-300"
                    style={{ width: `${winRate}%` }}
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Balance Card */}
          <Card className="trading-card">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center text-base">
                <BarChart3 className="h-5 w-5 mr-2" />
                Saldo da Conta
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center">
                <div className="text-2xl font-bold">
                  ${balance.toFixed(2)}
                </div>
                <div className="mt-2 text-sm text-muted-foreground">
                  Deriv Account
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Activity Feed */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <Card className="lg:col-span-2 trading-card">
            <CardHeader>
              <CardTitle>Feed de Atividades em Tempo Real</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {activityLog.map((activity) => (
                  <div 
                    key={activity.id}
                    className="flex items-start space-x-3 p-3 rounded-lg bg-muted/30"
                  >
                    {getActivityIcon(activity.type)}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium">
                        {activity.message}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {activity.timestamp}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Quick Actions */}
          <Card className="trading-card">
            <CardHeader>
              <CardTitle>Ações Rápidas</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Button 
                variant="outline" 
                className="w-full justify-start"
                onClick={() => window.location.href = '/settings'}
              >
                <Settings className="h-4 w-4 mr-2" />
                Configurar Estratégia
              </Button>
              <Button 
                variant="outline" 
                className="w-full justify-start"
                onClick={() => window.location.href = '/history'}
              >
                <BarChart3 className="h-4 w-4 mr-2" />
                Ver Histórico
              </Button>
              <Button 
                variant="outline" 
                className="w-full justify-start"
                onClick={() => window.location.href = '/performance'}
              >
                <Activity className="h-4 w-4 mr-2" />
                Relatório de Performance
              </Button>
              <Button 
                variant="default" 
                className="w-full justify-start bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 text-white"
                onClick={() => window.location.href = '/trading'}
              >
                <DollarSign className="h-4 w-4 mr-2" />
                Trading Real (Deriv API)
              </Button>
              {connectionStatus === 'authenticated' && (
                <Button 
                  variant="outline" 
                  className="w-full justify-start"
                  onClick={handleBuyCall}
                  disabled={!botRunning}
                >
                  <TrendingUp className="h-4 w-4 mr-2" />
                  Teste: Comprar CALL
                </Button>
              )}
            </CardContent>
          </Card>

          {/* Deriv API Status */}
          <DerivStatus />
        </div>
      </div>
    </Layout>
  );
}