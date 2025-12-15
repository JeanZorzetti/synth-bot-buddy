import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
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
  Square,
  RotateCcw,
  TrendingUp,
  TrendingDown,
  DollarSign,
  Activity,
  Target,
  AlertCircle,
  Clock,
  BarChart3,
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  AreaChart,
  Area,
} from 'recharts';

// Types
interface PaperTradingMetrics {
  status: string;
  uptime_seconds: number;
  initial_capital: number;
  current_capital: number;
  total_pnl: number;
  total_pnl_pct: number;
  peak_capital: number;
  max_drawdown_pct: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate_pct: number;
  profit_factor: number;
  sharpe_ratio: number;
  open_positions: number;
  avg_profit_per_trade: number;
}

interface Position {
  id: string;
  symbol: string;
  position_type: string;
  entry_price: number;
  size: number;
  entry_time: string;
  status: string;
  stop_loss?: number;
  take_profit?: number;
  profit_loss: number;
  profit_loss_pct: number;
}

interface Trade {
  id: string;
  symbol: string;
  position_type: string;
  entry_price: number;
  exit_price: number;
  size: number;
  entry_time: string;
  exit_time: string;
  profit_loss: number;
  profit_loss_pct: number;
  is_winner: boolean;
}

interface EquityPoint {
  timestamp: string;
  capital: number;
  profit_loss: number;
  trade_id: string;
}

export default function PaperTrading() {
  const { toast } = useToast();

  // State
  const [metrics, setMetrics] = useState<PaperTradingMetrics | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [equityCurve, setEquityCurve] = useState<EquityPoint[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isRunning, setIsRunning] = useState(false);

  // API Base URL
  const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  useEffect(() => {
    loadStatus();

    // Atualizar a cada 5 segundos se estiver rodando
    const interval = setInterval(() => {
      if (isRunning) {
        loadStatus();
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [isRunning]);

  const loadStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/paper-trading/status`);
      if (!response.ok) throw new Error('Failed to load status');

      const data = await response.json();
      setMetrics(data.metrics);
      setPositions(data.open_positions || []);
      setTrades(data.recent_trades || []);
      setIsRunning(data.metrics.status === 'running');

      // Carregar equity curve
      if (data.metrics.total_trades > 0) {
        loadEquityCurve();
      }
    } catch (error) {
      console.error('Error loading status:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const loadEquityCurve = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/paper-trading/equity-curve`);
      if (!response.ok) return;

      const data = await response.json();
      setEquityCurve(data.equity_curve || []);
    } catch (error) {
      console.error('Error loading equity curve:', error);
    }
  };

  const handleStart = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/paper-trading/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          reset: true,
          initial_capital: 10000,
          execution_latency_ms: 100,
          slippage_pct: 0.1,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to start');
      }

      const data = await response.json();
      setMetrics(data.metrics);
      setIsRunning(true);

      toast({
        title: 'Paper Trading Iniciado',
        description: `Capital inicial: ${formatCurrency(data.config.initial_capital)}`,
      });

      await loadStatus();
    } catch (error: any) {
      toast({
        title: 'Erro ao Iniciar',
        description: error.message,
        variant: 'destructive',
      });
    }
  };

  const handleStop = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/paper-trading/stop`, {
        method: 'POST',
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to stop');
      }

      const data = await response.json();
      setMetrics(data.final_metrics);
      setIsRunning(false);

      toast({
        title: 'Paper Trading Parado',
        description: `P&L Final: ${formatCurrency(data.final_metrics.total_pnl)} (${formatPercent(data.final_metrics.total_pnl_pct)})`,
      });

      await loadStatus();
    } catch (error: any) {
      toast({
        title: 'Erro ao Parar',
        description: error.message,
        variant: 'destructive',
      });
    }
  };

  const handleReset = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/paper-trading/reset`, {
        method: 'POST',
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to reset');
      }

      setMetrics(null);
      setPositions([]);
      setTrades([]);
      setEquityCurve([]);
      setIsRunning(false);

      toast({
        title: 'Paper Trading Resetado',
        description: 'Todas as métricas foram zeradas.',
      });

      await loadStatus();
    } catch (error: any) {
      toast({
        title: 'Erro ao Resetar',
        description: error.message,
        variant: 'destructive',
      });
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('pt-BR', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatPercent = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <Activity className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p className="text-sm text-muted-foreground">Carregando Paper Trading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Paper Trading</h1>
          <p className="text-muted-foreground">
            Simulação realista de trading com capital virtual
          </p>
        </div>

        <div className="flex gap-2">
          <Button
            onClick={handleStart}
            disabled={isRunning}
            className="gap-2"
          >
            <Play className="h-4 w-4" />
            Iniciar
          </Button>
          <Button
            onClick={handleStop}
            disabled={!isRunning}
            variant="destructive"
            className="gap-2"
          >
            <Square className="h-4 w-4" />
            Parar
          </Button>
          <Button
            onClick={handleReset}
            variant="outline"
            className="gap-2"
          >
            <RotateCcw className="h-4 w-4" />
            Resetar
          </Button>
        </div>
      </div>

      {/* Status Badge */}
      <div className="flex items-center gap-4">
        <Badge variant={isRunning ? 'default' : 'secondary'} className="text-sm">
          {isRunning ? '🟢 Rodando' : '🔴 Parado'}
        </Badge>
        {metrics && metrics.uptime_seconds > 0 && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Clock className="h-4 w-4" />
            <span>Tempo: {formatDuration(metrics.uptime_seconds)}</span>
          </div>
        )}
      </div>

      {/* Metrics Cards */}
      {metrics && (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Capital Atual</CardTitle>
              <DollarSign className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{formatCurrency(metrics.current_capital)}</div>
              <p className="text-xs text-muted-foreground">
                Inicial: {formatCurrency(metrics.initial_capital)}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">P&L Total</CardTitle>
              {metrics.total_pnl >= 0 ? (
                <TrendingUp className="h-4 w-4 text-green-500" />
              ) : (
                <TrendingDown className="h-4 w-4 text-red-500" />
              )}
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold ${metrics.total_pnl >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                {formatCurrency(metrics.total_pnl)}
              </div>
              <p className="text-xs text-muted-foreground">
                {formatPercent(metrics.total_pnl_pct)}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Win Rate</CardTitle>
              <Target className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{(metrics?.win_rate_pct ?? 0).toFixed(1)}%</div>
              <p className="text-xs text-muted-foreground">
                {metrics?.winning_trades ?? 0}W / {metrics?.losing_trades ?? 0}L
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Sharpe Ratio</CardTitle>
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{(metrics?.sharpe_ratio ?? 0).toFixed(2)}</div>
              <p className="text-xs text-muted-foreground">
                Drawdown: {(metrics?.max_drawdown_pct ?? 0).toFixed(2)}%
              </p>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Equity Curve */}
      {equityCurve.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Curva de Equity</CardTitle>
            <CardDescription>
              Evolução do capital ao longo do tempo
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={400}>
              <AreaChart data={equityCurve}>
                <defs>
                  <linearGradient id="colorCapital" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#8884d8" stopOpacity={0.8} />
                    <stop offset="95%" stopColor="#8884d8" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="timestamp"
                  tickFormatter={(value) => {
                    const date = new Date(value);
                    return date.toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit' });
                  }}
                />
                <YAxis
                  tickFormatter={(value) => formatCurrency(value)}
                />
                <Tooltip
                  formatter={(value: number) => formatCurrency(value)}
                  labelFormatter={(label) => new Date(label).toLocaleString('pt-BR')}
                />
                <Area
                  type="monotone"
                  dataKey="capital"
                  stroke="#8884d8"
                  fillOpacity={1}
                  fill="url(#colorCapital)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Open Positions */}
      {positions.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Posições Abertas ({positions.length})</CardTitle>
            <CardDescription>
              Posições ativas no momento
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Símbolo</TableHead>
                  <TableHead>Tipo</TableHead>
                  <TableHead>Tamanho</TableHead>
                  <TableHead>Entrada</TableHead>
                  <TableHead>Stop Loss</TableHead>
                  <TableHead>Take Profit</TableHead>
                  <TableHead>Abertura</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {positions.map((position) => (
                  <TableRow key={position.id}>
                    <TableCell className="font-medium">{position.symbol}</TableCell>
                    <TableCell>
                      <Badge variant={position.position_type === 'LONG' ? 'default' : 'destructive'}>
                        {position.position_type}
                      </Badge>
                    </TableCell>
                    <TableCell>{formatCurrency(position?.size ?? 0)}</TableCell>
                    <TableCell>{(position?.entry_price ?? 0).toFixed(4)}</TableCell>
                    <TableCell>
                      {position?.stop_loss ? position.stop_loss.toFixed(4) : '-'}
                    </TableCell>
                    <TableCell>
                      {position?.take_profit ? position.take_profit.toFixed(4) : '-'}
                    </TableCell>
                    <TableCell>
                      {new Date(position.entry_time).toLocaleTimeString('pt-BR')}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}

      {/* Recent Trades */}
      {trades.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Últimos Trades ({metrics?.total_trades || 0})</CardTitle>
            <CardDescription>
              Histórico de trades executados
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Símbolo</TableHead>
                  <TableHead>Tipo</TableHead>
                  <TableHead>Entrada</TableHead>
                  <TableHead>Saída</TableHead>
                  <TableHead>P&L</TableHead>
                  <TableHead>P&L %</TableHead>
                  <TableHead>Resultado</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {trades.map((trade) => (
                  <TableRow key={trade.id}>
                    <TableCell className="font-medium">{trade.symbol}</TableCell>
                    <TableCell>
                      <Badge variant={trade.position_type === 'LONG' ? 'default' : 'destructive'}>
                        {trade.position_type}
                      </Badge>
                    </TableCell>
                    <TableCell>{(trade?.entry_price ?? 0).toFixed(4)}</TableCell>
                    <TableCell>{(trade?.exit_price ?? 0).toFixed(4)}</TableCell>
                    <TableCell className={(trade?.profit_loss ?? 0) >= 0 ? 'text-green-500' : 'text-red-500'}>
                      {formatCurrency(trade?.profit_loss ?? 0)}
                    </TableCell>
                    <TableCell className={(trade?.profit_loss_pct ?? 0) >= 0 ? 'text-green-500' : 'text-red-500'}>
                      {formatPercent(trade?.profit_loss_pct ?? 0)}
                    </TableCell>
                    <TableCell>
                      <Badge variant={trade.is_winner ? 'default' : 'destructive'}>
                        {trade.is_winner ? '✓ Win' : '✗ Loss'}
                      </Badge>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}

      {/* Info Note */}
      {!metrics || metrics.total_trades === 0 ? (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertCircle className="h-5 w-5" />
              Como Usar
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <p className="text-sm text-muted-foreground">
              1. Clique em <strong>Iniciar</strong> para começar uma sessão de paper trading com $10,000 de capital virtual
            </p>
            <p className="text-sm text-muted-foreground">
              2. O sistema simulará latência de execução (~100ms) e slippage (0.1%) realistas
            </p>
            <p className="text-sm text-muted-foreground">
              3. Trades serão executados automaticamente baseados nos sinais ML do bot
            </p>
            <p className="text-sm text-muted-foreground">
              4. Clique em <strong>Parar</strong> para encerrar a sessão e ver resultados finais
            </p>
            <p className="text-sm text-muted-foreground">
              5. Use <strong>Resetar</strong> para zerar todas as métricas e começar do zero
            </p>
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}
