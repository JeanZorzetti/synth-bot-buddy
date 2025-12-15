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
  LineChart,
  BarChart3,
  TrendingUp,
  TrendingDown,
  RefreshCw,
  AlertCircle,
  CheckCircle2,
  Activity,
  DollarSign,
  Target,
  Zap,
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import {
  backtestingApi,
  type EquityCurveResponse,
  type BacktestWindowsResponse,
  type BacktestWindow,
} from '@/services/api';
import {
  LineChart as RechartsLineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Area,
  AreaChart,
} from 'recharts';

export default function BacktestingVisual() {
  const { toast } = useToast();

  // Data states
  const [equityCurve, setEquityCurve] = useState<EquityCurveResponse | null>(null);
  const [windows, setWindows] = useState<BacktestWindowsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);

  useEffect(() => {
    loadBacktestingData();
  }, []);

  const loadBacktestingData = async () => {
    try {
      setIsRefreshing(true);

      const [equityData, windowsData] = await Promise.all([
        backtestingApi.getEquityCurve(),
        backtestingApi.getWindows(),
      ]);

      setEquityCurve(equityData);
      setWindows(windowsData);

      toast({
        title: 'Dados carregados',
        description: 'Resultados de backtesting atualizados com sucesso.',
      });
    } catch (error) {
      console.error('Erro ao carregar dados de backtesting:', error);
      toast({
        title: 'Erro ao carregar dados',
        description: 'Não foi possível carregar os resultados de backtesting.',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
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

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-4 text-primary" />
          <p className="text-muted-foreground">Carregando resultados de backtesting...</p>
        </div>
      </div>
    );
  }

  if (!equityCurve || !windows) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <AlertCircle className="h-12 w-12 mx-auto mb-4 text-destructive" />
          <h3 className="text-lg font-semibold mb-2">Dados não disponíveis</h3>
          <p className="text-muted-foreground mb-4">
            Não foram encontrados resultados de backtesting.
          </p>
          <Button onClick={loadBacktestingData}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Tentar novamente
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Backtesting Visual</h1>
          <p className="text-muted-foreground mt-1">
            Análise de performance do modelo ML em walk-forward validation
          </p>
        </div>
        <Button onClick={loadBacktestingData} disabled={isRefreshing}>
          <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
          Atualizar
        </Button>
      </div>

      {/* Summary Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Capital Inicial</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatCurrency(equityCurve.summary.initial_capital)}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Base de cálculo
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Capital Final</CardTitle>
            <TrendingUp className="h-4 w-4 text-green-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">
              {formatCurrency(equityCurve.summary.final_capital)}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Após {equityCurve.summary.n_windows} janelas
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Retorno Total</CardTitle>
            <Target className="h-4 w-4 text-blue-500" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${equityCurve.summary.total_return_pct >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {formatPercent(equityCurve.summary.total_return_pct)}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Período: {equityCurve.summary.period}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Max Drawdown</CardTitle>
            <TrendingDown className="h-4 w-4 text-red-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-600">
              {formatPercent(-equityCurve.summary.max_drawdown_pct)}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Maior queda acumulada
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Equity Curve Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <LineChart className="h-5 w-5" />
            Curva de Equity (Capital ao Longo do Tempo)
          </CardTitle>
          <CardDescription>
            Evolução do capital com efeito composto dos lucros/perdas de cada janela walk-forward
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={400}>
            <AreaChart data={equityCurve.equity_points}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="date"
                tick={{ fontSize: 12 }}
              />
              <YAxis
                tick={{ fontSize: 12 }}
                tickFormatter={(value) => `$${value.toFixed(0)}`}
              />
              <Tooltip
                formatter={(value: number, name: string) => {
                  if (name === 'capital') return [formatCurrency(value), 'Capital'];
                  if (name === 'total_return_pct') return [formatPercent(value), 'Retorno Acumulado'];
                  return [value, name];
                }}
                labelFormatter={(label) => `Data: ${label}`}
              />
              <Legend />
              <Area
                type="monotone"
                dataKey="capital"
                stroke="#2563eb"
                fill="#3b82f6"
                fillOpacity={0.3}
                name="Capital"
              />
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Performance Metrics Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Métricas Agregadas (Média de {windows.summary.total_windows} Janelas)
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Accuracy</span>
                <Badge variant="outline">{windows.summary.avg_metrics.accuracy.toFixed(2)}%</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Precision</span>
                <Badge variant="outline">{windows.summary.avg_metrics.precision.toFixed(2)}%</Badge>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Recall</span>
                <Badge variant="outline">{windows.summary.avg_metrics.recall.toFixed(2)}%</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">F1-Score</span>
                <Badge variant="outline">{windows.summary.avg_metrics.f1.toFixed(2)}%</Badge>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Win Rate</span>
                <Badge variant={windows.summary.avg_trading.win_rate >= 50 ? "default" : "secondary"}>
                  {windows.summary.avg_trading.win_rate.toFixed(2)}%
                </Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Profit Médio</span>
                <Badge variant={windows.summary.avg_trading.total_profit >= 0 ? "default" : "destructive"}>
                  {formatPercent(windows.summary.avg_trading.total_profit)}
                </Badge>
              </div>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Sharpe Ratio</span>
                <Badge variant={windows.summary.avg_trading.sharpe_ratio >= 1 ? "default" : "secondary"}>
                  {windows.summary.avg_trading.sharpe_ratio.toFixed(2)}
                </Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Max DD Médio</span>
                <Badge variant="outline">
                  {formatPercent(-windows.summary.avg_trading.max_drawdown)}
                </Badge>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Windows Table */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Resultados por Janela Walk-Forward
          </CardTitle>
          <CardDescription>
            Performance detalhada de cada janela de validação ({windows.windows.length} janelas)
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="relative overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Janela</TableHead>
                  <TableHead>Período</TableHead>
                  <TableHead className="text-right">Accuracy</TableHead>
                  <TableHead className="text-right">Recall</TableHead>
                  <TableHead className="text-right">Trades</TableHead>
                  <TableHead className="text-right">Win Rate</TableHead>
                  <TableHead className="text-right">Profit</TableHead>
                  <TableHead className="text-right">Sharpe</TableHead>
                  <TableHead className="text-center">Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {windows.windows.map((window: BacktestWindow, index: number) => (
                  <TableRow key={index}>
                    <TableCell className="font-medium">{window.window}</TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {window.start_date} - {window.end_date}
                    </TableCell>
                    <TableCell className="text-right">
                      <Badge variant="outline" className="font-mono">
                        {window.metrics.accuracy.toFixed(2)}%
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right">
                      <Badge variant="outline" className="font-mono">
                        {window.metrics.recall.toFixed(2)}%
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right font-mono">
                      {window.trading.total_trades}
                    </TableCell>
                    <TableCell className="text-right">
                      <Badge
                        variant={window.trading.win_rate >= 50 ? "default" : "secondary"}
                        className="font-mono"
                      >
                        {window.trading.win_rate.toFixed(2)}%
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right">
                      <span className={`font-mono font-semibold ${
                        window.trading.total_profit >= 0 ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {formatPercent(window.trading.total_profit)}
                      </span>
                    </TableCell>
                    <TableCell className="text-right font-mono">
                      {window.trading.sharpe_ratio.toFixed(2)}
                    </TableCell>
                    <TableCell className="text-center">
                      {window.trading.total_profit >= 0 ? (
                        <CheckCircle2 className="h-5 w-5 text-green-500 inline" />
                      ) : (
                        <AlertCircle className="h-5 w-5 text-red-500 inline" />
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      {/* Info Note */}
      <Card className="border-blue-200 bg-blue-50 dark:bg-blue-950 dark:border-blue-800">
        <CardContent className="pt-6">
          <div className="flex gap-3">
            <Zap className="h-5 w-5 text-blue-600 dark:text-blue-400 flex-shrink-0 mt-0.5" />
            <div className="space-y-1">
              <p className="text-sm font-medium text-blue-900 dark:text-blue-100">
                Sobre os Resultados
              </p>
              <p className="text-sm text-blue-700 dark:text-blue-300">
                {equityCurve.notes}
              </p>
              <p className="text-sm text-blue-700 dark:text-blue-300">
                {windows.notes}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
