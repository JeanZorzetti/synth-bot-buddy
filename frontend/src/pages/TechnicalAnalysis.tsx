/**
 * Technical Analysis Page - Indicadores Técnicos, Backtesting e Validação Manual
 * Fase 1.4: Testar em dados históricos e criar visualização de indicadores
 */

import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  BarChart3,
  TrendingUp,
  Activity,
  CheckCircle,
  XCircle,
  Play,
  RefreshCw,
  AlertTriangle,
} from 'lucide-react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

interface BacktestResult {
  summary: {
    initial_balance: number;
    final_balance: number;
    total_profit: number;
    total_profit_percent: number;
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    win_rate: number;
    profit_factor: number;
    max_drawdown: number;
    sharpe_ratio: number;
  };
  trade_stats: {
    avg_win: number;
    avg_loss: number;
    largest_win: number;
    largest_loss: number;
    avg_profit_per_trade: number;
  };
  trades: any[];
  metadata: {
    symbol: string;
    timeframe: string;
    data_source: string;
    candles_analyzed: number;
  };
}

const TechnicalAnalysis = () => {
  const [backtestResult, setBacktestResult] = useState<BacktestResult | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Backtest settings
  const [settings, setSettings] = useState({
    symbol: 'R_100',
    timeframe: '1m',
    count: 1000,
    initialBalance: 1000,
    positionSize: 10,
    stopLoss: 2,
    takeProfit: 4,
  });

  const runBacktest = async () => {
    setIsRunning(true);
    setError(null);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'https://botderivapi.roilabs.com.br';
      const token = localStorage.getItem('deriv_api_key') || localStorage.getItem('deriv_primary_token');

      const headers: HeadersInit = {
        'Content-Type': 'application/json',
      };

      if (token) {
        headers['X-API-Token'] = token;
      }

      const params = new URLSearchParams({
        timeframe: settings.timeframe,
        count: settings.count.toString(),
        initial_balance: settings.initialBalance.toString(),
        position_size_percent: settings.positionSize.toString(),
        stop_loss_percent: settings.stopLoss.toString(),
        take_profit_percent: settings.takeProfit.toString(),
      });

      const response = await fetch(`${apiUrl}/api/backtest/${settings.symbol}?${params}`, {
        method: 'POST',
        headers,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      setBacktestResult(data);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Erro desconhecido';
      setError(`Erro ao executar backtest: ${errorMessage}`);
      console.error('Error running backtest:', err);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <BarChart3 className="h-8 w-8" />
          Análise Técnica
        </h1>
        <p className="text-muted-foreground">
          Backtesting de indicadores técnicos e validação manual de sinais
        </p>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="backtest" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="backtest">
            <BarChart3 className="h-4 w-4 mr-2" />
            Backtesting
          </TabsTrigger>
          <TabsTrigger value="indicators">
            <Activity className="h-4 w-4 mr-2" />
            Indicadores
          </TabsTrigger>
          <TabsTrigger value="validation">
            <CheckCircle className="h-4 w-4 mr-2" />
            Validação Manual
          </TabsTrigger>
        </TabsList>

        {/* Tab 1: Backtesting */}
        <TabsContent value="backtest" className="space-y-4">
          {/* Error Alert */}
          {error && (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>Erro</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {/* Settings Card */}
          <Card>
            <CardHeader>
              <CardTitle>Configurações do Backtest</CardTitle>
              <CardDescription>
                Configure os parâmetros para testar sua estratégia em dados históricos
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="symbol">Símbolo</Label>
                  <Input
                    id="symbol"
                    value={settings.symbol}
                    onChange={(e) => setSettings({ ...settings, symbol: e.target.value })}
                    placeholder="R_100"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="timeframe">Timeframe</Label>
                  <Input
                    id="timeframe"
                    value={settings.timeframe}
                    onChange={(e) => setSettings({ ...settings, timeframe: e.target.value })}
                    placeholder="1m"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="count">Candles</Label>
                  <Input
                    id="count"
                    type="number"
                    value={settings.count}
                    onChange={(e) => setSettings({ ...settings, count: parseInt(e.target.value) })}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="balance">Saldo Inicial ($)</Label>
                  <Input
                    id="balance"
                    type="number"
                    value={settings.initialBalance}
                    onChange={(e) => setSettings({ ...settings, initialBalance: parseFloat(e.target.value) })}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="position">Position Size (%)</Label>
                  <Input
                    id="position"
                    type="number"
                    value={settings.positionSize}
                    onChange={(e) => setSettings({ ...settings, positionSize: parseFloat(e.target.value) })}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="sl">Stop Loss (%)</Label>
                  <Input
                    id="sl"
                    type="number"
                    value={settings.stopLoss}
                    onChange={(e) => setSettings({ ...settings, stopLoss: parseFloat(e.target.value) })}
                  />
                </div>
              </div>

              <Button onClick={runBacktest} disabled={isRunning} size="lg" className="w-full gap-2">
                {isRunning ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    Executando Backtest...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4" />
                    Executar Backtest
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {/* Results */}
          {backtestResult && (
            <>
              {/* Summary Cards */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium">Profit Total</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold text-green-600">
                      ${backtestResult.summary.total_profit.toFixed(2)}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {backtestResult.summary.total_profit_percent.toFixed(2)}%
                    </p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium">Win Rate</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">
                      {backtestResult.summary.win_rate.toFixed(2)}%
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {backtestResult.summary.winning_trades}/{backtestResult.summary.total_trades} trades
                    </p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium">Profit Factor</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">
                      {backtestResult.summary.profit_factor.toFixed(2)}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Razão profit/loss
                    </p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm font-medium">Sharpe Ratio</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">
                      {backtestResult.summary.sharpe_ratio.toFixed(2)}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Risk-adjusted return
                    </p>
                  </CardContent>
                </Card>
              </div>

              {/* Detailed Stats */}
              <Card>
                <CardHeader>
                  <CardTitle>Estatísticas Detalhadas</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div>
                      <p className="text-sm text-muted-foreground">Saldo Inicial</p>
                      <p className="text-lg font-semibold">${backtestResult.summary.initial_balance}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Saldo Final</p>
                      <p className="text-lg font-semibold">${backtestResult.summary.final_balance.toFixed(2)}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Max Drawdown</p>
                      <p className="text-lg font-semibold text-red-600">
                        {backtestResult.summary.max_drawdown.toFixed(2)}%
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Avg Profit/Trade</p>
                      <p className="text-lg font-semibold">
                        ${backtestResult.trade_stats.avg_profit_per_trade.toFixed(2)}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Avg Win</p>
                      <p className="text-lg font-semibold text-green-600">
                        ${backtestResult.trade_stats.avg_win.toFixed(2)}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Avg Loss</p>
                      <p className="text-lg font-semibold text-red-600">
                        ${backtestResult.trade_stats.avg_loss.toFixed(2)}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Largest Win</p>
                      <p className="text-lg font-semibold text-green-600">
                        ${backtestResult.trade_stats.largest_win.toFixed(2)}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">Largest Loss</p>
                      <p className="text-lg font-semibold text-red-600">
                        ${backtestResult.trade_stats.largest_loss.toFixed(2)}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Trade List */}
              <Card>
                <CardHeader>
                  <CardTitle>Últimos Trades (20 mais recentes)</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2 max-h-96 overflow-y-auto">
                    {backtestResult.trades.map((trade, idx) => (
                      <div
                        key={idx}
                        className="flex items-center justify-between p-3 border rounded-lg hover:bg-accent"
                      >
                        <div className="flex items-center gap-3">
                          {trade.profit > 0 ? (
                            <TrendingUp className="h-5 w-5 text-green-600" />
                          ) : (
                            <Activity className="h-5 w-5 text-red-600" />
                          )}
                          <div>
                            <Badge variant={trade.signal_type === 'BUY' ? 'default' : 'secondary'}>
                              {trade.signal_type}
                            </Badge>
                            <p className="text-xs text-muted-foreground mt-1">
                              {trade.exit_reason}
                            </p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className={`font-semibold ${trade.profit > 0 ? 'text-green-600' : 'text-red-600'}`}>
                            ${trade.profit.toFixed(2)}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {trade.profit_percent.toFixed(2)}%
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </>
          )}
        </TabsContent>

        {/* Tab 2: Indicadores */}
        <TabsContent value="indicators" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Visualização de Indicadores</CardTitle>
              <CardDescription>
                Gráficos de indicadores técnicos em tempo real
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Alert>
                <Activity className="h-4 w-4" />
                <AlertDescription>
                  Funcionalidade em desenvolvimento. Em breve você poderá visualizar RSI, MACD, Bollinger Bands e outros indicadores.
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Tab 3: Validação Manual */}
        <TabsContent value="validation" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Validação Manual de Sinais</CardTitle>
              <CardDescription>
                Aprove ou rejeite sinais antes de executá-los automaticamente
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Alert>
                <CheckCircle className="h-4 w-4" />
                <AlertDescription>
                  Funcionalidade em desenvolvimento. Em breve você poderá validar manualmente cada sinal gerado pelo sistema antes de executá-lo.
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default TechnicalAnalysis;
