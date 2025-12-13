import { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Shield,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  DollarSign,
  Activity,
  BarChart3,
  RefreshCw
} from 'lucide-react';

interface RiskMetrics {
  current_capital: number;
  initial_capital: number;
  total_pnl: number;
  total_pnl_percent: number;
  daily_pnl: number;
  daily_loss_percent: number;
  weekly_pnl: number;
  weekly_loss_percent: number;
  drawdown_percent: number;
  peak_capital: number;
  active_trades_count: number;
  total_trades: number;
  win_rate: number;
  consecutive_losses: number;
  is_circuit_breaker_active: boolean;
  avg_win: number;
  avg_loss: number;
  kelly_criterion: number;
  limits: {
    max_daily_loss: number;
    max_weekly_loss: number;
    max_drawdown: number;
    max_position_size: number;
    max_concurrent_trades: number;
    circuit_breaker_losses: number;
    min_risk_reward: number;
  };
}

export default function RiskManagement() {
  const [metrics, setMetrics] = useState<RiskMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const fetchMetrics = async () => {
    try {
      const response = await fetch('https://botderivapi.roilabs.com.br/api/risk/metrics');
      const data = await response.json();
      setMetrics(data.metrics);
      setLastUpdate(new Date());
    } catch (error) {
      console.error('Error fetching risk metrics:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
    // Auto-refresh every 5 seconds
    const interval = setInterval(fetchMetrics, 5000);
    return () => clearInterval(interval);
  }, []);

  const resetCircuitBreaker = async () => {
    try {
      await fetch('https://botderivapi.roilabs.com.br/api/risk/reset-circuit-breaker', {
        method: 'POST'
      });
      fetchMetrics();
    } catch (error) {
      console.error('Error resetting circuit breaker:', error);
    }
  };

  if (loading || !metrics) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <RefreshCw className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  const pnlColor = metrics.total_pnl >= 0 ? 'text-green-500' : 'text-red-500';
  const dailyPnlColor = metrics.daily_pnl >= 0 ? 'text-green-500' : 'text-red-500';
  const weeklyPnlColor = metrics.weekly_pnl >= 0 ? 'text-green-500' : 'text-red-500';

  const dailyLossPercent = (metrics.daily_loss_percent / metrics.limits.max_daily_loss) * 100;
  const weeklyLossPercent = (metrics.weekly_loss_percent / metrics.limits.max_weekly_loss) * 100;
  const drawdownPercent = (metrics.drawdown_percent / metrics.limits.max_drawdown) * 100;
  const tradesPercent = (metrics.active_trades_count / metrics.limits.max_concurrent_trades) * 100;

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Shield className="w-8 h-8 text-primary" />
            Risk Management Dashboard
          </h1>
          <p className="text-muted-foreground mt-1">
            Last updated: {lastUpdate.toLocaleTimeString()}
          </p>
        </div>
        <Button onClick={fetchMetrics} variant="outline" size="sm">
          <RefreshCw className="w-4 h-4 mr-2" />
          Refresh
        </Button>
      </div>

      {/* Circuit Breaker Alert */}
      {metrics.is_circuit_breaker_active && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Circuit Breaker Activated</AlertTitle>
          <AlertDescription className="flex items-center justify-between">
            <span>
              Trading paused after {metrics.consecutive_losses} consecutive losses
            </span>
            <Button
              onClick={resetCircuitBreaker}
              variant="outline"
              size="sm"
              className="ml-4"
            >
              Reset
            </Button>
          </AlertDescription>
        </Alert>
      )}

      {/* Capital Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Current Capital
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              ${metrics.current_capital.toFixed(2)}
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Initial: ${metrics.initial_capital.toFixed(2)}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Total P&L
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${pnlColor} flex items-center gap-2`}>
              {metrics.total_pnl >= 0 ? (
                <TrendingUp className="w-5 h-5" />
              ) : (
                <TrendingDown className="w-5 h-5" />
              )}
              ${Math.abs(metrics.total_pnl).toFixed(2)}
            </div>
            <p className={`text-xs mt-1 ${pnlColor}`}>
              {metrics.total_pnl_percent >= 0 ? '+' : ''}
              {metrics.total_pnl_percent.toFixed(2)}%
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Win Rate
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {metrics.win_rate.toFixed(1)}%
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              {metrics.total_trades} total trades
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Kelly Criterion
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-primary">
              {metrics.kelly_criterion.toFixed(2)}%
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              Recommended risk per trade
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="limits" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="limits">
            <Activity className="w-4 h-4 mr-2" />
            Risk Limits
          </TabsTrigger>
          <TabsTrigger value="performance">
            <BarChart3 className="w-4 h-4 mr-2" />
            Performance
          </TabsTrigger>
          <TabsTrigger value="settings">
            <Shield className="w-4 h-4 mr-2" />
            Protection Settings
          </TabsTrigger>
        </TabsList>

        {/* Risk Limits Tab */}
        <TabsContent value="limits" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Daily Loss Limit</CardTitle>
              <CardDescription>
                Maximum loss allowed per day: {metrics.limits.max_daily_loss}%
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm">Current Daily Loss</span>
                <span className={`text-sm font-semibold ${dailyPnlColor}`}>
                  ${Math.abs(metrics.daily_pnl).toFixed(2)} ({metrics.daily_loss_percent.toFixed(2)}%)
                </span>
              </div>
              <Progress
                value={dailyLossPercent}
                className={dailyLossPercent > 80 ? 'bg-red-100' : ''}
              />
              <p className="text-xs text-muted-foreground">
                {dailyLossPercent.toFixed(0)}% of daily limit used
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Weekly Loss Limit</CardTitle>
              <CardDescription>
                Maximum loss allowed per week: {metrics.limits.max_weekly_loss}%
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm">Current Weekly Loss</span>
                <span className={`text-sm font-semibold ${weeklyPnlColor}`}>
                  ${Math.abs(metrics.weekly_pnl).toFixed(2)} ({metrics.weekly_loss_percent.toFixed(2)}%)
                </span>
              </div>
              <Progress
                value={weeklyLossPercent}
                className={weeklyLossPercent > 80 ? 'bg-red-100' : ''}
              />
              <p className="text-xs text-muted-foreground">
                {weeklyLossPercent.toFixed(0)}% of weekly limit used
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Drawdown</CardTitle>
              <CardDescription>
                Maximum drawdown allowed: {metrics.limits.max_drawdown}%
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm">Current Drawdown</span>
                <span className="text-sm font-semibold text-red-500">
                  {metrics.drawdown_percent.toFixed(2)}%
                </span>
              </div>
              <Progress
                value={drawdownPercent}
                className={drawdownPercent > 80 ? 'bg-red-100' : ''}
              />
              <p className="text-xs text-muted-foreground">
                Peak capital: ${metrics.peak_capital.toFixed(2)}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Active Trades</CardTitle>
              <CardDescription>
                Maximum concurrent trades: {metrics.limits.max_concurrent_trades}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm">Currently Active</span>
                <Badge variant={metrics.active_trades_count >= metrics.limits.max_concurrent_trades ? 'destructive' : 'default'}>
                  {metrics.active_trades_count} / {metrics.limits.max_concurrent_trades}
                </Badge>
              </div>
              <Progress value={tradesPercent} />
              <p className="text-xs text-muted-foreground">
                {metrics.limits.max_concurrent_trades - metrics.active_trades_count} slots available
              </p>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Performance Tab */}
        <TabsContent value="performance" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Average Win</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-green-500">
                  ${metrics.avg_win.toFixed(2)}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Average Loss</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-red-500">
                  ${metrics.avg_loss.toFixed(2)}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Win/Loss Ratio</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">
                  {metrics.avg_loss > 0
                    ? (metrics.avg_win / metrics.avg_loss).toFixed(2)
                    : 'N/A'
                  }
                </div>
                <p className="text-xs text-muted-foreground mt-2">
                  {metrics.avg_loss > 0 && metrics.avg_win / metrics.avg_loss >= 2
                    ? 'Excellent risk/reward ratio'
                    : 'Keep improving'
                  }
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Consecutive Losses</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-orange-500">
                  {metrics.consecutive_losses}
                </div>
                <p className="text-xs text-muted-foreground mt-2">
                  Circuit breaker triggers at {metrics.limits.circuit_breaker_losses}
                </p>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Trading Statistics</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Total Trades</span>
                <span className="font-semibold">{metrics.total_trades}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Win Rate</span>
                <span className="font-semibold">{metrics.win_rate.toFixed(1)}%</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Current Capital</span>
                <span className="font-semibold">${metrics.current_capital.toFixed(2)}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-muted-foreground">Peak Capital</span>
                <span className="font-semibold">${metrics.peak_capital.toFixed(2)}</span>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Settings Tab */}
        <TabsContent value="settings" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Protection Settings</CardTitle>
              <CardDescription>
                Current risk management configuration
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <p className="text-sm font-medium">Max Daily Loss</p>
                  <p className="text-2xl font-bold">{metrics.limits.max_daily_loss}%</p>
                </div>
                <div className="space-y-1">
                  <p className="text-sm font-medium">Max Weekly Loss</p>
                  <p className="text-2xl font-bold">{metrics.limits.max_weekly_loss}%</p>
                </div>
                <div className="space-y-1">
                  <p className="text-sm font-medium">Max Drawdown</p>
                  <p className="text-2xl font-bold">{metrics.limits.max_drawdown}%</p>
                </div>
                <div className="space-y-1">
                  <p className="text-sm font-medium">Max Position Size</p>
                  <p className="text-2xl font-bold">{metrics.limits.max_position_size}%</p>
                </div>
                <div className="space-y-1">
                  <p className="text-sm font-medium">Max Concurrent Trades</p>
                  <p className="text-2xl font-bold">{metrics.limits.max_concurrent_trades}</p>
                </div>
                <div className="space-y-1">
                  <p className="text-sm font-medium">Circuit Breaker</p>
                  <p className="text-2xl font-bold">{metrics.limits.circuit_breaker_losses} losses</p>
                </div>
                <div className="space-y-1 col-span-2">
                  <p className="text-sm font-medium">Min Risk/Reward Ratio</p>
                  <p className="text-2xl font-bold">1:{metrics.limits.min_risk_reward}</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Alert>
            <DollarSign className="h-4 w-4" />
            <AlertTitle>Kelly Criterion</AlertTitle>
            <AlertDescription>
              Based on your current win rate ({metrics.win_rate.toFixed(1)}%),
              the optimal risk per trade is {metrics.kelly_criterion.toFixed(2)}% of your capital.
              This uses Quarter Kelly for conservative position sizing.
            </AlertDescription>
          </Alert>
        </TabsContent>
      </Tabs>
    </div>
  );
}
