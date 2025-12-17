import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, TrendingDown, Clock, Target, AlertCircle, CheckCircle2 } from 'lucide-react';

interface LiveMetrics {
  performance: {
    win_rate: number;
    sharpe_ratio: number;
    profit_factor: number;
    max_drawdown_pct: number;
    win_rates_by_period?: {
      last_10?: number;
      last_20?: number;
      last_50?: number;
    };
  };
  execution: {
    avg_duration_minutes: number;
    timeout_rate: number;
    sl_hit_rate: number;
    tp_hit_rate: number;
    total_trades: number;
    open_positions: number;
  };
  trades_breakdown: {
    winning: number;
    losing: number;
    by_exit_reason: {
      timeout: number;
      stop_loss: number;
      take_profit: number;
    };
  };
}

interface LiveMetricsGridProps {
  metrics: LiveMetrics;
}

export function LiveMetricsGrid({ metrics }: LiveMetricsGridProps) {
  // Helper para colorir valores
  const getColorClass = (value: number, threshold: number, higherIsBetter: boolean = true) => {
    if (higherIsBetter) {
      return value >= threshold ? 'text-green-600' : 'text-red-600';
    } else {
      return value <= threshold ? 'text-green-600' : 'text-red-600';
    }
  };

  // Helper para ícone de trend
  const TrendIcon = ({ value, threshold, higherIsBetter = true }: any) => {
    const isGood = higherIsBetter ? value >= threshold : value <= threshold;
    return isGood ? (
      <TrendingUp className="h-4 w-4 text-green-600" />
    ) : (
      <TrendingDown className="h-4 w-4 text-red-600" />
    );
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {/* Win Rate */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Win Rate</CardTitle>
          <CheckCircle2 className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="text-2xl font-bold">
              <span className={getColorClass(metrics.performance.win_rate, 55)}>
                {metrics.performance.win_rate.toFixed(1)}%
              </span>
            </div>
            <TrendIcon value={metrics.performance.win_rate} threshold={55} />
          </div>
          <div className="mt-2 flex gap-2">
            {metrics.performance.win_rates_by_period?.last_10 !== undefined && (
              <Badge variant="outline" className="text-xs">
                L10: {metrics.performance.win_rates_by_period.last_10.toFixed(0)}%
              </Badge>
            )}
            {metrics.performance.win_rates_by_period?.last_20 !== undefined && (
              <Badge variant="outline" className="text-xs">
                L20: {metrics.performance.win_rates_by_period.last_20.toFixed(0)}%
              </Badge>
            )}
          </div>
          <p className="text-xs text-muted-foreground mt-1">
            {metrics.trades_breakdown.winning}W / {metrics.trades_breakdown.losing}L
          </p>
        </CardContent>
      </Card>

      {/* Avg Duration */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Avg Trade Duration</CardTitle>
          <Clock className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="text-2xl font-bold">
              <span className={getColorClass(metrics.execution.avg_duration_minutes, 10, false)}>
                {metrics.execution.avg_duration_minutes.toFixed(1)}
              </span>
              <span className="text-base font-normal text-muted-foreground ml-1">min</span>
            </div>
            <TrendIcon value={metrics.execution.avg_duration_minutes} threshold={10} higherIsBetter={false} />
          </div>
          <p className="text-xs text-muted-foreground mt-3">
            {metrics.execution.total_trades} total trades
          </p>
        </CardContent>
      </Card>

      {/* Timeout Rate */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Timeout Rate</CardTitle>
          <AlertCircle className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="text-2xl font-bold">
              <span className={getColorClass(metrics.execution.timeout_rate, 20, false)}>
                {metrics.execution.timeout_rate.toFixed(1)}%
              </span>
            </div>
            <TrendIcon value={metrics.execution.timeout_rate} threshold={20} higherIsBetter={false} />
          </div>
          <p className="text-xs text-muted-foreground mt-3">
            {metrics.trades_breakdown.by_exit_reason.timeout} of {metrics.execution.total_trades} trades
          </p>
        </CardContent>
      </Card>

      {/* SL Hit Rate */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Stop Loss Hit Rate</CardTitle>
          <Target className="h-4 w-4 text-red-500" />
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="text-2xl font-bold">
              <span className={getColorClass(metrics.execution.sl_hit_rate, 40, false)}>
                {metrics.execution.sl_hit_rate.toFixed(1)}%
              </span>
            </div>
          </div>
          <p className="text-xs text-muted-foreground mt-3">
            {metrics.trades_breakdown.by_exit_reason.stop_loss} SL hits
          </p>
        </CardContent>
      </Card>

      {/* TP Hit Rate */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Take Profit Hit Rate</CardTitle>
          <Target className="h-4 w-4 text-green-500" />
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="text-2xl font-bold">
              <span className={getColorClass(metrics.execution.tp_hit_rate, 40)}>
                {metrics.execution.tp_hit_rate.toFixed(1)}%
              </span>
            </div>
          </div>
          <p className="text-xs text-muted-foreground mt-3">
            {metrics.trades_breakdown.by_exit_reason.take_profit} TP hits
          </p>
        </CardContent>
      </Card>

      {/* Sharpe Ratio */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">Sharpe Ratio</CardTitle>
          <TrendingUp className="h-4 w-4 text-muted-foreground" />
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div className="text-2xl font-bold">
              <span className={getColorClass(metrics.performance.sharpe_ratio, 1.5)}>
                {metrics.performance.sharpe_ratio.toFixed(2)}
              </span>
            </div>
            <TrendIcon value={metrics.performance.sharpe_ratio} threshold={1.5} />
          </div>
          <p className="text-xs text-muted-foreground mt-3">
            Target: &gt; 1.5 (good risk-adjusted returns)
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
