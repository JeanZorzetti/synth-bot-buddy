import { useEffect, useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { toast } from 'sonner';
import {
  Shield,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  DollarSign,
  Activity,
  BarChart3,
  RefreshCw,
  LineChart,
  Brain,
  Zap,
  Play,
  Pause
} from 'lucide-react';
import {
  LineChart as RechartsLineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  AreaChart,
  BarChart,
  Bar
} from 'recharts';

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

interface EquityPoint {
  timestamp: string;
  capital: number;
  pnl: number;
  drawdown: number;
  trade_count: number;
  is_win?: boolean;
}

interface EquityHistory {
  equity_history: EquityPoint[];
  current_capital: number;
  initial_capital: number;
  peak_capital: number;
  total_trades: number;
}

interface MLPredictions {
  predicted_win_rate: number;
  predicted_avg_win: number;
  predicted_avg_loss: number;
  kelly_criterion: number;
  kelly_full: number;
  confidence: number;
}

interface MLStatus {
  ml_enabled: boolean;
  has_predictions: boolean;
  is_trained?: boolean;
  accuracy?: number;
  total_samples?: number;
}

interface FeatureImportance {
  feature: string;
  importance: number;
}

export default function RiskManagement() {
  const [metrics, setMetrics] = useState<RiskMetrics | null>(null);
  const [equityData, setEquityData] = useState<EquityHistory | null>(null);
  const [mlPredictions, setMlPredictions] = useState<MLPredictions | null>(null);
  const [mlStatus, setMlStatus] = useState<MLStatus>({ ml_enabled: false, has_predictions: false });
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance[]>([]);
  const [loading, setLoading] = useState(true);
  const [mlLoading, setMlLoading] = useState(false);
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

  const fetchEquityHistory = async () => {
    try {
      const response = await fetch('https://botderivapi.roilabs.com.br/api/risk/equity-history');
      const data = await response.json();
      setEquityData(data);
    } catch (error) {
      console.error('Error fetching equity history:', error);
    }
  };

  const fetchMLPredictions = async () => {
    try {
      const response = await fetch('https://botderivapi.roilabs.com.br/api/risk/predict-kelly-ml', {
        method: 'POST'
      });
      const data = await response.json();

      if (data.status === 'success') {
        setMlPredictions(data.predictions);
        setMlStatus({
          ml_enabled: data.ml_enabled,
          has_predictions: true,
          is_trained: true
        });
      }
    } catch (error) {
      console.error('Error fetching ML predictions:', error);
    }
  };

  const trainKellyML = async () => {
    setMlLoading(true);
    toast.info('Training ML model...', {
      description: 'This may take a few seconds'
    });

    try {
      const response = await fetch('https://botderivapi.roilabs.com.br/api/risk/train-kelly-ml', {
        method: 'POST'
      });
      const data = await response.json();

      if (data.status === 'success') {
        setMlStatus({
          ml_enabled: data.ml_enabled,
          has_predictions: false,
          is_trained: true,
          accuracy: data.metrics.accuracy,
          total_samples: data.metrics.total_samples
        });
        // Capturar feature importance
        if (data.feature_importance) {
          setFeatureImportance(data.feature_importance);
        }
        await fetchMLPredictions();

        toast.success('ML Model Trained Successfully!', {
          description: `Accuracy: ${(data.metrics.accuracy * 100).toFixed(1)}% | Samples: ${data.metrics.total_samples} trades`
        });
      } else if (data.status === 'insufficient_data') {
        toast.warning('Insufficient Data', {
          description: `${data.trades_remaining} more trades needed (minimum 50 trades required)`
        });
      } else {
        toast.error('Training Failed', {
          description: data.message || 'Unknown error occurred'
        });
      }
    } catch (error) {
      console.error('Error training Kelly ML:', error);
      toast.error('Training Failed', {
        description: 'Failed to connect to server. Please try again.'
      });
    } finally {
      setMlLoading(false);
    }
  };

  const toggleKellyML = async (enable: boolean) => {
    try {
      const response = await fetch(`https://botderivapi.roilabs.com.br/api/risk/toggle-kelly-ml?enable=${enable}`, {
        method: 'POST'
      });
      const data = await response.json();

      if (data.status === 'success') {
        setMlStatus(prev => ({
          ...prev,
          ml_enabled: data.ml_enabled,
          has_predictions: data.has_predictions
        }));

        toast.success(`ML Kelly ${enable ? 'Enabled' : 'Disabled'}`, {
          description: enable
            ? 'Position sizing now uses ML predictions'
            : 'Position sizing reverted to historical statistics'
        });
      } else {
        toast.error('Toggle Failed', {
          description: data.message || 'Failed to toggle ML Kelly'
        });
      }
    } catch (error) {
      console.error('Error toggling Kelly ML:', error);
      toast.error('Toggle Failed', {
        description: 'Failed to connect to server. Please try again.'
      });
    }
  };

  useEffect(() => {
    fetchMetrics();
    fetchEquityHistory();
    fetchMLPredictions();
    const interval = setInterval(() => {
      fetchMetrics();
      fetchEquityHistory();
      if (mlStatus.ml_enabled) {
        fetchMLPredictions();
      }
    }, 5000);
    return () => clearInterval(interval);
  }, [mlStatus.ml_enabled]);

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
      <Tabs defaultValue="charts" className="w-full">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="charts">
            <LineChart className="w-4 h-4 mr-2" />
            Charts
          </TabsTrigger>
          <TabsTrigger value="ml">
            <Brain className="w-4 h-4 mr-2" />
            ML Kelly
            {mlStatus.ml_enabled && (
              <Badge variant="default" className="ml-2 text-xs">ON</Badge>
            )}
          </TabsTrigger>
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

        {/* Charts Tab */}
        <TabsContent value="charts" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Equity Curve</CardTitle>
              <CardDescription>
                Capital growth over time
              </CardDescription>
            </CardHeader>
            <CardContent>
              {equityData && equityData.equity_history.length > 0 ? (
                <ResponsiveContainer width="100%" height={400}>
                  <AreaChart data={equityData.equity_history}>
                    <defs>
                      <linearGradient id="colorCapital" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#8884d8" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#8884d8" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="trade_count"
                      label={{ value: 'Trades', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis
                      label={{ value: 'Capital ($)', angle: -90, position: 'insideLeft' }}
                      domain={['auto', 'auto']}
                    />
                    <Tooltip
                      formatter={(value: number) => `$${value.toFixed(2)}`}
                      labelFormatter={(label) => `Trade ${label}`}
                    />
                    <Legend />
                    <Area
                      type="monotone"
                      dataKey="capital"
                      stroke="#8884d8"
                      fillOpacity={1}
                      fill="url(#colorCapital)"
                      name="Capital"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-[400px] flex items-center justify-center text-muted-foreground">
                  No trade data available yet. Start trading to see your equity curve.
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Drawdown Chart</CardTitle>
              <CardDescription>
                Drawdown percentage over time
              </CardDescription>
            </CardHeader>
            <CardContent>
              {equityData && equityData.equity_history.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={equityData.equity_history}>
                    <defs>
                      <linearGradient id="colorDrawdown" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8}/>
                        <stop offset="95%" stopColor="#ef4444" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="trade_count"
                      label={{ value: 'Trades', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis
                      label={{ value: 'Drawdown (%)', angle: -90, position: 'insideLeft' }}
                      domain={[0, 'auto']}
                    />
                    <Tooltip
                      formatter={(value: number) => `${value.toFixed(2)}%`}
                      labelFormatter={(label) => `Trade ${label}`}
                    />
                    <Legend />
                    <Area
                      type="monotone"
                      dataKey="drawdown"
                      stroke="#ef4444"
                      fillOpacity={1}
                      fill="url(#colorDrawdown)"
                      name="Drawdown"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-[300px] flex items-center justify-center text-muted-foreground">
                  No trade data available yet. Start trading to see drawdown metrics.
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>P&L per Trade</CardTitle>
              <CardDescription>
                Profit/Loss for each individual trade
              </CardDescription>
            </CardHeader>
            <CardContent>
              {equityData && equityData.equity_history.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <RechartsLineChart data={equityData.equity_history.slice(1)}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="trade_count"
                      label={{ value: 'Trade #', position: 'insideBottom', offset: -5 }}
                    />
                    <YAxis
                      label={{ value: 'P&L ($)', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip
                      formatter={(value: number) => `$${value.toFixed(2)}`}
                      labelFormatter={(label) => `Trade ${label}`}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="pnl"
                      stroke="#10b981"
                      strokeWidth={2}
                      dot={{ fill: '#10b981', r: 4 }}
                      name="P&L"
                    />
                  </RechartsLineChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-[300px] flex items-center justify-center text-muted-foreground">
                  No trade data available yet.
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* ML Kelly Tab */}
        <TabsContent value="ml" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="w-5 h-5" />
                Machine Learning Kelly Criterion
              </CardTitle>
              <CardDescription>
                Use ML to predict win_rate and adjust Kelly Criterion dynamically
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Status and Controls */}
              <div className="grid grid-cols-2 gap-4">
                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm">ML Status</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Trained</span>
                        <Badge variant={mlStatus.is_trained ? "default" : "secondary"}>
                          {mlStatus.is_trained ? "Yes" : "No"}
                        </Badge>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Enabled</span>
                        <Badge variant={mlStatus.ml_enabled ? "default" : "secondary"}>
                          {mlStatus.ml_enabled ? "ON" : "OFF"}
                        </Badge>
                      </div>
                      {mlStatus.accuracy && (
                        <div className="flex items-center justify-between">
                          <span className="text-sm">Accuracy</span>
                          <span className="font-semibold text-primary">
                            {(mlStatus.accuracy * 100).toFixed(1)}%
                          </span>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm">Controls</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    <Button
                      onClick={trainKellyML}
                      disabled={mlLoading}
                      className="w-full"
                      variant="default"
                    >
                      {mlLoading ? (
                        <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                      ) : (
                        <Zap className="w-4 h-4 mr-2" />
                      )}
                      Train Model
                    </Button>
                    <Button
                      onClick={() => toggleKellyML(!mlStatus.ml_enabled)}
                      disabled={!mlStatus.is_trained}
                      className="w-full"
                      variant={mlStatus.ml_enabled ? "destructive" : "default"}
                    >
                      {mlStatus.ml_enabled ? (
                        <><Pause className="w-4 h-4 mr-2" /> Disable ML</>
                      ) : (
                        <><Play className="w-4 h-4 mr-2" /> Enable ML</>
                      )}
                    </Button>
                  </CardContent>
                </Card>
              </div>

              {/* ML Predictions */}
              {mlPredictions && (
                <Card>
                  <CardHeader>
                    <CardTitle>Current Predictions</CardTitle>
                    <CardDescription>
                      ML model predictions based on current market conditions
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                      <div className="space-y-1">
                        <p className="text-sm text-muted-foreground">Predicted Win Rate</p>
                        <p className="text-2xl font-bold text-green-500">
                          {(mlPredictions.predicted_win_rate * 100).toFixed(1)}%
                        </p>
                      </div>
                      <div className="space-y-1">
                        <p className="text-sm text-muted-foreground">Predicted Avg Win</p>
                        <p className="text-2xl font-bold text-green-500">
                          ${mlPredictions.predicted_avg_win.toFixed(2)}
                        </p>
                      </div>
                      <div className="space-y-1">
                        <p className="text-sm text-muted-foreground">Predicted Avg Loss</p>
                        <p className="text-2xl font-bold text-red-500">
                          ${mlPredictions.predicted_avg_loss.toFixed(2)}
                        </p>
                      </div>
                      <div className="space-y-1">
                        <p className="text-sm text-muted-foreground">Kelly Criterion (ML)</p>
                        <p className="text-2xl font-bold text-primary">
                          {(mlPredictions.kelly_criterion * 100).toFixed(2)}%
                        </p>
                      </div>
                      <div className="space-y-1">
                        <p className="text-sm text-muted-foreground">Full Kelly</p>
                        <p className="text-2xl font-bold">
                          {(mlPredictions.kelly_full * 100).toFixed(2)}%
                        </p>
                      </div>
                      <div className="space-y-1">
                        <p className="text-sm text-muted-foreground">Confidence</p>
                        <p className="text-2xl font-bold">
                          {(mlPredictions.confidence * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Info Alert */}
              {!mlStatus.is_trained && (
                <Alert>
                  <Brain className="h-4 w-4" />
                  <AlertTitle>ML Model Not Trained</AlertTitle>
                  <AlertDescription>
                    You need at least 50 trades to train the ML model. Current trades: {metrics?.total_trades || 0}
                    {metrics && metrics.total_trades < 50 && (
                      <span className="block mt-2 font-semibold">
                        {50 - metrics.total_trades} more trades needed
                      </span>
                    )}
                  </AlertDescription>
                </Alert>
              )}

              {mlStatus.ml_enabled && (
                <Alert variant="default">
                  <Zap className="h-4 w-4" />
                  <AlertTitle>ML Kelly Active</AlertTitle>
                  <AlertDescription>
                    Position sizing is now using ML-predicted Kelly Criterion.
                    The model adapts to market conditions in real-time.
                  </AlertDescription>
                </Alert>
              )}

              {/* Feature Importance Chart */}
              {featureImportance.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Feature Importance</CardTitle>
                    <CardDescription>
                      Which factors influence the ML predictions the most
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={400}>
                      <BarChart data={featureImportance}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                          dataKey="feature"
                          angle={-45}
                          textAnchor="end"
                          height={120}
                          tick={{ fontSize: 12 }}
                        />
                        <YAxis
                          label={{ value: 'Importance', angle: -90, position: 'insideLeft' }}
                          tick={{ fontSize: 12 }}
                        />
                        <Tooltip
                          formatter={(value: number) => [(value * 100).toFixed(2) + '%', 'Importance']}
                          labelStyle={{ color: '#000' }}
                        />
                        <Bar dataKey="importance" fill="#8884d8" radius={[8, 8, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              )}
            </CardContent>
          </Card>
        </TabsContent>

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
