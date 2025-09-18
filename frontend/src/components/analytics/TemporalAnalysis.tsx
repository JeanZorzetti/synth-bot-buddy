'use client';

import React, { useState, useEffect } from 'react';
import { Calendar, Clock, TrendingUp, TrendingDown, BarChart3, ArrowRight, ArrowLeft, RotateCcw, Download, Filter } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { DatePickerWithRange } from '@/components/ui/date-range-picker';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';

interface TimeSeriesData {
  timestamp: string;
  value: number;
  category?: string;
  metadata?: { [key: string]: any };
}

interface ComparisonPeriod {
  id: string;
  name: string;
  current: {
    start: string;
    end: string;
    data: TimeSeriesData[];
  };
  previous: {
    start: string;
    end: string;
    data: TimeSeriesData[];
  };
  metrics: {
    growth: number;
    volatility: number;
    trend: 'up' | 'down' | 'stable';
    significance: 'high' | 'medium' | 'low';
  };
}

interface TrendAnalysis {
  metric: string;
  timeframe: string;
  pattern: 'seasonal' | 'cyclical' | 'linear' | 'exponential' | 'random';
  strength: number;
  direction: 'positive' | 'negative' | 'neutral';
  forecast: {
    nextPeriod: number;
    confidence: number;
    range: { min: number; max: number };
  };
  anomalies: Array<{
    timestamp: string;
    value: number;
    deviation: number;
    type: 'spike' | 'drop' | 'outlier';
  }>;
}

interface SeasonalPattern {
  period: 'daily' | 'weekly' | 'monthly' | 'quarterly' | 'yearly';
  peaks: Array<{ time: string; value: number }>;
  troughs: Array<{ time: string; value: number }>;
  cyclicity: number;
  amplitude: number;
}

const TemporalAnalysis: React.FC = () => {
  const [selectedMetric, setSelectedMetric] = useState('revenue');
  const [selectedTimeframe, setSelectedTimeframe] = useState('7d');
  const [comparisonType, setComparisonType] = useState('previous_period');
  const [showForecast, setShowForecast] = useState(true);
  const [showAnomalies, setShowAnomalies] = useState(true);
  const [comparisons, setComparisons] = useState<ComparisonPeriod[]>([]);
  const [trendAnalysis, setTrendAnalysis] = useState<TrendAnalysis[]>([]);
  const [seasonalPatterns, setSeasonalPatterns] = useState<SeasonalPattern[]>([]);
  const [selectedComparison, setSelectedComparison] = useState<ComparisonPeriod | null>(null);

  useEffect(() => {
    generateMockData();
  }, [selectedMetric, selectedTimeframe]);

  const generateMockData = () => {
    // Mock comparison periods
    const mockComparisons: ComparisonPeriod[] = [
      {
        id: 'revenue-comparison',
        name: 'Revenue Analysis',
        current: {
          start: '2024-01-08',
          end: '2024-01-15',
          data: generateTimeSeriesData('2024-01-08', 7, 50000, 150000)
        },
        previous: {
          start: '2024-01-01',
          end: '2024-01-07',
          data: generateTimeSeriesData('2024-01-01', 7, 45000, 120000)
        },
        metrics: {
          growth: 12.5,
          volatility: 8.3,
          trend: 'up',
          significance: 'high'
        }
      },
      {
        id: 'users-comparison',
        name: 'Active Users',
        current: {
          start: '2024-01-08',
          end: '2024-01-15',
          data: generateTimeSeriesData('2024-01-08', 7, 2000, 3500)
        },
        previous: {
          start: '2024-01-01',
          end: '2024-01-07',
          data: generateTimeSeriesData('2024-01-01', 7, 1800, 3200)
        },
        metrics: {
          growth: 8.7,
          volatility: 5.2,
          trend: 'up',
          significance: 'medium'
        }
      }
    ];

    // Mock trend analysis
    const mockTrendAnalysis: TrendAnalysis[] = [
      {
        metric: 'revenue',
        timeframe: '30d',
        pattern: 'exponential',
        strength: 0.85,
        direction: 'positive',
        forecast: {
          nextPeriod: 165000,
          confidence: 0.78,
          range: { min: 145000, max: 185000 }
        },
        anomalies: [
          {
            timestamp: '2024-01-12T14:30:00Z',
            value: 95000,
            deviation: -2.3,
            type: 'drop'
          },
          {
            timestamp: '2024-01-14T10:15:00Z',
            value: 185000,
            deviation: 2.8,
            type: 'spike'
          }
        ]
      },
      {
        metric: 'trading_volume',
        timeframe: '30d',
        pattern: 'cyclical',
        strength: 0.72,
        direction: 'positive',
        forecast: {
          nextPeriod: 2850000,
          confidence: 0.65,
          range: { min: 2600000, max: 3100000 }
        },
        anomalies: []
      }
    ];

    // Mock seasonal patterns
    const mockSeasonalPatterns: SeasonalPattern[] = [
      {
        period: 'weekly',
        peaks: [
          { time: 'Tuesday', value: 125000 },
          { time: 'Thursday', value: 135000 }
        ],
        troughs: [
          { time: 'Sunday', value: 85000 },
          { time: 'Saturday', value: 90000 }
        ],
        cyclicity: 0.68,
        amplitude: 0.45
      },
      {
        period: 'daily',
        peaks: [
          { time: '14:00', value: 15000 },
          { time: '20:00', value: 18000 }
        ],
        troughs: [
          { time: '04:00', value: 3000 },
          { time: '07:00', value: 4500 }
        ],
        cyclicity: 0.82,
        amplitude: 0.65
      }
    ];

    setComparisons(mockComparisons);
    setTrendAnalysis(mockTrendAnalysis);
    setSeasonalPatterns(mockSeasonalPatterns);
    setSelectedComparison(mockComparisons[0]);
  };

  const generateTimeSeriesData = (startDate: string, days: number, min: number, max: number): TimeSeriesData[] => {
    const data: TimeSeriesData[] = [];
    const start = new Date(startDate);

    for (let i = 0; i < days; i++) {
      const date = new Date(start.getTime() + i * 24 * 60 * 60 * 1000);
      const baseValue = min + (max - min) * Math.random();

      // Add some trend and noise
      const trend = (i / days) * (max - min) * 0.3;
      const noise = (Math.random() - 0.5) * (max - min) * 0.1;

      data.push({
        timestamp: date.toISOString(),
        value: Math.max(min, Math.min(max, baseValue + trend + noise)),
        category: selectedMetric
      });
    }

    return data;
  };

  const calculateGrowthRate = (current: number, previous: number) => {
    return ((current - previous) / previous) * 100;
  };

  const calculateVolatility = (data: TimeSeriesData[]) => {
    if (data.length < 2) return 0;

    const values = data.map(d => d.value);
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    return Math.sqrt(variance) / mean * 100;
  };

  const getMetricIcon = (metric: string) => {
    switch (metric) {
      case 'revenue': return '💰';
      case 'users': return '👥';
      case 'trades': return '📈';
      case 'volume': return '📊';
      default: return '📈';
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up': return <TrendingUp className="h-4 w-4 text-green-600" />;
      case 'down': return <TrendingDown className="h-4 w-4 text-red-600" />;
      default: return <BarChart3 className="h-4 w-4 text-gray-600" />;
    }
  };

  const getSignificanceColor = (significance: string) => {
    switch (significance) {
      case 'high': return 'bg-red-100 text-red-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'low': return 'bg-green-100 text-green-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const metrics = [
    { value: 'revenue', label: 'Revenue' },
    { value: 'users', label: 'Active Users' },
    { value: 'trades', label: 'Trading Volume' },
    { value: 'profit', label: 'Profit Margin' }
  ];

  const timeframes = [
    { value: '24h', label: 'Last 24 Hours' },
    { value: '7d', label: 'Last 7 Days' },
    { value: '30d', label: 'Last 30 Days' },
    { value: '90d', label: 'Last 3 Months' },
    { value: '1y', label: 'Last Year' }
  ];

  const comparisonTypes = [
    { value: 'previous_period', label: 'Previous Period' },
    { value: 'same_period_last_year', label: 'Same Period Last Year' },
    { value: 'baseline', label: 'Custom Baseline' },
    { value: 'rolling_average', label: 'Rolling Average' }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Temporal Analysis & Comparisons</h1>
          <p className="text-muted-foreground">
            Advanced time-based analytics with trend forecasting and pattern detection
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline">
            <Download className="h-4 w-4 mr-2" />
            Export Analysis
          </Button>
          <Button variant="outline">
            <Filter className="h-4 w-4 mr-2" />
            Advanced Filters
          </Button>
        </div>
      </div>

      {/* Controls */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Select value={selectedMetric} onValueChange={setSelectedMetric}>
          <SelectTrigger>
            <SelectValue placeholder="Select metric" />
          </SelectTrigger>
          <SelectContent>
            {metrics.map(metric => (
              <SelectItem key={metric.value} value={metric.value}>
                {getMetricIcon(metric.value)} {metric.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select value={selectedTimeframe} onValueChange={setSelectedTimeframe}>
          <SelectTrigger>
            <Clock className="h-4 w-4 mr-2" />
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {timeframes.map(timeframe => (
              <SelectItem key={timeframe.value} value={timeframe.value}>
                {timeframe.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select value={comparisonType} onValueChange={setComparisonType}>
          <SelectTrigger>
            <ArrowRight className="h-4 w-4 mr-2" />
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {comparisonTypes.map(type => (
              <SelectItem key={type.value} value={type.value}>
                {type.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <Switch id="forecast" checked={showForecast} onCheckedChange={setShowForecast} />
            <Label htmlFor="forecast" className="text-sm">Forecast</Label>
          </div>
          <div className="flex items-center space-x-2">
            <Switch id="anomalies" checked={showAnomalies} onCheckedChange={setShowAnomalies} />
            <Label htmlFor="anomalies" className="text-sm">Anomalies</Label>
          </div>
        </div>
      </div>

      {/* Period Comparisons */}
      {selectedComparison && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Current Period */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Calendar className="h-5 w-5" />
                <span>Current Period</span>
              </CardTitle>
              <CardDescription>
                {new Date(selectedComparison.current.start).toLocaleDateString()} - {new Date(selectedComparison.current.end).toLocaleDateString()}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <div className="text-2xl font-bold">
                    {selectedComparison.current.data.reduce((sum, d) => sum + d.value, 0).toLocaleString()}
                  </div>
                  <div className="text-sm text-muted-foreground">Total Value</div>
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <div className="font-medium">
                      {(selectedComparison.current.data.reduce((sum, d) => sum + d.value, 0) / selectedComparison.current.data.length).toLocaleString()}
                    </div>
                    <div className="text-muted-foreground">Daily Avg</div>
                  </div>
                  <div>
                    <div className="font-medium">
                      {Math.max(...selectedComparison.current.data.map(d => d.value)).toLocaleString()}
                    </div>
                    <div className="text-muted-foreground">Peak</div>
                  </div>
                </div>

                <div className="h-20 bg-muted/30 rounded-lg flex items-center justify-center">
                  <div className="text-center">
                    <BarChart3 className="h-6 w-6 mx-auto mb-1 text-muted-foreground" />
                    <div className="text-xs text-muted-foreground">Current Period Chart</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Previous Period */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <ArrowLeft className="h-5 w-5" />
                <span>Previous Period</span>
              </CardTitle>
              <CardDescription>
                {new Date(selectedComparison.previous.start).toLocaleDateString()} - {new Date(selectedComparison.previous.end).toLocaleDateString()}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <div className="text-2xl font-bold">
                    {selectedComparison.previous.data.reduce((sum, d) => sum + d.value, 0).toLocaleString()}
                  </div>
                  <div className="text-sm text-muted-foreground">Total Value</div>
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <div className="font-medium">
                      {(selectedComparison.previous.data.reduce((sum, d) => sum + d.value, 0) / selectedComparison.previous.data.length).toLocaleString()}
                    </div>
                    <div className="text-muted-foreground">Daily Avg</div>
                  </div>
                  <div>
                    <div className="font-medium">
                      {Math.max(...selectedComparison.previous.data.map(d => d.value)).toLocaleString()}
                    </div>
                    <div className="text-muted-foreground">Peak</div>
                  </div>
                </div>

                <div className="h-20 bg-muted/30 rounded-lg flex items-center justify-center">
                  <div className="text-center">
                    <BarChart3 className="h-6 w-6 mx-auto mb-1 text-muted-foreground" />
                    <div className="text-xs text-muted-foreground">Previous Period Chart</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Comparison Metrics */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <TrendingUp className="h-5 w-5" />
                <span>Comparison Analysis</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Growth Rate</span>
                  <div className="flex items-center space-x-2">
                    {getTrendIcon(selectedComparison.metrics.trend)}
                    <span className={`font-medium ${
                      selectedComparison.metrics.growth > 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {selectedComparison.metrics.growth > 0 ? '+' : ''}{selectedComparison.metrics.growth.toFixed(1)}%
                    </span>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Volatility</span>
                  <span className="text-sm">{selectedComparison.metrics.volatility.toFixed(1)}%</span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Significance</span>
                  <Badge className={getSignificanceColor(selectedComparison.metrics.significance)}>
                    {selectedComparison.metrics.significance}
                  </Badge>
                </div>

                <Separator />

                <div className="space-y-2">
                  <div className="text-sm font-medium">Period Progress</div>
                  <Progress value={75} className="h-2" />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Start</span>
                    <span>75% Complete</span>
                    <span>End</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Detailed Analysis Tabs */}
      <Tabs defaultValue="trends" className="space-y-4">
        <TabsList>
          <TabsTrigger value="trends">Trend Analysis</TabsTrigger>
          <TabsTrigger value="seasonal">Seasonal Patterns</TabsTrigger>
          <TabsTrigger value="forecast">Forecasting</TabsTrigger>
          <TabsTrigger value="anomalies">Anomaly Detection</TabsTrigger>
        </TabsList>

        <TabsContent value="trends" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {trendAnalysis.map(trend => (
              <Card key={trend.metric}>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <span>{trend.metric.replace('_', ' ').toUpperCase()}</span>
                    <Badge variant={trend.direction === 'positive' ? 'default' : trend.direction === 'negative' ? 'destructive' : 'secondary'}>
                      {trend.direction}
                    </Badge>
                  </CardTitle>
                  <CardDescription>
                    {trend.timeframe} analysis with {(trend.strength * 100).toFixed(0)}% confidence
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <div className="text-sm font-medium">Pattern Type</div>
                        <div className="text-sm text-muted-foreground capitalize">{trend.pattern}</div>
                      </div>
                      <div>
                        <div className="text-sm font-medium">Trend Strength</div>
                        <div className="flex items-center space-x-2">
                          <Progress value={trend.strength * 100} className="h-2 flex-1" />
                          <span className="text-sm">{(trend.strength * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    </div>

                    {showForecast && (
                      <div className="bg-muted/30 rounded-lg p-3">
                        <div className="text-sm font-medium mb-2">Next Period Forecast</div>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                          <div>
                            <div className="text-muted-foreground">Predicted Value</div>
                            <div className="font-medium">{trend.forecast.nextPeriod.toLocaleString()}</div>
                          </div>
                          <div>
                            <div className="text-muted-foreground">Confidence</div>
                            <div className="font-medium">{(trend.forecast.confidence * 100).toFixed(0)}%</div>
                          </div>
                          <div className="col-span-2">
                            <div className="text-muted-foreground">Range</div>
                            <div className="font-medium">
                              {trend.forecast.range.min.toLocaleString()} - {trend.forecast.range.max.toLocaleString()}
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="seasonal" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {seasonalPatterns.map((pattern, index) => (
              <Card key={index}>
                <CardHeader>
                  <CardTitle className="capitalize">{pattern.period} Patterns</CardTitle>
                  <CardDescription>
                    Cyclicity: {(pattern.cyclicity * 100).toFixed(0)}% | Amplitude: {(pattern.amplitude * 100).toFixed(0)}%
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <div className="text-sm font-medium text-green-600 mb-2">Peak Times</div>
                        <div className="space-y-1">
                          {pattern.peaks.map((peak, i) => (
                            <div key={i} className="flex justify-between text-sm">
                              <span>{peak.time}</span>
                              <span className="font-medium">{peak.value.toLocaleString()}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                      <div>
                        <div className="text-sm font-medium text-red-600 mb-2">Trough Times</div>
                        <div className="space-y-1">
                          {pattern.troughs.map((trough, i) => (
                            <div key={i} className="flex justify-between text-sm">
                              <span>{trough.time}</span>
                              <span className="font-medium">{trough.value.toLocaleString()}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>

                    <div className="h-32 bg-muted/30 rounded-lg flex items-center justify-center">
                      <div className="text-center">
                        <BarChart3 className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
                        <div className="text-sm text-muted-foreground">{pattern.period} Pattern Chart</div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="forecast">
          <Card>
            <CardHeader>
              <CardTitle>Predictive Forecasting</CardTitle>
              <CardDescription>
                AI-powered forecasts with confidence intervals and scenario analysis
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-96 flex items-center justify-center bg-muted/30 rounded-lg">
                <div className="text-center space-y-4">
                  <TrendingUp className="h-16 w-16 mx-auto text-muted-foreground" />
                  <div>
                    <p className="text-lg font-medium">Advanced Forecasting Dashboard</p>
                    <p className="text-sm text-muted-foreground max-w-md">
                      Integration with time series forecasting models (ARIMA, Prophet, LSTM) for predictive analytics
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="anomalies">
          <Card>
            <CardHeader>
              <CardTitle>Anomaly Detection</CardTitle>
              <CardDescription>
                Statistical and ML-based anomaly detection with root cause analysis
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {trendAnalysis
                  .filter(trend => trend.anomalies.length > 0)
                  .map(trend => (
                    <div key={trend.metric} className="space-y-3">
                      <h4 className="font-medium capitalize">{trend.metric.replace('_', ' ')} Anomalies</h4>
                      {trend.anomalies.map((anomaly, index) => (
                        <div key={index} className="flex items-center justify-between p-3 bg-muted/30 rounded-lg">
                          <div>
                            <div className="font-medium">
                              {anomaly.type === 'spike' ? '📈' : anomaly.type === 'drop' ? '📉' : '⚠️'} {anomaly.type.toUpperCase()}
                            </div>
                            <div className="text-sm text-muted-foreground">
                              {new Date(anomaly.timestamp).toLocaleString()}
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="font-medium">{anomaly.value.toLocaleString()}</div>
                            <div className={`text-sm ${anomaly.deviation > 0 ? 'text-red-600' : 'text-green-600'}`}>
                              {anomaly.deviation > 0 ? '+' : ''}{anomaly.deviation.toFixed(1)}σ
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ))}

                {trendAnalysis.every(trend => trend.anomalies.length === 0) && (
                  <div className="text-center py-8">
                    <div className="text-green-600 text-4xl mb-2">✅</div>
                    <p className="text-lg font-medium">No Anomalies Detected</p>
                    <p className="text-sm text-muted-foreground">
                      All metrics are within expected ranges for the selected time period
                    </p>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default TemporalAnalysis;