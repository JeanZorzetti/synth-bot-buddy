'use client';

import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, BarChart3, Activity, DollarSign, Users, Target, AlertTriangle, RefreshCw, Download, Filter, Calendar } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';

interface MetricData {
  id: string;
  name: string;
  value: number;
  previousValue: number;
  unit: string;
  category: string;
  trend: 'up' | 'down' | 'stable';
  changePercentage: number;
  target?: number;
  status: 'good' | 'warning' | 'critical';
  lastUpdated: string;
}

interface ChartData {
  id: string;
  title: string;
  type: 'line' | 'bar' | 'area' | 'pie' | 'gauge';
  data: Array<{
    timestamp: string;
    value: number;
    label?: string;
    category?: string;
  }>;
  config: {
    xAxis: string;
    yAxis: string;
    color: string;
    showTrend: boolean;
    showComparison: boolean;
  };
}

interface Alert {
  id: string;
  type: 'info' | 'warning' | 'error' | 'success';
  title: string;
  message: string;
  timestamp: string;
  isRead: boolean;
  actionRequired: boolean;
  relatedMetric?: string;
}

const RealTimeAnalytics: React.FC = () => {
  const [metrics, setMetrics] = useState<MetricData[]>([]);
  const [charts, setCharts] = useState<ChartData[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [isRealTimeEnabled, setIsRealTimeEnabled] = useState(true);
  const [selectedTimeRange, setSelectedTimeRange] = useState('24h');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [lastUpdate, setLastUpdate] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);

  // Simulate real-time data updates
  useEffect(() => {
    // Mock data initialization
    const mockMetrics: MetricData[] = [
      {
        id: '1',
        name: 'Active Trading Bots',
        value: 1247,
        previousValue: 1189,
        unit: 'bots',
        category: 'Trading',
        trend: 'up',
        changePercentage: 4.9,
        target: 1500,
        status: 'good',
        lastUpdated: new Date().toISOString()
      },
      {
        id: '2',
        name: 'Total Profit (24h)',
        value: 47582.34,
        previousValue: 43221.67,
        unit: 'USD',
        category: 'Financial',
        trend: 'up',
        changePercentage: 10.1,
        status: 'good',
        lastUpdated: new Date().toISOString()
      },
      {
        id: '3',
        name: 'Win Rate',
        value: 78.6,
        previousValue: 76.2,
        unit: '%',
        category: 'Performance',
        trend: 'up',
        changePercentage: 3.1,
        target: 80,
        status: 'good',
        lastUpdated: new Date().toISOString()
      },
      {
        id: '4',
        name: 'API Requests/min',
        value: 3847,
        previousValue: 4123,
        unit: 'req/min',
        category: 'System',
        trend: 'down',
        changePercentage: -6.7,
        target: 5000,
        status: 'warning',
        lastUpdated: new Date().toISOString()
      },
      {
        id: '5',
        name: 'Active Users',
        value: 2847,
        previousValue: 2756,
        unit: 'users',
        category: 'Users',
        trend: 'up',
        changePercentage: 3.3,
        target: 3000,
        status: 'good',
        lastUpdated: new Date().toISOString()
      },
      {
        id: '6',
        name: 'System Uptime',
        value: 99.97,
        previousValue: 99.89,
        unit: '%',
        category: 'System',
        trend: 'up',
        changePercentage: 0.08,
        target: 99.9,
        status: 'good',
        lastUpdated: new Date().toISOString()
      }
    ];

    const mockCharts: ChartData[] = [
      {
        id: 'chart1',
        title: 'Trading Volume (24h)',
        type: 'area',
        data: generateTimeSeriesData(24, 50000, 150000),
        config: {
          xAxis: 'time',
          yAxis: 'volume',
          color: '#3b82f6',
          showTrend: true,
          showComparison: true
        }
      },
      {
        id: 'chart2',
        title: 'Bot Performance Distribution',
        type: 'bar',
        data: [
          { timestamp: '0-20%', value: 45, label: 'Low Performance' },
          { timestamp: '20-40%', value: 123, label: 'Below Average' },
          { timestamp: '40-60%', value: 287, label: 'Average' },
          { timestamp: '60-80%', value: 456, label: 'Good' },
          { timestamp: '80-100%', value: 336, label: 'Excellent' }
        ],
        config: {
          xAxis: 'performance_range',
          yAxis: 'bot_count',
          color: '#10b981',
          showTrend: false,
          showComparison: false
        }
      },
      {
        id: 'chart3',
        title: 'Asset Class Distribution',
        type: 'pie',
        data: [
          { timestamp: 'Forex', value: 45, label: 'Forex', category: 'currency' },
          { timestamp: 'Crypto', value: 30, label: 'Cryptocurrency', category: 'crypto' },
          { timestamp: 'Stocks', value: 15, label: 'Stocks', category: 'equity' },
          { timestamp: 'Commodities', value: 10, label: 'Commodities', category: 'commodity' }
        ],
        config: {
          xAxis: 'asset_class',
          yAxis: 'percentage',
          color: '#f59e0b',
          showTrend: false,
          showComparison: false
        }
      }
    ];

    const mockAlerts: Alert[] = [
      {
        id: '1',
        type: 'warning',
        title: 'High API Usage',
        message: 'API requests have increased by 45% in the last hour. Consider scaling resources.',
        timestamp: new Date().toISOString(),
        isRead: false,
        actionRequired: true,
        relatedMetric: '4'
      },
      {
        id: '2',
        type: 'success',
        title: 'New Performance Record',
        message: 'Trading bots achieved a new daily profit record of $47,582.34',
        timestamp: new Date(Date.now() - 3600000).toISOString(),
        isRead: false,
        actionRequired: false,
        relatedMetric: '2'
      },
      {
        id: '3',
        type: 'info',
        title: 'Scheduled Maintenance',
        message: 'System maintenance scheduled for tomorrow 2:00 AM - 4:00 AM UTC',
        timestamp: new Date(Date.now() - 7200000).toISOString(),
        isRead: true,
        actionRequired: false
      }
    ];

    setMetrics(mockMetrics);
    setCharts(mockCharts);
    setAlerts(mockAlerts);
    setLastUpdate(new Date().toLocaleTimeString());

    // Real-time updates simulation
    if (isRealTimeEnabled) {
      const interval = setInterval(() => {
        updateMetrics();
        setLastUpdate(new Date().toLocaleTimeString());
      }, 5000); // Update every 5 seconds

      return () => clearInterval(interval);
    }
  }, [isRealTimeEnabled, selectedTimeRange]);

  const generateTimeSeriesData = (hours: number, min: number, max: number) => {
    const data = [];
    const now = new Date();

    for (let i = hours; i >= 0; i--) {
      const timestamp = new Date(now.getTime() - i * 60 * 60 * 1000);
      const value = Math.floor(Math.random() * (max - min) + min);
      data.push({
        timestamp: timestamp.toISOString(),
        value
      });
    }
    return data;
  };

  const updateMetrics = () => {
    setMetrics(prev => prev.map(metric => {
      const volatility = metric.category === 'Financial' ? 0.1 : 0.05;
      const change = (Math.random() - 0.5) * volatility;
      const newValue = Math.max(0, metric.value * (1 + change));

      return {
        ...metric,
        previousValue: metric.value,
        value: Math.round(newValue * 100) / 100,
        changePercentage: ((newValue - metric.value) / metric.value) * 100,
        trend: newValue > metric.value ? 'up' : newValue < metric.value ? 'down' : 'stable',
        lastUpdated: new Date().toISOString()
      };
    }));
  };

  const refreshData = async () => {
    setIsLoading(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1500));
    updateMetrics();
    setLastUpdate(new Date().toLocaleTimeString());
    setIsLoading(false);
  };

  const exportData = (format: 'csv' | 'excel' | 'pdf') => {
    // Implementation for data export
    console.log(`Exporting data in ${format} format`);
    // This would integrate with actual export functionality
  };

  const markAlertAsRead = (alertId: string) => {
    setAlerts(prev => prev.map(alert =>
      alert.id === alertId ? { ...alert, isRead: true } : alert
    ));
  };

  const getMetricIcon = (category: string) => {
    switch (category) {
      case 'Trading': return <BarChart3 className="h-5 w-5" />;
      case 'Financial': return <DollarSign className="h-5 w-5" />;
      case 'Performance': return <Target className="h-5 w-5" />;
      case 'System': return <Activity className="h-5 w-5" />;
      case 'Users': return <Users className="h-5 w-5" />;
      default: return <TrendingUp className="h-5 w-5" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'good': return 'text-green-600';
      case 'warning': return 'text-yellow-600';
      case 'critical': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const filteredMetrics = selectedCategory === 'all'
    ? metrics
    : metrics.filter(metric => metric.category === selectedCategory);

  const categories = ['all', 'Trading', 'Financial', 'Performance', 'System', 'Users'];
  const timeRanges = [
    { value: '1h', label: 'Last Hour' },
    { value: '24h', label: 'Last 24 Hours' },
    { value: '7d', label: 'Last 7 Days' },
    { value: '30d', label: 'Last 30 Days' }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Real-Time Analytics</h1>
          <p className="text-muted-foreground">
            Live monitoring dashboard with AI-powered insights
          </p>
        </div>
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <Switch
              id="real-time"
              checked={isRealTimeEnabled}
              onCheckedChange={setIsRealTimeEnabled}
            />
            <Label htmlFor="real-time" className="text-sm">
              Real-time updates
            </Label>
          </div>
          <Button variant="outline" onClick={refreshData} disabled={isLoading}>
            <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            {isLoading ? 'Refreshing...' : 'Refresh'}
          </Button>
          <Select>
            <SelectTrigger className="w-32">
              <Download className="h-4 w-4 mr-2" />
              <SelectValue placeholder="Export" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="csv" onClick={() => exportData('csv')}>CSV</SelectItem>
              <SelectItem value="excel" onClick={() => exportData('excel')}>Excel</SelectItem>
              <SelectItem value="pdf" onClick={() => exportData('pdf')}>PDF</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Select value={selectedTimeRange} onValueChange={setSelectedTimeRange}>
            <SelectTrigger className="w-48">
              <Calendar className="h-4 w-4 mr-2" />
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {timeRanges.map(range => (
                <SelectItem key={range.value} value={range.value}>
                  {range.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Select value={selectedCategory} onValueChange={setSelectedCategory}>
            <SelectTrigger className="w-48">
              <Filter className="h-4 w-4 mr-2" />
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {categories.map(category => (
                <SelectItem key={category} value={category}>
                  {category === 'all' ? 'All Categories' : category}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="text-sm text-muted-foreground">
          Last updated: {lastUpdate}
        </div>
      </div>

      {/* Alerts Bar */}
      {alerts.filter(alert => !alert.isRead).length > 0 && (
        <Card className="border-orange-200 bg-orange-50">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <AlertTriangle className="h-5 w-5 text-orange-600" />
                <span className="font-medium text-orange-800">
                  {alerts.filter(alert => !alert.isRead).length} Active Alert(s)
                </span>
              </div>
              <Button variant="outline" size="sm">
                View All Alerts
              </Button>
            </div>
            <div className="mt-2 space-y-1">
              {alerts.filter(alert => !alert.isRead).slice(0, 2).map(alert => (
                <div key={alert.id} className="text-sm text-orange-700">
                  <strong>{alert.title}:</strong> {alert.message}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredMetrics.map(metric => (
          <Card key={metric.id} className="hover:shadow-md transition-shadow">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  {getMetricIcon(metric.category)}
                  <CardTitle className="text-sm font-medium">{metric.name}</CardTitle>
                </div>
                <Badge variant="outline" className="text-xs">
                  {metric.category}
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-baseline space-x-2">
                  <span className="text-2xl font-bold">
                    {metric.unit === 'USD' ? '$' : ''}
                    {metric.value.toLocaleString()}
                    {metric.unit !== 'USD' ? metric.unit : ''}
                  </span>
                  <div className={`flex items-center space-x-1 ${
                    metric.trend === 'up' ? 'text-green-600' :
                    metric.trend === 'down' ? 'text-red-600' : 'text-gray-600'
                  }`}>
                    {metric.trend === 'up' ? <TrendingUp className="h-4 w-4" /> :
                     metric.trend === 'down' ? <TrendingDown className="h-4 w-4" /> : null}
                    <span className="text-sm font-medium">
                      {metric.changePercentage > 0 ? '+' : ''}
                      {metric.changePercentage.toFixed(1)}%
                    </span>
                  </div>
                </div>

                {metric.target && (
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>Progress to target</span>
                      <span>{Math.round((metric.value / metric.target) * 100)}%</span>
                    </div>
                    <Progress value={(metric.value / metric.target) * 100} className="h-2" />
                  </div>
                )}

                <div className="flex items-center justify-between text-xs text-muted-foreground">
                  <span>Previous: {metric.previousValue.toLocaleString()}</span>
                  <span className={getStatusColor(metric.status)}>
                    ● {metric.status.toUpperCase()}
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Charts Section */}
      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="trading">Trading Analytics</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="system">System Metrics</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {charts.map(chart => (
              <Card key={chart.id}>
                <CardHeader>
                  <CardTitle className="text-lg">{chart.title}</CardTitle>
                  <CardDescription>
                    Real-time data visualization with trend analysis
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-64 flex items-center justify-center bg-muted/30 rounded-lg">
                    <div className="text-center space-y-2">
                      <BarChart3 className="h-12 w-12 mx-auto text-muted-foreground" />
                      <p className="text-sm text-muted-foreground">
                        {chart.title} Chart
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Chart implementation with your preferred library (Chart.js, D3.js, etc.)
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="trading">
          <Card>
            <CardHeader>
              <CardTitle>Trading Performance Analytics</CardTitle>
              <CardDescription>
                Detailed trading metrics and performance indicators
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-96 flex items-center justify-center bg-muted/30 rounded-lg">
                <div className="text-center space-y-2">
                  <TrendingUp className="h-16 w-16 mx-auto text-muted-foreground" />
                  <p className="text-lg font-medium">Advanced Trading Charts</p>
                  <p className="text-sm text-muted-foreground max-w-md">
                    Integration with TradingView or custom charting solution for comprehensive trading analytics
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="performance">
          <Card>
            <CardHeader>
              <CardTitle>AI Model Performance</CardTitle>
              <CardDescription>
                Real-time AI model accuracy and performance metrics
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-96 flex items-center justify-center bg-muted/30 rounded-lg">
                <div className="text-center space-y-2">
                  <Target className="h-16 w-16 mx-auto text-muted-foreground" />
                  <p className="text-lg font-medium">Performance Dashboard</p>
                  <p className="text-sm text-muted-foreground max-w-md">
                    AI model accuracy, prediction confidence, and learning progress visualization
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="system">
          <Card>
            <CardHeader>
              <CardTitle>System Health Monitoring</CardTitle>
              <CardDescription>
                Infrastructure metrics and system performance indicators
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-96 flex items-center justify-center bg-muted/30 rounded-lg">
                <div className="text-center space-y-2">
                  <Activity className="h-16 w-16 mx-auto text-muted-foreground" />
                  <p className="text-lg font-medium">System Monitoring</p>
                  <p className="text-sm text-muted-foreground max-w-md">
                    CPU, memory, network, and database performance monitoring with alerting
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default RealTimeAnalytics;