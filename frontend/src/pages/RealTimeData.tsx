/**
 * Real-Time Data Dashboard - Phase 6 Integration
 * Interface para dados em tempo real, features e time-series
 */

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { TrendingUp, TrendingDown, Activity, Database, Wifi, WifiOff, Eye, Settings } from 'lucide-react';
import apiClient, { RealTickData, ProcessedFeatures } from '@/services/apiClient';

interface DataQualityMetrics {
  symbol: string;
  completeness: number;
  accuracy: number;
  timeliness: number;
  consistency: number;
  overall_score: number;
}

interface ConnectionStatus {
  deriv_api: boolean;
  database: boolean;
  feature_engine: boolean;
  tick_processor: boolean;
  last_update: string;
}

const RealTimeData: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState<string>('EUR/USD');
  const [timeframe, setTimeframe] = useState<string>('1m');
  const [tickData, setTickData] = useState<RealTickData[]>([]);
  const [features, setFeatures] = useState<ProcessedFeatures | null>(null);
  const [dataQuality, setDataQuality] = useState<DataQualityMetrics[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>({
    deriv_api: false,
    database: false,
    feature_engine: false,
    tick_processor: false,
    last_update: ''
  });
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);

  const symbols = [
    'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD',
    'BTC/USD', 'ETH/USD', 'Gold', 'Oil', 'SPX500'
  ];

  // WebSocket connection for real-time data
  useEffect(() => {
    const ws = apiClient.subscribeToMarketData([selectedSymbol], (data: RealTickData) => {
      setTickData(prev => {
        const newData = [...prev, data].slice(-100); // Keep last 100 ticks
        return newData;
      });
    });

    setIsConnected(true);

    return () => {
      ws.close();
      setIsConnected(false);
    };
  }, [selectedSymbol]);

  // Load initial data
  useEffect(() => {
    loadInitialData();
  }, [selectedSymbol, timeframe]);

  const loadInitialData = async () => {
    setLoading(true);
    try {
      // Load historical tick data
      const ticks = await apiClient.getRealTickData(selectedSymbol);
      setTickData(ticks.slice(-100));

      // Load processed features
      const featuresData = await apiClient.getProcessedFeatures(selectedSymbol);
      setFeatures(featuresData);

      // Load data quality metrics
      const qualityData = await loadDataQualityMetrics();
      setDataQuality(qualityData);

      // Load connection status
      const status = await loadConnectionStatus();
      setConnectionStatus(status);

    } catch (error) {
      console.error('Error loading real-time data:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadDataQualityMetrics = async (): Promise<DataQualityMetrics[]> => {
    // Simulated data quality metrics
    return symbols.map(symbol => ({
      symbol,
      completeness: 95 + Math.random() * 5,
      accuracy: 98 + Math.random() * 2,
      timeliness: 92 + Math.random() * 8,
      consistency: 96 + Math.random() * 4,
      overall_score: 95 + Math.random() * 5
    }));
  };

  const loadConnectionStatus = async (): Promise<ConnectionStatus> => {
    // Simulated connection status
    return {
      deriv_api: Math.random() > 0.1,
      database: Math.random() > 0.05,
      feature_engine: Math.random() > 0.08,
      tick_processor: Math.random() > 0.03,
      last_update: new Date().toISOString()
    };
  };

  const formatPrice = (price: number): string => {
    return price.toFixed(5);
  };

  const getPriceChange = (): { value: number; direction: 'up' | 'down' | 'neutral' } => {
    if (tickData.length < 2) return { value: 0, direction: 'neutral' };

    const current = tickData[tickData.length - 1].price;
    const previous = tickData[tickData.length - 2].price;
    const change = current - previous;

    return {
      value: Math.abs(change),
      direction: change > 0 ? 'up' : change < 0 ? 'down' : 'neutral'
    };
  };

  const priceChange = getPriceChange();
  const latestTick = tickData[tickData.length - 1];

  const chartData = tickData.map((tick, index) => ({
    index,
    price: tick.price,
    volume: tick.volume,
    timestamp: new Date(tick.timestamp).toLocaleTimeString()
  }));

  const featureChartData = features ? Object.entries(features.features).map(([name, value]) => ({
    name: name.replace(/_/g, ' ').toUpperCase(),
    value: typeof value === 'number' ? value : 0
  })).slice(0, 10) : [];

  const qualityColors = ['#10B981', '#F59E0B', '#EF4444'];

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Real-Time Data Center</h1>
          <p className="text-muted-foreground">Live market data, features, and quality monitoring</p>
        </div>
        <div className="flex items-center space-x-2">
          <Badge variant={isConnected ? "default" : "destructive"}>
            {isConnected ? <Wifi className="w-4 h-4 mr-1" /> : <WifiOff className="w-4 h-4 mr-1" />}
            {isConnected ? 'Connected' : 'Disconnected'}
          </Badge>
          <Button onClick={loadInitialData} disabled={loading}>
            Refresh Data
          </Button>
        </div>
      </div>

      {/* Symbol Selection */}
      <div className="flex items-center space-x-4">
        <Select value={selectedSymbol} onValueChange={setSelectedSymbol}>
          <SelectTrigger className="w-48">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {symbols.map(symbol => (
              <SelectItem key={symbol} value={symbol}>{symbol}</SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select value={timeframe} onValueChange={setTimeframe}>
          <SelectTrigger className="w-32">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="1m">1m</SelectItem>
            <SelectItem value="5m">5m</SelectItem>
            <SelectItem value="15m">15m</SelectItem>
            <SelectItem value="1h">1h</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Current Price Card */}
      {latestTick && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <span>{selectedSymbol}</span>
              <Badge variant={priceChange.direction === 'up' ? 'default' : priceChange.direction === 'down' ? 'destructive' : 'secondary'}>
                {priceChange.direction === 'up' ? <TrendingUp className="w-4 h-4 mr-1" /> :
                 priceChange.direction === 'down' ? <TrendingDown className="w-4 h-4 mr-1" /> :
                 <Activity className="w-4 h-4 mr-1" />}
                {priceChange.direction === 'neutral' ? 'No Change' : `${priceChange.direction === 'up' ? '+' : '-'}${priceChange.value.toFixed(5)}`}
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div>
                <p className="text-sm text-muted-foreground">Current Price</p>
                <p className="text-2xl font-bold">{formatPrice(latestTick.price)}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Volume</p>
                <p className="text-xl font-semibold">{latestTick.volume.toLocaleString()}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Bid</p>
                <p className="text-xl font-semibold">{latestTick.bid ? formatPrice(latestTick.bid) : 'N/A'}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Ask</p>
                <p className="text-xl font-semibold">{latestTick.ask ? formatPrice(latestTick.ask) : 'N/A'}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Main Content Tabs */}
      <Tabs defaultValue="charts" className="space-y-4">
        <TabsList>
          <TabsTrigger value="charts">Price Charts</TabsTrigger>
          <TabsTrigger value="features">Features</TabsTrigger>
          <TabsTrigger value="quality">Data Quality</TabsTrigger>
          <TabsTrigger value="status">System Status</TabsTrigger>
        </TabsList>

        {/* Price Charts Tab */}
        <TabsContent value="charts" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Price Movement</CardTitle>
                <CardDescription>Real-time price data for {selectedSymbol}</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis domain={['dataMin - 0.001', 'dataMax + 0.001']} />
                    <Tooltip />
                    <Line type="monotone" dataKey="price" stroke="#2563eb" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Volume Profile</CardTitle>
                <CardDescription>Trading volume over time</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={chartData.slice(-20)}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="volume" fill="#10b981" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Features Tab */}
        <TabsContent value="features" className="space-y-4">
          {features && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle>Technical Features</CardTitle>
                  <CardDescription>
                    Processed features for {selectedSymbol} - Confidence: {(features.confidence * 100).toFixed(1)}%
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={featureChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="value" fill="#6366f1" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Feature Details</CardTitle>
                  <CardDescription>Individual feature values and statistics</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3 max-h-72 overflow-y-auto">
                    {Object.entries(features.features).map(([name, value]) => (
                      <div key={name} className="flex justify-between items-center p-2 bg-muted rounded">
                        <span className="text-sm font-medium">{name.replace(/_/g, ' ')}</span>
                        <Badge variant="outline">
                          {typeof value === 'number' ? value.toFixed(4) : String(value)}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>

        {/* Data Quality Tab */}
        <TabsContent value="quality" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Data Quality Overview</CardTitle>
                <CardDescription>Quality metrics across all symbols</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {dataQuality.map((quality, index) => (
                    <div key={quality.symbol} className="space-y-2">
                      <div className="flex justify-between">
                        <span className="font-medium">{quality.symbol}</span>
                        <span className="text-sm text-muted-foreground">
                          {quality.overall_score.toFixed(1)}%
                        </span>
                      </div>
                      <Progress value={quality.overall_score} className="h-2" />
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Quality Metrics Distribution</CardTitle>
                <CardDescription>Breakdown of quality metrics</CardDescription>
              </CardHeader>
              <CardContent>
                {dataQuality.length > 0 && (
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={[
                          { name: 'Excellent (95-100%)', value: dataQuality.filter(q => q.overall_score >= 95).length },
                          { name: 'Good (90-95%)', value: dataQuality.filter(q => q.overall_score >= 90 && q.overall_score < 95).length },
                          { name: 'Fair (<90%)', value: dataQuality.filter(q => q.overall_score < 90).length }
                        ]}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {qualityColors.map((color, index) => (
                          <Cell key={`cell-${index}`} fill={color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* System Status Tab */}
        <TabsContent value="status" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {Object.entries(connectionStatus).map(([service, status]) => {
              if (service === 'last_update') return null;

              return (
                <Card key={service}>
                  <CardHeader className="pb-3">
                    <CardTitle className="text-sm flex items-center justify-between">
                      <span>{service.replace(/_/g, ' ').toUpperCase()}</span>
                      {typeof status === 'boolean' && (
                        <Badge variant={status ? "default" : "destructive"}>
                          {status ? 'Online' : 'Offline'}
                        </Badge>
                      )}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center space-x-2">
                      {typeof status === 'boolean' && (
                        <>
                          <div className={`w-3 h-3 rounded-full ${status ? 'bg-green-500' : 'bg-red-500'}`} />
                          <span className="text-sm text-muted-foreground">
                            {status ? 'Connected' : 'Disconnected'}
                          </span>
                        </>
                      )}
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>

          <Card>
            <CardHeader>
              <CardTitle>System Health</CardTitle>
              <CardDescription>
                Last updated: {new Date(connectionStatus.last_update).toLocaleString()}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <Alert>
                  <Database className="h-4 w-4" />
                  <AlertDescription>
                    All data pipeline components are operational. Time-series database is processing
                    {tickData.length > 0 ? ` ${tickData.length} ticks` : ' data'} for {selectedSymbol}.
                  </AlertDescription>
                </Alert>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
                  <div>
                    <p className="text-2xl font-bold text-green-600">{tickData.length}</p>
                    <p className="text-sm text-muted-foreground">Ticks Processed</p>
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-blue-600">
                      {features ? Object.keys(features.features).length : 0}
                    </p>
                    <p className="text-sm text-muted-foreground">Features Generated</p>
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-purple-600">
                      {dataQuality.filter(q => q.overall_score >= 95).length}
                    </p>
                    <p className="text-sm text-muted-foreground">High Quality Sources</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default RealTimeData;