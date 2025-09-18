'use client';

import React, { useState, useEffect } from 'react';
import { BarChart3, LineChart, PieChart, TrendingUp, ZoomIn, Download, Settings, Maximize2, Filter, Calendar, MousePointer } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';

interface ChartConfig {
  id: string;
  title: string;
  type: 'line' | 'bar' | 'area' | 'pie' | 'scatter' | 'heatmap' | 'candlestick';
  data: any[];
  dimensions: {
    width: number;
    height: number;
  };
  options: {
    showLegend: boolean;
    showGrid: boolean;
    showTooltip: boolean;
    enableZoom: boolean;
    enableDrillDown: boolean;
    animation: boolean;
    color: string;
    gradient: boolean;
  };
  drillDown?: {
    enabled: boolean;
    levels: string[];
    currentLevel: number;
  };
  filters: {
    dateRange: { start: string; end: string };
    categories: string[];
    metrics: string[];
  };
}

interface DrillDownData {
  level: number;
  dimension: string;
  data: any[];
  breadcrumb: string[];
}

interface ChartInteraction {
  type: 'click' | 'hover' | 'zoom' | 'filter';
  data: any;
  timestamp: string;
}

const InteractiveCharts: React.FC = () => {
  const [charts, setCharts] = useState<ChartConfig[]>([]);
  const [selectedChart, setSelectedChart] = useState<string>('');
  const [drillDownStack, setDrillDownStack] = useState<DrillDownData[]>([]);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [interactions, setInteractions] = useState<ChartInteraction[]>([]);
  const [showFilters, setShowFilters] = useState(false);
  const [selectedTimeRange, setSelectedTimeRange] = useState('7d');

  useEffect(() => {
    // Mock chart configurations
    const mockCharts: ChartConfig[] = [
      {
        id: 'trading-volume',
        title: 'Trading Volume Analysis',
        type: 'line',
        data: generateTradingVolumeData(),
        dimensions: { width: 800, height: 400 },
        options: {
          showLegend: true,
          showGrid: true,
          showTooltip: true,
          enableZoom: true,
          enableDrillDown: true,
          animation: true,
          color: '#3b82f6',
          gradient: true
        },
        drillDown: {
          enabled: true,
          levels: ['day', 'hour', 'minute'],
          currentLevel: 0
        },
        filters: {
          dateRange: { start: '2024-01-01', end: '2024-01-15' },
          categories: ['forex', 'crypto', 'stocks'],
          metrics: ['volume', 'value', 'trades']
        }
      },
      {
        id: 'profit-distribution',
        title: 'Profit Distribution by Asset Class',
        type: 'pie',
        data: generateProfitDistributionData(),
        dimensions: { width: 500, height: 400 },
        options: {
          showLegend: true,
          showGrid: false,
          showTooltip: true,
          enableZoom: false,
          enableDrillDown: true,
          animation: true,
          color: '#10b981',
          gradient: false
        },
        drillDown: {
          enabled: true,
          levels: ['asset_class', 'instrument', 'strategy'],
          currentLevel: 0
        },
        filters: {
          dateRange: { start: '2024-01-01', end: '2024-01-15' },
          categories: ['forex', 'crypto', 'stocks', 'commodities'],
          metrics: ['profit', 'trades', 'win_rate']
        }
      },
      {
        id: 'performance-heatmap',
        title: 'Bot Performance Heatmap',
        type: 'heatmap',
        data: generatePerformanceHeatmapData(),
        dimensions: { width: 800, height: 500 },
        options: {
          showLegend: true,
          showGrid: true,
          showTooltip: true,
          enableZoom: true,
          enableDrillDown: true,
          animation: false,
          color: '#f59e0b',
          gradient: true
        },
        drillDown: {
          enabled: true,
          levels: ['strategy', 'bot', 'trades'],
          currentLevel: 0
        },
        filters: {
          dateRange: { start: '2024-01-01', end: '2024-01-15' },
          categories: ['strategy_type', 'risk_level'],
          metrics: ['return', 'sharpe_ratio', 'max_drawdown']
        }
      },
      {
        id: 'candlestick-analysis',
        title: 'Price Action Analysis',
        type: 'candlestick',
        data: generateCandlestickData(),
        dimensions: { width: 900, height: 600 },
        options: {
          showLegend: true,
          showGrid: true,
          showTooltip: true,
          enableZoom: true,
          enableDrillDown: false,
          animation: false,
          color: '#ef4444',
          gradient: false
        },
        filters: {
          dateRange: { start: '2024-01-01', end: '2024-01-15' },
          categories: ['EURUSD', 'GBPUSD', 'USDJPY'],
          metrics: ['ohlc', 'volume', 'indicators']
        }
      }
    ];

    setCharts(mockCharts);
    setSelectedChart(mockCharts[0].id);
  }, []);

  const generateTradingVolumeData = () => {
    const data = [];
    const now = new Date();

    for (let i = 30; i >= 0; i--) {
      const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
      data.push({
        date: date.toISOString().split('T')[0],
        forex: Math.floor(Math.random() * 1000000) + 500000,
        crypto: Math.floor(Math.random() * 800000) + 200000,
        stocks: Math.floor(Math.random() * 600000) + 300000,
        total: 0
      });
    }

    return data.map(item => ({
      ...item,
      total: item.forex + item.crypto + item.stocks
    }));
  };

  const generateProfitDistributionData = () => {
    return [
      { category: 'Forex', value: 45.2, profit: 125840.50, trades: 2847 },
      { category: 'Cryptocurrency', value: 32.1, profit: 89234.20, trades: 1923 },
      { category: 'Stocks', value: 15.8, profit: 43892.10, trades: 856 },
      { category: 'Commodities', value: 6.9, profit: 19234.80, trades: 412 }
    ];
  };

  const generatePerformanceHeatmapData = () => {
    const strategies = ['Scalping', 'Swing', 'Trend Following', 'Mean Reversion', 'Arbitrage'];
    const timeframes = ['1m', '5m', '15m', '1h', '4h', '1d'];
    const data = [];

    strategies.forEach((strategy, i) => {
      timeframes.forEach((timeframe, j) => {
        data.push({
          strategy,
          timeframe,
          performance: Math.random() * 40 - 10, // -10% to +30%
          trades: Math.floor(Math.random() * 1000) + 100,
          winRate: Math.random() * 40 + 50 // 50% to 90%
        });
      });
    });

    return data;
  };

  const generateCandlestickData = () => {
    const data = [];
    let price = 1.1000;

    for (let i = 0; i < 100; i++) {
      const open = price;
      const change = (Math.random() - 0.5) * 0.0050;
      const close = open + change;
      const high = Math.max(open, close) + Math.random() * 0.0020;
      const low = Math.min(open, close) - Math.random() * 0.0020;

      data.push({
        timestamp: new Date(Date.now() - (100 - i) * 60000).toISOString(),
        open: parseFloat(open.toFixed(5)),
        high: parseFloat(high.toFixed(5)),
        low: parseFloat(low.toFixed(5)),
        close: parseFloat(close.toFixed(5)),
        volume: Math.floor(Math.random() * 10000) + 1000
      });

      price = close;
    }

    return data;
  };

  const handleChartClick = (chartId: string, data: any) => {
    const chart = charts.find(c => c.id === chartId);
    if (!chart?.drillDown?.enabled) return;

    const interaction: ChartInteraction = {
      type: 'click',
      data,
      timestamp: new Date().toISOString()
    };

    setInteractions(prev => [...prev, interaction]);

    // Implement drill-down logic
    if (chart.drillDown.currentLevel < chart.drillDown.levels.length - 1) {
      const newLevel = chart.drillDown.currentLevel + 1;
      const drillDownData: DrillDownData = {
        level: newLevel,
        dimension: chart.drillDown.levels[newLevel],
        data: generateDrillDownData(chartId, newLevel, data),
        breadcrumb: [...(drillDownStack[drillDownStack.length - 1]?.breadcrumb || []), data.category || data.label]
      };

      setDrillDownStack(prev => [...prev, drillDownData]);

      // Update chart config
      setCharts(prev => prev.map(c =>
        c.id === chartId
          ? { ...c, drillDown: { ...c.drillDown!, currentLevel: newLevel } }
          : c
      ));
    }
  };

  const generateDrillDownData = (chartId: string, level: number, context: any) => {
    // Mock drill-down data generation based on context
    switch (chartId) {
      case 'profit-distribution':
        if (level === 1) {
          // Drill from asset class to instruments
          return [
            { instrument: 'EURUSD', profit: 25680.50, trades: 847 },
            { instrument: 'GBPUSD', profit: 18920.30, trades: 623 },
            { instrument: 'USDJPY', profit: 22180.70, trades: 756 }
          ];
        }
        break;
      default:
        return [];
    }
    return [];
  };

  const handleDrillUp = () => {
    if (drillDownStack.length === 0) return;

    const newStack = drillDownStack.slice(0, -1);
    setDrillDownStack(newStack);

    const chart = charts.find(c => c.id === selectedChart);
    if (chart) {
      setCharts(prev => prev.map(c =>
        c.id === selectedChart
          ? { ...c, drillDown: { ...c.drillDown!, currentLevel: Math.max(0, c.drillDown!.currentLevel - 1) } }
          : c
      ));
    }
  };

  const updateChartConfig = (chartId: string, updates: Partial<ChartConfig>) => {
    setCharts(prev => prev.map(chart =>
      chart.id === chartId ? { ...chart, ...updates } : chart
    ));
  };

  const exportChart = (chartId: string, format: 'png' | 'svg' | 'pdf') => {
    console.log(`Exporting chart ${chartId} as ${format}`);
    // Implementation for chart export
  };

  const selectedChartData = charts.find(c => c.id === selectedChart);
  const currentDrillDown = drillDownStack[drillDownStack.length - 1];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Interactive Charts</h1>
          <p className="text-muted-foreground">
            Advanced data visualization with drill-down capabilities
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" onClick={() => setShowFilters(!showFilters)}>
            <Filter className="h-4 w-4 mr-2" />
            Filters
          </Button>
          <Button variant="outline" onClick={() => setIsFullscreen(!isFullscreen)}>
            <Maximize2 className="h-4 w-4 mr-2" />
            Fullscreen
          </Button>
        </div>
      </div>

      {/* Chart Selection */}
      <div className="flex items-center space-x-4">
        <Select value={selectedChart} onValueChange={setSelectedChart}>
          <SelectTrigger className="w-64">
            <SelectValue placeholder="Select chart" />
          </SelectTrigger>
          <SelectContent>
            {charts.map(chart => (
              <SelectItem key={chart.id} value={chart.id}>
                {chart.title}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select value={selectedTimeRange} onValueChange={setSelectedTimeRange}>
          <SelectTrigger className="w-48">
            <Calendar className="h-4 w-4 mr-2" />
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="1d">Last 24 Hours</SelectItem>
            <SelectItem value="7d">Last 7 Days</SelectItem>
            <SelectItem value="30d">Last 30 Days</SelectItem>
            <SelectItem value="90d">Last 3 Months</SelectItem>
          </SelectContent>
        </Select>

        {/* Breadcrumb */}
        {currentDrillDown && (
          <div className="flex items-center space-x-2">
            <Button variant="ghost" size="sm" onClick={handleDrillUp}>
              ← Back
            </Button>
            <div className="flex items-center space-x-1 text-sm text-muted-foreground">
              {currentDrillDown.breadcrumb.map((item, index) => (
                <React.Fragment key={index}>
                  {index > 0 && <span>/</span>}
                  <span>{item}</span>
                </React.Fragment>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Filters Panel */}
      {showFilters && selectedChartData && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Chart Filters</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label>Categories</Label>
                <div className="space-y-2">
                  {selectedChartData.filters.categories.map(category => (
                    <div key={category} className="flex items-center space-x-2">
                      <Switch id={category} defaultChecked />
                      <Label htmlFor={category} className="capitalize">{category}</Label>
                    </div>
                  ))}
                </div>
              </div>

              <div className="space-y-2">
                <Label>Metrics</Label>
                <div className="space-y-2">
                  {selectedChartData.filters.metrics.map(metric => (
                    <div key={metric} className="flex items-center space-x-2">
                      <Switch id={metric} defaultChecked />
                      <Label htmlFor={metric} className="capitalize">{metric.replace('_', ' ')}</Label>
                    </div>
                  ))}
                </div>
              </div>

              <div className="space-y-2">
                <Label>Date Range</Label>
                <div className="space-y-2">
                  <Input
                    type="date"
                    value={selectedChartData.filters.dateRange.start}
                    onChange={(e) => updateChartConfig(selectedChart, {
                      filters: {
                        ...selectedChartData.filters,
                        dateRange: { ...selectedChartData.filters.dateRange, start: e.target.value }
                      }
                    })}
                  />
                  <Input
                    type="date"
                    value={selectedChartData.filters.dateRange.end}
                    onChange={(e) => updateChartConfig(selectedChart, {
                      filters: {
                        ...selectedChartData.filters,
                        dateRange: { ...selectedChartData.filters.dateRange, end: e.target.value }
                      }
                    })}
                  />
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Main Chart Display */}
      {selectedChartData && (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Chart Area */}
          <div className="lg:col-span-3">
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="flex items-center space-x-2">
                      {selectedChartData.type === 'line' && <LineChart className="h-5 w-5" />}
                      {selectedChartData.type === 'bar' && <BarChart3 className="h-5 w-5" />}
                      {selectedChartData.type === 'pie' && <PieChart className="h-5 w-5" />}
                      <span>{selectedChartData.title}</span>
                    </CardTitle>
                    <CardDescription>
                      {currentDrillDown ?
                        `Drilling down to ${currentDrillDown.dimension} level` :
                        'Click on data points to drill down for more details'
                      }
                    </CardDescription>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Popover>
                      <PopoverTrigger asChild>
                        <Button variant="outline" size="sm">
                          <Settings className="h-4 w-4" />
                        </Button>
                      </PopoverTrigger>
                      <PopoverContent className="w-80">
                        <div className="space-y-4">
                          <h4 className="font-medium">Chart Options</h4>

                          <div className="space-y-3">
                            <div className="flex items-center justify-between">
                              <Label htmlFor="legend">Show Legend</Label>
                              <Switch
                                id="legend"
                                checked={selectedChartData.options.showLegend}
                                onCheckedChange={(checked) => updateChartConfig(selectedChart, {
                                  options: { ...selectedChartData.options, showLegend: checked }
                                })}
                              />
                            </div>

                            <div className="flex items-center justify-between">
                              <Label htmlFor="grid">Show Grid</Label>
                              <Switch
                                id="grid"
                                checked={selectedChartData.options.showGrid}
                                onCheckedChange={(checked) => updateChartConfig(selectedChart, {
                                  options: { ...selectedChartData.options, showGrid: checked }
                                })}
                              />
                            </div>

                            <div className="flex items-center justify-between">
                              <Label htmlFor="animation">Enable Animation</Label>
                              <Switch
                                id="animation"
                                checked={selectedChartData.options.animation}
                                onCheckedChange={(checked) => updateChartConfig(selectedChart, {
                                  options: { ...selectedChartData.options, animation: checked }
                                })}
                              />
                            </div>

                            <div className="flex items-center justify-between">
                              <Label htmlFor="zoom">Enable Zoom</Label>
                              <Switch
                                id="zoom"
                                checked={selectedChartData.options.enableZoom}
                                onCheckedChange={(checked) => updateChartConfig(selectedChart, {
                                  options: { ...selectedChartData.options, enableZoom: checked }
                                })}
                              />
                            </div>
                          </div>
                        </div>
                      </PopoverContent>
                    </Popover>

                    <Select onValueChange={(format) => exportChart(selectedChart, format as any)}>
                      <SelectTrigger className="w-24">
                        <Download className="h-4 w-4" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="png">PNG</SelectItem>
                        <SelectItem value="svg">SVG</SelectItem>
                        <SelectItem value="pdf">PDF</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div
                  className="relative bg-muted/30 rounded-lg cursor-pointer"
                  style={{
                    height: selectedChartData.dimensions.height,
                    minHeight: '400px'
                  }}
                  onClick={(e) => {
                    // Mock click data - replace with actual chart click handling
                    const mockClickData = { category: 'Forex', value: 45.2 };
                    handleChartClick(selectedChart, mockClickData);
                  }}
                >
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center space-y-4">
                      {selectedChartData.type === 'line' && <LineChart className="h-16 w-16 mx-auto text-muted-foreground" />}
                      {selectedChartData.type === 'bar' && <BarChart3 className="h-16 w-16 mx-auto text-muted-foreground" />}
                      {selectedChartData.type === 'pie' && <PieChart className="h-16 w-16 mx-auto text-muted-foreground" />}
                      {selectedChartData.type === 'heatmap' && <TrendingUp className="h-16 w-16 mx-auto text-muted-foreground" />}
                      {selectedChartData.type === 'candlestick' && <BarChart3 className="h-16 w-16 mx-auto text-muted-foreground" />}

                      <div>
                        <p className="text-lg font-medium">{selectedChartData.title}</p>
                        <p className="text-sm text-muted-foreground">
                          Interactive {selectedChartData.type} chart
                        </p>
                        <p className="text-xs text-muted-foreground mt-2">
                          Chart implementation with your preferred library
                        </p>
                        {selectedChartData.options.enableDrillDown && (
                          <Badge variant="outline" className="mt-2">
                            <MousePointer className="h-3 w-3 mr-1" />
                            Click to drill down
                          </Badge>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Chart Info Panel */}
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Chart Info</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div>
                  <Label className="text-sm font-medium">Type</Label>
                  <p className="text-sm text-muted-foreground capitalize">{selectedChartData.type}</p>
                </div>

                <div>
                  <Label className="text-sm font-medium">Data Points</Label>
                  <p className="text-sm text-muted-foreground">{selectedChartData.data.length}</p>
                </div>

                <div>
                  <Label className="text-sm font-medium">Drill-down</Label>
                  <p className="text-sm text-muted-foreground">
                    {selectedChartData.drillDown?.enabled ? 'Enabled' : 'Disabled'}
                  </p>
                </div>

                {selectedChartData.drillDown?.enabled && (
                  <div>
                    <Label className="text-sm font-medium">Levels</Label>
                    <div className="space-y-1">
                      {selectedChartData.drillDown.levels.map((level, index) => (
                        <Badge
                          key={level}
                          variant={index === selectedChartData.drillDown!.currentLevel ? 'default' : 'outline'}
                          className="text-xs"
                        >
                          {level}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Interactions Log */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Recent Interactions</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {interactions.slice(-5).reverse().map((interaction, index) => (
                    <div key={index} className="text-xs p-2 bg-muted/50 rounded">
                      <div className="flex items-center justify-between">
                        <Badge variant="outline" className="text-xs">
                          {interaction.type}
                        </Badge>
                        <span className="text-muted-foreground">
                          {new Date(interaction.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                      <p className="mt-1 text-muted-foreground">
                        {JSON.stringify(interaction.data, null, 2).slice(0, 100)}...
                      </p>
                    </div>
                  ))}
                  {interactions.length === 0 && (
                    <p className="text-sm text-muted-foreground text-center py-4">
                      No interactions yet
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
};

export default InteractiveCharts;