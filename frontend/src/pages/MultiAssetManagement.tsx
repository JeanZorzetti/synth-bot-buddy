/**
 * Multi-Asset Management - Phase 9 Integration
 * Interface completa para gerenciamento multi-ativos
 */

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';
import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ScatterChart, Scatter, Cell,
  BarChart, Bar, Heatmap
} from 'recharts';
import {
  TrendingUp, TrendingDown, BarChart3, Target,
  Globe, DollarSign, Activity, AlertTriangle,
  Plus, Minus, RefreshCw, Settings, Eye
} from 'lucide-react';
import { apiClient, MultiAssetSignal } from '@/services/apiClient';

interface AssetCluster {
  cluster_id: string;
  name: string;
  symbols: string[];
  avg_correlation: number;
  volatility_level: string;
  market_regime: string;
}

interface CorrelationData {
  symbol1: string;
  symbol2: string;
  correlation: number;
  timeframe: string;
}

interface CrossAssetMetrics {
  diversification_ratio: number;
  portfolio_var: number;
  concentration_risk: number;
  regime_stability: number;
  cross_asset_momentum: number;
}

interface PortfolioSymbol {
  symbol: string;
  asset_class: string;
  weight: number;
  current_price: number;
  daily_change: number;
  volatility: number;
  beta: number;
  correlation_score: number;
}

const MultiAssetManagement: React.FC = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [multiAssetStatus, setMultiAssetStatus] = useState<any>(null);
  const [correlationMatrix, setCorrelationMatrix] = useState<CorrelationData[]>([]);
  const [assetClusters, setAssetClusters] = useState<AssetCluster[]>([]);
  const [crossAssetSignals, setCrossAssetSignals] = useState<MultiAssetSignal[]>([]);
  const [portfolioSymbols, setPortfolioSymbols] = useState<PortfolioSymbol[]>([]);
  const [crossAssetMetrics, setCrossAssetMetrics] = useState<CrossAssetMetrics | null>(null);

  const [newSymbol, setNewSymbol] = useState('');
  const [newAssetClass, setNewAssetClass] = useState('');
  const [selectedTimeframe, setSelectedTimeframe] = useState('1H');
  const [selectedRegime, setSelectedRegime] = useState('all');

  const loadMultiAssetData = useCallback(async () => {
    setIsLoading(true);
    try {
      const [status, correlations, clusters, signals] = await Promise.all([
        apiClient.getMultiAssetStatus(),
        apiClient.getCorrelationMatrix(),
        apiClient.getAssetClusters(),
        apiClient.getCrossAssetSignals()
      ]);

      setMultiAssetStatus(status);
      setCorrelationMatrix(correlations.data || []);
      setAssetClusters(clusters);
      setCrossAssetSignals(signals);

      if (status.portfolio_symbols) {
        setPortfolioSymbols(status.portfolio_symbols);
      }

      if (status.cross_asset_metrics) {
        setCrossAssetMetrics(status.cross_asset_metrics);
      }
    } catch (error) {
      console.error('Erro ao carregar dados multi-ativos:', error);
    }
    setIsLoading(false);
  }, []);

  useEffect(() => {
    loadMultiAssetData();
    const interval = setInterval(loadMultiAssetData, 10000);
    return () => clearInterval(interval);
  }, [loadMultiAssetData]);

  const handleAddSymbol = async () => {
    if (!newSymbol || !newAssetClass) return;

    try {
      await apiClient.addSymbolToPortfolio(newSymbol, newAssetClass);
      setNewSymbol('');
      setNewAssetClass('');
      await loadMultiAssetData();
    } catch (error) {
      console.error('Erro ao adicionar símbolo:', error);
    }
  };

  const handleRemoveSymbol = async (symbol: string) => {
    try {
      await apiClient.removeSymbolFromPortfolio(symbol);
      await loadMultiAssetData();
    } catch (error) {
      console.error('Erro ao remover símbolo:', error);
    }
  };

  const getAssetClassColor = (assetClass: string) => {
    const colors: Record<string, string> = {
      'forex': '#3B82F6',
      'crypto': '#F59E0B',
      'commodities': '#10B981',
      'indices': '#8B5CF6',
      'stocks': '#EF4444'
    };
    return colors[assetClass.toLowerCase()] || '#6B7280';
  };

  const getCorrelationColor = (correlation: number) => {
    if (correlation > 0.7) return '#EF4444';
    if (correlation > 0.3) return '#F59E0B';
    if (correlation > -0.3) return '#10B981';
    return '#3B82F6';
  };

  const renderCorrelationHeatmap = () => {
    const uniqueSymbols = Array.from(new Set([
      ...correlationMatrix.map(d => d.symbol1),
      ...correlationMatrix.map(d => d.symbol2)
    ]));

    return (
      <div className="grid grid-cols-8 gap-1 text-xs">
        <div></div>
        {uniqueSymbols.slice(0, 7).map(symbol => (
          <div key={symbol} className="text-center font-medium p-1">
            {symbol}
          </div>
        ))}
        {uniqueSymbols.slice(0, 7).map(symbol1 => (
          <React.Fragment key={symbol1}>
            <div className="font-medium p-1">{symbol1}</div>
            {uniqueSymbols.slice(0, 7).map(symbol2 => {
              const corrData = correlationMatrix.find(
                d => (d.symbol1 === symbol1 && d.symbol2 === symbol2) ||
                     (d.symbol1 === symbol2 && d.symbol2 === symbol1)
              );
              const correlation = corrData?.correlation || (symbol1 === symbol2 ? 1 : 0);

              return (
                <div
                  key={symbol2}
                  className="p-1 text-center rounded text-white text-xs"
                  style={{ backgroundColor: getCorrelationColor(correlation) }}
                >
                  {correlation.toFixed(2)}
                </div>
              );
            })}
          </React.Fragment>
        ))}
      </div>
    );
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Multi-Asset Management</h1>
          <p className="text-muted-foreground">
            Gerenciamento avançado de portfólio multi-ativos
          </p>
        </div>
        <div className="flex gap-2">
          <Button onClick={loadMultiAssetData} disabled={isLoading}>
            <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Atualizar
          </Button>
          <Button variant="outline">
            <Settings className="h-4 w-4 mr-2" />
            Configurações
          </Button>
        </div>
      </div>

      {/* Status Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Ativos Ativos</p>
                <p className="text-2xl font-bold">
                  {portfolioSymbols.length}
                </p>
              </div>
              <Globe className="h-8 w-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Diversificação</p>
                <p className="text-2xl font-bold">
                  {crossAssetMetrics?.diversification_ratio?.toFixed(2) || '0.00'}
                </p>
              </div>
              <BarChart3 className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">VaR Portfolio</p>
                <p className="text-2xl font-bold">
                  {crossAssetMetrics?.portfolio_var?.toFixed(2) || '0.00'}%
                </p>
              </div>
              <AlertTriangle className="h-8 w-8 text-red-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Sinais Ativos</p>
                <p className="text-2xl font-bold">
                  {crossAssetSignals.length}
                </p>
              </div>
              <Activity className="h-8 w-8 text-purple-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="portfolio" className="space-y-4">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="portfolio">Portfolio</TabsTrigger>
          <TabsTrigger value="correlations">Correlações</TabsTrigger>
          <TabsTrigger value="clusters">Clusters</TabsTrigger>
          <TabsTrigger value="signals">Sinais</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>

        <TabsContent value="portfolio" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <Card>
                <CardHeader>
                  <CardTitle>Símbolos do Portfolio</CardTitle>
                  <CardDescription>
                    Distribuição atual de ativos no portfolio multi-classe
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {portfolioSymbols.map((symbol) => (
                      <div key={symbol.symbol} className="flex items-center justify-between p-4 border rounded-lg">
                        <div className="flex items-center space-x-4">
                          <Badge
                            style={{ backgroundColor: getAssetClassColor(symbol.asset_class) }}
                            className="text-white"
                          >
                            {symbol.asset_class.toUpperCase()}
                          </Badge>
                          <div>
                            <p className="font-medium">{symbol.symbol}</p>
                            <p className="text-sm text-muted-foreground">
                              ${symbol.current_price?.toFixed(4) || '0.0000'}
                            </p>
                          </div>
                        </div>

                        <div className="text-right space-x-4">
                          <div className="inline-block">
                            <p className="text-sm">Peso: {(symbol.weight * 100).toFixed(1)}%</p>
                            <Progress value={symbol.weight * 100} className="w-20" />
                          </div>

                          <div className="inline-block">
                            <p className={`text-sm ${symbol.daily_change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                              {symbol.daily_change >= 0 ? '+' : ''}{symbol.daily_change?.toFixed(2) || '0.00'}%
                            </p>
                          </div>

                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleRemoveSymbol(symbol.symbol)}
                          >
                            <Minus className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>

            <div>
              <Card>
                <CardHeader>
                  <CardTitle>Adicionar Ativo</CardTitle>
                  <CardDescription>
                    Expanda seu portfolio com novos ativos
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <Label htmlFor="symbol">Símbolo</Label>
                    <Input
                      id="symbol"
                      value={newSymbol}
                      onChange={(e) => setNewSymbol(e.target.value.toUpperCase())}
                      placeholder="ex: EURUSD, BTCUSD"
                    />
                  </div>

                  <div>
                    <Label htmlFor="assetClass">Classe de Ativo</Label>
                    <Select value={newAssetClass} onValueChange={setNewAssetClass}>
                      <SelectTrigger>
                        <SelectValue placeholder="Selecione a classe" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="forex">Forex</SelectItem>
                        <SelectItem value="crypto">Cripto</SelectItem>
                        <SelectItem value="commodities">Commodities</SelectItem>
                        <SelectItem value="indices">Índices</SelectItem>
                        <SelectItem value="stocks">Ações</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <Button
                    onClick={handleAddSymbol}
                    className="w-full"
                    disabled={!newSymbol || !newAssetClass}
                  >
                    <Plus className="h-4 w-4 mr-2" />
                    Adicionar Ativo
                  </Button>
                </CardContent>
              </Card>

              <Card className="mt-4">
                <CardHeader>
                  <CardTitle>Métricas Cross-Asset</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm">Risco de Concentração</span>
                    <span className="font-medium">
                      {crossAssetMetrics?.concentration_risk?.toFixed(2) || '0.00'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Estabilidade de Regime</span>
                    <span className="font-medium">
                      {crossAssetMetrics?.regime_stability?.toFixed(2) || '0.00'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-sm">Momentum Cross-Asset</span>
                    <span className="font-medium">
                      {crossAssetMetrics?.cross_asset_momentum?.toFixed(2) || '0.00'}
                    </span>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="correlations" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Matriz de Correlações</CardTitle>
              <CardDescription>
                Análise de correlações entre ativos do portfolio
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="mb-4">
                <Label>Timeframe</Label>
                <Select value={selectedTimeframe} onValueChange={setSelectedTimeframe}>
                  <SelectTrigger className="w-32">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="1H">1 Hora</SelectItem>
                    <SelectItem value="4H">4 Horas</SelectItem>
                    <SelectItem value="1D">1 Dia</SelectItem>
                    <SelectItem value="1W">1 Semana</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {correlationMatrix.length > 0 ? (
                renderCorrelationHeatmap()
              ) : (
                <Alert>
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>
                    Dados de correlação não disponíveis
                  </AlertDescription>
                </Alert>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="clusters" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {assetClusters.map((cluster) => (
              <Card key={cluster.cluster_id}>
                <CardHeader>
                  <CardTitle className="text-lg">{cluster.name}</CardTitle>
                  <CardDescription>
                    Correlação Média: {cluster.avg_correlation.toFixed(2)}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex justify-between text-sm">
                      <span>Volatilidade:</span>
                      <Badge variant={cluster.volatility_level === 'High' ? 'destructive' : 'secondary'}>
                        {cluster.volatility_level}
                      </Badge>
                    </div>

                    <div className="flex justify-between text-sm">
                      <span>Regime:</span>
                      <Badge variant="outline">{cluster.market_regime}</Badge>
                    </div>

                    <Separator />

                    <div>
                      <p className="text-sm font-medium mb-2">Símbolos:</p>
                      <div className="flex flex-wrap gap-1">
                        {cluster.symbols.map((symbol) => (
                          <Badge key={symbol} variant="secondary" className="text-xs">
                            {symbol}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="signals" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Sinais Cross-Asset</CardTitle>
              <CardDescription>
                Sinais de trading baseados em análise multi-ativos
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {crossAssetSignals.map((signal, index) => (
                  <div key={index} className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-4">
                        <Badge className="text-white" style={{
                          backgroundColor: signal.signal_direction > 0 ? '#10B981' : '#EF4444'
                        }}>
                          {signal.primary_symbol}
                        </Badge>
                        <div>
                          <p className="font-medium">
                            {signal.signal_direction > 0 ? 'COMPRA' : 'VENDA'}
                          </p>
                          <p className="text-sm text-muted-foreground">
                            {new Date(signal.timestamp).toLocaleTimeString()}
                          </p>
                        </div>
                      </div>

                      <div className="text-right space-y-1">
                        <p className="text-sm">
                          Confiança: <span className="font-medium">{(signal.confidence * 100).toFixed(1)}%</span>
                        </p>
                        <p className="text-sm">
                          Força Cross-Asset: <span className="font-medium">{(signal.cross_asset_strength * 100).toFixed(1)}%</span>
                        </p>
                        <p className="text-sm">
                          Consistência de Regime: <span className="font-medium">{(signal.regime_consistency * 100).toFixed(1)}%</span>
                        </p>
                      </div>
                    </div>

                    {signal.supporting_symbols.length > 0 && (
                      <div className="mt-3">
                        <p className="text-sm font-medium mb-2">Símbolos Suporte:</p>
                        <div className="flex flex-wrap gap-1">
                          {signal.supporting_symbols.map((symbol) => (
                            <Badge key={symbol} variant="outline" className="text-xs">
                              {symbol}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ))}

                {crossAssetSignals.length === 0 && (
                  <Alert>
                    <Eye className="h-4 w-4" />
                    <AlertDescription>
                      Nenhum sinal cross-asset ativo no momento
                    </AlertDescription>
                  </Alert>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Distribuição por Classe de Ativo</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={portfolioSymbols.reduce((acc, symbol) => {
                    const existing = acc.find(item => item.asset_class === symbol.asset_class);
                    if (existing) {
                      existing.weight += symbol.weight;
                      existing.count += 1;
                    } else {
                      acc.push({
                        asset_class: symbol.asset_class,
                        weight: symbol.weight,
                        count: 1
                      });
                    }
                    return acc;
                  }, [] as any[])}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="asset_class" />
                    <YAxis />
                    <Tooltip formatter={(value: any) => [`${(value * 100).toFixed(1)}%`, 'Peso']} />
                    <Bar dataKey="weight" fill="#3B82F6" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Volatilidade vs Beta</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <ScatterChart data={portfolioSymbols}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="volatility" name="Volatilidade" />
                    <YAxis dataKey="beta" name="Beta" />
                    <Tooltip
                      cursor={{ strokeDasharray: '3 3' }}
                      formatter={(value: any, name: string) => [value?.toFixed(3) || '0.000', name]}
                      labelFormatter={(value: any) => `${value}`}
                    />
                    <Scatter dataKey="beta" fill="#8B5CF6">
                      {portfolioSymbols.map((symbol, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={getAssetClassColor(symbol.asset_class)}
                        />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default MultiAssetManagement;