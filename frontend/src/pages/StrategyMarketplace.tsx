/**
 * Strategy Marketplace - Phase 10 Integration
 * Marketplace completo para estratégias de trading
 */

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';
import { Textarea } from '@/components/ui/textarea';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Progress } from '@/components/ui/progress';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, BarChart, Bar, AreaChart, Area
} from 'recharts';
import {
  Store, Star, Download, DollarSign, TrendingUp, TrendingDown,
  Users, Code, Shield, Award, Filter, Search, Plus,
  Heart, Share, Eye, CheckCircle, XCircle, Upload,
  RefreshCw, Settings, Tag, Calendar, Activity
} from 'lucide-react';
import { apiClient, Strategy } from '@/services/apiClient';

interface MarketplaceStats {
  total_strategies: number;
  total_downloads: number;
  total_revenue: number;
  active_traders: number;
  avg_rating: number;
  trending_categories: string[];
}

interface StrategyDetails extends Strategy {
  backtest_metrics?: {
    total_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate: number;
    profit_factor: number;
    total_trades: number;
  };
  performance_chart?: Array<{
    date: string;
    value: number;
    benchmark: number;
  }>;
  reviews?: Array<{
    user: string;
    rating: number;
    comment: string;
    date: string;
  }>;
  code_preview?: string;
  license_terms?: string;
}

interface MyStrategy extends Strategy {
  sales_count: number;
  revenue: number;
  status: 'draft' | 'pending' | 'approved' | 'rejected';
  rejection_reason?: string;
}

const StrategyMarketplace: React.FC = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [myStrategies, setMyStrategies] = useState<MyStrategy[]>([]);
  const [marketplaceStats, setMarketplaceStats] = useState<MarketplaceStats | null>(null);
  const [selectedStrategy, setSelectedStrategy] = useState<StrategyDetails | null>(null);

  const [searchQuery, setSearchQuery] = useState('');
  const [categoryFilter, setCategoryFilter] = useState('all');
  const [priceFilter, setPriceFilter] = useState('all');
  const [ratingFilter, setRatingFilter] = useState('all');
  const [sortBy, setSortBy] = useState('rating');

  const [submitStrategyDialog, setSubmitStrategyDialog] = useState(false);
  const [strategyDetailsDialog, setStrategyDetailsDialog] = useState(false);
  const [newStrategy, setNewStrategy] = useState({
    name: '',
    description: '',
    category: '',
    pricing_model: 'one_time',
    price: 0,
    supported_symbols: [] as string[],
    tags: [] as string[],
    code: '',
    license_terms: ''
  });

  const [newSymbol, setNewSymbol] = useState('');
  const [newTag, setNewTag] = useState('');

  const loadMarketplaceData = useCallback(async () => {
    setIsLoading(true);
    try {
      const filters = {
        category: categoryFilter !== 'all' ? categoryFilter : undefined,
        min_rating: ratingFilter !== 'all' ? parseInt(ratingFilter) : undefined,
        max_price: priceFilter !== 'all' ? parseInt(priceFilter) : undefined,
        sort_by: sortBy
      };

      const [strategiesData, myStrategiesData, statsData] = await Promise.all([
        apiClient.getStrategies(filters),
        apiClient.getUserStrategies(),
        apiClient.getMarketplaceStats()
      ]);

      setStrategies(strategiesData);
      // Load real user strategies with sales data
      const realStrategiesData = await apiClient.getUserStrategiesWithSales();
      setMyStrategies(realStrategiesData.map(s => ({
        ...s,
        sales_count: s.sales_count || 0,
        revenue: s.revenue || 0,
        status: s.status || 'draft'
      })));
      setMarketplaceStats(statsData);

    } catch (error) {
      console.error('Erro ao carregar marketplace:', error);
    }
    setIsLoading(false);
  }, [categoryFilter, priceFilter, ratingFilter, sortBy]);

  useEffect(() => {
    loadMarketplaceData();
  }, [loadMarketplaceData]);

  const handleViewStrategy = async (strategyId: string) => {
    try {
      const strategyDetails = await apiClient.getStrategy(strategyId);

      // Load real strategy details
      const [backtestData, performanceData, reviewsData] = await Promise.all([
        apiClient.getStrategyBacktestMetrics(strategyId),
        apiClient.getStrategyPerformanceChart(strategyId),
        apiClient.getStrategyReviews(strategyId)
      ]);

      const detailedStrategy: StrategyDetails = {
        ...strategyDetails,
        backtest_metrics: backtestData || {
          total_return: 0,
          sharpe_ratio: 0,
          max_drawdown: 0,
          win_rate: 0,
          profit_factor: 0,
          total_trades: 0
        },
        performance_chart: performanceData || [],
        reviews: reviewsData || [],
        code_preview: strategyDetails.code_preview || 'Código não disponível',
        license_terms: strategyDetails.license_terms || 'Termos não definidos'
      };

      setSelectedStrategy(detailedStrategy);
      setStrategyDetailsDialog(true);
    } catch (error) {
      console.error('Erro ao carregar detalhes da estratégia:', error);
    }
  };

  const handlePurchaseStrategy = async (strategyId: string) => {
    try {
      await apiClient.purchaseStrategy(strategyId, 'personal');
      await loadMarketplaceData();
    } catch (error) {
      console.error('Erro ao comprar estratégia:', error);
    }
  };

  const handleSubmitStrategy = async () => {
    try {
      await apiClient.submitStrategy(newStrategy);
      setSubmitStrategyDialog(false);
      setNewStrategy({
        name: '',
        description: '',
        category: '',
        pricing_model: 'one_time',
        price: 0,
        supported_symbols: [],
        tags: [],
        code: '',
        license_terms: ''
      });
      await loadMarketplaceData();
    } catch (error) {
      console.error('Erro ao submeter estratégia:', error);
    }
  };

  const addSymbol = () => {
    if (newSymbol && !newStrategy.supported_symbols.includes(newSymbol)) {
      setNewStrategy({
        ...newStrategy,
        supported_symbols: [...newStrategy.supported_symbols, newSymbol.toUpperCase()]
      });
      setNewSymbol('');
    }
  };

  const addTag = () => {
    if (newTag && !newStrategy.tags.includes(newTag)) {
      setNewStrategy({
        ...newStrategy,
        tags: [...newStrategy.tags, newTag.toLowerCase()]
      });
      setNewTag('');
    }
  };

  const removeSymbol = (symbol: string) => {
    setNewStrategy({
      ...newStrategy,
      supported_symbols: newStrategy.supported_symbols.filter(s => s !== symbol)
    });
  };

  const removeTag = (tag: string) => {
    setNewStrategy({
      ...newStrategy,
      tags: newStrategy.tags.filter(t => t !== tag)
    });
  };

  const getCategoryColor = (category: string) => {
    const colors: Record<string, string> = {
      'scalping': 'bg-red-500',
      'swing': 'bg-blue-500',
      'trend': 'bg-green-500',
      'mean_reversion': 'bg-purple-500',
      'arbitrage': 'bg-yellow-500',
      'ml': 'bg-pink-500'
    };
    return colors[category.toLowerCase()] || 'bg-gray-500';
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'approved': return 'bg-green-500';
      case 'pending': return 'bg-yellow-500';
      case 'rejected': return 'bg-red-500';
      case 'draft': return 'bg-gray-500';
      default: return 'bg-gray-500';
    }
  };

  const filteredStrategies = strategies.filter(strategy => {
    const matchesSearch = strategy.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         strategy.description.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesSearch;
  });

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Strategy Marketplace</h1>
          <p className="text-muted-foreground">
            Descubra, compre e venda estratégias de trading
          </p>
        </div>
        <div className="flex gap-2">
          <Button onClick={loadMarketplaceData} disabled={isLoading}>
            <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Atualizar
          </Button>
          <Dialog open={submitStrategyDialog} onOpenChange={setSubmitStrategyDialog}>
            <DialogTrigger asChild>
              <Button>
                <Plus className="h-4 w-4 mr-2" />
                Submeter Estratégia
              </Button>
            </DialogTrigger>
          </Dialog>
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-6 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Total Estratégias</p>
                <p className="text-2xl font-bold">
                  {marketplaceStats?.total_strategies || 0}
                </p>
              </div>
              <Store className="h-8 w-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Downloads</p>
                <p className="text-2xl font-bold">
                  {marketplaceStats?.total_downloads?.toLocaleString() || '0'}
                </p>
              </div>
              <Download className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Receita Total</p>
                <p className="text-2xl font-bold">
                  ${marketplaceStats?.total_revenue?.toLocaleString() || '0'}
                </p>
              </div>
              <DollarSign className="h-8 w-8 text-yellow-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Traders Ativos</p>
                <p className="text-2xl font-bold">
                  {marketplaceStats?.active_traders || 0}
                </p>
              </div>
              <Users className="h-8 w-8 text-purple-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Avaliação Média</p>
                <p className="text-2xl font-bold">
                  {marketplaceStats?.avg_rating?.toFixed(1) || '0.0'}
                </p>
              </div>
              <Star className="h-8 w-8 text-orange-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Minhas Estratégias</p>
                <p className="text-2xl font-bold">
                  {myStrategies.length}
                </p>
              </div>
              <Code className="h-8 w-8 text-indigo-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="marketplace" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="marketplace">Marketplace</TabsTrigger>
          <TabsTrigger value="my-strategies">Minhas Estratégias</TabsTrigger>
          <TabsTrigger value="purchased">Compradas</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>

        <TabsContent value="marketplace" className="space-y-4">
          {/* Filters */}
          <Card>
            <CardContent className="p-4">
              <div className="grid grid-cols-1 md:grid-cols-6 gap-4">
                <div className="md:col-span-2">
                  <Label>Buscar</Label>
                  <div className="relative">
                    <Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" />
                    <Input
                      placeholder="Buscar estratégias..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-10"
                    />
                  </div>
                </div>

                <div>
                  <Label>Categoria</Label>
                  <Select value={categoryFilter} onValueChange={setCategoryFilter}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">Todas</SelectItem>
                      <SelectItem value="scalping">Scalping</SelectItem>
                      <SelectItem value="swing">Swing</SelectItem>
                      <SelectItem value="trend">Trend</SelectItem>
                      <SelectItem value="mean_reversion">Mean Reversion</SelectItem>
                      <SelectItem value="arbitrage">Arbitrage</SelectItem>
                      <SelectItem value="ml">Machine Learning</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label>Preço</Label>
                  <Select value={priceFilter} onValueChange={setPriceFilter}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">Todos</SelectItem>
                      <SelectItem value="0">Grátis</SelectItem>
                      <SelectItem value="50">Até $50</SelectItem>
                      <SelectItem value="100">Até $100</SelectItem>
                      <SelectItem value="500">Até $500</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label>Avaliação</Label>
                  <Select value={ratingFilter} onValueChange={setRatingFilter}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">Todas</SelectItem>
                      <SelectItem value="4">4+ estrelas</SelectItem>
                      <SelectItem value="3">3+ estrelas</SelectItem>
                      <SelectItem value="2">2+ estrelas</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label>Ordenar por</Label>
                  <Select value={sortBy} onValueChange={setSortBy}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="rating">Avaliação</SelectItem>
                      <SelectItem value="downloads">Downloads</SelectItem>
                      <SelectItem value="price">Preço</SelectItem>
                      <SelectItem value="recent">Mais Recentes</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Strategy Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredStrategies.map((strategy) => (
              <Card key={strategy.strategy_id} className="cursor-pointer hover:shadow-lg transition-shadow">
                <CardHeader>
                  <div className="flex justify-between items-start">
                    <div>
                      <CardTitle className="text-lg">{strategy.name}</CardTitle>
                      <CardDescription className="line-clamp-2">
                        {strategy.description}
                      </CardDescription>
                    </div>
                    <Badge className={`text-white ${getCategoryColor(strategy.category)}`}>
                      {strategy.category}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <div className="flex items-center space-x-1">
                        {Array.from({ length: 5 }).map((_, i) => (
                          <Star
                            key={i}
                            className={`h-4 w-4 ${
                              i < Math.floor(strategy.rating)
                                ? 'text-yellow-400 fill-current'
                                : 'text-gray-300'
                            }`}
                          />
                        ))}
                        <span className="text-sm text-muted-foreground ml-2">
                          ({strategy.rating.toFixed(1)})
                        </span>
                      </div>
                      <div className="text-right">
                        <p className="text-lg font-bold">
                          {strategy.price > 0 ? `$${strategy.price}` : 'Grátis'}
                        </p>
                      </div>
                    </div>

                    <div className="flex justify-between text-sm text-muted-foreground">
                      <span>{strategy.downloads} downloads</span>
                      <span>Por: {strategy.creator_id}</span>
                    </div>

                    <div className="flex flex-wrap gap-1">
                      {strategy.supported_symbols.slice(0, 3).map((symbol) => (
                        <Badge key={symbol} variant="outline" className="text-xs">
                          {symbol}
                        </Badge>
                      ))}
                      {strategy.supported_symbols.length > 3 && (
                        <Badge variant="outline" className="text-xs">
                          +{strategy.supported_symbols.length - 3}
                        </Badge>
                      )}
                    </div>

                    <div className="flex flex-wrap gap-1">
                      {strategy.tags.slice(0, 3).map((tag) => (
                        <Badge key={tag} variant="secondary" className="text-xs">
                          {tag}
                        </Badge>
                      ))}
                    </div>

                    <div className="flex space-x-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleViewStrategy(strategy.strategy_id)}
                        className="flex-1"
                      >
                        <Eye className="h-4 w-4 mr-1" />
                        Ver Detalhes
                      </Button>
                      <Button
                        size="sm"
                        onClick={() => handlePurchaseStrategy(strategy.strategy_id)}
                        className="flex-1"
                      >
                        {strategy.price > 0 ? 'Comprar' : 'Download'}
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="my-strategies" className="space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold">Minhas Estratégias</h3>
            <Dialog open={submitStrategyDialog} onOpenChange={setSubmitStrategyDialog}>
              <DialogTrigger asChild>
                <Button>
                  <Plus className="h-4 w-4 mr-2" />
                  Nova Estratégia
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-2xl">
                <DialogHeader>
                  <DialogTitle>Submeter Nova Estratégia</DialogTitle>
                  <DialogDescription>
                    Compartilhe sua estratégia com a comunidade
                  </DialogDescription>
                </DialogHeader>
                <ScrollArea className="max-h-[60vh]">
                  <div className="space-y-4 pr-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <Label>Nome da Estratégia</Label>
                        <Input
                          value={newStrategy.name}
                          onChange={(e) => setNewStrategy({...newStrategy, name: e.target.value})}
                          placeholder="Ex: Scalping RSI Otimizado"
                        />
                      </div>
                      <div>
                        <Label>Categoria</Label>
                        <Select
                          value={newStrategy.category}
                          onValueChange={(value) => setNewStrategy({...newStrategy, category: value})}
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="Selecione a categoria" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="scalping">Scalping</SelectItem>
                            <SelectItem value="swing">Swing</SelectItem>
                            <SelectItem value="trend">Trend</SelectItem>
                            <SelectItem value="mean_reversion">Mean Reversion</SelectItem>
                            <SelectItem value="arbitrage">Arbitrage</SelectItem>
                            <SelectItem value="ml">Machine Learning</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>

                    <div>
                      <Label>Descrição</Label>
                      <Textarea
                        value={newStrategy.description}
                        onChange={(e) => setNewStrategy({...newStrategy, description: e.target.value})}
                        placeholder="Descreva sua estratégia, como funciona e seus benefícios..."
                        rows={3}
                      />
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <Label>Modelo de Preço</Label>
                        <Select
                          value={newStrategy.pricing_model}
                          onValueChange={(value) => setNewStrategy({...newStrategy, pricing_model: value})}
                        >
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="free">Grátis</SelectItem>
                            <SelectItem value="one_time">Pagamento Único</SelectItem>
                            <SelectItem value="subscription">Assinatura</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div>
                        <Label>Preço ($)</Label>
                        <Input
                          type="number"
                          value={newStrategy.price}
                          onChange={(e) => setNewStrategy({...newStrategy, price: parseFloat(e.target.value) || 0})}
                          disabled={newStrategy.pricing_model === 'free'}
                        />
                      </div>
                    </div>

                    <div>
                      <Label>Símbolos Suportados</Label>
                      <div className="flex space-x-2 mb-2">
                        <Input
                          placeholder="Ex: EURUSD"
                          value={newSymbol}
                          onChange={(e) => setNewSymbol(e.target.value)}
                          onKeyPress={(e) => e.key === 'Enter' && addSymbol()}
                        />
                        <Button onClick={addSymbol} size="sm">
                          <Plus className="h-4 w-4" />
                        </Button>
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {newStrategy.supported_symbols.map((symbol) => (
                          <Badge
                            key={symbol}
                            variant="secondary"
                            className="cursor-pointer"
                            onClick={() => removeSymbol(symbol)}
                          >
                            {symbol} ×
                          </Badge>
                        ))}
                      </div>
                    </div>

                    <div>
                      <Label>Tags</Label>
                      <div className="flex space-x-2 mb-2">
                        <Input
                          placeholder="Ex: rsi, momentum"
                          value={newTag}
                          onChange={(e) => setNewTag(e.target.value)}
                          onKeyPress={(e) => e.key === 'Enter' && addTag()}
                        />
                        <Button onClick={addTag} size="sm">
                          <Plus className="h-4 w-4" />
                        </Button>
                      </div>
                      <div className="flex flex-wrap gap-1">
                        {newStrategy.tags.map((tag) => (
                          <Badge
                            key={tag}
                            variant="outline"
                            className="cursor-pointer"
                            onClick={() => removeTag(tag)}
                          >
                            {tag} ×
                          </Badge>
                        ))}
                      </div>
                    </div>

                    <div>
                      <Label>Código da Estratégia</Label>
                      <Textarea
                        value={newStrategy.code}
                        onChange={(e) => setNewStrategy({...newStrategy, code: e.target.value})}
                        placeholder="# Código Python da sua estratégia..."
                        rows={8}
                        className="font-mono text-sm"
                      />
                    </div>

                    <div>
                      <Label>Termos de Licença</Label>
                      <Textarea
                        value={newStrategy.license_terms}
                        onChange={(e) => setNewStrategy({...newStrategy, license_terms: e.target.value})}
                        placeholder="Defina os termos de uso da sua estratégia..."
                        rows={3}
                      />
                    </div>

                    <Button onClick={handleSubmitStrategy} className="w-full">
                      <Upload className="h-4 w-4 mr-2" />
                      Submeter Estratégia
                    </Button>
                  </div>
                </ScrollArea>
              </DialogContent>
            </Dialog>
          </div>

          <div className="grid gap-4">
            {myStrategies.map((strategy) => (
              <Card key={strategy.strategy_id}>
                <CardContent className="p-4">
                  <div className="flex justify-between items-start">
                    <div className="space-y-2">
                      <div className="flex items-center space-x-2">
                        <h4 className="font-semibold">{strategy.name}</h4>
                        <Badge className={`text-white ${getStatusColor(strategy.status)}`}>
                          {strategy.status}
                        </Badge>
                        <Badge className={`text-white ${getCategoryColor(strategy.category)}`}>
                          {strategy.category}
                        </Badge>
                      </div>

                      <p className="text-sm text-muted-foreground line-clamp-2">
                        {strategy.description}
                      </p>

                      <div className="flex space-x-6 text-sm">
                        <div>
                          <span className="text-muted-foreground">Downloads: </span>
                          <span className="font-medium">{strategy.downloads}</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Vendas: </span>
                          <span className="font-medium">{strategy.sales_count}</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Receita: </span>
                          <span className="font-medium">${strategy.revenue.toFixed(2)}</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Avaliação: </span>
                          <span className="font-medium">{strategy.rating.toFixed(1)}</span>
                        </div>
                      </div>

                      {strategy.status === 'rejected' && strategy.rejection_reason && (
                        <Alert>
                          <XCircle className="h-4 w-4" />
                          <AlertDescription>{strategy.rejection_reason}</AlertDescription>
                        </Alert>
                      )}
                    </div>

                    <div className="flex space-x-2">
                      <Button variant="outline" size="sm">
                        <Settings className="h-4 w-4" />
                      </Button>
                      <Button variant="outline" size="sm">
                        <Activity className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="purchased" className="space-y-4">
          <Alert>
            <CheckCircle className="h-4 w-4" />
            <AlertDescription>
              Suas estratégias compradas aparecerão aqui
            </AlertDescription>
          </Alert>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Vendas por Categoria</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={[
                    { category: 'Scalping', sales: 45 },
                    { category: 'Swing', sales: 32 },
                    { category: 'Trend', sales: 28 },
                    { category: 'ML', sales: 18 }
                  ]}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="category" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="sales" fill="#3B82F6" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Receita Mensal</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={[
                    { month: 'Jan', revenue: 2340 },
                    { month: 'Fev', revenue: 3456 },
                    { month: 'Mar', revenue: 4123 },
                    { month: 'Abr', revenue: 3890 }
                  ]}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip formatter={(value: any) => [`$${value}`, 'Receita']} />
                    <Area type="monotone" dataKey="revenue" stroke="#10B981" fill="#10B981" fillOpacity={0.3} />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>

      {/* Strategy Details Dialog */}
      <Dialog open={strategyDetailsDialog} onOpenChange={setStrategyDetailsDialog}>
        <DialogContent className="max-w-4xl">
          {selectedStrategy && (
            <>
              <DialogHeader>
                <DialogTitle className="flex items-center space-x-2">
                  <span>{selectedStrategy.name}</span>
                  <Badge className={`text-white ${getCategoryColor(selectedStrategy.category)}`}>
                    {selectedStrategy.category}
                  </Badge>
                </DialogTitle>
                <DialogDescription>
                  {selectedStrategy.description}
                </DialogDescription>
              </DialogHeader>

              <Tabs defaultValue="overview" className="mt-4">
                <TabsList className="grid w-full grid-cols-4">
                  <TabsTrigger value="overview">Overview</TabsTrigger>
                  <TabsTrigger value="performance">Performance</TabsTrigger>
                  <TabsTrigger value="code">Código</TabsTrigger>
                  <TabsTrigger value="reviews">Reviews</TabsTrigger>
                </TabsList>

                <TabsContent value="overview" className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <h4 className="font-semibold mb-2">Informações Gerais</h4>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span>Preço:</span>
                          <span className="font-medium">
                            {selectedStrategy.price > 0 ? `$${selectedStrategy.price}` : 'Grátis'}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Downloads:</span>
                          <span className="font-medium">{selectedStrategy.downloads}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Avaliação:</span>
                          <span className="font-medium">{selectedStrategy.rating.toFixed(1)}/5.0</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Criador:</span>
                          <span className="font-medium">{selectedStrategy.creator_id}</span>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h4 className="font-semibold mb-2">Métricas de Backtest</h4>
                      {selectedStrategy.backtest_metrics && (
                        <div className="space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span>Retorno Total:</span>
                            <span className="font-medium text-green-600">
                              {selectedStrategy.backtest_metrics.total_return.toFixed(2)}%
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span>Sharpe Ratio:</span>
                            <span className="font-medium">{selectedStrategy.backtest_metrics.sharpe_ratio.toFixed(2)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>Max Drawdown:</span>
                            <span className="font-medium text-red-600">
                              {selectedStrategy.backtest_metrics.max_drawdown.toFixed(2)}%
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span>Win Rate:</span>
                            <span className="font-medium">{selectedStrategy.backtest_metrics.win_rate.toFixed(1)}%</span>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  <div>
                    <h4 className="font-semibold mb-2">Símbolos Suportados</h4>
                    <div className="flex flex-wrap gap-1">
                      {selectedStrategy.supported_symbols.map((symbol) => (
                        <Badge key={symbol} variant="outline">
                          {symbol}
                        </Badge>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h4 className="font-semibold mb-2">Tags</h4>
                    <div className="flex flex-wrap gap-1">
                      {selectedStrategy.tags.map((tag) => (
                        <Badge key={tag} variant="secondary">
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="performance">
                  {selectedStrategy.performance_chart && (
                    <div className="space-y-4">
                      <h4 className="font-semibold">Performance vs Benchmark</h4>
                      <ResponsiveContainer width="100%" height={400}>
                        <LineChart data={selectedStrategy.performance_chart}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="date" />
                          <YAxis />
                          <Tooltip />
                          <Line
                            type="monotone"
                            dataKey="value"
                            stroke="#3B82F6"
                            strokeWidth={2}
                            name="Estratégia"
                          />
                          <Line
                            type="monotone"
                            dataKey="benchmark"
                            stroke="#6B7280"
                            strokeWidth={1}
                            strokeDasharray="5 5"
                            name="Benchmark"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </TabsContent>

                <TabsContent value="code">
                  <div className="space-y-4">
                    <h4 className="font-semibold">Preview do Código</h4>
                    <pre className="bg-muted p-4 rounded-lg text-sm overflow-x-auto">
                      <code>{selectedStrategy.code_preview}</code>
                    </pre>
                    <Alert>
                      <Shield className="h-4 w-4" />
                      <AlertDescription>
                        O código completo estará disponível após a compra
                      </AlertDescription>
                    </Alert>
                  </div>
                </TabsContent>

                <TabsContent value="reviews">
                  <div className="space-y-4">
                    {selectedStrategy.reviews?.map((review, index) => (
                      <div key={index} className="border-b pb-4 last:border-b-0">
                        <div className="flex justify-between items-start mb-2">
                          <div className="flex items-center space-x-2">
                            <span className="font-medium">{review.user}</span>
                            <div className="flex items-center">
                              {Array.from({ length: 5 }).map((_, i) => (
                                <Star
                                  key={i}
                                  className={`h-4 w-4 ${
                                    i < review.rating
                                      ? 'text-yellow-400 fill-current'
                                      : 'text-gray-300'
                                  }`}
                                />
                              ))}
                            </div>
                          </div>
                          <span className="text-sm text-muted-foreground">
                            {new Date(review.date).toLocaleDateString()}
                          </span>
                        </div>
                        <p className="text-sm">{review.comment}</p>
                      </div>
                    ))}
                  </div>
                </TabsContent>
              </Tabs>

              <div className="flex justify-end space-x-2 mt-6">
                <Button variant="outline">
                  <Heart className="h-4 w-4 mr-2" />
                  Favoritar
                </Button>
                <Button variant="outline">
                  <Share className="h-4 w-4 mr-2" />
                  Compartilhar
                </Button>
                <Button onClick={() => handlePurchaseStrategy(selectedStrategy.strategy_id)}>
                  {selectedStrategy.price > 0 ? `Comprar - $${selectedStrategy.price}` : 'Download Grátis'}
                </Button>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default StrategyMarketplace;