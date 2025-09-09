import { useState, useEffect } from 'react';
import { Layout } from '@/components/Layout';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { 
  History as HistoryIcon,
  TrendingUp,
  TrendingDown,
  Filter,
  Download,
  Search,
  Calendar,
  Loader2,
  RefreshCw
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { apiService } from '@/services/api';
import { useToast } from '@/hooks/use-toast';

interface Trade {
  trade_id: string;
  symbol: string;
  contract_type: 'CALL' | 'PUT';
  amount: number;
  entry_price: number;
  exit_price: number | null;
  entry_time: number;
  exit_time: number | null;
  duration: number;
  status: 'won' | 'lost' | 'active' | 'cancelled';
  pnl: number | null;
  contract_id: string | null;
}

interface HistoryResponse {
  trades: Trade[];
  total_trades: number;
  summary: {
    total_pnl: number;
    wins: number;
    losses: number;
    win_rate: number;
  };
}

export default function History() {
  const { toast } = useToast();
  const [filterAsset, setFilterAsset] = useState<string>('all');
  const [filterOutcome, setFilterOutcome] = useState<string>('all');
  const [searchDate, setSearchDate] = useState('');
  const [historyData, setHistoryData] = useState<HistoryResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const loadTradeHistory = async () => {
    try {
      // Try to load trade history directly - if 404, endpoints aren't available yet
      const data = await apiService.get<HistoryResponse>('/trading/history?limit=100');
      setHistoryData(data);
    } catch (error: any) {
      console.error('Failed to load trade history:', error);
      
      if (error.message.includes('Not Found') || error.message.includes('404')) {
        // Trading endpoints not deployed yet - this is expected during gradual deployment
        console.log('Trading endpoints not available, showing empty state');
        setHistoryData({
          trades: [],
          total_trades: 0,
          summary: { total_pnl: 0, wins: 0, losses: 0, win_rate: 0 }
        });
        
        // Only show toast on manual refresh, not initial load
        if (isRefreshing) {
          toast({
            title: "Funcionalidade em Desenvolvimento",
            description: "Os endpoints de trading estão sendo deployados. Em breve você poderá visualizar o histórico completo.",
            variant: "default",
          });
        }
      } else {
        // Other errors (network, server, etc.)
        let errorMessage = "Não foi possível carregar o histórico de trades.";
        if (error.message.includes('Failed to fetch')) {
          errorMessage = "Não foi possível conectar ao backend. Verifique sua conexão.";
        }
        
        toast({
          title: "Erro ao Carregar Histórico",
          description: errorMessage,
          variant: "destructive",
        });
        
        // Set empty data on error
        setHistoryData({
          trades: [],
          total_trades: 0,
          summary: { total_pnl: 0, wins: 0, losses: 0, win_rate: 0 }
        });
      }
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  };

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await loadTradeHistory();
  };

  useEffect(() => {
    loadTradeHistory();
  }, []);

  // Show loading state
  if (isLoading) {
    return (
      <Layout>
        <div className="flex items-center justify-center h-64">
          <div className="flex flex-col items-center space-y-4">
            <Loader2 className="h-8 w-8 animate-spin" />
            <p className="text-muted-foreground">Carregando histórico de trades...</p>
          </div>
        </div>
      </Layout>
    );
  }

  if (!historyData) {
    return (
      <Layout>
        <div className="flex items-center justify-center h-64">
          <p className="text-muted-foreground">Erro ao carregar dados</p>
        </div>
      </Layout>
    );
  }

  // Convert symbol to readable asset name
  const getAssetName = (symbol: string) => {
    const symbolMap: Record<string, string> = {
      'R_10': 'Volatility 10 Index',
      'R_25': 'Volatility 25 Index',
      'R_50': 'Volatility 50 Index',
      'R_75': 'Volatility 75 Index',
      'R_100': 'Volatility 100 Index',
      'JD25': 'Jump 25 Index',
      'JD50': 'Jump 50 Index',
      'JD75': 'Jump 75 Index',
      'JD100': 'Jump 100 Index',
      'RDBULL': 'Boom 1000 Index',
      'RDBEAR': 'Crash 1000 Index'
    };
    return symbolMap[symbol] || symbol;
  };

  // Filter trades based on current filters
  const filteredTrades = historyData.trades.filter(trade => {
    const assetName = getAssetName(trade.symbol);
    const matchesAsset = filterAsset === 'all' || assetName === filterAsset;
    const matchesOutcome = filterOutcome === 'all' || trade.status === filterOutcome;
    
    let matchesDate = true;
    if (searchDate) {
      const tradeDate = new Date(trade.entry_time * 1000).toISOString().split('T')[0];
      matchesDate = tradeDate === searchDate;
    }
    
    return matchesAsset && matchesOutcome && matchesDate;
  });

  // Get unique assets for filter
  const uniqueAssets = [...new Set(historyData.trades.map(t => getAssetName(t.symbol)))];

  // Calculate statistics for filtered trades
  const totalTrades = filteredTrades.length;
  const winningTrades = filteredTrades.filter(t => t.status === 'won').length;
  const totalPnL = filteredTrades.reduce((sum, t) => sum + (t.pnl || 0), 0);
  const winRate = totalTrades > 0 ? (winningTrades / totalTrades) * 100 : 0;
  const largestWin = Math.max(...filteredTrades.filter(t => (t.pnl || 0) > 0).map(t => t.pnl || 0), 0);
  const largestLoss = Math.min(...filteredTrades.filter(t => (t.pnl || 0) < 0).map(t => t.pnl || 0), 0);


  return (
    <Layout>
      <div className="space-y-6">
        {/* Page Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Histórico de Operações</h1>
            <p className="text-muted-foreground">
              Analise todas as operações realizadas pelo bot ({historyData.total_trades} trades)
            </p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" onClick={handleRefresh} disabled={isRefreshing}>
              <RefreshCw className={cn("h-4 w-4 mr-2", isRefreshing && "animate-spin")} />
              Atualizar
            </Button>
          </div>
        </div>

        {/* Statistics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <Card className="trading-card">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">P/L Total</p>
                  <p className={cn(
                    "text-2xl font-bold",
                    totalPnL >= 0 ? "text-success" : "text-danger"
                  )}>
                    {totalPnL >= 0 ? '+' : ''}${totalPnL.toFixed(2)}
                  </p>
                </div>
                {totalPnL >= 0 ? (
                  <TrendingUp className="h-8 w-8 text-success" />
                ) : (
                  <TrendingDown className="h-8 w-8 text-danger" />
                )}
              </div>
            </CardContent>
          </Card>

          <Card className="trading-card">
            <CardContent className="p-6">
              <div>
                <p className="text-sm text-muted-foreground">Taxa de Vitórias</p>
                <p className="text-2xl font-bold">{winRate.toFixed(1)}%</p>
                <p className="text-sm text-muted-foreground">
                  {winningTrades}/{totalTrades} trades
                </p>
              </div>
            </CardContent>
          </Card>

          <Card className="trading-card">
            <CardContent className="p-6">
              <div>
                <p className="text-sm text-muted-foreground">Maior Lucro</p>
                <p className="text-2xl font-bold text-success">
                  +${largestWin.toFixed(2)}
                </p>
              </div>
            </CardContent>
          </Card>

          <Card className="trading-card">
            <CardContent className="p-6">
              <div>
                <p className="text-sm text-muted-foreground">Maior Perda</p>
                <p className="text-2xl font-bold text-danger">
                  ${largestLoss.toFixed(2)}
                </p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Filters and Actions */}
        <Card className="trading-card">
          <CardHeader>
            <CardTitle className="flex items-center">
              <Filter className="h-5 w-5 mr-2" />
              Filtros e Ações
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-4 items-end">
              <div className="flex-1 min-w-48">
                <label className="text-sm font-medium mb-2 block">Ativo</label>
                <Select value={filterAsset} onValueChange={setFilterAsset}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">Todos os Ativos</SelectItem>
                    {uniqueAssets.map(asset => (
                      <SelectItem key={asset} value={asset}>{asset}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="flex-1 min-w-48">
                <label className="text-sm font-medium mb-2 block">Resultado</label>
                <Select value={filterOutcome} onValueChange={setFilterOutcome}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">Todos</SelectItem>
                    <SelectItem value="won">Apenas Vitórias</SelectItem>
                    <SelectItem value="lost">Apenas Perdas</SelectItem>
                    <SelectItem value="active">Trades Ativos</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex-1 min-w-48">
                <label className="text-sm font-medium mb-2 block">Data</label>
                <Input
                  type="date"
                  value={searchDate}
                  onChange={(e) => setSearchDate(e.target.value)}
                  className="w-full"
                />
              </div>

              <Button variant="outline">
                <Download className="h-4 w-4 mr-2" />
                Exportar CSV
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Trades Table */}
        <Card className="trading-card">
          <CardHeader>
            <CardTitle className="flex items-center">
              <HistoryIcon className="h-5 w-5 mr-2" />
              Operações ({filteredTrades.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left p-4 font-medium">Ativo</th>
                    <th className="text-left p-4 font-medium">Data/Hora</th>
                    <th className="text-left p-4 font-medium">Tipo</th>
                    <th className="text-left p-4 font-medium">Stake</th>
                    <th className="text-left p-4 font-medium">Entrada</th>
                    <th className="text-left p-4 font-medium">Saída</th>
                    <th className="text-left p-4 font-medium">Duração</th>
                    <th className="text-left p-4 font-medium">P/L</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredTrades.map((trade) => (
                    <tr key={trade.trade_id} className="border-b border-border/50 hover:bg-muted/30">
                      <td className="p-4">
                        <div className="font-medium">{getAssetName(trade.symbol)}</div>
                      </td>
                      <td className="p-4">
                        <div className="text-sm">
                          {new Date(trade.entry_time * 1000).toLocaleString('pt-BR')}
                        </div>
                      </td>
                      <td className="p-4">
                        <Badge 
                          variant={trade.contract_type === 'CALL' ? 'default' : 'secondary'}
                          className={cn(
                            trade.contract_type === 'CALL' 
                              ? 'bg-success text-success-foreground' 
                              : 'bg-danger text-danger-foreground'
                          )}
                        >
                          {trade.contract_type}
                        </Badge>
                      </td>
                      <td className="p-4">
                        <div className="font-medium">${trade.amount.toFixed(2)}</div>
                      </td>
                      <td className="p-4">
                        <div className="font-mono text-sm">{trade.entry_price.toFixed(2)}</div>
                      </td>
                      <td className="p-4">
                        <div className="font-mono text-sm">
                          {trade.exit_price ? trade.exit_price.toFixed(2) : '-'}
                        </div>
                      </td>
                      <td className="p-4">
                        <div className="text-sm">{trade.duration} ticks</div>
                      </td>
                      <td className="p-4">
                        <div className={cn(
                          "font-bold",
                          (trade.pnl || 0) >= 0 ? "text-success" : "text-danger"
                        )}>
                          {trade.pnl !== null ? (
                            `${trade.pnl >= 0 ? '+' : ''}$${trade.pnl.toFixed(2)}`
                          ) : (
                            trade.status === 'active' ? 'Ativo' : '-'
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {filteredTrades.length === 0 && (
                <div className="text-center py-12 text-muted-foreground">
                  <HistoryIcon className="h-16 w-16 mx-auto mb-6 opacity-50" />
                  <h3 className="text-lg font-semibold mb-2">
                    {historyData.total_trades === 0 
                      ? "Histórico de Trading em Desenvolvimento" 
                      : "Nenhuma operação encontrada"
                    }
                  </h3>
                  <p className="mb-4 max-w-md mx-auto">
                    {historyData.total_trades === 0 
                      ? "O sistema de histórico de trades está sendo implementado. Em breve você poderá visualizar todas suas operações com detalhes completos." 
                      : "Nenhuma operação encontrada com os filtros selecionados. Tente ajustar os filtros ou aguarde novas operações."
                    }
                  </p>
                  {historyData.total_trades === 0 && (
                    <div className="mt-6 p-4 bg-muted/50 rounded-lg max-w-lg mx-auto">
                      <p className="text-sm font-medium mb-2">Funcionalidades em breve:</p>
                      <div className="grid grid-cols-1 gap-2 text-sm">
                        <div className="flex items-center justify-center space-x-2">
                          <TrendingUp className="h-4 w-4 text-primary" />
                          <span>Histórico detalhado de trades</span>
                        </div>
                        <div className="flex items-center justify-center space-x-2">
                          <Filter className="h-4 w-4 text-primary" />
                          <span>Filtros avançados por período</span>
                        </div>
                        <div className="flex items-center justify-center space-x-2">
                          <Download className="h-4 w-4 text-primary" />
                          <span>Exportação de relatórios</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </Layout>
  );
}