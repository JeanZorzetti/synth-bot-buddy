import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  History,
  TrendingUp,
  TrendingDown,
  Clock,
  Filter,
  Download,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
  DollarSign,
  BarChart3,
  Activity,
  FileText,
  FileSpreadsheet,
  ChevronDown
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { tradesApi, type Trade } from '@/services/api';
import { format } from 'date-fns';
import { exportUtils } from '@/utils/exporters';

export default function TradeHistory() {
  const { toast } = useToast();

  // Data states
  const [trades, setTrades] = useState<Trade[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Pagination states
  const [page, setPage] = useState(1);
  const [limit, setLimit] = useState(25);
  const [totalPages, setTotalPages] = useState(1);
  const [total, setTotal] = useState(0);

  // Filter states
  const [symbolFilter, setSymbolFilter] = useState('');
  const [typeFilter, setTypeFilter] = useState('');
  const [resultFilter, setResultFilter] = useState('');
  const [strategyFilter, setStrategyFilter] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');

  // Sort states
  const [sortBy, setSortBy] = useState('timestamp');
  const [sortOrder, setSortOrder] = useState('DESC');

  useEffect(() => {
    loadTrades();
    loadStats();
  }, [page, limit, symbolFilter, typeFilter, resultFilter, strategyFilter, startDate, endDate, sortBy, sortOrder]);

  const loadTrades = async () => {
    try {
      setIsRefreshing(true);

      const response = await tradesApi.getHistory({
        page,
        limit,
        symbol: symbolFilter || undefined,
        trade_type: typeFilter || undefined,
        result: resultFilter || undefined,
        strategy: strategyFilter || undefined,
        start_date: startDate || undefined,
        end_date: endDate || undefined,
        sort_by: sortBy,
        sort_order: sortOrder,
      });

      setTrades(response.data.trades);
      setTotalPages(response.data.pagination.total_pages);
      setTotal(response.data.pagination.total);
    } catch (error: any) {
      console.error('Failed to load trades:', error);
      toast({
        title: "Erro ao carregar histórico",
        description: "Não foi possível carregar o histórico de trades. Tente novamente.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  };

  const loadStats = async () => {
    try {
      const response = await tradesApi.getStats();
      setStats(response.data);
    } catch (error: any) {
      console.error('Failed to load stats:', error);
    }
  };

  const handleRefresh = () => {
    loadTrades();
    loadStats();
  };

  const handleClearFilters = () => {
    setSymbolFilter('');
    setTypeFilter('');
    setResultFilter('');
    setStrategyFilter('');
    setStartDate('');
    setEndDate('');
    setPage(1);
  };

  const handleExportCSV = () => {
    try {
      const exportData = trades.map(trade => ({
        ID: trade.id,
        'Data/Hora': formatDate(trade.timestamp),
        'Símbolo': trade.symbol,
        'Tipo': trade.trade_type,
        'Entrada': trade.entry_price,
        'Saída': trade.exit_price || '-',
        'Stake': trade.stake,
        'P&L': trade.profit_loss || 0,
        'Resultado': trade.result,
        'Confiança': trade.confidence ? `${trade.confidence}%` : '-',
        'Estratégia': trade.strategy || '-',
      }));

      exportUtils.toCSV(exportData, `trade-history-${new Date().toISOString().split('T')[0]}`);

      toast({
        title: "Exportação concluída",
        description: `${trades.length} trades exportados para CSV`,
      });
    } catch (error) {
      toast({
        title: "Erro ao exportar",
        description: "Não foi possível exportar os dados para CSV",
        variant: "destructive",
      });
    }
  };

  const handleExportExcel = () => {
    try {
      const exportData = trades.map(trade => ({
        ID: trade.id,
        'Data/Hora': formatDate(trade.timestamp),
        'Símbolo': trade.symbol,
        'Tipo': trade.trade_type,
        'Entrada': trade.entry_price,
        'Saída': trade.exit_price || '-',
        'Stake': trade.stake,
        'P&L': trade.profit_loss || 0,
        'Resultado': trade.result,
        'Confiança (%)': trade.confidence || '-',
        'Estratégia': trade.strategy || '-',
      }));

      exportUtils.toExcel(exportData, `trade-history-${new Date().toISOString().split('T')[0]}`, 'Trades');

      toast({
        title: "Exportação concluída",
        description: `${trades.length} trades exportados para Excel`,
      });
    } catch (error) {
      toast({
        title: "Erro ao exportar",
        description: "Não foi possível exportar os dados para Excel",
        variant: "destructive",
      });
    }
  };

  const handleExportPDF = () => {
    try {
      const exportData = trades.map(trade => ({
        ID: trade.id,
        'Data/Hora': formatDate(trade.timestamp),
        'Símbolo': trade.symbol,
        'Tipo': trade.trade_type,
        'Entrada': formatPrice(trade.entry_price),
        'Saída': formatPrice(trade.exit_price),
        'Stake': formatPrice(trade.stake),
        'P&L': formatPnL(trade.profit_loss),
        'Resultado': trade.result,
        'Confiança': trade.confidence ? `${trade.confidence.toFixed(1)}%` : '-',
        'Estratégia': trade.strategy || '-',
      }));

      exportUtils.dataToPDF(
        exportData,
        `trade-history-${new Date().toISOString().split('T')[0]}`,
        'Trade History Report'
      );

      toast({
        title: "Exportação concluída",
        description: `${trades.length} trades exportados para PDF`,
      });
    } catch (error) {
      toast({
        title: "Erro ao exportar",
        description: "Não foi possível exportar os dados para PDF",
        variant: "destructive",
      });
    }
  };

  const formatPrice = (price?: number) => {
    if (price === undefined || price === null) return '-';
    return `$${price.toFixed(2)}`;
  };

  const formatPnL = (pnl?: number) => {
    if (pnl === undefined || pnl === null) return '-';
    const formatted = `$${Math.abs(pnl).toFixed(2)}`;
    return pnl >= 0 ? `+${formatted}` : `-${formatted}`;
  };

  const formatDate = (dateString: string) => {
    try {
      return format(new Date(dateString), 'dd/MM/yyyy HH:mm:ss');
    } catch {
      return dateString;
    }
  };

  const getResultBadge = (result: string) => {
    switch (result) {
      case 'win':
        return <Badge className="bg-success text-success-foreground">Win</Badge>;
      case 'loss':
        return <Badge className="bg-destructive text-destructive-foreground">Loss</Badge>;
      case 'pending':
        return <Badge variant="outline">Pending</Badge>;
      default:
        return <Badge variant="secondary">{result}</Badge>;
    }
  };

  const getStrategyBadge = (strategy?: string) => {
    if (!strategy) return null;

    const strategyColors: Record<string, string> = {
      ml: 'bg-blue-500',
      technical: 'bg-purple-500',
      hybrid: 'bg-green-500',
      order_flow: 'bg-orange-500',
    };

    return (
      <Badge className={strategyColors[strategy] || 'bg-gray-500'} style={{ color: 'white' }}>
        {strategy.toUpperCase()}
      </Badge>
    );
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex flex-col items-center space-y-4">
          <RefreshCw className="h-8 w-8 animate-spin" />
          <p className="text-muted-foreground">Carregando histórico de trades...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center">
            <History className="h-8 w-8 mr-3 text-primary" />
            Histórico de Trades
          </h1>
          <p className="text-muted-foreground">
            Visualize e analise todos os trades executados pelo bot
          </p>
        </div>
        <Button onClick={handleRefresh} disabled={isRefreshing} variant="outline">
          <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
          Atualizar
        </Button>
      </div>

      {/* Statistics Cards */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Total de Trades</p>
                  <p className="text-2xl font-bold">{stats.overall.total_trades}</p>
                </div>
                <BarChart3 className="h-8 w-8 text-blue-500" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Win Rate</p>
                  <p className="text-2xl font-bold">{stats.overall.win_rate.toFixed(1)}%</p>
                </div>
                <Activity className="h-8 w-8 text-green-500" />
              </div>
              <div className="mt-2 text-xs text-muted-foreground">
                {stats.overall.wins}W / {stats.overall.losses}L
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">P&L Total</p>
                  <p className={`text-2xl font-bold ${stats.overall.total_pnl >= 0 ? 'text-success' : 'text-destructive'}`}>
                    {formatPnL(stats.overall.total_pnl)}
                  </p>
                </div>
                {stats.overall.total_pnl >= 0 ? (
                  <TrendingUp className="h-8 w-8 text-success" />
                ) : (
                  <TrendingDown className="h-8 w-8 text-destructive" />
                )}
              </div>
              <div className="mt-2 text-xs text-muted-foreground">
                Média: {formatPnL(stats.overall.avg_pnl)}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Confiança Média</p>
                  <p className="text-2xl font-bold">
                    {stats.overall.avg_confidence ? `${stats.overall.avg_confidence.toFixed(1)}%` : '-'}
                  </p>
                </div>
                <DollarSign className="h-8 w-8 text-purple-500" />
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Filter className="h-5 w-5 mr-2" />
            Filtros
          </CardTitle>
          <CardDescription>Filtre os trades por diversos critérios</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-4">
            <div className="space-y-2">
              <Label>Símbolo</Label>
              <Input
                placeholder="Ex: R_75"
                value={symbolFilter}
                onChange={(e) => {
                  setSymbolFilter(e.target.value);
                  setPage(1);
                }}
              />
            </div>

            <div className="space-y-2">
              <Label>Tipo</Label>
              <Select value={typeFilter} onValueChange={(value) => { setTypeFilter(value); setPage(1); }}>
                <SelectTrigger>
                  <SelectValue placeholder="Todos" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">Todos</SelectItem>
                  <SelectItem value="BUY">Buy</SelectItem>
                  <SelectItem value="SELL">Sell</SelectItem>
                  <SelectItem value="CALL">Call</SelectItem>
                  <SelectItem value="PUT">Put</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Resultado</Label>
              <Select value={resultFilter} onValueChange={(value) => { setResultFilter(value); setPage(1); }}>
                <SelectTrigger>
                  <SelectValue placeholder="Todos" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">Todos</SelectItem>
                  <SelectItem value="win">Win</SelectItem>
                  <SelectItem value="loss">Loss</SelectItem>
                  <SelectItem value="pending">Pending</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Estratégia</Label>
              <Select value={strategyFilter} onValueChange={(value) => { setStrategyFilter(value); setPage(1); }}>
                <SelectTrigger>
                  <SelectValue placeholder="Todas" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">Todas</SelectItem>
                  <SelectItem value="ml">ML</SelectItem>
                  <SelectItem value="technical">Technical</SelectItem>
                  <SelectItem value="hybrid">Hybrid</SelectItem>
                  <SelectItem value="order_flow">Order Flow</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <Label>Data Início</Label>
              <Input
                type="date"
                value={startDate}
                onChange={(e) => { setStartDate(e.target.value); setPage(1); }}
              />
            </div>

            <div className="space-y-2">
              <Label>Data Fim</Label>
              <Input
                type="date"
                value={endDate}
                onChange={(e) => { setEndDate(e.target.value); setPage(1); }}
              />
            </div>

            <div className="space-y-2">
              <Label>Itens por página</Label>
              <Select value={limit.toString()} onValueChange={(value) => { setLimit(parseInt(value)); setPage(1); }}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="10">10</SelectItem>
                  <SelectItem value="25">25</SelectItem>
                  <SelectItem value="50">50</SelectItem>
                  <SelectItem value="100">100</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-end">
              <Button onClick={handleClearFilters} variant="outline" className="w-full">
                Limpar Filtros
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Trades Table */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Trades</CardTitle>
              <CardDescription>
                {total} trades encontrados {symbolFilter || typeFilter || resultFilter ? '(filtrado)' : ''}
              </CardDescription>
            </div>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm">
                  <Download className="h-4 w-4 mr-2" />
                  Exportar
                  <ChevronDown className="h-4 w-4 ml-2" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem onClick={handleExportCSV}>
                  <FileText className="h-4 w-4 mr-2" />
                  Exportar CSV
                </DropdownMenuItem>
                <DropdownMenuItem onClick={handleExportExcel}>
                  <FileSpreadsheet className="h-4 w-4 mr-2" />
                  Exportar Excel
                </DropdownMenuItem>
                <DropdownMenuItem onClick={handleExportPDF}>
                  <FileText className="h-4 w-4 mr-2" />
                  Exportar PDF
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </CardHeader>
        <CardContent>
          <div className="rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>ID</TableHead>
                  <TableHead>Data/Hora</TableHead>
                  <TableHead>Símbolo</TableHead>
                  <TableHead>Tipo</TableHead>
                  <TableHead>Entrada</TableHead>
                  <TableHead>Saída</TableHead>
                  <TableHead>Stake</TableHead>
                  <TableHead>P&L</TableHead>
                  <TableHead>Resultado</TableHead>
                  <TableHead>Confiança</TableHead>
                  <TableHead>Estratégia</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {trades.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={11} className="text-center text-muted-foreground py-8">
                      <Clock className="h-12 w-12 mx-auto mb-2 opacity-50" />
                      <p>Nenhum trade encontrado</p>
                      <p className="text-sm mt-1">Os trades executados aparecerão aqui</p>
                    </TableCell>
                  </TableRow>
                ) : (
                  trades.map((trade) => (
                    <TableRow key={trade.id}>
                      <TableCell className="font-mono text-sm">#{trade.id}</TableCell>
                      <TableCell className="text-sm">{formatDate(trade.timestamp)}</TableCell>
                      <TableCell>
                        <Badge variant="outline">{trade.symbol}</Badge>
                      </TableCell>
                      <TableCell>
                        <Badge variant={trade.trade_type === 'BUY' || trade.trade_type === 'CALL' ? 'default' : 'secondary'}>
                          {trade.trade_type}
                        </Badge>
                      </TableCell>
                      <TableCell className="font-mono text-sm">{formatPrice(trade.entry_price)}</TableCell>
                      <TableCell className="font-mono text-sm">{formatPrice(trade.exit_price)}</TableCell>
                      <TableCell className="font-mono text-sm">{formatPrice(trade.stake)}</TableCell>
                      <TableCell className={`font-mono text-sm font-semibold ${trade.profit_loss && trade.profit_loss >= 0 ? 'text-success' : 'text-destructive'}`}>
                        {formatPnL(trade.profit_loss)}
                      </TableCell>
                      <TableCell>{getResultBadge(trade.result)}</TableCell>
                      <TableCell className="text-sm">
                        {trade.confidence ? `${trade.confidence.toFixed(1)}%` : '-'}
                      </TableCell>
                      <TableCell>{getStrategyBadge(trade.strategy)}</TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between mt-4">
              <div className="text-sm text-muted-foreground">
                Página {page} de {totalPages} ({total} total)
              </div>
              <div className="flex items-center space-x-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage(page - 1)}
                  disabled={page === 1}
                >
                  <ChevronLeft className="h-4 w-4" />
                  Anterior
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage(page + 1)}
                  disabled={page === totalPages}
                >
                  Próxima
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
