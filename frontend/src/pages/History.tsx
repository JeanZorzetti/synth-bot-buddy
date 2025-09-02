import { useState } from 'react';
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
  Calendar
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface Trade {
  id: string;
  asset: string;
  timestamp: string;
  type: 'CALL' | 'PUT';
  stake: number;
  entryPrice: number;
  exitPrice: number;
  pnl: number;
  duration: string;
  outcome: 'win' | 'loss';
}

export default function History() {
  const [filterAsset, setFilterAsset] = useState<string>('all');
  const [filterOutcome, setFilterOutcome] = useState<string>('all');
  const [searchDate, setSearchDate] = useState('');

  // Mock trade data
  const [trades] = useState<Trade[]>([
    {
      id: '1',
      asset: 'Volatility 75 Index',
      timestamp: '2024-01-31 14:35:22',
      type: 'CALL',
      stake: 10.00,
      entryPrice: 1234.56,
      exitPrice: 1245.78,
      pnl: 8.50,
      duration: '5 min',
      outcome: 'win'
    },
    {
      id: '2', 
      asset: 'Volatility 100 Index',
      timestamp: '2024-01-31 14:28:15',
      type: 'PUT',
      stake: 10.00,
      entryPrice: 2345.67,
      exitPrice: 2320.45,
      pnl: 7.80,
      duration: '3 min',
      outcome: 'win'
    },
    {
      id: '3',
      asset: 'Jump 25 Index',
      timestamp: '2024-01-31 14:22:08',
      type: 'CALL',
      stake: 10.00,
      entryPrice: 987.65,
      exitPrice: 982.34,
      pnl: -10.00,
      duration: '2 min',
      outcome: 'loss'
    },
    {
      id: '4',
      asset: 'Volatility 75 Index',
      timestamp: '2024-01-31 14:15:33',
      type: 'PUT',
      stake: 10.00,
      entryPrice: 1245.78,
      exitPrice: 1256.89,
      pnl: 9.20,
      duration: '4 min',
      outcome: 'win'
    },
    {
      id: '5',
      asset: 'Volatility 100 Index',
      timestamp: '2024-01-31 14:08:45',
      type: 'CALL',
      stake: 10.00,
      entryPrice: 2320.45,
      exitPrice: 2315.67,
      pnl: -10.00,
      duration: '5 min',
      outcome: 'loss'
    }
  ]);

  // Filter trades
  const filteredTrades = trades.filter(trade => {
    const matchesAsset = filterAsset === 'all' || trade.asset === filterAsset;
    const matchesOutcome = filterOutcome === 'all' || trade.outcome === filterOutcome;
    const matchesDate = !searchDate || trade.timestamp.includes(searchDate);
    
    return matchesAsset && matchesOutcome && matchesDate;
  });

  // Calculate statistics
  const totalTrades = filteredTrades.length;
  const winningTrades = filteredTrades.filter(t => t.outcome === 'win').length;
  const totalPnL = filteredTrades.reduce((sum, t) => sum + t.pnl, 0);
  const winRate = totalTrades > 0 ? (winningTrades / totalTrades) * 100 : 0;
  const largestWin = Math.max(...filteredTrades.filter(t => t.pnl > 0).map(t => t.pnl), 0);
  const largestLoss = Math.min(...filteredTrades.filter(t => t.pnl < 0).map(t => t.pnl), 0);

  const uniqueAssets = [...new Set(trades.map(t => t.asset))];

  return (
    <Layout>
      <div className="space-y-6">
        {/* Page Header */}
        <div>
          <h1 className="text-3xl font-bold">Histórico de Operações</h1>
          <p className="text-muted-foreground">
            Analise todas as operações realizadas pelo bot
          </p>
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
                    <SelectItem value="win">Apenas Vitórias</SelectItem>
                    <SelectItem value="loss">Apenas Perdas</SelectItem>
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
                    <tr key={trade.id} className="border-b border-border/50 hover:bg-muted/30">
                      <td className="p-4">
                        <div className="font-medium">{trade.asset}</div>
                      </td>
                      <td className="p-4">
                        <div className="text-sm">
                          {new Date(trade.timestamp).toLocaleString('pt-BR')}
                        </div>
                      </td>
                      <td className="p-4">
                        <Badge 
                          variant={trade.type === 'CALL' ? 'default' : 'secondary'}
                          className={cn(
                            trade.type === 'CALL' 
                              ? 'bg-success text-success-foreground' 
                              : 'bg-danger text-danger-foreground'
                          )}
                        >
                          {trade.type}
                        </Badge>
                      </td>
                      <td className="p-4">
                        <div className="font-medium">${trade.stake.toFixed(2)}</div>
                      </td>
                      <td className="p-4">
                        <div className="font-mono text-sm">{trade.entryPrice.toFixed(2)}</div>
                      </td>
                      <td className="p-4">
                        <div className="font-mono text-sm">{trade.exitPrice.toFixed(2)}</div>
                      </td>
                      <td className="p-4">
                        <div className="text-sm">{trade.duration}</div>
                      </td>
                      <td className="p-4">
                        <div className={cn(
                          "font-bold",
                          trade.pnl >= 0 ? "text-success" : "text-danger"
                        )}>
                          {trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {filteredTrades.length === 0 && (
                <div className="text-center py-8 text-muted-foreground">
                  <HistoryIcon className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>Nenhuma operação encontrada com os filtros selecionados.</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </Layout>
  );
}