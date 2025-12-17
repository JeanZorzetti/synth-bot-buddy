import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  TrendingUp,
  TrendingDown,
  Clock,
  Target,
  XCircle,
  CheckCircle2,
  History,
  Filter,
  BarChart3,
} from 'lucide-react';

interface Trade {
  id: string;
  symbol: string;
  position_type: string;
  entry_price: number;
  exit_price: number;
  size: number;
  entry_time: string;
  exit_time: string;
  profit_loss: number;
  profit_loss_pct: number;
  is_winner: boolean;
  exit_reason: string | null;
  duration_seconds: number;
  duration_minutes: number;
}

interface TradeStatistics {
  total: number;
  winning: number;
  losing: number;
  best_trade: {
    id: string;
    profit_loss: number;
    profit_loss_pct: number;
  } | null;
  worst_trade: {
    id: string;
    profit_loss: number;
    profit_loss_pct: number;
  } | null;
  avg_profit: number;
  avg_loss: number;
  avg_profit_pct: number;
  avg_loss_pct: number;
}

interface TradeHistoryTableProps {
  apiBaseUrl: string;
  isRunning: boolean;
  pollInterval?: number; // ms
}

export function TradeHistoryTable({
  apiBaseUrl,
  isRunning,
  pollInterval = 30000 // 30 segundos
}: TradeHistoryTableProps) {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [statistics, setStatistics] = useState<TradeStatistics | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [resultFilter, setResultFilter] = useState<string>('all'); // 'all', 'win', 'loss'
  const [limit, setLimit] = useState<number>(50);

  // Buscar trades
  const loadTrades = async () => {
    try {
      setIsLoading(true);

      const params = new URLSearchParams({
        limit: limit.toString(),
      });

      if (resultFilter !== 'all') {
        params.append('result_filter', resultFilter);
      }

      const response = await fetch(`${apiBaseUrl}/api/forward-testing/trades?${params}`);
      const data = await response.json();

      if (data.status === 'success') {
        setTrades(data.data.reverse()); // Mais recentes primeiro
        setStatistics(data.statistics);
      }
    } catch (error) {
      console.error('Erro ao carregar trades:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Formatar duração
  const formatDuration = (minutes: number): string => {
    if (minutes < 1) return '<1 min';
    if (minutes < 60) return `${Math.round(minutes)} min`;

    const hours = Math.floor(minutes / 60);
    const mins = Math.round(minutes % 60);
    return `${hours}h ${mins}m`;
  };

  // Ícone do exit reason
  const getExitReasonBadge = (reason: string | null) => {
    switch (reason) {
      case 'take_profit':
        return (
          <Badge variant="default" className="bg-green-500">
            <Target className="h-3 w-3 mr-1" />
            TP
          </Badge>
        );
      case 'stop_loss':
        return (
          <Badge variant="destructive">
            <XCircle className="h-3 w-3 mr-1" />
            SL
          </Badge>
        );
      case 'timeout':
        return (
          <Badge variant="secondary">
            <Clock className="h-3 w-3 mr-1" />
            Timeout
          </Badge>
        );
      case 'manual':
        return (
          <Badge variant="outline">
            Manual
          </Badge>
        );
      default:
        return (
          <Badge variant="outline">
            {reason || 'N/A'}
          </Badge>
        );
    }
  };

  // Polling quando sistema está rodando
  useEffect(() => {
    if (!isRunning) {
      return;
    }

    loadTrades(); // Carregar imediatamente
    const interval = setInterval(loadTrades, pollInterval);
    return () => clearInterval(interval);
  }, [isRunning, resultFilter, limit, pollInterval]);

  // Recarregar ao mudar filtros
  useEffect(() => {
    if (isRunning) {
      loadTrades();
    }
  }, [resultFilter, limit]);

  // Não exibir se não estiver rodando e não houver trades
  if (!isRunning && trades.length === 0) {
    return null;
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <History className="h-5 w-5" />
              Histórico de Trades
              {statistics && (
                <Badge variant="secondary">
                  {statistics.total} total
                </Badge>
              )}
            </CardTitle>
            <CardDescription>
              Todos os trades executados com detalhes completos
            </CardDescription>
          </div>

          <div className="flex gap-2">
            {/* Filtro de Resultado */}
            <Select value={resultFilter} onValueChange={setResultFilter}>
              <SelectTrigger className="w-[140px]">
                <Filter className="h-4 w-4 mr-2" />
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">Todos ({statistics?.total || 0})</SelectItem>
                <SelectItem value="win">
                  <span className="flex items-center">
                    <CheckCircle2 className="h-3 w-3 mr-1 text-green-600" />
                    Ganhos ({statistics?.winning || 0})
                  </span>
                </SelectItem>
                <SelectItem value="loss">
                  <span className="flex items-center">
                    <XCircle className="h-3 w-3 mr-1 text-red-600" />
                    Perdas ({statistics?.losing || 0})
                  </span>
                </SelectItem>
              </SelectContent>
            </Select>

            {/* Limite */}
            <Select value={limit.toString()} onValueChange={(val) => setLimit(parseInt(val))}>
              <SelectTrigger className="w-[100px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="20">20</SelectItem>
                <SelectItem value="50">50</SelectItem>
                <SelectItem value="100">100</SelectItem>
                <SelectItem value="200">200</SelectItem>
              </SelectContent>
            </Select>

            <Button
              variant="outline"
              size="sm"
              onClick={loadTrades}
              disabled={isLoading}
            >
              <BarChart3 className={`h-4 w-4 mr-1 ${isLoading ? 'animate-spin' : ''}`} />
              Atualizar
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent>
        {/* Estatísticas Agregadas */}
        {statistics && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6 p-4 bg-gray-50 rounded-lg">
            <div className="text-center">
              <p className="text-sm text-muted-foreground mb-1">Melhor Trade</p>
              {statistics.best_trade ? (
                <>
                  <p className="text-lg font-bold text-green-600">
                    +${statistics.best_trade.profit_loss.toFixed(2)}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    ({statistics.best_trade.profit_loss_pct.toFixed(2)}%)
                  </p>
                </>
              ) : (
                <p className="text-sm text-muted-foreground">-</p>
              )}
            </div>

            <div className="text-center">
              <p className="text-sm text-muted-foreground mb-1">Pior Trade</p>
              {statistics.worst_trade ? (
                <>
                  <p className="text-lg font-bold text-red-600">
                    ${statistics.worst_trade.profit_loss.toFixed(2)}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    ({statistics.worst_trade.profit_loss_pct.toFixed(2)}%)
                  </p>
                </>
              ) : (
                <p className="text-sm text-muted-foreground">-</p>
              )}
            </div>

            <div className="text-center">
              <p className="text-sm text-muted-foreground mb-1">Lucro Médio</p>
              <p className="text-lg font-bold text-green-600">
                +${statistics.avg_profit.toFixed(2)}
              </p>
              <p className="text-xs text-muted-foreground">
                ({statistics.avg_profit_pct.toFixed(2)}%)
              </p>
            </div>

            <div className="text-center">
              <p className="text-sm text-muted-foreground mb-1">Perda Média</p>
              <p className="text-lg font-bold text-red-600">
                ${statistics.avg_loss.toFixed(2)}
              </p>
              <p className="text-xs text-muted-foreground">
                ({statistics.avg_loss_pct.toFixed(2)}%)
              </p>
            </div>
          </div>
        )}

        {/* Tabela de Trades */}
        {trades.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <History className="h-12 w-12 mx-auto mb-3 opacity-30" />
            <p className="text-sm">Nenhum trade executado ainda</p>
            <p className="text-xs">Os trades aparecerão aqui quando o sistema estiver rodando</p>
          </div>
        ) : (
          <div className="border rounded-lg overflow-hidden">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[100px]">ID</TableHead>
                  <TableHead>Tipo</TableHead>
                  <TableHead>Entry → Exit</TableHead>
                  <TableHead>P&L</TableHead>
                  <TableHead>Duração</TableHead>
                  <TableHead>Exit Reason</TableHead>
                  <TableHead>Timestamp</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {trades.map((trade) => (
                  <TableRow
                    key={trade.id}
                    className={trade.is_winner ? 'bg-green-50' : 'bg-red-50'}
                  >
                    <TableCell className="font-mono text-xs">
                      {trade.id.slice(-8)}
                    </TableCell>

                    <TableCell>
                      <Badge
                        variant={trade.position_type === 'LONG' ? 'default' : 'destructive'}
                      >
                        {trade.position_type === 'LONG' ? (
                          <TrendingUp className="h-3 w-3 mr-1" />
                        ) : (
                          <TrendingDown className="h-3 w-3 mr-1" />
                        )}
                        {trade.position_type}
                      </Badge>
                    </TableCell>

                    <TableCell className="font-mono text-sm">
                      ${trade.entry_price.toFixed(2)} → ${trade.exit_price.toFixed(2)}
                    </TableCell>

                    <TableCell>
                      <div className={`font-bold ${trade.profit_loss >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {trade.profit_loss >= 0 ? '+' : ''}${trade.profit_loss.toFixed(2)}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        ({trade.profit_loss_pct >= 0 ? '+' : ''}{trade.profit_loss_pct.toFixed(2)}%)
                      </div>
                    </TableCell>

                    <TableCell className="text-sm">
                      <div className="flex items-center gap-1">
                        <Clock className="h-3 w-3 text-muted-foreground" />
                        {formatDuration(trade.duration_minutes)}
                      </div>
                    </TableCell>

                    <TableCell>
                      {getExitReasonBadge(trade.exit_reason)}
                    </TableCell>

                    <TableCell className="text-xs text-muted-foreground">
                      {new Date(trade.exit_time).toLocaleString('pt-BR', {
                        day: '2-digit',
                        month: '2-digit',
                        hour: '2-digit',
                        minute: '2-digit'
                      })}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
