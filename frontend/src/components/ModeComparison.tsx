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
  TrendingUp,
  Trophy,
  Target,
  Clock,
  BarChart3,
  Lightbulb,
  RefreshCw,
} from 'lucide-react';

interface SymbolStats {
  total_trades: number;
  win_rate_pct: number;
  total_pnl: number;
  total_pnl_pct: number;
  sharpe_ratio: number;
  avg_duration_minutes: number;
  timeout_rate_pct: number;
  winning_trades: number;
  losing_trades: number;
}

interface Recommendation {
  type: string;
  symbol: string;
  value: number;
  message: string;
}

interface ModeComparisonProps {
  apiBaseUrl: string;
  isRunning: boolean;
}

export function ModeComparison({ apiBaseUrl, isRunning }: ModeComparisonProps) {
  const [comparisonData, setComparisonData] = useState<Record<string, SymbolStats>>({});
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [currentSymbol, setCurrentSymbol] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);

  // Buscar dados de comparação
  const loadComparison = async () => {
    try {
      setIsLoading(true);

      const response = await fetch(`${apiBaseUrl}/api/forward-testing/mode-comparison`);
      const data = await response.json();

      if (data.status === 'success') {
        setComparisonData(data.data.by_symbol || {});
        setRecommendations(data.data.recommendations || []);
        setCurrentSymbol(data.current_symbol || '');
      }
    } catch (error) {
      console.error('Erro ao carregar comparação de modos:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Carregar ao montar e quando sistema parar (para ver resultados finais)
  useEffect(() => {
    loadComparison();
  }, []);

  // Ícone baseado no tipo de recomendação
  const getRecommendationIcon = (type: string) => {
    switch (type) {
      case 'best_win_rate':
        return <Trophy className="h-4 w-4 text-yellow-600" />;
      case 'best_pnl':
        return <TrendingUp className="h-4 w-4 text-green-600" />;
      case 'best_sharpe':
        return <Target className="h-4 w-4 text-blue-600" />;
      case 'fastest':
        return <Clock className="h-4 w-4 text-purple-600" />;
      default:
        return <Lightbulb className="h-4 w-4" />;
    }
  };

  // Cor do badge de win rate
  const getWinRateColor = (winRate: number): string => {
    if (winRate >= 60) return 'text-green-600';
    if (winRate >= 50) return 'text-yellow-600';
    return 'text-red-600';
  };

  // Cor do badge de P&L
  const getPnlColor = (pnl: number): string => {
    if (pnl > 0) return 'text-green-600';
    if (pnl < 0) return 'text-red-600';
    return 'text-gray-600';
  };

  const symbols = Object.keys(comparisonData);

  if (symbols.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Comparador de Performance por Ativo
          </CardTitle>
          <CardDescription>
            Execute trades em diferentes ativos para comparar a performance
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-muted-foreground">
            <BarChart3 className="h-12 w-12 mx-auto mb-3 opacity-30" />
            <p className="text-sm">Aguardando trades para gerar comparação</p>
            <p className="text-xs mt-2">
              Troque de ativo no seletor e execute mais trades para ver comparações
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Comparador de Performance por Ativo
              <Badge variant="secondary">{symbols.length} ativos</Badge>
            </CardTitle>
            <CardDescription>
              Compare a performance de diferentes ativos testados
            </CardDescription>
          </div>

          <Button
            variant="outline"
            size="sm"
            onClick={loadComparison}
            disabled={isLoading}
          >
            <RefreshCw className={`h-4 w-4 mr-1 ${isLoading ? 'animate-spin' : ''}`} />
            Atualizar
          </Button>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Recomendações */}
        {recommendations.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 p-4 bg-blue-50 rounded-lg border-2 border-blue-200">
            <div className="md:col-span-2 flex items-center gap-2 mb-2">
              <Lightbulb className="h-5 w-5 text-blue-600" />
              <h3 className="font-semibold text-blue-900">Recomendações Baseadas em Dados</h3>
            </div>

            {recommendations.map((rec, idx) => (
              <div key={idx} className="flex items-start gap-2 bg-white p-3 rounded border border-blue-200">
                {getRecommendationIcon(rec.type)}
                <div className="flex-1">
                  <p className="text-sm font-medium text-gray-900">{rec.message}</p>
                  {rec.symbol === currentSymbol && (
                    <Badge variant="default" className="mt-1 text-xs">
                      Ativo Atual
                    </Badge>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Tabela Comparativa */}
        <div className="border rounded-lg overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Ativo</TableHead>
                <TableHead className="text-center">Trades</TableHead>
                <TableHead className="text-center">Win Rate</TableHead>
                <TableHead className="text-center">P&L Total</TableHead>
                <TableHead className="text-center">Sharpe Ratio</TableHead>
                <TableHead className="text-center">Duração Média</TableHead>
                <TableHead className="text-center">Timeout Rate</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {symbols
                .sort((a, b) => comparisonData[b].total_pnl_pct - comparisonData[a].total_pnl_pct) // Ordenar por P&L
                .map((symbol) => {
                  const stats = comparisonData[symbol];
                  const isCurrent = symbol === currentSymbol;

                  return (
                    <TableRow
                      key={symbol}
                      className={isCurrent ? 'bg-blue-50 border-2 border-blue-300' : ''}
                    >
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <span className="font-semibold">{symbol}</span>
                          {isCurrent && (
                            <Badge variant="default" className="text-xs">
                              Atual
                            </Badge>
                          )}
                        </div>
                      </TableCell>

                      <TableCell className="text-center">
                        <div className="text-sm font-medium">{stats.total_trades}</div>
                        <div className="text-xs text-muted-foreground">
                          {stats.winning_trades}W / {stats.losing_trades}L
                        </div>
                      </TableCell>

                      <TableCell className="text-center">
                        <div className={`text-sm font-bold ${getWinRateColor(stats.win_rate_pct)}`}>
                          {stats.win_rate_pct.toFixed(1)}%
                        </div>
                        {stats.win_rate_pct >= 55 && (
                          <Trophy className="h-3 w-3 text-yellow-600 mx-auto mt-1" />
                        )}
                      </TableCell>

                      <TableCell className="text-center">
                        <div className={`text-sm font-bold ${getPnlColor(stats.total_pnl_pct)}`}>
                          {stats.total_pnl_pct >= 0 ? '+' : ''}{stats.total_pnl_pct.toFixed(2)}%
                        </div>
                        <div className="text-xs text-muted-foreground">
                          ${stats.total_pnl.toFixed(2)}
                        </div>
                      </TableCell>

                      <TableCell className="text-center">
                        <div className={`text-sm font-medium ${
                          stats.sharpe_ratio >= 1.5 ? 'text-green-600' :
                          stats.sharpe_ratio >= 1.0 ? 'text-yellow-600' :
                          'text-red-600'
                        }`}>
                          {stats.sharpe_ratio.toFixed(2)}
                        </div>
                        {stats.sharpe_ratio >= 1.5 && (
                          <Target className="h-3 w-3 text-blue-600 mx-auto mt-1" />
                        )}
                      </TableCell>

                      <TableCell className="text-center">
                        <div className="text-sm">
                          {stats.avg_duration_minutes < 1
                            ? '<1 min'
                            : stats.avg_duration_minutes < 60
                            ? `${Math.round(stats.avg_duration_minutes)} min`
                            : `${(stats.avg_duration_minutes / 60).toFixed(1)}h`}
                        </div>
                      </TableCell>

                      <TableCell className="text-center">
                        <div className={`text-sm ${
                          stats.timeout_rate_pct > 30 ? 'text-red-600' :
                          stats.timeout_rate_pct > 20 ? 'text-yellow-600' :
                          'text-green-600'
                        }`}>
                          {stats.timeout_rate_pct.toFixed(1)}%
                        </div>
                      </TableCell>
                    </TableRow>
                  );
                })}
            </TableBody>
          </Table>
        </div>

        {/* Legenda */}
        <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
          <div className="flex items-center gap-1">
            <Trophy className="h-3 w-3 text-yellow-600" />
            Win Rate ≥ 55%
          </div>
          <div className="flex items-center gap-1">
            <Target className="h-3 w-3 text-blue-600" />
            Sharpe ≥ 1.5
          </div>
          <div className="flex items-center gap-1">
            <div className="w-3 h-3 bg-blue-100 border-2 border-blue-300 rounded" />
            Ativo Atual
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
