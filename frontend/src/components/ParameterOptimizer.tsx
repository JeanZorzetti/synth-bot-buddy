import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Settings, TrendingUp, AlertCircle, Loader2, Award } from 'lucide-react';
import { useToast } from './ui/use-toast';

interface OptimizationResult {
  stop_loss_pct: number;
  take_profit_pct: number;
  timeout_minutes: number;
  total_trades: number;
  win_rate: number;
  total_profit_loss: number;
  profit_loss_pct: number;
  sharpe_ratio: number;
  max_drawdown_pct: number;
  avg_trade_duration_minutes: number;
  timeout_rate: number;
  score: number;
}

interface OptimizationData {
  results: OptimizationResult[];
  total_combinations_tested: number;
  historical_trades_used: number;
  symbol: string;
  best_params: OptimizationResult | null;
}

interface ParameterOptimizerProps {
  apiBaseUrl: string;
  isRunning: boolean;
}

export function ParameterOptimizer({ apiBaseUrl, isRunning }: ParameterOptimizerProps) {
  const [data, setData] = useState<OptimizationData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState<string>('ALL');
  const { toast } = useToast();

  const runOptimization = async () => {
    setIsLoading(true);
    try {
      const symbolParam = selectedSymbol === 'ALL' ? '' : `?symbol=${selectedSymbol}`;
      const response = await fetch(`${apiBaseUrl}/api/forward-testing/optimize-parameters${symbolParam}&top_n=10`);
      const result = await response.json();

      if (result.status === 'error') {
        toast({
          title: 'Erro',
          description: result.message,
          variant: 'destructive'
        });
        return;
      }

      setData(result.data);
      toast({
        title: 'Otimização Concluída! 🎯',
        description: `${result.data.total_combinations_tested} combinações testadas`
      });
    } catch (error) {
      console.error('Error running optimization:', error);
      toast({
        title: 'Erro ao otimizar',
        description: 'Falha ao executar otimização de parâmetros',
        variant: 'destructive'
      });
    } finally {
      setIsLoading(false);
    }
  };

  const getSharpeColor = (sharpe: number) => {
    if (sharpe >= 2.0) return 'text-green-600 dark:text-green-400';
    if (sharpe >= 1.0) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  const getWinRateColor = (winRate: number) => {
    if (winRate >= 55) return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200';
    if (winRate >= 45) return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200';
    return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200';
  };

  return (
    <Card className="mt-4">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            <CardTitle>Otimizador de Parâmetros (Grid Search)</CardTitle>
          </div>
          <Button
            onClick={runOptimization}
            disabled={isLoading || !isRunning}
            variant="default"
          >
            {isLoading ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Otimizando...
              </>
            ) : (
              <>
                <TrendingUp className="h-4 w-4 mr-2" />
                Otimizar Parâmetros
              </>
            )}
          </Button>
        </div>

        {!isRunning && (
          <div className="flex items-center gap-2 mt-2 text-sm text-yellow-600 dark:text-yellow-400">
            <AlertCircle className="h-4 w-4" />
            Inicie o Forward Testing para usar o otimizador
          </div>
        )}
      </CardHeader>

      <CardContent>
        {data && (
          <div className="space-y-4">
            {/* Informações da Otimização */}
            <div className="grid grid-cols-3 gap-4">
              <Card className="p-4">
                <div className="text-sm text-muted-foreground">Combinações Testadas</div>
                <div className="text-2xl font-bold">{data.total_combinations_tested}</div>
              </Card>
              <Card className="p-4">
                <div className="text-sm text-muted-foreground">Trades Analisados</div>
                <div className="text-2xl font-bold">{data.historical_trades_used}</div>
              </Card>
              <Card className="p-4">
                <div className="text-sm text-muted-foreground">Ativo</div>
                <div className="text-2xl font-bold">{data.symbol}</div>
              </Card>
            </div>

            {/* Melhor Resultado em Destaque */}
            {data.best_params && (
              <Card className="p-4 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-950 dark:to-emerald-950 border-green-300 dark:border-green-700">
                <div className="flex items-center gap-2 mb-3">
                  <Award className="h-5 w-5 text-green-600 dark:text-green-400" />
                  <h3 className="font-bold text-lg">🏆 Melhores Parâmetros Encontrados</h3>
                </div>
                <div className="grid grid-cols-5 gap-4">
                  <div>
                    <div className="text-sm text-muted-foreground">Stop Loss</div>
                    <div className="text-xl font-bold text-green-600 dark:text-green-400">
                      {data.best_params.stop_loss_pct}%
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground">Take Profit</div>
                    <div className="text-xl font-bold text-green-600 dark:text-green-400">
                      {data.best_params.take_profit_pct}%
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground">Timeout</div>
                    <div className="text-xl font-bold text-green-600 dark:text-green-400">
                      {data.best_params.timeout_minutes}min
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground">Sharpe Ratio</div>
                    <div className="text-xl font-bold text-green-600 dark:text-green-400">
                      {data.best_params.sharpe_ratio.toFixed(3)}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground">Win Rate</div>
                    <div className="text-xl font-bold text-green-600 dark:text-green-400">
                      {data.best_params.win_rate.toFixed(1)}%
                    </div>
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-4 mt-3">
                  <div>
                    <div className="text-sm text-muted-foreground">P&L Total</div>
                    <div className={`text-lg font-bold ${data.best_params.total_profit_loss >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                      ${data.best_params.total_profit_loss.toFixed(2)}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground">Max Drawdown</div>
                    <div className="text-lg font-bold text-red-600 dark:text-red-400">
                      {data.best_params.max_drawdown_pct.toFixed(2)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-muted-foreground">Duração Média</div>
                    <div className="text-lg font-bold">
                      {data.best_params.avg_trade_duration_minutes.toFixed(1)}min
                    </div>
                  </div>
                </div>
              </Card>
            )}

            {/* Tabela de Resultados */}
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-muted">
                  <tr>
                    <th className="p-2 text-left font-semibold">Rank</th>
                    <th className="p-2 text-center font-semibold">SL %</th>
                    <th className="p-2 text-center font-semibold">TP %</th>
                    <th className="p-2 text-center font-semibold">Timeout (min)</th>
                    <th className="p-2 text-center font-semibold">Trades</th>
                    <th className="p-2 text-center font-semibold">Win Rate</th>
                    <th className="p-2 text-center font-semibold">P&L %</th>
                    <th className="p-2 text-center font-semibold">Sharpe</th>
                    <th className="p-2 text-center font-semibold">Max DD %</th>
                    <th className="p-2 text-center font-semibold">Score</th>
                  </tr>
                </thead>
                <tbody>
                  {data.results.map((result, idx) => (
                    <tr
                      key={idx}
                      className={`border-b ${idx === 0 ? 'bg-green-50 dark:bg-green-950/30' : 'hover:bg-muted/50'}`}
                    >
                      <td className="p-2">
                        {idx === 0 ? (
                          <Badge className="bg-green-600 text-white">🏆 #{idx + 1}</Badge>
                        ) : idx === 1 ? (
                          <Badge className="bg-gray-400 text-white">🥈 #{idx + 1}</Badge>
                        ) : idx === 2 ? (
                          <Badge className="bg-orange-600 text-white">🥉 #{idx + 1}</Badge>
                        ) : (
                          <Badge variant="outline">#{idx + 1}</Badge>
                        )}
                      </td>
                      <td className="p-2 text-center font-mono">{result.stop_loss_pct}</td>
                      <td className="p-2 text-center font-mono">{result.take_profit_pct}</td>
                      <td className="p-2 text-center font-mono">{result.timeout_minutes}</td>
                      <td className="p-2 text-center">{result.total_trades}</td>
                      <td className="p-2 text-center">
                        <Badge className={getWinRateColor(result.win_rate)}>
                          {result.win_rate.toFixed(1)}%
                        </Badge>
                      </td>
                      <td className={`p-2 text-center font-bold ${result.profit_loss_pct >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`}>
                        {result.profit_loss_pct >= 0 ? '+' : ''}{result.profit_loss_pct.toFixed(2)}%
                      </td>
                      <td className={`p-2 text-center font-bold ${getSharpeColor(result.sharpe_ratio)}`}>
                        {result.sharpe_ratio.toFixed(3)}
                      </td>
                      <td className="p-2 text-center text-red-600 dark:text-red-400">
                        {result.max_drawdown_pct.toFixed(2)}%
                      </td>
                      <td className="p-2 text-center font-bold">
                        {result.score.toFixed(3)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Legenda */}
            <div className="text-xs text-muted-foreground space-y-1">
              <div className="font-semibold">📊 Métricas:</div>
              <div><strong>Score:</strong> Métrica combinada (50% Sharpe + 20% Win Rate + 20% P&L% - 10% Max DD)</div>
              <div><strong>Sharpe Ratio:</strong> Retorno ajustado ao risco (≥2.0 excelente, ≥1.0 bom)</div>
              <div><strong>Max DD:</strong> Maior drawdown histórico em %</div>
              <div><strong>Win Rate:</strong> Percentual de trades vencedores (≥55% excelente)</div>
            </div>
          </div>
        )}

        {!data && !isLoading && (
          <div className="text-center py-8 text-muted-foreground">
            <Settings className="h-12 w-12 mx-auto mb-3 opacity-50" />
            <p>Clique em "Otimizar Parâmetros" para encontrar os melhores valores de SL/TP/Timeout</p>
            <p className="text-sm mt-2">O Grid Search testará 150 combinações diferentes</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
