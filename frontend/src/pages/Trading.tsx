import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Play,
  Square,
  Pause,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Brain,
  Shield,
  Activity,
  DollarSign,
  Clock,
  Target
} from 'lucide-react';

interface AIStatus {
  status: 'idle' | 'analyzing' | 'executing' | 'paused' | 'error';
  confidence: number;
  last_signal: {
    action: 'buy' | 'sell' | 'hold';
    strength: number;
    timestamp: string;
    reasoning: string;
  };
  active_positions: number;
  total_trades_today: number;
}

interface TradingDecision {
  id: string;
  timestamp: string;
  action: 'buy' | 'sell' | 'close';
  confidence: number;
  reasoning: string;
  symbol: string;
  amount: number;
  status: 'pending' | 'executed' | 'failed';
}

interface RiskMetrics {
  position_size_pct: number;
  max_drawdown: number;
  var_95: number;
  sharpe_ratio: number;
  risk_score: number;
}

const Trading: React.FC = () => {
  const [aiStatus, setAiStatus] = useState<AIStatus>({
    status: 'idle',
    confidence: 0,
    last_signal: {
      action: 'hold',
      strength: 0,
      timestamp: new Date().toISOString(),
      reasoning: 'Aguardando sinais do mercado'
    },
    active_positions: 0,
    total_trades_today: 0
  });

  const [isAutonomousMode, setIsAutonomousMode] = useState(false);
  const [emergencyStop, setEmergencyStop] = useState(false);
  const [recentDecisions, setRecentDecisions] = useState<TradingDecision[]>([]);
  const [riskMetrics, setRiskMetrics] = useState<RiskMetrics>({
    position_size_pct: 2.5,
    max_drawdown: 5.2,
    var_95: 8.1,
    sharpe_ratio: 1.47,
    risk_score: 0.32
  });

  // Simular dados em tempo real
  useEffect(() => {
    const interval = setInterval(() => {
      if (isAutonomousMode && !emergencyStop) {
        // Simular mudanças no status da IA
        const statuses: AIStatus['status'][] = ['analyzing', 'executing', 'analyzing'];
        const randomStatus = statuses[Math.floor(Math.random() * statuses.length)];

        setAiStatus(prev => ({
          ...prev,
          status: randomStatus,
          confidence: Math.random() * 100,
          last_signal: {
            ...prev.last_signal,
            action: Math.random() > 0.6 ? (Math.random() > 0.5 ? 'buy' : 'sell') : 'hold',
            strength: Math.random(),
            timestamp: new Date().toISOString()
          }
        }));

        // Simular novas decisões
        if (Math.random() > 0.8) {
          const newDecision: TradingDecision = {
            id: Math.random().toString(36).substr(2, 9),
            timestamp: new Date().toISOString(),
            action: Math.random() > 0.5 ? 'buy' : 'sell',
            confidence: Math.random() * 100,
            reasoning: 'Padrão de momentum detectado com alta confiança',
            symbol: 'R_100',
            amount: Math.random() * 100 + 10,
            status: 'executed'
          };

          setRecentDecisions(prev => [newDecision, ...prev.slice(0, 9)]);
        }
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [isAutonomousMode, emergencyStop]);

  const getStatusColor = (status: AIStatus['status']) => {
    switch (status) {
      case 'idle': return 'text-gray-500';
      case 'analyzing': return 'text-blue-500';
      case 'executing': return 'text-green-500';
      case 'paused': return 'text-yellow-500';
      case 'error': return 'text-red-500';
      default: return 'text-gray-500';
    }
  };

  const getStatusIcon = (status: AIStatus['status']) => {
    switch (status) {
      case 'idle': return <Clock className="h-4 w-4" />;
      case 'analyzing': return <Brain className="h-4 w-4" />;
      case 'executing': return <Activity className="h-4 w-4" />;
      case 'paused': return <Pause className="h-4 w-4" />;
      case 'error': return <AlertTriangle className="h-4 w-4" />;
      default: return <Clock className="h-4 w-4" />;
    }
  };

  const handleStartAutonomous = () => {
    setIsAutonomousMode(true);
    setEmergencyStop(false);
    setAiStatus(prev => ({ ...prev, status: 'analyzing' }));
  };

  const handleStopAutonomous = () => {
    setIsAutonomousMode(false);
    setAiStatus(prev => ({ ...prev, status: 'idle' }));
  };

  const handleEmergencyStop = () => {
    setEmergencyStop(true);
    setIsAutonomousMode(false);
    setAiStatus(prev => ({ ...prev, status: 'paused' }));
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">🚀 Trading Autônomo</h1>
          <p className="text-muted-foreground">IA operando de forma independente</p>
        </div>

        {/* Controles Principais */}
        <div className="flex items-center space-x-4">
          {!isAutonomousMode ? (
            <Button onClick={handleStartAutonomous} className="bg-green-600 hover:bg-green-700">
              <Play className="h-4 w-4 mr-2" />
              Iniciar IA
            </Button>
          ) : (
            <Button onClick={handleStopAutonomous} variant="outline">
              <Square className="h-4 w-4 mr-2" />
              Parar IA
            </Button>
          )}

          <Button
            onClick={handleEmergencyStop}
            variant="destructive"
            disabled={!isAutonomousMode}
          >
            <AlertTriangle className="h-4 w-4 mr-2" />
            EMERGÊNCIA
          </Button>
        </div>
      </div>

      {emergencyStop && (
        <Alert className="border-red-500 bg-red-50">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription className="font-medium">
            PARADA DE EMERGÊNCIA ATIVADA - Todas as operações foram interrompidas
          </AlertDescription>
        </Alert>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Status da IA */}
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Brain className="h-5 w-5" />
                <span>Status da IA</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className={`flex items-center space-x-2 ${getStatusColor(aiStatus.status)}`}>
                    {getStatusIcon(aiStatus.status)}
                    <span className="font-medium capitalize">
                      {aiStatus.status === 'idle' ? 'Inativo' :
                       aiStatus.status === 'analyzing' ? 'Analisando' :
                       aiStatus.status === 'executing' ? 'Executando' :
                       aiStatus.status === 'paused' ? 'Pausado' : 'Erro'}
                    </span>
                  </div>
                  <Badge variant={isAutonomousMode ? "default" : "secondary"}>
                    {isAutonomousMode ? 'AUTÔNOMO' : 'MANUAL'}
                  </Badge>
                </div>

                <div className="text-right">
                  <div className="text-sm text-muted-foreground">Confiança</div>
                  <div className="text-2xl font-bold">{aiStatus.confidence.toFixed(1)}%</div>
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Nível de Confiança</span>
                  <span>{aiStatus.confidence.toFixed(1)}%</span>
                </div>
                <Progress value={aiStatus.confidence} className="h-2" />
              </div>

              <Separator />

              <div className="grid grid-cols-2 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold">{aiStatus.active_positions}</div>
                  <div className="text-sm text-muted-foreground">Posições Ativas</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold">{aiStatus.total_trades_today}</div>
                  <div className="text-sm text-muted-foreground">Trades Hoje</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Último Sinal */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Target className="h-5 w-5" />
                <span>Último Sinal da IA</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  {aiStatus.last_signal.action === 'buy' ? (
                    <TrendingUp className="h-6 w-6 text-green-500" />
                  ) : aiStatus.last_signal.action === 'sell' ? (
                    <TrendingDown className="h-6 w-6 text-red-500" />
                  ) : (
                    <Clock className="h-6 w-6 text-gray-500" />
                  )}
                  <div>
                    <div className="font-semibold capitalize">
                      {aiStatus.last_signal.action === 'buy' ? 'COMPRA' :
                       aiStatus.last_signal.action === 'sell' ? 'VENDA' : 'AGUARDAR'}
                    </div>
                    <div className="text-sm text-muted-foreground">
                      Força: {(aiStatus.last_signal.strength * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm text-muted-foreground">
                    {new Date(aiStatus.last_signal.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              </div>

              <div className="bg-gray-50 p-3 rounded-lg">
                <div className="text-sm font-medium mb-1">Raciocínio da IA:</div>
                <div className="text-sm text-muted-foreground">
                  {aiStatus.last_signal.reasoning}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Log de Decisões */}
          <Card>
            <CardHeader>
              <CardTitle>📝 Log de Decisões em Tempo Real</CardTitle>
              <CardDescription>
                Histórico das últimas decisões tomadas pela IA
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-80">
                <div className="space-y-3">
                  {recentDecisions.length === 0 ? (
                    <div className="text-center py-8 text-muted-foreground">
                      Nenhuma decisão registrada ainda
                    </div>
                  ) : (
                    recentDecisions.map((decision) => (
                      <div key={decision.id} className="border rounded-lg p-3">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center space-x-2">
                            {decision.action === 'buy' ? (
                              <TrendingUp className="h-4 w-4 text-green-500" />
                            ) : (
                              <TrendingDown className="h-4 w-4 text-red-500" />
                            )}
                            <span className="font-medium uppercase">
                              {decision.action}
                            </span>
                            <Badge variant="outline">{decision.symbol}</Badge>
                          </div>
                          <div className="text-sm text-muted-foreground">
                            {new Date(decision.timestamp).toLocaleTimeString()}
                          </div>
                        </div>

                        <div className="text-sm text-muted-foreground mb-2">
                          {decision.reasoning}
                        </div>

                        <div className="flex justify-between text-sm">
                          <span>Confiança: {decision.confidence.toFixed(1)}%</span>
                          <span>Valor: ${decision.amount.toFixed(2)}</span>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </ScrollArea>
            </CardContent>
          </Card>
        </div>

        {/* Sidebar - Controles e Métricas */}
        <div className="space-y-6">
          {/* Controles de Risco */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Shield className="h-5 w-5" />
                <span>Risk Management</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm">Modo Conservador</span>
                <Switch />
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm">Auto Stop-Loss</span>
                <Switch defaultChecked />
              </div>

              <div className="flex items-center justify-between">
                <span className="text-sm">Risk Alerts</span>
                <Switch defaultChecked />
              </div>

              <Separator />

              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-sm">Position Size</span>
                  <span className="text-sm font-medium">{riskMetrics.position_size_pct}%</span>
                </div>

                <div className="flex justify-between">
                  <span className="text-sm">Max Drawdown</span>
                  <span className="text-sm font-medium">{riskMetrics.max_drawdown}%</span>
                </div>

                <div className="flex justify-between">
                  <span className="text-sm">VaR (95%)</span>
                  <span className="text-sm font-medium">{riskMetrics.var_95}%</span>
                </div>

                <div className="flex justify-between">
                  <span className="text-sm">Sharpe Ratio</span>
                  <span className="text-sm font-medium">{riskMetrics.sharpe_ratio}</span>
                </div>
              </div>

              <div className="pt-2">
                <div className="flex justify-between mb-2">
                  <span className="text-sm">Risk Score</span>
                  <span className="text-sm font-medium">
                    {(riskMetrics.risk_score * 100).toFixed(0)}%
                  </span>
                </div>
                <Progress value={riskMetrics.risk_score * 100} className="h-2" />
                <div className="text-xs text-muted-foreground mt-1">
                  {riskMetrics.risk_score < 0.3 ? 'Baixo Risco' :
                   riskMetrics.risk_score < 0.7 ? 'Risco Moderado' : 'Alto Risco'}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Performance Rápida */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <DollarSign className="h-5 w-5" />
                <span>Performance Hoje</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">+$247.80</div>
                <div className="text-sm text-muted-foreground">P&L Total</div>
              </div>

              <div className="grid grid-cols-2 gap-3 text-center">
                <div>
                  <div className="text-lg font-semibold">73%</div>
                  <div className="text-xs text-muted-foreground">Win Rate</div>
                </div>
                <div>
                  <div className="text-lg font-semibold">18</div>
                  <div className="text-xs text-muted-foreground">Trades</div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 text-center">
                <div>
                  <div className="text-lg font-semibold text-green-600">13</div>
                  <div className="text-xs text-muted-foreground">Wins</div>
                </div>
                <div>
                  <div className="text-lg font-semibold text-red-600">5</div>
                  <div className="text-xs text-muted-foreground">Losses</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Status do Sistema */}
          <Card>
            <CardHeader>
              <CardTitle>🔧 Status do Sistema</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex justify-between">
                <span className="text-sm">Deriv API</span>
                <Badge variant="default" className="bg-green-600">Online</Badge>
              </div>

              <div className="flex justify-between">
                <span className="text-sm">WebSocket</span>
                <Badge variant="default" className="bg-green-600">Conectado</Badge>
              </div>

              <div className="flex justify-between">
                <span className="text-sm">Modelo IA</span>
                <Badge variant="default" className="bg-blue-600">v2.1.3</Badge>
              </div>

              <div className="flex justify-between">
                <span className="text-sm">Risk Engine</span>
                <Badge variant="default" className="bg-green-600">Ativo</Badge>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Trading;