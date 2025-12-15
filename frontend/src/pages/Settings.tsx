import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Slider } from '@/components/ui/slider';
import {
  Shield,
  DollarSign,
  TrendingUp,
  Settings as SettingsIcon,
  AlertTriangle,
  Target,
  BarChart3,
  Save,
  Loader2,
  Brain,
  Layers,
  Zap,
  ChevronDown,
  Info
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { apiService } from '@/services/api';

export default function Settings() {
  const { toast } = useToast();

  // Loading states
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);

  // Risk Management Settings
  const [stopLoss, setStopLoss] = useState('50');
  const [takeProfit, setTakeProfit] = useState('100');
  const [stakeAmount, setStakeAmount] = useState('10');
  const [maxDailyLoss, setMaxDailyLoss] = useState('5');
  const [maxConcurrentTrades, setMaxConcurrentTrades] = useState('3');
  const [positionSizing, setPositionSizing] = useState('fixed_fractional');
  const [stopLossType, setStopLossType] = useState('fixed');
  const [takeProfitType, setTakeProfitType] = useState('fixed');

  // Strategy Settings
  const [aggressiveness, setAggressiveness] = useState('moderate');
  const [useRSI, setUseRSI] = useState(true);
  const [useMovingAverages, setUseMovingAverages] = useState(true);
  const [useBollinger, setUseBollinger] = useState(false);
  const [useMACD, setUseMACD] = useState(true);
  const [useCandlestickPatterns, setUseCandlestickPatterns] = useState(true);
  const [useChartPatterns, setUseChartPatterns] = useState(false);

  // ML Configuration
  const [mlEnabled, setMlEnabled] = useState(true);
  const [mlModel, setMlModel] = useState('ensemble');
  const [mlConfidenceThreshold, setMlConfidenceThreshold] = useState(70);

  // Order Flow Configuration
  const [orderFlowEnabled, setOrderFlowEnabled] = useState(true);
  const [orderFlowWeight, setOrderFlowWeight] = useState(30);

  // Execution Settings
  const [autoTrade, setAutoTrade] = useState(false);
  const [minSignalConfidence, setMinSignalConfidence] = useState(75);
  const [orderType, setOrderType] = useState('market');
  const [slippageTolerance, setSlippageTolerance] = useState('0.5');

  // Asset Selection
  const [selectedAssets, setSelectedAssets] = useState({
    volatility75: true,
    volatility100: true,
    jump25: false,
    jump50: false,
    jump75: false,
    jump100: false,
    boom1000: false,
    crash1000: false
  });

  // Load settings from backend on mount
  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    try {
      setIsLoading(true);
      const response = await apiService.getSettings();
      const settings = response.settings;

      // Update state with loaded settings
      setStopLoss(settings.stop_loss?.toString() || '50');
      setTakeProfit(settings.take_profit?.toString() || '100');
      setStakeAmount(settings.stake_amount?.toString() || '10');
      setAggressiveness(settings.aggressiveness || 'moderate');

      // Risk Management Advanced
      setMaxDailyLoss(settings.max_daily_loss?.toString() || '5');
      setMaxConcurrentTrades(settings.max_concurrent_trades?.toString() || '3');
      setPositionSizing(settings.position_sizing || 'fixed_fractional');
      setStopLossType(settings.stop_loss_type || 'fixed');
      setTakeProfitType(settings.take_profit_type || 'fixed');

      // Indicators
      setUseRSI(settings.indicators?.use_rsi ?? true);
      setUseMovingAverages(settings.indicators?.use_moving_averages ?? true);
      setUseBollinger(settings.indicators?.use_bollinger ?? false);
      setUseMACD(settings.indicators?.use_macd ?? true);
      setUseCandlestickPatterns(settings.indicators?.use_candlestick_patterns ?? true);
      setUseChartPatterns(settings.indicators?.use_chart_patterns ?? false);

      // ML Config
      setMlEnabled(settings.ml?.enabled ?? true);
      setMlModel(settings.ml?.model || 'ensemble');
      setMlConfidenceThreshold(settings.ml?.confidence_threshold || 70);

      // Order Flow
      setOrderFlowEnabled(settings.order_flow?.enabled ?? true);
      setOrderFlowWeight(settings.order_flow?.weight || 30);

      // Execution
      setAutoTrade(settings.execution?.auto_trade ?? false);
      setMinSignalConfidence(settings.execution?.min_signal_confidence || 75);
      setOrderType(settings.execution?.order_type || 'market');
      setSlippageTolerance(settings.execution?.slippage_tolerance?.toString() || '0.5');

      // Assets
      setSelectedAssets(settings.selected_assets || selectedAssets);

    } catch (error: any) {
      console.error('Failed to load settings:', error);
      toast({
        title: "Erro ao carregar configurações",
        description: "Usando configurações padrão. Verifique a conexão com o servidor.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleAssetToggle = (asset: keyof typeof selectedAssets) => {
    setSelectedAssets(prev => ({
      ...prev,
      [asset]: !prev[asset]
    }));
  };

  const handleSaveSettings = async () => {
    if (isSaving) return;

    // Validate settings
    const stopLossValue = parseFloat(stopLoss);
    const takeProfitValue = parseFloat(takeProfit);
    const stakeValue = parseFloat(stakeAmount);
    const maxDailyLossValue = parseFloat(maxDailyLoss);
    const maxConcurrentTradesValue = parseInt(maxConcurrentTrades);
    const slippageToleranceValue = parseFloat(slippageTolerance);

    if (stopLossValue <= 0 || takeProfitValue <= 0 || stakeValue <= 0) {
      toast({
        title: "Erro de Validação",
        description: "Todos os valores monetários devem ser maiores que zero.",
        variant: "destructive",
      });
      return;
    }

    if (stopLossValue >= takeProfitValue) {
      toast({
        title: "Erro de Validação",
        description: "O Take Profit deve ser maior que o Stop Loss.",
        variant: "destructive",
      });
      return;
    }

    const selectedAssetCount = Object.values(selectedAssets).filter(Boolean).length;
    if (selectedAssetCount === 0) {
      toast({
        title: "Erro de Validação",
        description: "Selecione pelo menos um ativo para operar.",
        variant: "destructive",
      });
      return;
    }

    try {
      setIsSaving(true);

      const settingsData = {
        // Basic Risk Management
        stop_loss: stopLossValue,
        take_profit: takeProfitValue,
        stake_amount: stakeValue,
        aggressiveness,

        // Advanced Risk Management
        max_daily_loss: maxDailyLossValue,
        max_concurrent_trades: maxConcurrentTradesValue,
        position_sizing: positionSizing,
        stop_loss_type: stopLossType,
        take_profit_type: takeProfitType,

        // Strategy Indicators
        indicators: {
          use_rsi: useRSI,
          use_moving_averages: useMovingAverages,
          use_bollinger: useBollinger,
          use_macd: useMACD,
          use_candlestick_patterns: useCandlestickPatterns,
          use_chart_patterns: useChartPatterns
        },

        // ML Configuration
        ml: {
          enabled: mlEnabled,
          model: mlModel,
          confidence_threshold: mlConfidenceThreshold
        },

        // Order Flow
        order_flow: {
          enabled: orderFlowEnabled,
          weight: orderFlowWeight
        },

        // Execution
        execution: {
          auto_trade: autoTrade,
          min_signal_confidence: minSignalConfidence,
          order_type: orderType,
          slippage_tolerance: slippageToleranceValue
        },

        // Assets
        selected_assets: selectedAssets
      };

      const response = await apiService.updateSettings(settingsData);

      toast({
        title: "Configurações Salvas",
        description: response.message || "Suas configurações foram aplicadas com sucesso.",
      });

    } catch (error: any) {
      console.error('Failed to save settings:', error);
      toast({
        title: "Erro ao salvar",
        description: error.message || "Não foi possível salvar as configurações. Tente novamente.",
        variant: "destructive",
      });
    } finally {
      setIsSaving(false);
    }
  };

  const assets = [
    { id: 'volatility75', name: 'Volatility 75 Index', description: 'Mercado sintético com volatilidade de 75%' },
    { id: 'volatility100', name: 'Volatility 100 Index', description: 'Mercado sintético com volatilidade de 100%' },
    { id: 'jump25', name: 'Jump 25 Index', description: 'Mercado com saltos ocasionais de 25%' },
    { id: 'jump50', name: 'Jump 50 Index', description: 'Mercado com saltos ocasionais de 50%' },
    { id: 'jump75', name: 'Jump 75 Index', description: 'Mercado com saltos ocasionais de 75%' },
    { id: 'jump100', name: 'Jump 100 Index', description: 'Mercado com saltos ocasionais de 100%' },
    { id: 'boom1000', name: 'Boom 1000 Index', description: 'Mercado com picos de alta' },
    { id: 'crash1000', name: 'Crash 1000 Index', description: 'Mercado com quedas abruptas' },
  ];

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex flex-col items-center space-y-4">
          <Loader2 className="h-8 w-8 animate-spin" />
          <p className="text-muted-foreground">Carregando configurações...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold">Configurações do Bot</h1>
        <p className="text-muted-foreground">
          Configure todos os parâmetros de operação, estratégia e gerenciamento de risco
        </p>
      </div>

      <Tabs defaultValue="risk" className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="risk">
            <Shield className="h-4 w-4 mr-2" />
            Risco
          </TabsTrigger>
          <TabsTrigger value="strategy">
            <BarChart3 className="h-4 w-4 mr-2" />
            Estratégia
          </TabsTrigger>
          <TabsTrigger value="ml">
            <Brain className="h-4 w-4 mr-2" />
            ML & IA
          </TabsTrigger>
          <TabsTrigger value="execution">
            <Zap className="h-4 w-4 mr-2" />
            Execução
          </TabsTrigger>
          <TabsTrigger value="assets">
            <Target className="h-4 w-4 mr-2" />
            Ativos
          </TabsTrigger>
        </TabsList>

        {/* Risk Management Tab */}
        <TabsContent value="risk" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Basic Risk Management */}
            <Card className="trading-card">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Shield className="h-5 w-5 mr-2 text-primary" />
                  Gerenciamento de Risco Básico
                </CardTitle>
                <CardDescription>
                  Configure limites financeiros por sessão
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <Label htmlFor="stopLoss">Stop Loss (USD)</Label>
                  <Input
                    id="stopLoss"
                    type="number"
                    value={stopLoss}
                    onChange={(e) => setStopLoss(e.target.value)}
                    placeholder="50.00"
                  />
                  <p className="text-sm text-muted-foreground">
                    Perda máxima permitida por sessão
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="takeProfit">Take Profit (USD)</Label>
                  <Input
                    id="takeProfit"
                    type="number"
                    value={takeProfit}
                    onChange={(e) => setTakeProfit(e.target.value)}
                    placeholder="100.00"
                  />
                  <p className="text-sm text-muted-foreground">
                    Meta de lucro para encerrar a sessão
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="stakeAmount">Valor por Entrada (USD)</Label>
                  <Input
                    id="stakeAmount"
                    type="number"
                    value={stakeAmount}
                    onChange={(e) => setStakeAmount(e.target.value)}
                    placeholder="10.00"
                  />
                  <p className="text-sm text-muted-foreground">
                    Valor investido em cada operação
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="maxDailyLoss">Perda Diária Máxima (%)</Label>
                  <Input
                    id="maxDailyLoss"
                    type="number"
                    value={maxDailyLoss}
                    onChange={(e) => setMaxDailyLoss(e.target.value)}
                    placeholder="5.0"
                  />
                  <p className="text-sm text-muted-foreground">
                    % máximo de perda do capital por dia
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Advanced Risk Management */}
            <Card className="trading-card">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Shield className="h-5 w-5 mr-2 text-orange-500" />
                  Gerenciamento Avançado
                </CardTitle>
                <CardDescription>
                  Configurações avançadas de risco e posicionamento
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <Label htmlFor="maxConcurrentTrades">Trades Simultâneas</Label>
                  <Input
                    id="maxConcurrentTrades"
                    type="number"
                    value={maxConcurrentTrades}
                    onChange={(e) => setMaxConcurrentTrades(e.target.value)}
                    placeholder="3"
                    min="1"
                    max="10"
                  />
                  <p className="text-sm text-muted-foreground">
                    Máximo de operações abertas ao mesmo tempo
                  </p>
                </div>

                <div className="space-y-2">
                  <Label>Tipo de Dimensionamento</Label>
                  <Select value={positionSizing} onValueChange={setPositionSizing}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="fixed_fractional">Fixed Fractional</SelectItem>
                      <SelectItem value="kelly">Kelly Criterion</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-sm text-muted-foreground">
                    Método de cálculo do tamanho da posição
                  </p>
                </div>

                <div className="space-y-2">
                  <Label>Tipo de Stop Loss</Label>
                  <Select value={stopLossType} onValueChange={setStopLossType}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="fixed">Fixo</SelectItem>
                      <SelectItem value="atr">ATR-Based</SelectItem>
                      <SelectItem value="trailing">Trailing Stop</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-sm text-muted-foreground">
                    Estratégia de stop loss a ser utilizada
                  </p>
                </div>

                <div className="space-y-2">
                  <Label>Tipo de Take Profit</Label>
                  <Select value={takeProfitType} onValueChange={setTakeProfitType}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="fixed">Fixo</SelectItem>
                      <SelectItem value="partial">Saída Parcial</SelectItem>
                      <SelectItem value="trailing">Trailing Profit</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-sm text-muted-foreground">
                    Estratégia de realização de lucro
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Warning */}
          <Card className="border-warning bg-warning/5">
            <CardContent className="p-4">
              <div className="flex items-start space-x-3">
                <AlertTriangle className="h-5 w-5 text-warning mt-0.5" />
                <div>
                  <p className="text-sm font-medium">Aviso Importante</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    O trading envolve riscos significativos. Configure limites adequados ao seu perfil de investidor e capital disponível. Nunca invista dinheiro que não pode perder.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Strategy Tab */}
        <TabsContent value="strategy" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Strategy Configuration */}
            <Card className="trading-card">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <BarChart3 className="h-5 w-5 mr-2 text-primary" />
                  Configuração da Estratégia
                </CardTitle>
                <CardDescription>
                  Defina o comportamento e agressividade do bot
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <Label>Nível de Agressividade</Label>
                  <Select value={aggressiveness} onValueChange={setAggressiveness}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="conservative">Conservador</SelectItem>
                      <SelectItem value="moderate">Moderado</SelectItem>
                      <SelectItem value="aggressive">Agressivo</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-sm text-muted-foreground">
                    Define a frequência e intensidade das operações
                  </p>
                </div>

                <Separator />

                <div>
                  <h4 className="text-sm font-medium mb-4">Indicadores Técnicos</h4>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>RSI (Relative Strength Index)</Label>
                        <p className="text-sm text-muted-foreground">
                          Identifica condições de sobrecompra/sobrevenda
                        </p>
                      </div>
                      <Switch
                        checked={useRSI}
                        onCheckedChange={setUseRSI}
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Médias Móveis (SMA/EMA)</Label>
                        <p className="text-sm text-muted-foreground">
                          Analisa tendências de curto e longo prazo
                        </p>
                      </div>
                      <Switch
                        checked={useMovingAverages}
                        onCheckedChange={setUseMovingAverages}
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Bandas de Bollinger</Label>
                        <p className="text-sm text-muted-foreground">
                          Detecta volatilidade e pontos de reversão
                        </p>
                      </div>
                      <Switch
                        checked={useBollinger}
                        onCheckedChange={setUseBollinger}
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>MACD</Label>
                        <p className="text-sm text-muted-foreground">
                          Convergência e divergência de médias móveis
                        </p>
                      </div>
                      <Switch
                        checked={useMACD}
                        onCheckedChange={setUseMACD}
                      />
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Pattern Recognition */}
            <Card className="trading-card">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <TrendingUp className="h-5 w-5 mr-2 text-primary" />
                  Reconhecimento de Padrões
                </CardTitle>
                <CardDescription>
                  Ative análise de padrões de candlestick e gráficos
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Padrões de Candlestick</Label>
                    <p className="text-sm text-muted-foreground">
                      Doji, Hammer, Engulfing, etc.
                    </p>
                  </div>
                  <Switch
                    checked={useCandlestickPatterns}
                    onCheckedChange={setUseCandlestickPatterns}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Padrões de Gráfico</Label>
                    <p className="text-sm text-muted-foreground">
                      Cabeça e Ombros, Triângulos, etc.
                    </p>
                  </div>
                  <Switch
                    checked={useChartPatterns}
                    onCheckedChange={setUseChartPatterns}
                  />
                </div>

                <Separator />

                <div className="p-4 bg-blue-50 dark:bg-blue-950/20 rounded-lg border border-blue-200 dark:border-blue-900">
                  <div className="flex items-start space-x-2">
                    <Info className="h-5 w-5 text-blue-600 dark:text-blue-400 mt-0.5" />
                    <div>
                      <p className="text-sm font-medium text-blue-900 dark:text-blue-100">Dica</p>
                      <p className="text-sm text-blue-700 dark:text-blue-300 mt-1">
                        Combine múltiplos indicadores para sinais mais confiáveis. Estratégias híbridas geralmente têm melhor desempenho.
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* ML & AI Tab */}
        <TabsContent value="ml" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* ML Configuration */}
            <Card className="trading-card">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Brain className="h-5 w-5 mr-2 text-primary" />
                  Machine Learning
                </CardTitle>
                <CardDescription>
                  Configure o modelo de IA XGBoost (68.14% accuracy)
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Ativar Machine Learning</Label>
                    <p className="text-sm text-muted-foreground">
                      Usar modelo XGBoost treinado para predições
                    </p>
                  </div>
                  <Switch
                    checked={mlEnabled}
                    onCheckedChange={setMlEnabled}
                  />
                </div>

                <Separator />

                <div className="space-y-2">
                  <Label>Modelo de ML</Label>
                  <Select value={mlModel} onValueChange={setMlModel} disabled={!mlEnabled}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="ensemble">Ensemble (XGBoost + RF)</SelectItem>
                      <SelectItem value="xgboost">XGBoost Only</SelectItem>
                      <SelectItem value="random_forest">Random Forest</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-sm text-muted-foreground">
                    Modelo de IA a ser utilizado nas predições
                  </p>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label>Threshold de Confiança ML: {mlConfidenceThreshold}%</Label>
                    <Badge variant="outline">{mlConfidenceThreshold}%</Badge>
                  </div>
                  <Slider
                    value={[mlConfidenceThreshold]}
                    onValueChange={(value) => setMlConfidenceThreshold(value[0])}
                    min={60}
                    max={95}
                    step={5}
                    disabled={!mlEnabled}
                    className="w-full"
                  />
                  <p className="text-sm text-muted-foreground">
                    Confiança mínima do modelo ML para gerar sinais
                  </p>
                </div>

                <div className="p-4 bg-green-50 dark:bg-green-950/20 rounded-lg border border-green-200 dark:border-green-900">
                  <div className="flex items-start space-x-2">
                    <Brain className="h-5 w-5 text-green-600 dark:text-green-400 mt-0.5" />
                    <div>
                      <p className="text-sm font-medium text-green-900 dark:text-green-100">Status do Modelo</p>
                      <p className="text-sm text-green-700 dark:text-green-300 mt-1">
                        XGBoost treinado com 68.14% de acurácia em 10.000+ samples
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Order Flow Configuration */}
            <Card className="trading-card">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Layers className="h-5 w-5 mr-2 text-primary" />
                  Order Flow Analysis
                </CardTitle>
                <CardDescription>
                  Análise institucional de fluxo de ordens
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Ativar Order Flow</Label>
                    <p className="text-sm text-muted-foreground">
                      Análise de volume e desequilíbrio de ordens
                    </p>
                  </div>
                  <Switch
                    checked={orderFlowEnabled}
                    onCheckedChange={setOrderFlowEnabled}
                  />
                </div>

                <Separator />

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label>Peso do Order Flow: {orderFlowWeight}%</Label>
                    <Badge variant="outline">{orderFlowWeight}%</Badge>
                  </div>
                  <Slider
                    value={[orderFlowWeight]}
                    onValueChange={(value) => setOrderFlowWeight(value[0])}
                    min={10}
                    max={50}
                    step={5}
                    disabled={!orderFlowEnabled}
                    className="w-full"
                  />
                  <p className="text-sm text-muted-foreground">
                    Influência do Order Flow na decisão final (complementa ML)
                  </p>
                </div>

                <div className="p-4 bg-purple-50 dark:bg-purple-950/20 rounded-lg border border-purple-200 dark:border-purple-900">
                  <div className="flex items-start space-x-2">
                    <Layers className="h-5 w-5 text-purple-600 dark:text-purple-400 mt-0.5" />
                    <div>
                      <p className="text-sm font-medium text-purple-900 dark:text-purple-100">Análise Institucional</p>
                      <p className="text-sm text-purple-700 dark:text-purple-300 mt-1">
                        Order Flow detecta movimentos de grandes players e desequilíbrios de mercado
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Execution Tab */}
        <TabsContent value="execution" className="space-y-6">
          <Card className="trading-card">
            <CardHeader>
              <CardTitle className="flex items-center">
                <Zap className="h-5 w-5 mr-2 text-primary" />
                Configurações de Execução
              </CardTitle>
              <CardDescription>
                Controle como as ordens são executadas no mercado
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="space-y-6">
                  <div className="flex items-center justify-between p-4 border rounded-lg">
                    <div>
                      <Label className="text-base">Auto Trading</Label>
                      <p className="text-sm text-muted-foreground mt-1">
                        Executar trades automaticamente (sem aprovação manual)
                      </p>
                    </div>
                    <Switch
                      checked={autoTrade}
                      onCheckedChange={setAutoTrade}
                    />
                  </div>

                  {autoTrade && (
                    <div className="p-4 bg-yellow-50 dark:bg-yellow-950/20 rounded-lg border border-yellow-200 dark:border-yellow-900">
                      <div className="flex items-start space-x-2">
                        <AlertTriangle className="h-5 w-5 text-yellow-600 dark:text-yellow-400 mt-0.5" />
                        <div>
                          <p className="text-sm font-medium text-yellow-900 dark:text-yellow-100">Atenção</p>
                          <p className="text-sm text-yellow-700 dark:text-yellow-300 mt-1">
                            Auto Trading ativado. O bot executará ordens automaticamente sem confirmação.
                          </p>
                        </div>
                      </div>
                    </div>
                  )}

                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label>Confiança Mínima do Sinal: {minSignalConfidence}%</Label>
                      <Badge variant="outline">{minSignalConfidence}%</Badge>
                    </div>
                    <Slider
                      value={[minSignalConfidence]}
                      onValueChange={(value) => setMinSignalConfidence(value[0])}
                      min={65}
                      max={95}
                      step={5}
                      className="w-full"
                    />
                    <p className="text-sm text-muted-foreground">
                      Confiança mínima combinada (ML + Indicators + Order Flow)
                    </p>
                  </div>
                </div>

                <div className="space-y-6">
                  <div className="space-y-2">
                    <Label>Tipo de Ordem</Label>
                    <Select value={orderType} onValueChange={setOrderType}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="market">Market (Execução Imediata)</SelectItem>
                        <SelectItem value="limit">Limit (Preço Específico)</SelectItem>
                      </SelectContent>
                    </Select>
                    <p className="text-sm text-muted-foreground">
                      Como as ordens serão executadas no mercado
                    </p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="slippageTolerance">Tolerância de Slippage (%)</Label>
                    <Input
                      id="slippageTolerance"
                      type="number"
                      value={slippageTolerance}
                      onChange={(e) => setSlippageTolerance(e.target.value)}
                      placeholder="0.5"
                      step="0.1"
                    />
                    <p className="text-sm text-muted-foreground">
                      Diferença máxima aceitável entre preço esperado e executado
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Assets Tab */}
        <TabsContent value="assets" className="space-y-6">
          <Card className="trading-card">
            <CardHeader>
              <CardTitle className="flex items-center">
                <Target className="h-5 w-5 mr-2 text-primary" />
                Seleção de Ativos
              </CardTitle>
              <CardDescription>
                Escolha quais índices sintéticos o bot deve operar
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {assets.map((asset) => (
                  <div
                    key={asset.id}
                    className="flex items-center justify-between p-4 border border-border rounded-lg hover:bg-muted/30 transition-colors"
                  >
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <h4 className="font-medium">{asset.name}</h4>
                        {selectedAssets[asset.id as keyof typeof selectedAssets] && (
                          <Badge variant="default" className="bg-success text-success-foreground">
                            Ativo
                          </Badge>
                        )}
                      </div>
                      <p className="text-sm text-muted-foreground">
                        {asset.description}
                      </p>
                    </div>
                    <Switch
                      checked={selectedAssets[asset.id as keyof typeof selectedAssets]}
                      onCheckedChange={() => handleAssetToggle(asset.id as keyof typeof selectedAssets)}
                    />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Save Settings Button */}
      <div className="flex justify-end sticky bottom-4">
        <Button
          onClick={handleSaveSettings}
          disabled={isSaving || isLoading}
          size="lg"
          className="success-gradient hover:opacity-90 disabled:opacity-50 shadow-lg"
        >
          {isSaving ? (
            <>
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Salvando...
            </>
          ) : (
            <>
              <Save className="h-4 w-4 mr-2" />
              Salvar Todas as Configurações
            </>
          )}
        </Button>
      </div>
    </div>
  );
}
