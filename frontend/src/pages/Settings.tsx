import { useState } from 'react';
import { Layout } from '@/components/Layout';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { 
  Shield, 
  DollarSign, 
  TrendingUp, 
  Settings as SettingsIcon,
  AlertTriangle,
  Target,
  BarChart3,
  Save
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

export default function Settings() {
  const { toast } = useToast();
  
  // Risk Management Settings
  const [stopLoss, setStopLoss] = useState('50');
  const [takeProfit, setTakeProfit] = useState('100');
  const [stakeAmount, setStakeAmount] = useState('10');
  
  // Strategy Settings
  const [aggressiveness, setAggressiveness] = useState('moderate');
  const [useRSI, setUseRSI] = useState(true);
  const [useMovingAverages, setUseMovingAverages] = useState(true);
  const [useBollinger, setUseBollinger] = useState(false);
  
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

  const handleAssetToggle = (asset: keyof typeof selectedAssets) => {
    setSelectedAssets(prev => ({
      ...prev,
      [asset]: !prev[asset]
    }));
  };

  const handleSaveSettings = () => {
    // Validate settings
    const stopLossValue = parseFloat(stopLoss);
    const takeProfitValue = parseFloat(takeProfit);
    const stakeValue = parseFloat(stakeAmount);

    if (stopLossValue <= 0 || takeProfitValue <= 0 || stakeValue <= 0) {
      toast({
        title: "Erro de Validação",
        description: "Todos os valores devem ser maiores que zero.",
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

    // Save settings (in a real app, this would save to backend/localStorage)
    toast({
      title: "Configurações Salvas",
      description: "Suas configurações foram aplicadas com sucesso.",
    });
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

  return (
    <Layout>
      <div className="space-y-6">
        {/* Page Header */}
        <div>
          <h1 className="text-3xl font-bold">Configurações do Bot</h1>
          <p className="text-muted-foreground">
            Configure os parâmetros de operação e gerenciamento de risco
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Risk Management */}
          <Card className="trading-card">
            <CardHeader>
              <CardTitle className="flex items-center">
                <Shield className="h-5 w-5 mr-2 text-primary" />
                Gerenciamento de Risco
              </CardTitle>
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
                  Valor fixo investido em cada operação
                </p>
              </div>

              <div className="p-4 bg-warning/10 rounded-lg border border-warning/20">
                <div className="flex items-start space-x-2">
                  <AlertTriangle className="h-5 w-5 text-warning mt-0.5" />
                  <div>
                    <p className="text-sm font-medium">Aviso Importante</p>
                    <p className="text-sm text-muted-foreground mt-1">
                      O trading envolve riscos. Configure limites adequados ao seu perfil de investidor.
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Strategy Configuration */}
          <Card className="trading-card">
            <CardHeader>
              <CardTitle className="flex items-center">
                <BarChart3 className="h-5 w-5 mr-2 text-primary" />
                Configuração da Estratégia
              </CardTitle>
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
                      <Label>Médias Móveis</Label>
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
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Asset Selection */}
        <Card className="trading-card">
          <CardHeader>
            <CardTitle className="flex items-center">
              <Target className="h-5 w-5 mr-2 text-primary" />
              Seleção de Ativos
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground mb-4">
              Escolha quais índices sintéticos o bot deve operar
            </p>
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

        {/* Save Settings */}
        <div className="flex justify-end">
          <Button 
            onClick={handleSaveSettings}
            className="success-gradient hover:opacity-90"
          >
            <Save className="h-4 w-4 mr-2" />
            Salvar Configurações
          </Button>
        </div>
      </div>
    </Layout>
  );
}