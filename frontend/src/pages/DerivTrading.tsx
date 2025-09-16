import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Activity, 
  DollarSign, 
  TrendingUp, 
  TrendingDown, 
  BarChart3, 
  Clock, 
  Zap,
  AlertCircle,
  CheckCircle,
  RefreshCw,
  Play,
  Square,
  Eye,
  Wallet
} from 'lucide-react';
import { toast } from 'sonner';
import apiService from '@/services/api';
import TradingBuyForm from '@/components/TradingBuyForm';
import TradeResultModal from '@/components/TradeResultModal';

interface DerivConnection {
  is_connected: boolean;
  is_authenticated: boolean;
  balance: number;
  loginid: string;
  api_status: string;
}

interface DerivTick {
  price: number;
  timestamp: number;
  epoch: number;
}

interface DerivContract {
  contract_id: number;
  symbol: string;
  contract_type: string;
  buy_price: number;
  current_spot?: number;
  profit: number;
  payout: number;
  is_sold: boolean;
  longcode: string;
  date_start: number;
  date_expiry: number;
}

interface DerivTransaction {
  transaction_id: number;
  contract_id: number;
  symbol: string;
  contract_type: string;
  buy_price: number;
  sell_price?: number;
  profit: number;
  duration: number;
  purchase_time: number;
  sell_time?: number;
  longcode: string;
}

const DerivTrading: React.FC = () => {
  // Estados de conexão
  const [connection, setConnection] = useState<DerivConnection | null>(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [apiToken, setApiToken] = useState('');
  const [isDemoMode, setIsDemoMode] = useState(true);

  // Estados de trading
  const [symbols, setSymbols] = useState<string[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState('R_50');
  const [contractType, setContractType] = useState('CALL');
  const [amount, setAmount] = useState(10);
  const [duration, setDuration] = useState(5);
  const [durationUnit, setDurationUnit] = useState('m');
  
  // Estados de dados
  const [currentTick, setCurrentTick] = useState<DerivTick | null>(null);
  const [portfolio, setPortfolio] = useState<DerivContract[]>([]);
  const [history, setHistory] = useState<DerivTransaction[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [endpointsAvailable, setEndpointsAvailable] = useState<boolean | null>(null);

  // Estados para o novo componente de buy
  const [tradeResult, setTradeResult] = useState<any>(null);
  const [tradeError, setTradeError] = useState<string | null>(null);
  const [showResultModal, setShowResultModal] = useState(false);

  // Carregar dados iniciais
  useEffect(() => {
    checkEndpointsAvailability();
    checkConnectionStatus();
    loadSymbols();
  }, []);

  const checkEndpointsAvailability = async () => {
    try {
      const available = await apiService.checkDerivEndpointsAvailable();
      setEndpointsAvailable(available);
    } catch (error) {
      setEndpointsAvailable(false);
    }
  };

  // Atualizar tick periodicamente se conectado
  useEffect(() => {
    if (connection?.is_authenticated && selectedSymbol) {
      const interval = setInterval(() => {
        updateCurrentTick();
      }, 2000);
      
      return () => clearInterval(interval);
    }
  }, [connection, selectedSymbol]);

  const checkConnectionStatus = async () => {
    try {
      const status = await apiService.derivGetStatus();
      if (status.status === 'success') {
        setConnection(status.connection_info);
      }
    } catch (error) {
      console.log('Not connected to Deriv API');
    }
  };

  const loadSymbols = async () => {
    try {
      const response = await apiService.derivGetSymbols();
      if (response.status === 'success') {
        setSymbols(response.symbols);
      }
    } catch (error) {
      // Usar símbolos padrão se não conseguir carregar
      setSymbols(['R_10', 'R_25', 'R_50', 'R_75', 'R_100']);
    }
  };

  const connectToDerivAPI = async () => {
    if (!apiToken.trim()) {
      toast.error('Por favor, insira seu token da API Deriv');
      return;
    }

    setIsConnecting(true);
    try {
      const response = await apiService.derivConnect(apiToken, isDemoMode);
      if (response.status === 'success') {
        setConnection(response.connection_info);
        toast.success('Conectado à API Deriv com sucesso!');
        
        // Carregar dados iniciais
        await loadSymbols();
        await loadPortfolio();
        await loadHistory();
        
        // Subscrever ao símbolo selecionado
        if (selectedSymbol) {
          await subscribeToSymbol(selectedSymbol);
        }
      }
    } catch (error: any) {
      if (error.message === 'DERIV_ENDPOINTS_NOT_AVAILABLE') {
        toast.error('🚧 Endpoints da Deriv API ainda não foram deployados no servidor');
      } else {
        toast.error(`Erro ao conectar: ${error.message}`);
      }
    } finally {
      setIsConnecting(false);
    }
  };

  const disconnectFromDerivAPI = async () => {
    try {
      await apiService.derivDisconnect();
      setConnection(null);
      setCurrentTick(null);
      setPortfolio([]);
      toast.success('Desconectado da API Deriv');
    } catch (error: any) {
      toast.error(`Erro ao desconectar: ${error.message}`);
    }
  };

  const subscribeToSymbol = async (symbol: string) => {
    try {
      await apiService.derivSubscribeTicks(symbol);
      await updateCurrentTick();
    } catch (error: any) {
      console.error('Error subscribing to symbol:', error);
    }
  };

  const updateCurrentTick = async () => {
    try {
      const response = await apiService.derivGetLastTick(selectedSymbol);
      if (response.status === 'success') {
        setCurrentTick(response.tick);
      }
    } catch (error) {
      // Silently handle tick update errors
    }
  };

  const loadPortfolio = async () => {
    try {
      const response = await apiService.derivGetPortfolio();
      if (response.status === 'success') {
        setPortfolio(response.contracts);
      }
    } catch (error) {
      console.error('Error loading portfolio:', error);
    }
  };

  const loadHistory = async () => {
    try {
      const response = await apiService.derivGetHistory(20);
      if (response.status === 'success') {
        setHistory(response.transactions);
      }
    } catch (error) {
      console.error('Error loading history:', error);
    }
  };

  const executeTrade = async () => {
    if (!connection?.is_authenticated) {
      toast.error('Conecte-se à API Deriv primeiro');
      return;
    }

    if (amount <= 0) {
      toast.error('Valor do stake deve ser maior que zero');
      return;
    }

    setIsLoading(true);
    try {
      const response = await apiService.derivBuyContract(
        contractType,
        selectedSymbol,
        amount,
        duration,
        durationUnit
      );

      if (response.status === 'success') {
        toast.success(`Contrato comprado! ID: ${response.contract.contract_id}`);
        
        // Atualizar dados
        await loadPortfolio();
        await updateBalance();
      }
    } catch (error: any) {
      toast.error(`Erro ao executar trade: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const sellContract = async (contractId: number) => {
    setIsLoading(true);
    try {
      const response = await apiService.derivSellContract(contractId);
      if (response.status === 'success') {
        toast.success(`Contrato vendido por ${response.sale.sold_for}`);
        
        // Atualizar dados
        await loadPortfolio();
        await loadHistory();
        await updateBalance();
      }
    } catch (error: any) {
      toast.error(`Erro ao vender contrato: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const updateBalance = async () => {
    try {
      const response = await apiService.derivGetBalance();
      if (response.status === 'success' && connection) {
        setConnection({
          ...connection,
          balance: response.balance
        });
      }
    } catch (error) {
      console.error('Error updating balance:', error);
    }
  };

  // Callbacks para o novo componente de trading
  const handleTradeSuccess = async (result: any) => {
    setTradeResult(result);
    setTradeError(null);
    setShowResultModal(true);

    // Atualizar dados após sucesso
    await loadPortfolio();
    await loadHistory();

    toast.success(`Contrato comprado! ID: ${result.contract.contract_id}`);
  };

  const handleTradeError = (error: string) => {
    setTradeResult(null);
    setTradeError(error);
    setShowResultModal(true);

    toast.error(`Erro ao executar trade: ${error}`);
  };

  const closeResultModal = () => {
    setShowResultModal(false);
    setTradeResult(null);
    setTradeError(null);
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('pt-BR', {
      style: 'currency',
      currency: 'USD'
    }).format(amount);
  };

  const formatTime = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleString('pt-BR');
  };

  if (!connection?.is_authenticated) {
    return (
      <div className="container mx-auto p-6">
        <div className="max-w-md mx-auto">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Conectar à API Deriv
              </CardTitle>
              <CardDescription>
                Conecte-se à API real da Deriv para operar em conta fictícia
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="api-token">Token da API Deriv</Label>
                <Input
                  id="api-token"
                  type="password"
                  placeholder="Insira seu token da API Deriv"
                  value={apiToken}
                  onChange={(e) => setApiToken(e.target.value)}
                />
                <p className="text-sm text-muted-foreground">
                  Obtenha seu token em{' '}
                  <a 
                    href="https://app.deriv.com/account/api-token" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:underline"
                  >
                    app.deriv.com
                  </a>
                </p>
              </div>

              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="demo-mode"
                  checked={isDemoMode}
                  onChange={(e) => setIsDemoMode(e.target.checked)}
                />
                <Label htmlFor="demo-mode">Usar conta demo</Label>
              </div>

              {endpointsAvailable === false && (
                <Alert className="mb-4 border-blue-200 bg-blue-50">
                  <AlertCircle className="h-4 w-4 text-blue-600" />
                  <AlertDescription className="text-blue-800">
                    🚧 <strong>Preview da Interface</strong> - Os endpoints da Deriv API estão sendo deployados no servidor. 
                    Esta é uma prévia da interface que funcionará após o deploy completo.
                  </AlertDescription>
                </Alert>
              )}

              <Alert>
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  {isDemoMode ? 
                    'Modo demo: Use para testar sem risco financeiro' :
                    'Modo real: Cuidado! Você está usando dinheiro real'
                  }
                </AlertDescription>
              </Alert>

              <Button 
                onClick={connectToDerivAPI}
                disabled={isConnecting}
                className="w-full"
              >
                {isConnecting ? (
                  <>
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                    Conectando...
                  </>
                ) : (
                  <>
                    <Zap className="h-4 w-4 mr-2" />
                    Conectar
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Banner de desenvolvimento */}
      {endpointsAvailable === false && (
        <Alert className="border-blue-200 bg-blue-50">
          <AlertCircle className="h-4 w-4 text-blue-600" />
          <AlertDescription className="text-blue-800">
            🚧 <strong>Interface em Preview</strong> - Esta é uma demonstração da interface de trading real. 
            Os endpoints estão sendo deployados e estarão funcionais em breve.
          </AlertDescription>
        </Alert>
      )}

      {/* Header com informações de conexão */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Trading Real - Deriv API</h1>
          <p className="text-muted-foreground">
            Conectado como {connection.loginid} • {isDemoMode ? 'Demo' : 'Real'}
          </p>
        </div>
        <div className="flex items-center gap-4">
          <Badge variant="outline" className="text-green-600">
            <CheckCircle className="h-3 w-3 mr-1" />
            Conectado
          </Badge>
          <Button variant="outline" onClick={disconnectFromDerivAPI}>
            Desconectar
          </Button>
        </div>
      </div>

      {/* Cards de informações principais */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Saldo</p>
                <p className="text-2xl font-bold">{formatCurrency(connection.balance)}</p>
              </div>
              <Wallet className="h-8 w-8 text-green-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Contratos Ativos</p>
                <p className="text-2xl font-bold">{portfolio.length}</p>
              </div>
              <BarChart3 className="h-8 w-8 text-blue-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Preço Atual</p>
                <p className="text-2xl font-bold">
                  {currentTick ? currentTick.price.toFixed(3) : '--'}
                </p>
              </div>
              <Activity className="h-8 w-8 text-purple-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-muted-foreground">Símbolo</p>
                <p className="text-2xl font-bold">{selectedSymbol}</p>
              </div>
              <TrendingUp className="h-8 w-8 text-orange-600" />
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="trade" className="space-y-4">
        <TabsList>
          <TabsTrigger value="trade">Trading</TabsTrigger>
          <TabsTrigger value="portfolio">Portfólio</TabsTrigger>
          <TabsTrigger value="history">Histórico</TabsTrigger>
        </TabsList>

        <TabsContent value="trade" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Novo componente de compra */}
            <TradingBuyForm
              symbol={selectedSymbol}
              onTradeSuccess={handleTradeSuccess}
              onTradeError={handleTradeError}
              disabled={!connection?.is_authenticated || !endpointsAvailable}
            />

            {/* Informações do último tick */}
            {currentTick && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Activity className="h-5 w-5" />
                    Preço Atual - {selectedSymbol}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="text-center">
                      <div className="text-3xl font-bold text-blue-600">
                        {currentTick.price}
                      </div>
                      <p className="text-sm text-muted-foreground">
                        Atualizado em: {new Date(currentTick.timestamp * 1000).toLocaleTimeString()}
                      </p>
                    </div>

                    <Button
                      variant="outline"
                      onClick={updateCurrentTick}
                      className="w-full"
                    >
                      <RefreshCw className="h-4 w-4 mr-2" />
                      Atualizar Preço
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>

        <TabsContent value="portfolio">
          <Card>
            <CardHeader>
              <CardTitle>Portfólio de Contratos</CardTitle>
              <CardDescription>
                Seus contratos ativos na plataforma Deriv
              </CardDescription>
            </CardHeader>
            <CardContent>
              {portfolio.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>Nenhum contrato ativo</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {portfolio.map((contract) => (
                    <div 
                      key={contract.contract_id}
                      className="flex items-center justify-between p-4 border rounded-lg"
                    >
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <Badge variant="outline">{contract.symbol}</Badge>
                          <Badge variant={contract.contract_type === 'CALL' ? 'default' : 'secondary'}>
                            {contract.contract_type}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground mt-1">
                          ID: {contract.contract_id}
                        </p>
                      </div>
                      
                      <div className="text-right">
                        <p className="font-medium">
                          Lucro: <span className={contract.profit >= 0 ? 'text-green-600' : 'text-red-600'}>
                            {formatCurrency(contract.profit)}
                          </span>
                        </p>
                        <p className="text-sm text-muted-foreground">
                          Stake: {formatCurrency(contract.buy_price)}
                        </p>
                      </div>

                      {!contract.is_sold && (
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => sellContract(contract.contract_id)}
                          disabled={isLoading}
                        >
                          <Square className="h-4 w-4 mr-1" />
                          Vender
                        </Button>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="history">
          <Card>
            <CardHeader>
              <CardTitle>Histórico de Trades</CardTitle>
              <CardDescription>
                Seus últimos trades executados
              </CardDescription>
            </CardHeader>
            <CardContent>
              {history.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <Clock className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>Nenhum trade no histórico</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {history.map((transaction) => (
                    <div 
                      key={transaction.transaction_id}
                      className="flex items-center justify-between p-4 border rounded-lg"
                    >
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <Badge variant="outline">{transaction.symbol}</Badge>
                          <Badge variant={transaction.contract_type === 'CALL' ? 'default' : 'secondary'}>
                            {transaction.contract_type}
                          </Badge>
                          <Badge variant={transaction.profit >= 0 ? 'default' : 'destructive'}>
                            {transaction.profit >= 0 ? (
                              <TrendingUp className="h-3 w-3 mr-1" />
                            ) : (
                              <TrendingDown className="h-3 w-3 mr-1" />
                            )}
                            {transaction.profit >= 0 ? 'Ganho' : 'Perda'}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground mt-1">
                          {formatTime(transaction.purchase_time)}
                        </p>
                      </div>
                      
                      <div className="text-right">
                        <p className="font-medium">
                          <span className={transaction.profit >= 0 ? 'text-green-600' : 'text-red-600'}>
                            {formatCurrency(transaction.profit)}
                          </span>
                        </p>
                        <p className="text-sm text-muted-foreground">
                          Stake: {formatCurrency(transaction.buy_price)}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Modal de resultado */}
      <TradeResultModal
        result={tradeResult}
        error={tradeError}
        isOpen={showResultModal}
        onClose={closeResultModal}
      />
    </div>
  );
};

export default DerivTrading;