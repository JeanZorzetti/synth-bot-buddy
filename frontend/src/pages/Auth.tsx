import { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { 
  TrendingUp,
  Shield,
  AlertTriangle,
  CheckCircle2,
  Wifi,
  WifiOff,
  Loader2,
  LogIn,
  Users,
  ChevronDown,
  ChevronUp,
  Key,
  Eye,
  EyeOff
} from 'lucide-react';
import { useAuth } from '@/hooks/useAuth';
import { apiService } from '@/services/api';

interface DerivAccount {
  loginid: string;
  currency: string;
  account_type: string;
  is_virtual: number;
  token: string;
}

export default function Auth() {
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [accounts, setAccounts] = useState<DerivAccount[]>([]);
  const [selectedAccount, setSelectedAccount] = useState<string>('');
  const [isProcessingOAuth, setIsProcessingOAuth] = useState(false);
  const [showManualLogin, setShowManualLogin] = useState(false);
  const [apiKey, setApiKey] = useState('');
  const [showKey, setShowKey] = useState(false);
  
  const { login, isValidating, isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();

  // Check backend status on mount
  useEffect(() => {
    const checkBackendStatus = async () => {
      try {
        const isOnline = await apiService.isAvailable();
        setBackendStatus(isOnline ? 'online' : 'offline');
      } catch {
        setBackendStatus('offline');
      }
    };

    checkBackendStatus();
    // Check every 30 seconds
    const interval = setInterval(checkBackendStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  // Process OAuth callback parameters on mount
  useEffect(() => {
    const processOAuthCallback = async () => {
      // Check if we have OAuth callback parameters
      const oAuthAccounts: DerivAccount[] = [];
      let accountIndex = 1;
      
      while (searchParams.has(`acct${accountIndex}`)) {
        const loginid = searchParams.get(`acct${accountIndex}`);
        const token = searchParams.get(`token${accountIndex}`);
        const currency = searchParams.get(`cur${accountIndex}`) || 'USD';
        
        if (loginid && token) {
          // Determine account type and if it's virtual
          const isVirtual = loginid.toLowerCase().includes('vrt') || loginid.toLowerCase().includes('virtual') ? 1 : 0;
          const accountType = loginid.toLowerCase().includes('cr') ? 'trading' : 'wallet';
          
          oAuthAccounts.push({
            loginid,
            token,
            currency: currency.toUpperCase(),
            account_type: accountType,
            is_virtual: isVirtual
          });
        }
        accountIndex++;
      }
      
      if (oAuthAccounts.length > 0) {
        setAccounts(oAuthAccounts);
        setIsProcessingOAuth(true);
        
        // Clean the URL
        window.history.replaceState({}, document.title, window.location.pathname);
      }
    };
    
    processOAuthCallback();
  }, [searchParams]);

  // Redirect if already authenticated
  useEffect(() => {
    if (isAuthenticated) {
      navigate('/dashboard');
    }
  }, [isAuthenticated, navigate]);

  const handleOAuthLogin = () => {
    const APP_ID = import.meta.env.VITE_DERIV_APP_ID || '99188';
    const oAuthUrl = `https://oauth.deriv.com/oauth2/authorize?app_id=${APP_ID}`;
    window.location.href = oAuthUrl;
  };

  const handleAccountSelect = async () => {
    if (!selectedAccount) return;
    
    const account = accounts.find(acc => acc.loginid === selectedAccount);
    if (!account) return;

    const success = await login(account.token);
    if (success) {
      navigate('/dashboard');
    }
  };

  const handleManualLogin = async () => {
    if (!apiKey.trim()) return;

    const success = await login(apiKey);
    if (success) {
      navigate('/dashboard');
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !isValidating) {
      handleManualLogin();
    }
  };

  const getStatusInfo = () => {
    switch (backendStatus) {
      case 'checking':
        return { icon: Loader2, text: 'Verificando servidor...', color: 'text-muted-foreground', spin: true };
      case 'online':
        return { icon: Wifi, text: 'Servidor online', color: 'text-success', spin: false };
      case 'offline':
        return { icon: WifiOff, text: 'Servidor offline', color: 'text-danger', spin: false };
    }
  };

  const statusInfo = getStatusInfo();
  const StatusIcon = statusInfo.icon;

  // If we have accounts from OAuth, show account selection
  if (accounts.length > 0 && isProcessingOAuth) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center p-4">
        <div className="w-full max-w-md space-y-6">
          {/* Logo and Title */}
          <div className="text-center space-y-2">
            <div className="flex items-center justify-center space-x-2">
              <TrendingUp className="h-8 w-8 text-primary" />
              <span className="text-2xl font-bold">BotDeriv</span>
            </div>
            <p className="text-muted-foreground">
              Selecione a conta para trading
            </p>
          </div>

          {/* Account Selection Card */}
          <Card className="trading-card">
            <CardHeader>
              <CardTitle className="flex items-center">
                <Users className="h-5 w-5 mr-2 text-primary" />
                Suas Contas Deriv
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                {accounts.map((account) => (
                  <div 
                    key={account.loginid} 
                    className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                      selectedAccount === account.loginid 
                        ? 'border-primary bg-primary/5' 
                        : 'border-border hover:bg-muted/30'
                    }`}
                    onClick={() => setSelectedAccount(account.loginid)}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="flex items-center space-x-2">
                          <span className="font-medium">{account.loginid}</span>
                          <Badge variant={account.is_virtual ? "secondary" : "default"}>
                            {account.is_virtual ? "Demo" : "Real"}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground">
                          {account.currency} • {account.account_type === 'trading' ? 'Conta de Trading' : 'Carteira'}
                        </p>
                      </div>
                      {selectedAccount === account.loginid && (
                        <CheckCircle2 className="h-5 w-5 text-primary" />
                      )}
                    </div>
                  </div>
                ))}
              </div>

              <Button 
                onClick={handleAccountSelect}
                disabled={!selectedAccount || isValidating}
                className="w-full success-gradient hover:opacity-90"
              >
                {isValidating ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Conectando...
                  </>
                ) : (
                  <>
                    <CheckCircle2 className="h-4 w-4 mr-2" />
                    Conectar com {selectedAccount}
                  </>
                )}
              </Button>

              <Alert>
                <Shield className="h-4 w-4" />
                <AlertDescription className="text-sm">
                  <strong>OAuth Seguro:</strong> Autenticação segura através da Deriv, sem exposição de tokens permanentes.
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="w-full max-w-md space-y-6">
        {/* Logo and Title */}
        <div className="text-center space-y-2">
          <div className="flex items-center justify-center space-x-2">
            <TrendingUp className="h-8 w-8 text-primary" />
            <span className="text-2xl font-bold">BotDeriv</span>
          </div>
          <p className="text-muted-foreground">
            Sistema de trading com gestão inteligente de capital
          </p>
          
          {/* Server Status */}
          <div className="flex items-center justify-center space-x-2 pt-2">
            <StatusIcon className={`h-4 w-4 ${statusInfo.color} ${statusInfo.spin ? 'animate-spin' : ''}`} />
            <span className={`text-sm ${statusInfo.color}`}>
              {statusInfo.text}
            </span>
          </div>
        </div>

        {/* OAuth Authentication Card */}
        <Card className="trading-card">
          <CardHeader>
            <CardTitle className="flex items-center">
              <LogIn className="h-5 w-5 mr-2 text-primary" />
              Login Seguro via Deriv
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="text-center space-y-4">
              <p className="text-sm text-muted-foreground">
                Faça login de forma segura usando sua conta Deriv oficial. 
                Não é necessário gerar tokens manualmente.
              </p>

              <Button 
                onClick={handleOAuthLogin}
                disabled={backendStatus === 'offline'}
                className="w-full success-gradient hover:opacity-90"
                size="lg"
              >
                <LogIn className="h-5 w-5 mr-2" />
                Fazer Login com Deriv
              </Button>
            </div>

            {/* Status Information */}
            {backendStatus === 'offline' && (
              <Alert>
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription className="text-sm">
                  <strong>Servidor offline:</strong> Aguarde o servidor ficar online para fazer login.
                </AlertDescription>
              </Alert>
            )}

            {/* Security Notice */}
            <Alert>
              <Shield className="h-4 w-4" />
              <AlertDescription className="text-sm">
                <strong>OAuth Seguro:</strong> Login oficial da Deriv usando protocolo OAuth. 
                Seus dados permanecem seguros e privados.
              </AlertDescription>
            </Alert>

            {/* OAuth Process Explanation */}
            <div className="pt-4 border-t border-border">
              <h4 className="text-sm font-medium mb-3">🔐 Como funciona o login OAuth:</h4>
              <div className="text-sm text-muted-foreground space-y-2 mb-4">
                <div className="flex items-start space-x-2">
                  <span className="text-primary font-medium">1.</span>
                  <span>Você será redirecionado para a Deriv oficial</span>
                </div>
                <div className="flex items-start space-x-2">
                  <span className="text-primary font-medium">2.</span>
                  <span>Faça login com suas credenciais da Deriv</span>
                </div>
                <div className="flex items-start space-x-2">
                  <span className="text-primary font-medium">3.</span>
                  <span>Autorize o acesso às suas contas</span>
                </div>
                <div className="flex items-start space-x-2">
                  <span className="text-primary font-medium">4.</span>
                  <span>Retorne automaticamente e selecione sua conta</span>
                </div>
              </div>
              
              <h4 className="text-sm font-medium mb-3">Vantagens do OAuth:</h4>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li className="flex items-center">
                  <CheckCircle2 className="h-4 w-4 text-success mr-2" />
                  Maior segurança (sem tokens permanentes)
                </li>
                <li className="flex items-center">
                  <CheckCircle2 className="h-4 w-4 text-success mr-2" />
                  Processo automatizado
                </li>
                <li className="flex items-center">
                  <CheckCircle2 className="h-4 w-4 text-success mr-2" />
                  Suporte a múltiplas contas
                </li>
                <li className="flex items-center">
                  <CheckCircle2 className="h-4 w-4 text-success mr-2" />
                  Recomendado oficialmente pela Deriv
                </li>
              </ul>
            </div>

            {/* Manual Login Option (Collapsible) */}
            <Collapsible open={showManualLogin} onOpenChange={setShowManualLogin}>
              <CollapsibleTrigger asChild>
                <Button variant="ghost" className="w-full justify-between" size="sm">
                  <span className="flex items-center">
                    <Key className="h-4 w-4 mr-2" />
                    Login Manual (Avançado)
                  </span>
                  {showManualLogin ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="space-y-4 pt-4">
                <Alert>
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription className="text-sm">
                    <strong>Método Legacy:</strong> Use apenas se tiver problemas com OAuth. 
                    O método OAuth é mais seguro e recomendado.
                  </AlertDescription>
                </Alert>
                
                <div className="space-y-2">
                  <Label htmlFor="apiKey">Token de API da Deriv</Label>
                  <div className="relative">
                    <Input
                      id="apiKey"
                      type={showKey ? "text" : "password"}
                      value={apiKey}
                      onChange={(e) => setApiKey(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder="Cole seu token aqui..."
                      className="pr-10"
                      disabled={isValidating}
                    />
                    <button
                      type="button"
                      onClick={() => setShowKey(!showKey)}
                      className="absolute right-3 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground"
                    >
                      {showKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </button>
                  </div>
                </div>

                <Button 
                  onClick={handleManualLogin}
                  disabled={isValidating || !apiKey.trim()}
                  className="w-full"
                  variant="outline"
                >
                  {isValidating ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Validando...
                    </>
                  ) : (
                    <>
                      <Key className="h-4 w-4 mr-2" />
                      Login com Token Manual
                    </>
                  )}
                </Button>
              </CollapsibleContent>
            </Collapsible>
          </CardContent>
        </Card>

        {/* Token Help */}
        <Card>
          <CardContent className="pt-6">
            <div className="text-center space-y-4">
              <h3 className="text-lg font-semibold">🔑 Precisa de um Token da Deriv?</h3>
              <p className="text-sm text-muted-foreground">
                Para usar nosso bot, você precisa de um token API da Deriv com permissões de trading.
              </p>
              <div className="flex flex-col gap-2">
                <Button 
                  variant="outline" 
                  onClick={() => window.open('https://app.deriv.com/account/api-token', '_blank')}
                  className="w-full"
                >
                  <Key className="h-4 w-4 mr-2" />
                  Gerar Token da Deriv (Real)
                </Button>
                <Button 
                  variant="outline" 
                  onClick={() => window.open('https://app.deriv.com/account/api-token', '_blank')}
                  className="w-full"
                >
                  <TrendingUp className="h-4 w-4 mr-2" />
                  Criar Conta Demo Deriv
                </Button>
              </div>
              <div className="text-xs text-muted-foreground space-y-1">
                <p>✅ Selecione escopo "Admin" para funcionalidade completa</p>
                <p>✅ Use conta demo para testes seguros</p>
                <p>⚠️ Mantenha seu token sempre seguro</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Footer */}
        <p className="text-center text-sm text-muted-foreground">
          Ao conectar, você concorda com nossos termos de uso e política de privacidade
        </p>
      </div>
    </div>
  );
}