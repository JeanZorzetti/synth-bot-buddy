import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { 
  TrendingUp,
  Key,
  Shield,
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  Eye,
  EyeOff,
  Wifi,
  WifiOff,
  Loader2
} from 'lucide-react';
import { useAuth } from '@/hooks/useAuth';
import { apiService } from '@/services/api';

export default function Auth() {
  const [apiKey, setApiKey] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  
  const { login, isValidating, isAuthenticated } = useAuth();
  const navigate = useNavigate();

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

  // Redirect if already authenticated
  useEffect(() => {
    if (isAuthenticated) {
      navigate('/dashboard');
    }
  }, [isAuthenticated, navigate]);

  const handleValidateKey = async () => {
    if (!apiKey.trim()) {
      return;
    }

    const success = await login(apiKey);
    if (success) {
      navigate('/dashboard');
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !isValidating) {
      handleValidateKey();
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

  const ApiGuideContent = () => (
    <div className="space-y-4 max-h-96 overflow-y-auto">
      <div className="space-y-3">
        <h4 className="font-semibold">Passo 1: Acesse sua conta Deriv</h4>
        <p className="text-sm text-muted-foreground">
          Faça login em sua conta na plataforma Deriv (app.deriv.com)
        </p>
      </div>

      <div className="space-y-3">
        <h4 className="font-semibold">Passo 2: Navegue para API Token</h4>
        <p className="text-sm text-muted-foreground">
          Vá para "Settings" → "API Token" no menu da sua conta
        </p>
      </div>

      <div className="space-y-3">
        <h4 className="font-semibold">Passo 3: Crie um novo token</h4>
        <p className="text-sm text-muted-foreground">
          Clique em "Create" e configure os escopos necessários:
        </p>
        <ul className="text-sm text-muted-foreground space-y-1 ml-4">
          <li>• <strong>Read:</strong> Para consultar informações da conta</li>
          <li>• <strong>Trade:</strong> Para executar operações</li>
          <li>• <strong>Payments:</strong> Para acessar histórico financeiro</li>
          <li>• <strong>Trading Information:</strong> Para dados de mercado</li>
        </ul>
      </div>

      <div className="space-y-3">
        <h4 className="font-semibold">Passo 4: Copie o token</h4>
        <p className="text-sm text-muted-foreground">
          Após criar, copie o token gerado e cole no campo acima
        </p>
      </div>

      <Alert>
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription className="text-sm">
          <strong>Importante:</strong> Nunca compartilhe seu token de API com terceiros. 
          Este aplicativo armazena o token de forma segura apenas no seu dispositivo.
        </AlertDescription>
      </Alert>
    </div>
  );

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

        {/* Authentication Card */}
        <Card className="trading-card">
          <CardHeader>
            <CardTitle className="flex items-center">
              <Key className="h-5 w-5 mr-2 text-primary" />
              Conectar à Deriv
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="apiKey">Chave de API da Deriv</Label>
              <div className="relative">
                <Input
                  id="apiKey"
                  type={showKey ? "text" : "password"}
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Cole seu token aqui... (ex: a1-Abc123...)"
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
              
              <div className="flex items-center space-x-2">
                <Dialog>
                  <DialogTrigger asChild>
                    <Button variant="link" size="sm" className="p-0 h-auto">
                      <HelpCircle className="h-4 w-4 mr-1" />
                      Como obter minha chave de API?
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="max-w-lg">
                    <DialogHeader>
                      <DialogTitle>Como obter sua chave de API da Deriv</DialogTitle>
                    </DialogHeader>
                    <ApiGuideContent />
                  </DialogContent>
                </Dialog>
              </div>
            </div>

            <Button 
              onClick={handleValidateKey}
              disabled={isValidating || !apiKey.trim()}
              className="w-full success-gradient hover:opacity-90"
            >
              {isValidating ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Validando token...
                </>
              ) : (
                <>
                  <CheckCircle2 className="h-4 w-4 mr-2" />
                  {backendStatus === 'offline' ? 'Salvar Token (Servidor Offline)' : 'Validar e Conectar'}
                </>
              )}
            </Button>

            {/* Status Information */}
            {backendStatus === 'offline' && (
              <Alert>
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription className="text-sm">
                  <strong>Servidor offline:</strong> O token será salvo localmente e validado 
                  quando o servidor estiver disponível.
                </AlertDescription>
              </Alert>
            )}

            {/* Security Notice */}
            <Alert>
              <Shield className="h-4 w-4" />
              <AlertDescription className="text-sm">
                <strong>Segurança Garantida:</strong> Sua chave de API é armazenada de forma 
                criptografada apenas no seu dispositivo e nunca é compartilhada com terceiros.
              </AlertDescription>
            </Alert>

            {/* How to Get Token */}
            <div className="pt-4 border-t border-border">
              <h4 className="text-sm font-medium mb-3">🔑 Como obter seu token da API:</h4>
              <div className="text-sm text-muted-foreground space-y-2 mb-4">
                <div className="flex items-start space-x-2">
                  <span className="text-primary font-medium">1.</span>
                  <span>Acesse <strong>app.deriv.com</strong></span>
                </div>
                <div className="flex items-start space-x-2">
                  <span className="text-primary font-medium">2.</span>
                  <span>Settings → API Token</span>
                </div>
                <div className="flex items-start space-x-2">
                  <span className="text-primary font-medium">3.</span>
                  <span>Crie token com escopos: <strong>Read, Trade, Payments</strong></span>
                </div>
                <div className="flex items-start space-x-2">
                  <span className="text-primary font-medium">4.</span>
                  <span>Copie o token (começa com <code className="bg-muted px-1 rounded">a1-</code>)</span>
                </div>
              </div>
              
              <h4 className="text-sm font-medium mb-3">Funcionalidades do Sistema:</h4>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li className="flex items-center">
                  <CheckCircle2 className="h-4 w-4 text-success mr-2" />
                  Gestão inteligente de capital
                </li>
                <li className="flex items-center">
                  <CheckCircle2 className="h-4 w-4 text-success mr-2" />
                  Reinvestimento progressivo (20%)
                </li>
                <li className="flex items-center">
                  <CheckCircle2 className="h-4 w-4 text-success mr-2" />
                  Martingale controlado (1.25x)
                </li>
                <li className="flex items-center">
                  <CheckCircle2 className="h-4 w-4 text-success mr-2" />
                  Monitoramento em tempo real
                </li>
              </ul>
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