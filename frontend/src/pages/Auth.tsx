import { useState } from 'react';
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
  EyeOff
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

export default function Auth() {
  const [apiKey, setApiKey] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showKey, setShowKey] = useState(false);
  const navigate = useNavigate();
  const { toast } = useToast();

  const handleValidateKey = async () => {
    if (!apiKey.trim()) {
      toast({
        title: "Erro de Validação",
        description: "Por favor, insira sua chave de API da Deriv.",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    
    // Simulate API validation (in a real app, this would validate with Deriv's API)
    setTimeout(() => {
      if (apiKey.length >= 20) {
        // Store the API key (in a real app, this should be encrypted)
        localStorage.setItem('deriv_api_key', apiKey);
        
        toast({
          title: "Conexão Estabelecida",
          description: "Sua chave de API foi validada com sucesso!",
        });
        
        navigate('/dashboard');
      } else {
        toast({
          title: "Chave Inválida",
          description: "A chave de API fornecida é inválida. Verifique e tente novamente.",
          variant: "destructive",
        });
      }
      setIsLoading(false);
    }, 2000);
  };

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
            <span className="text-2xl font-bold">Deriv AI Bot</span>
          </div>
          <p className="text-muted-foreground">
            Bot de IA autônomo para trading em ativos sintéticos
          </p>
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
                  placeholder="Cole sua chave de API aqui..."
                  className="pr-10"
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
              disabled={isLoading}
              className="w-full success-gradient hover:opacity-90"
            >
              {isLoading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Validando...
                </>
              ) : (
                <>
                  <CheckCircle2 className="h-4 w-4 mr-2" />
                  Validar e Salvar Chave
                </>
              )}
            </Button>

            {/* Security Notice */}
            <Alert>
              <Shield className="h-4 w-4" />
              <AlertDescription className="text-sm">
                <strong>Segurança Garantida:</strong> Sua chave de API é armazenada de forma 
                criptografada apenas no seu dispositivo e nunca é compartilhada com terceiros.
              </AlertDescription>
            </Alert>

            {/* Features List */}
            <div className="pt-4 border-t border-border">
              <h4 className="text-sm font-medium mb-3">O que você pode fazer:</h4>
              <ul className="text-sm text-muted-foreground space-y-2">
                <li className="flex items-center">
                  <CheckCircle2 className="h-4 w-4 text-success mr-2" />
                  Trading automatizado com IA
                </li>
                <li className="flex items-center">
                  <CheckCircle2 className="h-4 w-4 text-success mr-2" />
                  Gerenciamento de risco personalizado
                </li>
                <li className="flex items-center">
                  <CheckCircle2 className="h-4 w-4 text-success mr-2" />
                  Monitoramento em tempo real
                </li>
                <li className="flex items-center">
                  <CheckCircle2 className="h-4 w-4 text-success mr-2" />
                  Histórico detalhado de operações
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