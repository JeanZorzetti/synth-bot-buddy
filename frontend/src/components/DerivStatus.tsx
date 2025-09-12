import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  Activity, 
  CheckCircle, 
  XCircle, 
  AlertCircle, 
  RefreshCw,
  ExternalLink,
  Eye
} from 'lucide-react';
import apiService from '@/services/api';

interface DerivStatusData {
  is_connected: boolean;
  is_authenticated: boolean;
  balance?: number;
  loginid?: string;
  api_status?: string;
  subscribed_symbols?: string[];
}

const DerivStatus: React.FC = () => {
  const [status, setStatus] = useState<DerivStatusData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);

  useEffect(() => {
    checkStatus();
    
    // Check status every 30 seconds
    const interval = setInterval(checkStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const checkStatus = async () => {
    setIsLoading(true);
    try {
      const response = await apiService.derivGetStatus();
      if (response.status === 'success' || response.status === 'development') {
        setStatus(response.connection_info);
      }
      setLastChecked(new Date());
    } catch (error: any) {
      if (error.message === 'DERIV_ENDPOINTS_NOT_AVAILABLE') {
        setStatus({
          is_connected: false,
          is_authenticated: false,
          api_status: 'endpoints_not_available'
        });
      } else {
        setStatus({
          is_connected: false,
          is_authenticated: false
        });
      }
      setLastChecked(new Date());
    } finally {
      setIsLoading(false);
    }
  };

  const formatCurrency = (amount?: number) => {
    if (amount === undefined) return '--';
    return new Intl.NumberFormat('pt-BR', {
      style: 'currency',
      currency: 'USD'
    }).format(amount);
  };

  const getStatusBadge = () => {
    if (!status) return null;

    // Verificar se os endpoints estão indisponíveis
    if (status.api_status === 'endpoints_not_available') {
      return (
        <Badge className="bg-blue-100 text-blue-800 border-blue-200">
          <AlertCircle className="h-3 w-3 mr-1" />
          Em Desenvolvimento
        </Badge>
      );
    }

    if (status.is_authenticated) {
      return (
        <Badge className="bg-green-100 text-green-800 border-green-200">
          <CheckCircle className="h-3 w-3 mr-1" />
          Conectado
        </Badge>
      );
    }

    if (status.is_connected) {
      return (
        <Badge className="bg-yellow-100 text-yellow-800 border-yellow-200">
          <AlertCircle className="h-3 w-3 mr-1" />
          Conectado (Não Auth)
        </Badge>
      );
    }

    return (
      <Badge className="bg-red-100 text-red-800 border-red-200">
        <XCircle className="h-3 w-3 mr-1" />
        Desconectado
      </Badge>
    );
  };

  const getConnectionIcon = () => {
    if (!status) return <Activity className="h-4 w-4 text-gray-400" />;
    
    if (status.is_authenticated) {
      return <Activity className="h-4 w-4 text-green-600" />;
    }
    
    if (status.is_connected) {
      return <Activity className="h-4 w-4 text-yellow-600" />;
    }
    
    return <Activity className="h-4 w-4 text-red-600" />;
  };

  return (
    <Card className="w-full">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium flex items-center gap-2">
          {getConnectionIcon()}
          Status Deriv API
        </CardTitle>
        <div className="flex items-center gap-2">
          {getStatusBadge()}
          <Button
            variant="ghost"
            size="sm"
            onClick={checkStatus}
            disabled={isLoading}
            className="h-8 w-8 p-0"
          >
            <RefreshCw className={`h-3 w-3 ${isLoading ? 'animate-spin' : ''}`} />
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {status?.is_authenticated ? (
            <>
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Login ID:</span>
                <span className="font-mono">{status.loginid}</span>
              </div>
              
              {status.balance !== undefined && (
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Saldo:</span>
                  <span className="font-semibold">{formatCurrency(status.balance)}</span>
                </div>
              )}
              
              {status.subscribed_symbols && status.subscribed_symbols.length > 0 && (
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Símbolos:</span>
                  <span className="text-xs">{status.subscribed_symbols.length} ativos</span>
                </div>
              )}
              
              <Button 
                variant="outline" 
                size="sm" 
                className="w-full mt-3"
                onClick={() => window.location.href = '/trading'}
              >
                <ExternalLink className="h-3 w-3 mr-2" />
                Abrir Trading
              </Button>
            </>
          ) : (
            <div className="text-center py-4">
              {status?.api_status === 'endpoints_not_available' ? (
                <>
                  <div className="text-sm text-muted-foreground mb-3">
                    🚧 Funcionalidade em desenvolvimento
                  </div>
                  <div className="text-xs text-muted-foreground mb-3">
                    Os endpoints da Deriv API estão sendo deployados no servidor
                  </div>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => window.location.href = '/trading'}
                    className="w-full"
                  >
                    <Eye className="h-3 w-3 mr-2" />
                    Ver Interface (Preview)
                  </Button>
                </>
              ) : (
                <>
                  <div className="text-sm text-muted-foreground mb-3">
                    Conecte-se à API Deriv para trading real
                  </div>
                  <Button 
                    variant="default" 
                    size="sm"
                    onClick={() => window.location.href = '/trading'}
                    className="w-full"
                  >
                    <Activity className="h-3 w-3 mr-2" />
                    Conectar Deriv API
                  </Button>
                </>
              )}
            </div>
          )}
          
          {lastChecked && (
            <div className="text-xs text-muted-foreground text-center mt-2">
              Última verificação: {lastChecked.toLocaleTimeString('pt-BR')}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default DerivStatus;