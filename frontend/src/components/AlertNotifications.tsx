import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { AlertCircle, CheckCircle2, Info, XCircle, Bell, BellOff } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface Alert {
  id: string;
  level: 'CRITICAL' | 'WARNING' | 'INFO';
  type: string;
  message: string;
  details: Record<string, any>;
  timestamp: string;
  read: boolean;
}

interface AlertNotificationsProps {
  apiBaseUrl: string;
  isRunning: boolean;
  pollInterval?: number; // ms
}

export function AlertNotifications({
  apiBaseUrl,
  isRunning,
  pollInterval = 10000 // 10 segundos por padrão
}: AlertNotificationsProps) {
  const { toast } = useToast();
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [unreadCount, setUnreadCount] = useState(0);
  const [showAll, setShowAll] = useState(false);
  const [lastAlertId, setLastAlertId] = useState<string | null>(null);

  // Buscar alertas
  const loadAlerts = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/api/forward-testing/alerts?limit=20`);
      const data = await response.json();

      if (data.status === 'success') {
        const newAlerts = data.data as Alert[];
        setAlerts(newAlerts);

        // Contar não lidos
        const unreadAlerts = newAlerts.filter(a => !a.read);
        setUnreadCount(unreadAlerts.length);

        // Verificar se há novos alertas CRITICAL ou WARNING
        if (newAlerts.length > 0 && lastAlertId !== newAlerts[0].id) {
          const latestAlert = newAlerts[0];

          // Mostrar toast apenas para CRITICAL e WARNING não lidos
          if (!latestAlert.read && (latestAlert.level === 'CRITICAL' || latestAlert.level === 'WARNING')) {
            showToastForAlert(latestAlert);
          }

          setLastAlertId(latestAlert.id);
        }
      }
    } catch (error) {
      console.error('Erro ao carregar alertas:', error);
    }
  };

  // Mostrar toast para alerta
  const showToastForAlert = (alert: Alert) => {
    const icon = getAlertIcon(alert.level);
    const variant = alert.level === 'CRITICAL' ? 'destructive' : 'default';

    toast({
      title: alert.message,
      description: formatAlertDetails(alert),
      variant,
      duration: alert.level === 'CRITICAL' ? 10000 : 5000, // CRITICAL fica 10s
    });
  };

  // Formatar detalhes do alerta
  const formatAlertDetails = (alert: Alert): string => {
    const details = alert.details;

    switch (alert.type) {
      case 'HIGH_DRAWDOWN':
        return `Capital: $${details.current_capital?.toLocaleString()} (Peak: $${details.peak_capital?.toLocaleString()})`;

      case 'CONSECUTIVE_LOSSES':
        return `Último prejuízo: $${Math.abs(details.last_loss || 0).toFixed(2)}`;

      case 'LOW_WIN_RATE':
        return `Win Rate atual: ${details.win_rate_pct?.toFixed(1)}% (${details.total_trades} trades)`;

      case 'HIGH_TIMEOUT_RATE':
        return `${details.timeout_rate_pct?.toFixed(1)}% dos trades fecharam por timeout`;

      case 'HIGH_SL_HIT_RATE':
        return `${details.sl_hit_rate_pct?.toFixed(1)}% dos trades atingiram Stop Loss`;

      case 'CAPITAL_RECORD':
        return `Novo recorde: $${details.new_peak?.toLocaleString()}`;

      case 'PROFIT_MILESTONE':
        return `Capital atual: $${details.current_capital?.toLocaleString()} (+${details.profit_pct?.toFixed(1)}%)`;

      case 'TAKE_PROFIT_HIT':
        return `Trade ID: ${details.trade_id?.slice(-8)} | Lucro: $${details.profit_loss?.toFixed(2)}`;

      default:
        return new Date(alert.timestamp).toLocaleString('pt-BR');
    }
  };

  // Marcar alerta como lido
  const markAsRead = async (alertId: string) => {
    try {
      await fetch(`${apiBaseUrl}/api/forward-testing/alerts/${alertId}/mark-read`, {
        method: 'POST',
      });

      // Atualizar localmente
      setAlerts(prev => prev.map(a =>
        a.id === alertId ? { ...a, read: true } : a
      ));
      setUnreadCount(prev => Math.max(0, prev - 1));
    } catch (error) {
      console.error('Erro ao marcar alerta como lido:', error);
    }
  };

  // Marcar todos como lidos
  const markAllAsRead = async () => {
    try {
      await fetch(`${apiBaseUrl}/api/forward-testing/alerts/mark-all-read`, {
        method: 'POST',
      });

      // Atualizar localmente
      setAlerts(prev => prev.map(a => ({ ...a, read: true })));
      setUnreadCount(0);

      toast({
        title: 'Alertas Marcados',
        description: 'Todos os alertas foram marcados como lidos',
      });
    } catch (error) {
      console.error('Erro ao marcar todos como lidos:', error);
    }
  };

  // Ícone baseado no nível
  const getAlertIcon = (level: string) => {
    switch (level) {
      case 'CRITICAL':
        return <XCircle className="h-4 w-4 text-red-600" />;
      case 'WARNING':
        return <AlertCircle className="h-4 w-4 text-yellow-600" />;
      case 'INFO':
        return <Info className="h-4 w-4 text-blue-600" />;
      default:
        return <Bell className="h-4 w-4" />;
    }
  };

  // Badge variant baseado no nível
  const getAlertBadgeVariant = (level: string): 'default' | 'destructive' | 'secondary' => {
    switch (level) {
      case 'CRITICAL':
        return 'destructive';
      case 'WARNING':
        return 'default';
      case 'INFO':
        return 'secondary';
      default:
        return 'secondary';
    }
  };

  // Polling quando sistema está rodando
  useEffect(() => {
    if (!isRunning) {
      setAlerts([]);
      setUnreadCount(0);
      setLastAlertId(null);
      return;
    }

    loadAlerts(); // Carregar imediatamente
    const interval = setInterval(loadAlerts, pollInterval);
    return () => clearInterval(interval);
  }, [isRunning, pollInterval]);

  // Não exibir se não estiver rodando
  if (!isRunning) {
    return null;
  }

  // Filtrar alertas baseado em showAll
  const displayedAlerts = showAll ? alerts : alerts.filter(a => !a.read).slice(0, 5);

  return (
    <Card className="border-blue-200">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Bell className="h-5 w-5 text-blue-600" />
            <CardTitle className="text-blue-900">
              Alertas do Sistema
              {unreadCount > 0 && (
                <Badge variant="destructive" className="ml-2">
                  {unreadCount}
                </Badge>
              )}
            </CardTitle>
          </div>

          <div className="flex gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowAll(!showAll)}
            >
              {showAll ? <BellOff className="h-4 w-4 mr-1" /> : <Bell className="h-4 w-4 mr-1" />}
              {showAll ? 'Mostrar Não Lidos' : 'Mostrar Todos'}
            </Button>

            {unreadCount > 0 && (
              <Button
                variant="outline"
                size="sm"
                onClick={markAllAsRead}
              >
                <CheckCircle2 className="h-4 w-4 mr-1" />
                Marcar Todos
              </Button>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent>
        {displayedAlerts.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <Bell className="h-12 w-12 mx-auto mb-3 opacity-30" />
            <p className="text-sm">
              {showAll ? 'Nenhum alerta gerado ainda' : 'Nenhum alerta não lido'}
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {displayedAlerts.map((alert) => (
              <div
                key={alert.id}
                className={`p-4 rounded-lg border-2 transition-all ${
                  alert.read
                    ? 'border-gray-200 bg-gray-50 opacity-60'
                    : alert.level === 'CRITICAL'
                    ? 'border-red-300 bg-red-50'
                    : alert.level === 'WARNING'
                    ? 'border-yellow-300 bg-yellow-50'
                    : 'border-blue-200 bg-blue-50'
                }`}
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {getAlertIcon(alert.level)}
                    <Badge variant={getAlertBadgeVariant(alert.level)}>
                      {alert.level}
                    </Badge>
                    {!alert.read && (
                      <Badge variant="outline" className="text-xs">
                        Novo
                      </Badge>
                    )}
                  </div>

                  <span className="text-xs text-muted-foreground">
                    {new Date(alert.timestamp).toLocaleTimeString('pt-BR')}
                  </span>
                </div>

                <p className={`font-semibold mb-1 ${
                  alert.level === 'CRITICAL' ? 'text-red-900' :
                  alert.level === 'WARNING' ? 'text-yellow-900' :
                  'text-blue-900'
                }`}>
                  {alert.message}
                </p>

                <p className="text-sm text-muted-foreground mb-3">
                  {formatAlertDetails(alert)}
                </p>

                {!alert.read && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => markAsRead(alert.id)}
                    className="w-full"
                  >
                    <CheckCircle2 className="h-3 w-3 mr-1" />
                    Marcar como lido
                  </Button>
                )}
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
