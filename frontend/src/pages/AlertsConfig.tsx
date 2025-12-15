import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Separator } from '@/components/ui/separator';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Bell,
  Mail,
  MessageSquare,
  Send,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  Info,
  Shield,
  Save,
  TestTube,
  History as HistoryIcon,
} from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import {
  alertsApi,
  type AlertConfig,
  type AlertHistoryItem,
} from '@/services/api';
import { format } from 'date-fns';

export default function AlertsConfig() {
  const { toast } = useToast();

  // Data states
  const [config, setConfig] = useState<AlertConfig | null>(null);
  const [history, setHistory] = useState<AlertHistoryItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [isTesting, setIsTesting] = useState<string | null>(null);

  // Form states - Discord
  const [discordWebhook, setDiscordWebhook] = useState('');
  const [discordEnabled, setDiscordEnabled] = useState(false);

  // Form states - Telegram
  const [telegramToken, setTelegramToken] = useState('');
  const [telegramChatId, setTelegramChatId] = useState('');
  const [telegramEnabled, setTelegramEnabled] = useState(false);

  // Form states - Email
  const [emailSmtpServer, setEmailSmtpServer] = useState('');
  const [emailSmtpPort, setEmailSmtpPort] = useState('587');
  const [emailUsername, setEmailUsername] = useState('');
  const [emailPassword, setEmailPassword] = useState('');
  const [emailFrom, setEmailFrom] = useState('');
  const [emailTo, setEmailTo] = useState('');
  const [emailEnabled, setEmailEnabled] = useState(false);

  // General settings
  const [minLevel, setMinLevel] = useState('WARNING');

  // Test alert states
  const [testChannel, setTestChannel] = useState('discord');
  const [testTitle, setTestTitle] = useState('Teste de Alerta');
  const [testMessage, setTestMessage] = useState('Este é um alerta de teste do Deriv Bot Buddy.');

  useEffect(() => {
    loadConfig();
    loadHistory();
  }, []);

  const loadConfig = async () => {
    try {
      setIsLoading(true);
      const response = await alertsApi.getConfig();
      const cfg = response.config;

      setConfig(cfg);

      // Populate form fields
      setDiscordWebhook(cfg.discord.webhook_url || '');
      setDiscordEnabled(cfg.discord.enabled);

      setTelegramToken(cfg.telegram.bot_token || '');
      setTelegramChatId(cfg.telegram.chat_id || '');
      setTelegramEnabled(cfg.telegram.enabled);

      setEmailSmtpServer(cfg.email.smtp_server || '');
      setEmailSmtpPort(String(cfg.email.smtp_port || 587));
      setEmailUsername(cfg.email.smtp_username || '');
      setEmailPassword(cfg.email.smtp_password || '');
      setEmailFrom(cfg.email.email_from || '');
      setEmailTo(cfg.email.email_to?.join(', ') || '');
      setEmailEnabled(cfg.email.enabled);

      setMinLevel(cfg.settings.min_level);
    } catch (error) {
      console.error('Erro ao carregar configuração:', error);
      toast({
        title: 'Erro ao carregar configuração',
        description: 'Não foi possível carregar a configuração de alertas.',
        variant: 'destructive',
      });
    } finally {
      setIsLoading(false);
    }
  };

  const loadHistory = async () => {
    try {
      const response = await alertsApi.getHistory();
      setHistory(response.history);
    } catch (error) {
      console.error('Erro ao carregar histórico:', error);
    }
  };

  const handleSaveConfig = async () => {
    try {
      setIsSaving(true);

      const newConfig: Partial<AlertConfig> = {
        discord: {
          enabled: discordEnabled,
          webhook_configured: Boolean(discordWebhook),
          webhook_url: discordWebhook || undefined,
        },
        telegram: {
          enabled: telegramEnabled,
          bot_configured: Boolean(telegramToken),
          bot_token: telegramToken || undefined,
          chat_id: telegramChatId || undefined,
        },
        email: {
          enabled: emailEnabled,
          smtp_server: emailSmtpServer || undefined,
          smtp_port: parseInt(emailSmtpPort) || 587,
          smtp_username: emailUsername || undefined,
          smtp_password: emailPassword || undefined,
          email_from: emailFrom || undefined,
          email_to: emailTo ? emailTo.split(',').map(e => e.trim()) : undefined,
        },
        settings: {
          enabled_channels: [],
          min_level: minLevel,
        },
      };

      await alertsApi.updateConfig(newConfig);

      toast({
        title: 'Configuração salva',
        description: 'As configurações de alertas foram atualizadas com sucesso.',
      });

      await loadConfig();
    } catch (error) {
      console.error('Erro ao salvar configuração:', error);
      toast({
        title: 'Erro ao salvar',
        description: 'Não foi possível salvar a configuração.',
        variant: 'destructive',
      });
    } finally {
      setIsSaving(false);
    }
  };

  const handleTestAlert = async (channel: string) => {
    try {
      setIsTesting(channel);

      await alertsApi.sendTest({
        channel,
        title: testTitle,
        message: testMessage,
        level: 'INFO',
      });

      toast({
        title: 'Alerta enviado',
        description: `Alerta de teste enviado via ${channel}. Verifique seu ${channel}.`,
      });

      await loadHistory();
    } catch (error) {
      console.error('Erro ao enviar teste:', error);
      toast({
        title: 'Erro ao enviar teste',
        description: `Não foi possível enviar o alerta de teste via ${channel}.`,
        variant: 'destructive',
      });
    } finally {
      setIsTesting(null);
    }
  };

  const getLevelBadge = (level: string) => {
    const variants: Record<string, any> = {
      INFO: 'outline',
      WARNING: 'secondary',
      ERROR: 'destructive',
      CRITICAL: 'destructive',
    };
    return variants[level.toUpperCase()] || 'outline';
  };

  const getLevelIcon = (level: string) => {
    switch (level.toUpperCase()) {
      case 'INFO':
        return <Info className="h-4 w-4 text-blue-500" />;
      case 'WARNING':
        return <AlertCircle className="h-4 w-4 text-yellow-500" />;
      case 'ERROR':
      case 'CRITICAL':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      default:
        return <Info className="h-4 w-4" />;
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-4 text-primary" />
          <p className="text-muted-foreground">Carregando configurações de alertas...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Configuração de Alertas</h1>
          <p className="text-muted-foreground mt-1">
            Configure notificações via Discord, Telegram e Email
          </p>
        </div>
        <Button onClick={handleSaveConfig} disabled={isSaving}>
          {isSaving ? (
            <>
              <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
              Salvando...
            </>
          ) : (
            <>
              <Save className="h-4 w-4 mr-2" />
              Salvar Configuração
            </>
          )}
        </Button>
      </div>

      {/* Status Cards */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Discord</CardTitle>
            <MessageSquare className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div>
                {config?.discord.enabled ? (
                  <Badge variant="default" className="gap-1">
                    <CheckCircle2 className="h-3 w-3" />
                    Configurado
                  </Badge>
                ) : (
                  <Badge variant="secondary">Desabilitado</Badge>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Telegram</CardTitle>
            <Send className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div>
                {config?.telegram.enabled ? (
                  <Badge variant="default" className="gap-1">
                    <CheckCircle2 className="h-3 w-3" />
                    Configurado
                  </Badge>
                ) : (
                  <Badge variant="secondary">Desabilitado</Badge>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Email</CardTitle>
            <Mail className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div>
                {config?.email.enabled ? (
                  <Badge variant="default" className="gap-1">
                    <CheckCircle2 className="h-3 w-3" />
                    Configurado
                  </Badge>
                ) : (
                  <Badge variant="secondary">Desabilitado</Badge>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Discord Configuration */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <MessageSquare className="h-5 w-5" />
              <CardTitle>Discord Webhook</CardTitle>
            </div>
            <Switch checked={discordEnabled} onCheckedChange={setDiscordEnabled} />
          </div>
          <CardDescription>
            Configure webhook do Discord para receber notificações
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="discord-webhook">Webhook URL</Label>
            <Input
              id="discord-webhook"
              type="url"
              placeholder="https://discord.com/api/webhooks/..."
              value={discordWebhook}
              onChange={(e) => setDiscordWebhook(e.target.value)}
              disabled={!discordEnabled}
            />
            <p className="text-xs text-muted-foreground">
              Crie um webhook nas configurações do seu servidor Discord
            </p>
          </div>
          <Button
            variant="outline"
            onClick={() => handleTestAlert('discord')}
            disabled={!discordEnabled || !discordWebhook || isTesting === 'discord'}
          >
            {isTesting === 'discord' ? (
              <>
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                Enviando...
              </>
            ) : (
              <>
                <TestTube className="h-4 w-4 mr-2" />
                Testar Discord
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Telegram Configuration */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Send className="h-5 w-5" />
              <CardTitle>Telegram Bot</CardTitle>
            </div>
            <Switch checked={telegramEnabled} onCheckedChange={setTelegramEnabled} />
          </div>
          <CardDescription>
            Configure bot do Telegram para receber notificações
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="telegram-token">Bot Token</Label>
              <Input
                id="telegram-token"
                type="password"
                placeholder="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
                value={telegramToken}
                onChange={(e) => setTelegramToken(e.target.value)}
                disabled={!telegramEnabled}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="telegram-chat">Chat ID</Label>
              <Input
                id="telegram-chat"
                placeholder="-1001234567890"
                value={telegramChatId}
                onChange={(e) => setTelegramChatId(e.target.value)}
                disabled={!telegramEnabled}
              />
            </div>
          </div>
          <p className="text-xs text-muted-foreground">
            Crie um bot com @BotFather e obtenha o Chat ID com @userinfobot
          </p>
          <Button
            variant="outline"
            onClick={() => handleTestAlert('telegram')}
            disabled={!telegramEnabled || !telegramToken || !telegramChatId || isTesting === 'telegram'}
          >
            {isTesting === 'telegram' ? (
              <>
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                Enviando...
              </>
            ) : (
              <>
                <TestTube className="h-4 w-4 mr-2" />
                Testar Telegram
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* Email Configuration */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Mail className="h-5 w-5" />
              <CardTitle>Email SMTP</CardTitle>
            </div>
            <Switch checked={emailEnabled} onCheckedChange={setEmailEnabled} />
          </div>
          <CardDescription>
            Configure servidor SMTP para enviar emails
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="smtp-server">Servidor SMTP</Label>
              <Input
                id="smtp-server"
                placeholder="smtp.gmail.com"
                value={emailSmtpServer}
                onChange={(e) => setEmailSmtpServer(e.target.value)}
                disabled={!emailEnabled}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="smtp-port">Porta</Label>
              <Input
                id="smtp-port"
                type="number"
                placeholder="587"
                value={emailSmtpPort}
                onChange={(e) => setEmailSmtpPort(e.target.value)}
                disabled={!emailEnabled}
              />
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="smtp-username">Username</Label>
              <Input
                id="smtp-username"
                placeholder="seu-email@gmail.com"
                value={emailUsername}
                onChange={(e) => setEmailUsername(e.target.value)}
                disabled={!emailEnabled}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="smtp-password">Password</Label>
              <Input
                id="smtp-password"
                type="password"
                placeholder="senha ou app password"
                value={emailPassword}
                onChange={(e) => setEmailPassword(e.target.value)}
                disabled={!emailEnabled}
              />
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="email-from">Email Remetente</Label>
              <Input
                id="email-from"
                type="email"
                placeholder="bot@example.com"
                value={emailFrom}
                onChange={(e) => setEmailFrom(e.target.value)}
                disabled={!emailEnabled}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="email-to">Destinatários (separados por vírgula)</Label>
              <Input
                id="email-to"
                placeholder="usuario1@email.com, usuario2@email.com"
                value={emailTo}
                onChange={(e) => setEmailTo(e.target.value)}
                disabled={!emailEnabled}
              />
            </div>
          </div>

          <p className="text-xs text-muted-foreground">
            Para Gmail, use App Password: https://myaccount.google.com/apppasswords
          </p>
          <Button
            variant="outline"
            onClick={() => handleTestAlert('email')}
            disabled={!emailEnabled || !emailSmtpServer || !emailUsername || isTesting === 'email'}
          >
            {isTesting === 'email' ? (
              <>
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                Enviando...
              </>
            ) : (
              <>
                <TestTube className="h-4 w-4 mr-2" />
                Testar Email
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      {/* General Settings */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Shield className="h-5 w-5" />
            <CardTitle>Configurações Gerais</CardTitle>
          </div>
          <CardDescription>
            Nível mínimo de alerta e outras configurações
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="min-level">Nível Mínimo de Alerta</Label>
            <Select value={minLevel} onValueChange={setMinLevel}>
              <SelectTrigger id="min-level">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="INFO">INFO - Todos os alertas</SelectItem>
                <SelectItem value="WARNING">WARNING - Avisos e erros</SelectItem>
                <SelectItem value="ERROR">ERROR - Apenas erros</SelectItem>
                <SelectItem value="CRITICAL">CRITICAL - Apenas críticos</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground">
              Alertas abaixo deste nível serão ignorados
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Test Alert Form */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <TestTube className="h-5 w-5" />
            <CardTitle>Testar Alertas</CardTitle>
          </div>
          <CardDescription>
            Envie um alerta de teste personalizado
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="test-title">Título do Alerta</Label>
            <Input
              id="test-title"
              value={testTitle}
              onChange={(e) => setTestTitle(e.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="test-message">Mensagem</Label>
            <Textarea
              id="test-message"
              rows={3}
              value={testMessage}
              onChange={(e) => setTestMessage(e.target.value)}
            />
          </div>
        </CardContent>
      </Card>

      {/* Alerts History */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <HistoryIcon className="h-5 w-5" />
            <CardTitle>Histórico de Alertas ({history.length})</CardTitle>
          </div>
          <CardDescription>
            Últimos alertas enviados pelo sistema
          </CardDescription>
        </CardHeader>
        <CardContent>
          {history.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <Bell className="h-12 w-12 mx-auto mb-2 opacity-20" />
              <p>Nenhum alerta enviado ainda</p>
            </div>
          ) : (
            <div className="relative overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Data/Hora</TableHead>
                    <TableHead>Nível</TableHead>
                    <TableHead>Título</TableHead>
                    <TableHead>Canais</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {history.slice().reverse().slice(0, 20).map((alert, index) => (
                    <TableRow key={index}>
                      <TableCell className="text-sm">
                        {format(new Date(alert.timestamp), 'dd/MM/yyyy HH:mm:ss')}
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          {getLevelIcon(alert.level)}
                          <Badge variant={getLevelBadge(alert.level)}>
                            {alert.level}
                          </Badge>
                        </div>
                      </TableCell>
                      <TableCell className="font-medium">{alert.title}</TableCell>
                      <TableCell>
                        <div className="flex gap-1">
                          {alert.channels.map((ch) => (
                            <Badge key={ch} variant="outline" className="text-xs">
                              {ch}
                            </Badge>
                          ))}
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
