import { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Alert, AlertDescription } from '@/components/ui/alert'
import {
  Play,
  StopCircle,
  RefreshCw,
  TrendingUp,
  AlertCircle,
  CheckCircle2,
  Target,
  AlertTriangle,
  Activity,
} from 'lucide-react'
import EquityCurve from '@/components/abutre/EquityCurve'
import CurrentPosition from '@/components/abutre/CurrentPosition'
import MarketMonitor from '@/components/abutre/MarketMonitor'
import TradesTable from '@/components/abutre/TradesTable'
import { useDashboard } from '@/hooks/useDashboard'
import { useWebSocket } from '@/hooks/useWebSocket'
import { useToast } from '@/hooks/use-toast'

export default function AbutreDashboard() {
  const [mounted, setMounted] = useState(false)
  const { toast } = useToast()

  // Zustand store state
  const {
    isConnected,
    botStatus,
    currentBalance,
    position,
    marketData,
    riskStats,
    trades,
    balanceHistory,
  } = useDashboard()

  // Initialize WebSocket connection
  useWebSocket()

  useEffect(() => {
    setMounted(true)
  }, [])

  // Calculate metrics from risk stats or use defaults
  const balance = currentBalance || riskStats?.current_balance || 2000
  const roi = riskStats?.roi_pct || 0
  const winRate = riskStats?.win_rate_pct || 0
  const maxDrawdown = riskStats?.max_drawdown_pct ? riskStats.max_drawdown_pct * 100 : 0
  const totalTrades = riskStats?.total_trades || 0

  const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://botderivapi.roilabs.com.br'

  const handleStart = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/abutre/start`, {
        method: 'POST',
      })

      const data = await response.json()

      if (data.status === 'success') {
        toast({
          title: 'Bot Iniciado',
          description: 'Aguardando gatilho (8+ velas consecutivas)',
        })
      } else {
        throw new Error(data.message || 'Erro ao iniciar bot')
      }
    } catch (error: any) {
      toast({
        title: 'Erro ao Iniciar',
        description: error.message,
        variant: 'destructive',
      })
    }
  }

  const handleStop = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/abutre/stop`, {
        method: 'POST',
      })

      const data = await response.json()

      if (data.status === 'success') {
        toast({
          title: 'Bot Parado',
          description: 'Sistema interrompido com segurança',
          variant: 'destructive',
        })
      } else {
        throw new Error(data.message || 'Erro ao parar bot')
      }
    } catch (error: any) {
      toast({
        title: 'Erro ao Parar',
        description: error.message,
        variant: 'destructive',
      })
    }
  }

  if (!mounted) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <RefreshCw className="h-8 w-8 animate-spin mx-auto mb-4 text-primary" />
          <p className="text-muted-foreground">Carregando Abutre Bot...</p>
        </div>
      </div>
    )
  }

  const isRunning = botStatus === 'running'

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
            🦅 Abutre Bot
          </h1>
          <p className="text-muted-foreground mt-1">
            Delayed Martingale Strategy - Paper Trading em tempo real
          </p>
        </div>

        <div className="flex gap-2">
          {isRunning ? (
            <>
              <Button onClick={handleStop} variant="destructive">
                <StopCircle className="h-4 w-4 mr-2" />
                Parar
              </Button>
            </>
          ) : (
            <Button onClick={handleStart}>
              <Play className="h-4 w-4 mr-2" />
              Iniciar Bot
            </Button>
          )}
        </div>
      </div>

      {/* Status Banner */}
      <Alert className={isRunning ? 'border-green-500 bg-green-50' : 'border-gray-300'}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {isRunning ? (
              <CheckCircle2 className="h-5 w-5 text-green-600" />
            ) : (
              <AlertCircle className="h-5 w-5 text-gray-600" />
            )}
            <div>
              <AlertDescription className="font-semibold text-lg">
                {isRunning ? '🟢 Bot Rodando' : '⏸️ Bot Parado'}
              </AlertDescription>
              <p className="text-sm text-muted-foreground mt-1">
                {isConnected ? (
                  <>✅ Connected • Paper Trading Mode</>
                ) : (
                  <>⚠️ Disconnected - Aguardando conexão...</>
                )}
              </p>
            </div>
          </div>

          {isRunning && (
            <div className="flex gap-4">
              <div className="text-center">
                <p className="text-2xl font-bold text-green-600">{totalTrades}</p>
                <p className="text-xs text-muted-foreground">Trades</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-blue-600">{winRate.toFixed(1)}%</p>
                <p className="text-xs text-muted-foreground">Win Rate</p>
              </div>
            </div>
          )}
        </div>
      </Alert>

      {/* Metrics Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Current Balance</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              ${balance.toLocaleString()}
            </div>
            <p className={`text-xs ${(balance - 2000) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {(balance - 2000) >= 0 ? '+' : ''}
              ${(balance - 2000).toFixed(2)} from initial
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">ROI</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {roi.toFixed(2)}%
            </div>
            <p className="text-xs text-muted-foreground">
              vs. $2,000 initial
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Win Rate</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {winRate.toFixed(1)}%
            </div>
            <p className="text-xs text-muted-foreground">
              {totalTrades} trades
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Max Drawdown</CardTitle>
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-600">
              {maxDrawdown.toFixed(2)}%
            </div>
            <p className="text-xs text-muted-foreground">
              Limit: 25%
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Charts & Position Grid */}
      <div className="grid gap-6 md:grid-cols-3">
        {/* Equity Curve - Takes 2 columns */}
        <div className="md:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle>Equity Curve</CardTitle>
              <CardDescription>
                Saldo ao longo do tempo - Alvo: +40.25% ROI
              </CardDescription>
            </CardHeader>
            <CardContent>
              <EquityCurve data={balanceHistory} />
            </CardContent>
          </Card>
        </div>

        {/* Current Position */}
        <div className="md:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle>Current Position</CardTitle>
              <CardDescription>
                Monitorando V100 para streak de 8+ velas
              </CardDescription>
            </CardHeader>
            <CardContent>
              <CurrentPosition position={position} />
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Market Monitor & Trades */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Market Monitor */}
        <Card>
          <CardHeader>
            <CardTitle>Market Monitor</CardTitle>
            <CardDescription>
              Detector de streaks - Aguardando 8+ velas consecutivas
            </CardDescription>
          </CardHeader>
          <CardContent>
            <MarketMonitor data={marketData} />
          </CardContent>
        </Card>

        {/* Recent Trades */}
        <Card>
          <CardHeader>
            <CardTitle>Recent Trades</CardTitle>
            <CardDescription>
              Últimas operações executadas
            </CardDescription>
          </CardHeader>
          <CardContent>
            <TradesTable trades={trades} maxRows={8} />
          </CardContent>
        </Card>
      </div>

      {/* Info Card */}
      <Card className="border-blue-200 bg-blue-50">
        <CardContent className="pt-6">
          <div className="flex gap-3">
            <CheckCircle2 className="h-5 w-5 text-blue-600 flex-shrink-0 mt-0.5" />
            <div className="space-y-2">
              <p className="text-sm font-medium text-blue-900">
                Sobre o Abutre Bot
              </p>
              <p className="text-sm text-blue-700">
                Bot de trading automatizado usando estratégia <strong>Delayed Martingale</strong>.
                Aguarda 8+ velas consecutivas da mesma cor antes de entrar na direção oposta.
                Validado em backtest: <strong>+40.25% ROI (180 dias), 100% Win Rate, 0 busts</strong>.
              </p>
              <p className="text-sm text-blue-700">
                <strong>Fase Atual:</strong> Forward Test (30 dias) em Paper Trading.
                <strong>Critérios:</strong> ROI &gt; 5%, Win Rate &gt; 90%, Max DD &lt; 30%.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
