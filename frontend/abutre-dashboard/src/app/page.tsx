'use client'

import { useEffect, useState } from 'react'
import { TrendingUp, Activity, Target, AlertTriangle } from 'lucide-react'
import MetricsCard from '@/components/MetricsCard'
import EquityCurve from '@/components/EquityCurve'
import CurrentPosition from '@/components/CurrentPosition'
import MarketMonitor from '@/components/MarketMonitor'
import TradesTable from '@/components/TradesTable'
import { useDashboard } from '@/hooks/useDashboard'
import { useWebSocket } from '@/hooks/useWebSocket'

export default function DashboardPage() {
  const [mounted, setMounted] = useState(false)

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

  if (!mounted) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-pulse text-slate-400">Loading...</div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/95 backdrop-blur supports-[backdrop-filter]:bg-slate-900/75 sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-sky-500 to-blue-600 flex items-center justify-center">
                <span className="text-xl font-bold">🦅</span>
              </div>
              <div>
                <h1 className="text-xl font-bold">Abutre Dashboard</h1>
                <p className="text-xs text-slate-400">Delayed Martingale Strategy</p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              {/* Connection Status */}
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'} animate-pulse`} />
                <span className="text-sm text-slate-400">
                  {isConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>

              {/* Bot Status */}
              <div className="px-3 py-1.5 rounded-lg bg-slate-800 text-sm font-medium">
                Status: <span className="text-slate-300">{botStatus.toUpperCase()}</span>
              </div>

              {/* Balance */}
              <div className="px-4 py-2 rounded-lg bg-gradient-to-r from-sky-500/10 to-blue-600/10 border border-sky-500/20">
                <div className="text-xs text-slate-400">Balance</div>
                <div className="text-lg font-bold text-sky-400">
                  ${balance.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        {/* Metrics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          {/* Balance Card */}
          <MetricCard
            title="Current Balance"
            value={`$${balance.toLocaleString('en-US', { minimumFractionDigits: 2 })}`}
            change={`+$${(balance - 2000).toFixed(2)}`}
            changeType="positive"
            icon={<TrendingUp className="w-5 h-5" />}
            iconColor="text-sky-500"
            iconBg="bg-sky-500/10"
          />

          {/* ROI Card */}
          <MetricCard
            title="ROI"
            value={`${roi.toFixed(2)}%`}
            change="vs. $2,000 initial"
            changeType="positive"
            icon={<Activity className="w-5 h-5" />}
            iconColor="text-emerald-500"
            iconBg="bg-emerald-500/10"
          />

          {/* Win Rate Card */}
          <MetricCard
            title="Win Rate"
            value={`${winRate.toFixed(1)}%`}
            change={`${totalTrades} trades`}
            changeType="positive"
            icon={<Target className="w-5 h-5" />}
            iconColor="text-green-500"
            iconBg="bg-green-500/10"
          />

          {/* Max Drawdown Card */}
          <MetricCard
            title="Max Drawdown"
            value={`${maxDrawdown.toFixed(2)}%`}
            change="Limit: 25%"
            changeType="warning"
            icon={<AlertTriangle className="w-5 h-5" />}
            iconColor="text-amber-500"
            iconBg="bg-amber-500/10"
          />
        </div>

        {/* Charts & Data Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          {/* Equity Curve - Takes 2 columns */}
          <div className="lg:col-span-2">
            <div className="rounded-xl bg-slate-800/50 border border-slate-700/50 p-6">
              <h2 className="text-lg font-semibold mb-4">Equity Curve</h2>
              <EquityCurve data={balanceHistory} />
            </div>
          </div>

          {/* Current Position */}
          <div className="lg:col-span-1">
            <div className="rounded-xl bg-slate-800/50 border border-slate-700/50 p-6">
              <h2 className="text-lg font-semibold mb-4">Current Position</h2>
              <CurrentPosition position={position} />
            </div>
          </div>
        </div>

        {/* Market Monitor & Recent Trades */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Market Monitor */}
          <div className="rounded-xl bg-slate-800/50 border border-slate-700/50 p-6">
            <h2 className="text-lg font-semibold mb-4">Market Monitor</h2>
            <MarketMonitor data={marketData} />
          </div>

          {/* Recent Trades */}
          <div className="rounded-xl bg-slate-800/50 border border-slate-700/50 p-6">
            <h2 className="text-lg font-semibold mb-4">Recent Trades</h2>
            <TradesTable trades={trades} maxRows={8} />
          </div>
        </div>

        {/* Footer Info */}
        <div className="mt-8 p-4 rounded-lg bg-slate-800/30 border border-slate-700/50 text-center text-sm text-slate-400">
          <p>
            🦅 Abutre Bot v1.0.0 | Validated: +40.25% ROI (180 days) | 100% Win Rate
          </p>
          <p className="mt-1 text-xs">
            {isConnected ? (
              <>✅ Connected to backend - Real-time data active</>
            ) : (
              <>⚠️ Disconnected - Waiting for backend connection...</>
            )}
          </p>
        </div>
      </main>
    </div>
  )
}
