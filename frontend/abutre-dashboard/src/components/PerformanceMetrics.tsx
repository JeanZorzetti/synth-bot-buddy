'use client'

import { TrendingUp, TrendingDown, Target, DollarSign, AlertTriangle, Zap } from 'lucide-react'
import type { PerformanceMetrics as PerformanceMetricsType } from '@/hooks/useAnalytics'

interface PerformanceMetricsProps {
  metrics: PerformanceMetricsType | null
  isLoading?: boolean
}

export default function PerformanceMetrics({ metrics, isLoading = false }: PerformanceMetricsProps) {
  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[1, 2, 3, 4].map(i => (
          <div key={i} className="rounded-xl bg-slate-800/50 border border-slate-700/50 p-6 animate-pulse">
            <div className="h-4 bg-slate-700 rounded w-1/2 mb-4"></div>
            <div className="h-8 bg-slate-700 rounded w-3/4"></div>
          </div>
        ))}
      </div>
    )
  }

  if (!metrics) {
    return (
      <div className="rounded-xl bg-slate-800/50 border border-slate-700/50 p-6 text-center">
        <p className="text-slate-400">Nenhum dado disponível</p>
      </div>
    )
  }

  const winRateColor = metrics.win_rate >= 50 ? 'text-emerald-400' : metrics.win_rate >= 40 ? 'text-amber-400' : 'text-red-400'
  const profitFactorColor = metrics.profit_factor >= 1.5 ? 'text-emerald-400' : metrics.profit_factor >= 1 ? 'text-amber-400' : 'text-red-400'
  const totalProfitColor = metrics.total_profit >= 0 ? 'text-emerald-400' : 'text-red-400'

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {/* Win Rate */}
      <MetricCard
        icon={<Target className="w-5 h-5" />}
        label="Win Rate"
        value={`${metrics.win_rate.toFixed(1)}%`}
        valueColor={winRateColor}
        subtitle={`${metrics.total_trades} trades total`}
        iconBg="bg-sky-500/10"
        iconColor="text-sky-400"
      />

      {/* Profit Factor */}
      <MetricCard
        icon={<Zap className="w-5 h-5" />}
        label="Profit Factor"
        value={metrics.profit_factor.toFixed(2)}
        valueColor={profitFactorColor}
        subtitle={metrics.profit_factor >= 1 ? 'Positivo' : 'Negativo'}
        iconBg="bg-purple-500/10"
        iconColor="text-purple-400"
      />

      {/* Total Profit */}
      <MetricCard
        icon={<DollarSign className="w-5 h-5" />}
        label="Total Profit"
        value={`$${metrics.total_profit.toFixed(2)}`}
        valueColor={totalProfitColor}
        subtitle={`Drawdown: $${metrics.max_drawdown.toFixed(2)}`}
        iconBg={metrics.total_profit >= 0 ? 'bg-emerald-500/10' : 'bg-red-500/10'}
        iconColor={metrics.total_profit >= 0 ? 'text-emerald-400' : 'text-red-400'}
      />

      {/* Streaks */}
      <MetricCard
        icon={<TrendingUp className="w-5 h-5" />}
        label="Max Streaks"
        value={`${metrics.max_win_streak}W / ${metrics.max_loss_streak}L`}
        valueColor="text-slate-100"
        subtitle={`Avg Win: $${metrics.avg_win.toFixed(2)}`}
        iconBg="bg-amber-500/10"
        iconColor="text-amber-400"
      />

      {/* Sharpe Ratio (secondary row) */}
      {metrics.sharpe_ratio !== null && (
        <div className="md:col-span-2 lg:col-span-4">
          <div className="rounded-xl bg-slate-800/50 border border-slate-700/50 p-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-indigo-500/10">
                  <TrendingUp className="w-4 h-4 text-indigo-400" />
                </div>
                <div>
                  <p className="text-sm text-slate-400">Sharpe Ratio</p>
                  <p className="text-lg font-semibold text-slate-100">{metrics.sharpe_ratio.toFixed(2)}</p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-xs text-slate-400">Performance Ajustada ao Risco</p>
                <p className="text-sm text-slate-300">
                  {metrics.sharpe_ratio >= 2 ? 'Excelente' : metrics.sharpe_ratio >= 1 ? 'Bom' : 'Ruim'}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

interface MetricCardProps {
  icon: React.ReactNode
  label: string
  value: string
  valueColor: string
  subtitle: string
  iconBg: string
  iconColor: string
}

function MetricCard({ icon, label, value, valueColor, subtitle, iconBg, iconColor }: MetricCardProps) {
  return (
    <div className="rounded-xl bg-slate-800/50 border border-slate-700/50 p-6 hover:border-slate-600/50 transition-colors">
      <div className="flex items-start justify-between mb-3">
        <p className="text-sm font-medium text-slate-400">{label}</p>
        <div className={`p-2 rounded-lg ${iconBg}`}>
          <div className={iconColor}>{icon}</div>
        </div>
      </div>
      <p className={`text-2xl font-bold ${valueColor} mb-1`}>{value}</p>
      <p className="text-xs text-slate-500">{subtitle}</p>
    </div>
  )
}
