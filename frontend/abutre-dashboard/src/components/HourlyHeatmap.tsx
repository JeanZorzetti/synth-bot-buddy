'use client'

import { Clock, TrendingUp, TrendingDown, AlertTriangle } from 'lucide-react'
import type { HourlyAnalysis } from '@/hooks/useAnalytics'

interface HourlyHeatmapProps {
  data: HourlyAnalysis | null
  isLoading?: boolean
}

export default function HourlyHeatmap({ data, isLoading = false }: HourlyHeatmapProps) {
  if (isLoading) {
    return (
      <div className="rounded-xl bg-slate-800/50 border border-slate-700/50 p-6">
        <div className="animate-pulse">
          <div className="h-6 bg-slate-700 rounded w-1/3 mb-4"></div>
          <div className="grid grid-cols-8 gap-2">
            {Array.from({ length: 24 }).map((_, i) => (
              <div key={i} className="h-20 bg-slate-700 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  if (!data || data.hourly_stats.length === 0) {
    return (
      <div className="rounded-xl bg-slate-800/50 border border-slate-700/50 p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-slate-700/50">
            <Clock className="w-5 h-5 text-slate-400" />
          </div>
          <h3 className="text-lg font-semibold text-slate-300">Análise por Horário</h3>
        </div>
        <p className="text-sm text-slate-400">Nenhum dado disponível</p>
      </div>
    )
  }

  const { hourly_stats, best_hour, worst_hour } = data

  // Get color based on risk score (0-10)
  const getRiskColor = (riskScore: number) => {
    if (riskScore >= 8) return 'bg-red-500'
    if (riskScore >= 6) return 'bg-orange-500'
    if (riskScore >= 4) return 'bg-yellow-500'
    if (riskScore >= 2) return 'bg-blue-500'
    return 'bg-green-500'
  }

  const getRiskBorderColor = (riskScore: number) => {
    if (riskScore >= 8) return 'border-red-500/50'
    if (riskScore >= 6) return 'border-orange-500/50'
    if (riskScore >= 4) return 'border-yellow-500/50'
    if (riskScore >= 2) return 'border-blue-500/50'
    return 'border-green-500/50'
  }

  const getRiskLabel = (riskScore: number) => {
    if (riskScore >= 8) return 'Muito Alto'
    if (riskScore >= 6) return 'Alto'
    if (riskScore >= 4) return 'Médio'
    if (riskScore >= 2) return 'Baixo'
    return 'Muito Baixo'
  }

  // Get opacity based on trade count (more trades = more opacity)
  const getOpacity = (tradeCount: number, maxTrades: number) => {
    if (tradeCount === 0) return 0.1
    return Math.max(0.3, Math.min(1, tradeCount / maxTrades))
  }

  const maxTrades = Math.max(...hourly_stats.map(h => h.trade_count))

  return (
    <div className="rounded-xl bg-slate-800/50 border border-slate-700/50 p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-blue-500/10">
            <Clock className="w-5 h-5 text-blue-500" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-slate-300">Análise por Horário</h3>
            <p className="text-xs text-slate-400">Heatmap de 24 horas</p>
          </div>
        </div>
      </div>

      {/* Best and Worst Hours */}
      <div className="grid grid-cols-2 gap-4">
        {best_hour && (
          <div className="p-4 rounded-lg bg-green-500/5 border border-green-500/20">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="w-4 h-4 text-green-500" />
              <span className="text-xs font-medium text-slate-400">Melhor Horário</span>
            </div>
            <div className="text-2xl font-bold text-green-400 mb-1">
              {best_hour.hour.toString().padStart(2, '0')}:00h
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-400">Win Rate:</span>
              <span className="text-green-400 font-semibold">
                {(best_hour.win_rate * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-400">Trades:</span>
              <span className="text-slate-300">{best_hour.trade_count}</span>
            </div>
          </div>
        )}

        {worst_hour && (
          <div className="p-4 rounded-lg bg-red-500/5 border border-red-500/20">
            <div className="flex items-center gap-2 mb-2">
              <TrendingDown className="w-4 h-4 text-red-500" />
              <span className="text-xs font-medium text-slate-400">Pior Horário</span>
            </div>
            <div className="text-2xl font-bold text-red-400 mb-1">
              {worst_hour.hour.toString().padStart(2, '0')}:00h
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-400">Win Rate:</span>
              <span className="text-red-400 font-semibold">
                {(worst_hour.win_rate * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex items-center justify-between text-xs">
              <span className="text-slate-400">Trades:</span>
              <span className="text-slate-300">{worst_hour.trade_count}</span>
            </div>
          </div>
        )}
      </div>

      {/* Heatmap Grid */}
      <div>
        <div className="text-xs font-medium text-slate-400 mb-3">
          Risco por Horário (0-10)
        </div>

        <div className="grid grid-cols-6 sm:grid-cols-8 md:grid-cols-12 gap-2">
          {hourly_stats.map(hour => {
            const opacity = getOpacity(hour.trade_count, maxTrades)
            const riskColor = getRiskColor(hour.risk_score)
            const riskBorder = getRiskBorderColor(hour.risk_score)

            return (
              <div
                key={hour.hour}
                className={`group relative rounded-lg border ${riskBorder} p-3 hover:scale-105 transition-all cursor-pointer`}
                style={{
                  backgroundColor: `${riskColor.replace('bg-', 'rgba(')}`,
                  opacity: opacity
                }}
              >
                {/* Hour label */}
                <div className="text-xs font-bold text-white text-center mb-1">
                  {hour.hour.toString().padStart(2, '0')}h
                </div>

                {/* Risk score */}
                <div className="text-center">
                  <div className="text-lg font-bold text-white">
                    {hour.risk_score.toFixed(1)}
                  </div>
                </div>

                {/* Tooltip on hover */}
                <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 hidden group-hover:block z-10">
                  <div className="bg-slate-800 border border-slate-700 rounded-lg p-3 shadow-xl whitespace-nowrap">
                    <div className="text-xs font-semibold text-slate-300 mb-2">
                      {hour.hour.toString().padStart(2, '0')}:00h
                    </div>

                    <div className="space-y-1">
                      <div className="flex items-center justify-between gap-4">
                        <span className="text-xs text-slate-400">Trades:</span>
                        <span className="text-xs font-semibold text-white">
                          {hour.trade_count}
                        </span>
                      </div>

                      <div className="flex items-center justify-between gap-4">
                        <span className="text-xs text-slate-400">Win Rate:</span>
                        <span className={`text-xs font-semibold ${
                          hour.win_rate >= 0.5 ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {(hour.win_rate * 100).toFixed(1)}%
                        </span>
                      </div>

                      <div className="flex items-center justify-between gap-4">
                        <span className="text-xs text-slate-400">Avg Profit:</span>
                        <span className={`text-xs font-semibold ${
                          hour.avg_profit >= 0 ? 'text-green-400' : 'text-red-400'
                        }`}>
                          ${hour.avg_profit.toFixed(2)}
                        </span>
                      </div>

                      <div className="flex items-center justify-between gap-4">
                        <span className="text-xs text-slate-400">Risco:</span>
                        <span className={`text-xs font-semibold ${
                          hour.risk_score >= 6 ? 'text-red-400' :
                          hour.risk_score >= 4 ? 'text-yellow-400' : 'text-green-400'
                        }`}>
                          {getRiskLabel(hour.risk_score)}
                        </span>
                      </div>
                    </div>

                    {/* Arrow pointer */}
                    <div className="absolute top-full left-1/2 transform -translate-x-1/2 -mt-px">
                      <div className="border-4 border-transparent border-t-slate-700"></div>
                    </div>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Legend */}
      <div className="space-y-3">
        <div className="text-xs font-medium text-slate-400">Legenda de Risco</div>

        <div className="flex items-center gap-4 flex-wrap">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-green-500"></div>
            <span className="text-xs text-slate-400">Muito Baixo (0-2)</span>
          </div>

          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-blue-500"></div>
            <span className="text-xs text-slate-400">Baixo (2-4)</span>
          </div>

          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-yellow-500"></div>
            <span className="text-xs text-slate-400">Médio (4-6)</span>
          </div>

          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-orange-500"></div>
            <span className="text-xs text-slate-400">Alto (6-8)</span>
          </div>

          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded bg-red-500"></div>
            <span className="text-xs text-slate-400">Muito Alto (8-10)</span>
          </div>
        </div>

        <div className="flex items-start gap-2 p-3 rounded-lg bg-blue-500/5 border border-blue-500/20">
          <AlertTriangle className="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-xs text-slate-400">
              <span className="font-semibold text-blue-400">Dica:</span> Opacidade indica volume de trades.
              Células mais opacas = mais dados para análise.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
