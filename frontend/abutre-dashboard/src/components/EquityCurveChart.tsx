'use client'

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Area, AreaChart } from 'recharts'
import { TrendingUp, TrendingDown, DollarSign } from 'lucide-react'
import type { EquityCurveData } from '@/hooks/useAnalytics'

interface EquityCurveChartProps {
  data: EquityCurveData | null
  isLoading?: boolean
}

export default function EquityCurveChart({ data, isLoading = false }: EquityCurveChartProps) {
  if (isLoading) {
    return (
      <div className="rounded-xl bg-slate-800/50 border border-slate-700/50 p-6">
        <div className="animate-pulse">
          <div className="h-6 bg-slate-700 rounded w-1/3 mb-4"></div>
          <div className="h-64 bg-slate-700 rounded"></div>
        </div>
      </div>
    )
  }

  if (!data || data.equity_curve.length === 0) {
    return (
      <div className="rounded-xl bg-slate-800/50 border border-slate-700/50 p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-slate-700/50">
            <DollarSign className="w-5 h-5 text-slate-400" />
          </div>
          <h3 className="text-lg font-semibold text-slate-300">Curva de Equity</h3>
        </div>
        <p className="text-sm text-slate-400">Nenhum dado disponível</p>
      </div>
    )
  }

  const { equity_curve, peak_balance, lowest_balance } = data

  // Calculate overall trend
  const firstBalance = equity_curve[0]?.balance || 0
  const lastBalance = equity_curve[equity_curve.length - 1]?.balance || 0
  const totalChange = lastBalance - firstBalance
  const totalChangePercent = firstBalance !== 0 ? (totalChange / firstBalance) * 100 : 0
  const isPositive = totalChange >= 0

  // Format chart data
  const chartData = equity_curve.map(point => ({
    time: new Date(point.time).toLocaleDateString('pt-BR', {
      day: '2-digit',
      month: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    }),
    timestamp: new Date(point.time).getTime(),
    balance: parseFloat(point.balance.toFixed(2)),
    profit: parseFloat(point.cumulative_profit.toFixed(2))
  }))

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload
      return (
        <div className="bg-slate-800 border border-slate-700 rounded-lg p-3 shadow-xl">
          <p className="text-xs text-slate-400 mb-2">{data.time}</p>
          <div className="space-y-1">
            <div className="flex items-center justify-between gap-4">
              <span className="text-xs text-slate-400">Balance:</span>
              <span className="text-sm font-semibold text-blue-400">
                ${data.balance.toFixed(2)}
              </span>
            </div>
            <div className="flex items-center justify-between gap-4">
              <span className="text-xs text-slate-400">Profit:</span>
              <span className={`text-sm font-semibold ${
                data.profit >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {data.profit >= 0 ? '+' : ''}${data.profit.toFixed(2)}
              </span>
            </div>
          </div>
        </div>
      )
    }
    return null
  }

  return (
    <div className="rounded-xl bg-slate-800/50 border border-slate-700/50 p-6 space-y-4">
      {/* Header with summary stats */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg ${
            isPositive ? 'bg-green-500/10' : 'bg-red-500/10'
          }`}>
            {isPositive ? (
              <TrendingUp className="w-5 h-5 text-green-500" />
            ) : (
              <TrendingDown className="w-5 h-5 text-red-500" />
            )}
          </div>
          <div>
            <h3 className="text-lg font-semibold text-slate-300">Curva de Equity</h3>
            <p className="text-xs text-slate-400">
              {equity_curve.length} pontos de dados
            </p>
          </div>
        </div>

        <div className="text-right">
          <div className={`text-2xl font-bold ${
            isPositive ? 'text-green-500' : 'text-red-500'
          }`}>
            {isPositive ? '+' : ''}${totalChange.toFixed(2)}
          </div>
          <div className={`text-sm ${
            isPositive ? 'text-green-400' : 'text-red-400'
          }`}>
            {isPositive ? '+' : ''}{totalChangePercent.toFixed(2)}%
          </div>
        </div>
      </div>

      {/* Peak and Lowest Balance Cards */}
      <div className="grid grid-cols-2 gap-4">
        <div className="p-3 rounded-lg bg-green-500/5 border border-green-500/20">
          <div className="flex items-center gap-2 mb-1">
            <TrendingUp className="w-4 h-4 text-green-500" />
            <span className="text-xs font-medium text-slate-400">Pico</span>
          </div>
          <div className="text-lg font-bold text-green-400">
            ${peak_balance.toFixed(2)}
          </div>
        </div>

        <div className="p-3 rounded-lg bg-red-500/5 border border-red-500/20">
          <div className="flex items-center gap-2 mb-1">
            <TrendingDown className="w-4 h-4 text-red-500" />
            <span className="text-xs font-medium text-slate-400">Menor</span>
          </div>
          <div className="text-lg font-bold text-red-400">
            ${lowest_balance.toFixed(2)}
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={chartData}
            margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
          >
            <defs>
              <linearGradient id="colorBalance" x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="5%"
                  stopColor={isPositive ? '#10b981' : '#ef4444'}
                  stopOpacity={0.3}
                />
                <stop
                  offset="95%"
                  stopColor={isPositive ? '#10b981' : '#ef4444'}
                  stopOpacity={0}
                />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />

            <XAxis
              dataKey="time"
              stroke="#94a3b8"
              fontSize={11}
              tickLine={false}
              axisLine={{ stroke: '#334155' }}
              angle={-45}
              textAnchor="end"
              height={80}
              interval="preserveStartEnd"
            />

            <YAxis
              stroke="#94a3b8"
              fontSize={11}
              tickLine={false}
              axisLine={{ stroke: '#334155' }}
              tickFormatter={(value) => `$${value}`}
              width={60}
            />

            <Tooltip content={<CustomTooltip />} />

            {/* Reference line at initial balance */}
            <ReferenceLine
              y={firstBalance}
              stroke="#64748b"
              strokeDasharray="3 3"
              label={{
                value: 'Inicial',
                position: 'right',
                fill: '#64748b',
                fontSize: 11
              }}
            />

            {/* Area under the curve */}
            <Area
              type="monotone"
              dataKey="balance"
              stroke={isPositive ? '#10b981' : '#ef4444'}
              strokeWidth={2}
              fill="url(#colorBalance)"
              animationDuration={1000}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-6 pt-2 border-t border-slate-700/50">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-blue-500"></div>
          <span className="text-xs text-slate-400">Balance</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-0.5 bg-slate-500"></div>
          <span className="text-xs text-slate-400">Linha Base</span>
        </div>
      </div>
    </div>
  )
}
