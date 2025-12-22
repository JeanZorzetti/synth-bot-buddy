'use client'

/**
 * ABUTRE DASHBOARD - Equity Curve Chart Component
 *
 * Interactive line chart showing balance evolution over time
 * Built with Recharts library
 */

import { useMemo } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
} from 'recharts'
import type { BalanceSnapshot } from '@/types'

interface EquityCurveProps {
  data: BalanceSnapshot[]
  height?: number
}

export default function EquityCurve({ data, height = 256 }: EquityCurveProps) {
  // Transform data for Recharts
  const chartData = useMemo(() => {
    return data.map((snapshot) => ({
      timestamp: new Date(snapshot.timestamp).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
      }),
      balance: snapshot.balance,
      peak: snapshot.peak_balance,
      drawdown: snapshot.drawdown_pct * 100,
    }))
  }, [data])

  // Calculate statistics
  const stats = useMemo(() => {
    if (data.length === 0) {
      return { initial: 0, current: 0, peak: 0, roi: 0 }
    }

    const initial = data[0].balance
    const current = data[data.length - 1].balance
    const peak = Math.max(...data.map((d) => d.balance))
    const roi = ((current - initial) / initial) * 100

    return { initial, current, peak, roi }
  }, [data])

  if (data.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-slate-500">
        <div className="text-center">
          <div className="text-6xl mb-2">📊</div>
          <p className="font-medium">Waiting for Data</p>
          <p className="text-sm mt-1">Start trading to see equity curve</p>
        </div>
      </div>
    )
  }

  return (
    <div>
      {/* Stats Header */}
      <div className="grid grid-cols-4 gap-4 mb-4">
        <div>
          <p className="text-xs text-slate-400">Initial</p>
          <p className="text-sm font-semibold text-slate-300">
            ${stats.initial.toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-400">Current</p>
          <p className="text-sm font-semibold text-sky-400">
            ${stats.current.toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-400">Peak</p>
          <p className="text-sm font-semibold text-emerald-400">
            ${stats.peak.toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-400">ROI</p>
          <p
            className={`text-sm font-semibold ${
              stats.roi >= 0 ? 'text-emerald-400' : 'text-red-400'
            }`}
          >
            {stats.roi >= 0 ? '+' : ''}
            {stats.roi.toFixed(2)}%
          </p>
        </div>
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
          <defs>
            <linearGradient id="balanceGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#0ea5e9" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#0ea5e9" stopOpacity={0} />
            </linearGradient>
          </defs>

          <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.2} />

          <XAxis
            dataKey="timestamp"
            stroke="#64748b"
            fontSize={11}
            tickLine={false}
            axisLine={false}
          />

          <YAxis
            stroke="#64748b"
            fontSize={11}
            tickLine={false}
            axisLine={false}
            tickFormatter={(value) => `$${value.toLocaleString()}`}
          />

          <Tooltip
            content={<CustomTooltip />}
            cursor={{ stroke: '#475569', strokeWidth: 1, strokeDasharray: '5 5' }}
          />

          {/* Peak balance line (dashed) */}
          <Line
            type="monotone"
            dataKey="peak"
            stroke="#10b981"
            strokeWidth={1}
            strokeDasharray="5 5"
            dot={false}
            activeDot={false}
          />

          {/* Balance area */}
          <Area
            type="monotone"
            dataKey="balance"
            stroke="#0ea5e9"
            strokeWidth={2}
            fill="url(#balanceGradient)"
            dot={{ fill: '#0ea5e9', strokeWidth: 0, r: 3 }}
            activeDot={{ r: 5, stroke: '#0ea5e9', strokeWidth: 2, fill: '#0f172a' }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

// Custom Tooltip Component
function CustomTooltip({ active, payload }: any) {
  if (!active || !payload || !payload.length) return null

  const data = payload[0].payload

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg p-3 shadow-xl">
      <p className="text-xs text-slate-400 mb-2">{data.timestamp}</p>
      <div className="space-y-1">
        <div className="flex items-center justify-between gap-4">
          <span className="text-xs text-slate-400">Balance:</span>
          <span className="text-sm font-semibold text-sky-400">
            ${data.balance.toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </span>
        </div>
        <div className="flex items-center justify-between gap-4">
          <span className="text-xs text-slate-400">Peak:</span>
          <span className="text-sm font-semibold text-emerald-400">
            ${data.peak.toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </span>
        </div>
        <div className="flex items-center justify-between gap-4">
          <span className="text-xs text-slate-400">Drawdown:</span>
          <span
            className={`text-sm font-semibold ${
              data.drawdown > 15 ? 'text-red-400' : 'text-amber-400'
            }`}
          >
            {data.drawdown.toFixed(2)}%
          </span>
        </div>
      </div>
    </div>
  )
}
