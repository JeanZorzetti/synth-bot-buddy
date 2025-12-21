'use client'

/**
 * ABUTRE DASHBOARD - Trades Table Component
 *
 * Displays recent trading history with:
 * - Entry/exit times
 * - Direction (CALL/PUT)
 * - Max level reached
 * - Result (WIN/LOSS/STOP_LOSS)
 * - Profit/Loss
 * - Running balance
 */

import { ArrowUp, ArrowDown, TrendingUp, TrendingDown, AlertTriangle } from 'lucide-react'
import type { Trade } from '@/types'

interface TradesTableProps {
  trades: Trade[]
  maxRows?: number
}

export default function TradesTable({ trades, maxRows = 10 }: TradesTableProps) {
  // Show most recent trades first
  const recentTrades = [...trades].reverse().slice(0, maxRows)

  if (trades.length === 0) {
    return (
      <div className="text-center py-8 text-slate-500">
        <div className="w-16 h-16 mx-auto mb-3 rounded-full bg-slate-700/30 flex items-center justify-center">
          <span className="text-2xl">📋</span>
        </div>
        <p className="font-medium">No Trades Yet</p>
        <p className="text-sm mt-1">Trade history will appear here</p>
      </div>
    )
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-700/50">
            <th className="text-left py-3 px-2 text-xs font-medium text-slate-400">Time</th>
            <th className="text-left py-3 px-2 text-xs font-medium text-slate-400">Direction</th>
            <th className="text-center py-3 px-2 text-xs font-medium text-slate-400">Level</th>
            <th className="text-center py-3 px-2 text-xs font-medium text-slate-400">Result</th>
            <th className="text-right py-3 px-2 text-xs font-medium text-slate-400">P&L</th>
            <th className="text-right py-3 px-2 text-xs font-medium text-slate-400">Balance</th>
          </tr>
        </thead>
        <tbody>
          {recentTrades.map((trade) => (
            <TradeRow key={trade.id} trade={trade} />
          ))}
        </tbody>
      </table>

      {/* Show total count if more than maxRows */}
      {trades.length > maxRows && (
        <div className="mt-3 text-center text-xs text-slate-500">
          Showing {maxRows} of {trades.length} trades
        </div>
      )}
    </div>
  )
}

function TradeRow({ trade }: { trade: Trade }) {
  const isCall = trade.direction === 'CALL'
  const isWin = trade.result === 'WIN'
  const isStopLoss = trade.result === 'STOP_LOSS'

  // Format entry time
  const entryTime = new Date(trade.entry_time).toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
  })

  // Direction styling
  const directionColor = isCall ? 'text-emerald-500' : 'text-red-500'
  const directionBg = isCall ? 'bg-emerald-500/10' : 'bg-red-500/10'
  const directionIcon = isCall ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />

  // Result styling
  let resultBadge = null
  if (trade.result === 'WIN') {
    resultBadge = (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 text-xs font-medium">
        <TrendingUp className="w-3 h-3" />
        WIN
      </span>
    )
  } else if (trade.result === 'LOSS') {
    resultBadge = (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-red-500/10 text-red-400 text-xs font-medium">
        <TrendingDown className="w-3 h-3" />
        LOSS
      </span>
    )
  } else if (trade.result === 'STOP_LOSS') {
    resultBadge = (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 text-xs font-medium">
        <AlertTriangle className="w-3 h-3" />
        STOP
      </span>
    )
  } else {
    resultBadge = (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-slate-700/50 text-slate-400 text-xs font-medium">
        OPEN
      </span>
    )
  }

  // Level indicator color
  const levelColor =
    trade.max_level_reached <= 3
      ? 'text-emerald-400'
      : trade.max_level_reached <= 6
      ? 'text-amber-400'
      : 'text-red-400'

  return (
    <tr className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors">
      {/* Time */}
      <td className="py-3 px-2">
        <span className="text-xs text-slate-400">{entryTime}</span>
      </td>

      {/* Direction */}
      <td className="py-3 px-2">
        <div className={`inline-flex items-center gap-1.5 px-2 py-1 rounded ${directionBg}`}>
          <div className={directionColor}>{directionIcon}</div>
          <span className={`text-xs font-medium ${directionColor}`}>{trade.direction}</span>
        </div>
      </td>

      {/* Max Level */}
      <td className="py-3 px-2 text-center">
        <span className={`text-sm font-semibold ${levelColor}`}>
          L{trade.max_level_reached}
        </span>
      </td>

      {/* Result */}
      <td className="py-3 px-2 text-center">{resultBadge}</td>

      {/* P&L */}
      <td className="py-3 px-2 text-right">
        {trade.profit !== null ? (
          <span
            className={`text-sm font-semibold ${
              trade.profit >= 0 ? 'text-emerald-400' : 'text-red-400'
            }`}
          >
            {trade.profit >= 0 ? '+' : ''}${trade.profit.toFixed(2)}
          </span>
        ) : (
          <span className="text-xs text-slate-500">-</span>
        )}
      </td>

      {/* Balance After */}
      <td className="py-3 px-2 text-right">
        {trade.balance_after !== null ? (
          <span className="text-sm text-slate-300">
            ${trade.balance_after.toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </span>
        ) : (
          <span className="text-xs text-slate-500">-</span>
        )}
      </td>
    </tr>
  )
}
