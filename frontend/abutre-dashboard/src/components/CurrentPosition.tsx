'use client'

/**
 * ABUTRE DASHBOARD - Current Position Monitor
 *
 * Displays active Martingale position status with:
 * - Entry time and streak size
 * - Direction (CALL/PUT)
 * - Current level and stake
 * - Total accumulated loss
 * - Time in position
 */

import { useEffect, useState } from 'react'
import { ArrowUp, ArrowDown, Clock, TrendingDown, Target, DollarSign } from 'lucide-react'
import type { PositionState } from '@/types'

interface CurrentPositionProps {
  position: PositionState | null
}

export default function CurrentPosition({ position }: CurrentPositionProps) {
  const [timeInPosition, setTimeInPosition] = useState<string>('0s')

  // Update time in position every second
  useEffect(() => {
    if (!position?.in_position || !position.entry_time) {
      setTimeInPosition('0s')
      return
    }

    const updateTime = () => {
      const entryTime = new Date(position.entry_time!).getTime()
      const now = Date.now()
      const diffSeconds = Math.floor((now - entryTime) / 1000)

      if (diffSeconds < 60) {
        setTimeInPosition(`${diffSeconds}s`)
      } else if (diffSeconds < 3600) {
        const minutes = Math.floor(diffSeconds / 60)
        const seconds = diffSeconds % 60
        setTimeInPosition(`${minutes}m ${seconds}s`)
      } else {
        const hours = Math.floor(diffSeconds / 3600)
        const minutes = Math.floor((diffSeconds % 3600) / 60)
        setTimeInPosition(`${hours}h ${minutes}m`)
      }
    }

    updateTime()
    const interval = setInterval(updateTime, 1000)

    return () => clearInterval(interval)
  }, [position?.in_position, position?.entry_time])

  // No position - Idle state
  if (!position || !position.in_position) {
    return (
      <div className="text-center py-8 text-slate-500">
        <div className="w-16 h-16 mx-auto mb-3 rounded-full bg-slate-700/30 flex items-center justify-center">
          <span className="text-2xl">⏸️</span>
        </div>
        <p className="font-medium">Waiting for Signal</p>
        <p className="text-sm mt-1">No active position</p>
        <div className="mt-4 text-xs text-slate-600">
          <p>Monitoring V100 for 8+ streak</p>
        </div>
      </div>
    )
  }

  // Active position
  const isCall = position.direction === 'CALL'
  const directionColor = isCall ? 'text-emerald-500' : 'text-red-500'
  const directionBg = isCall ? 'bg-emerald-500/10' : 'bg-red-500/10'
  const directionIcon = isCall ? <ArrowUp className="w-5 h-5" /> : <ArrowDown className="w-5 h-5" />

  // Calculate next level stake (Martingale 2.0x)
  const nextStake = position.current_stake * 2

  return (
    <div className="space-y-4">
      {/* Header - Direction */}
      <div className="flex items-center justify-between">
        <div className={`flex items-center gap-2 px-3 py-2 rounded-lg ${directionBg}`}>
          <div className={directionColor}>{directionIcon}</div>
          <span className={`font-bold ${directionColor}`}>{position.direction}</span>
        </div>
        <div className="text-xs text-slate-400 flex items-center gap-1">
          <Clock className="w-3 h-3" />
          {timeInPosition}
        </div>
      </div>

      {/* Position Details Grid */}
      <div className="space-y-3">
        {/* Entry Info */}
        <div className="p-3 rounded-lg bg-slate-900/50 border border-slate-700/50">
          <div className="flex items-center justify-between mb-2">
            <span className="text-xs text-slate-400">Entry Streak</span>
            <span className="text-sm font-semibold text-slate-300">
              {position.entry_streak_size} candles
            </span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400">Entry Time</span>
            <span className="text-xs text-slate-400">
              {new Date(position.entry_time!).toLocaleTimeString('en-US', {
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
              })}
            </span>
          </div>
        </div>

        {/* Current Level */}
        <div className="p-3 rounded-lg bg-slate-900/50 border border-slate-700/50">
          <div className="flex items-center gap-2 mb-2">
            <Target className="w-4 h-4 text-sky-400" />
            <span className="text-xs text-slate-400">Martingale Level</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-2xl font-bold text-sky-400">
              Level {position.current_level}
            </span>
            <span className="text-xs text-slate-500">/ 10 max</span>
          </div>
          <div className="mt-2 h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all ${
                position.current_level <= 5
                  ? 'bg-emerald-500'
                  : position.current_level <= 8
                  ? 'bg-amber-500'
                  : 'bg-red-500'
              }`}
              style={{ width: `${(position.current_level / 10) * 100}%` }}
            />
          </div>
        </div>

        {/* Current Stake */}
        <div className="p-3 rounded-lg bg-slate-900/50 border border-slate-700/50">
          <div className="flex items-center gap-2 mb-2">
            <DollarSign className="w-4 h-4 text-amber-400" />
            <span className="text-xs text-slate-400">Current Stake</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xl font-bold text-amber-400">
              ${position.current_stake.toFixed(2)}
            </span>
            <span className="text-xs text-slate-500">
              Next: ${nextStake.toFixed(2)}
            </span>
          </div>
        </div>

        {/* Total Loss */}
        <div className="p-3 rounded-lg bg-red-500/5 border border-red-500/20">
          <div className="flex items-center gap-2 mb-2">
            <TrendingDown className="w-4 h-4 text-red-400" />
            <span className="text-xs text-slate-400">Accumulated Loss</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xl font-bold text-red-400">
              -${Math.abs(position.total_loss).toFixed(2)}
            </span>
            <span className="text-xs text-red-400/70">Must recover</span>
          </div>
        </div>
      </div>

      {/* Warning */}
      {position.current_level >= 7 && (
        <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/30">
          <p className="text-xs text-amber-400 text-center">
            ⚠️ Deep level - High risk exposure
          </p>
        </div>
      )}

      {/* Contract ID */}
      {position.contract_id && (
        <div className="text-center text-xs text-slate-600">
          <p>Contract: {position.contract_id}</p>
        </div>
      )}
    </div>
  )
}
