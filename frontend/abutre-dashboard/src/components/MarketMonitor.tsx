'use client'

/**
 * ABUTRE DASHBOARD - Market Monitor Component
 *
 * Real-time market data display showing:
 * - Current streak count and direction
 * - Countdown to trigger (8 candles)
 * - Current price
 * - Last candle color sequence
 */

import { ArrowUp, ArrowDown, TrendingUp, Activity } from 'lucide-react'
import type { MarketData } from '@/types'

interface MarketMonitorProps {
  data: MarketData | null
}

export default function MarketMonitor({ data }: MarketMonitorProps) {
  if (!data) {
    return (
      <div className="text-center py-8 text-slate-500">
        <div className="w-16 h-16 mx-auto mb-3 rounded-full bg-slate-700/30 flex items-center justify-center">
          <Activity className="w-8 h-8 opacity-50" />
        </div>
        <p className="font-medium">Connecting to Market</p>
        <p className="text-sm mt-1">WebSocket initializing...</p>
      </div>
    )
  }

  const isGreenStreak = data.current_streak_direction === 1
  const isRedStreak = data.current_streak_direction === -1
  const streakColor = isGreenStreak ? 'text-emerald-500' : 'text-red-500'
  const streakBg = isGreenStreak ? 'bg-emerald-500/10' : 'bg-red-500/10'
  const streakIcon = isGreenStreak ? <ArrowUp className="w-6 h-6" /> : <ArrowDown className="w-6 h-6" />

  // Trigger countdown
  const candlesUntilTrigger = Math.max(0, 8 - data.current_streak_count)
  const isTriggerNear = candlesUntilTrigger <= 2 && candlesUntilTrigger > 0
  const isTriggered = data.current_streak_count >= 8

  return (
    <div className="space-y-4">
      {/* Symbol & Price */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-700/50">
        <div>
          <p className="text-xs text-slate-400">Symbol</p>
          <p className="text-lg font-bold text-slate-200">{data.symbol}</p>
        </div>
        <div className="text-right">
          <p className="text-xs text-slate-400">Current Price</p>
          <p className="text-lg font-bold text-sky-400">
            {data.current_price.toFixed(2)}
          </p>
        </div>
      </div>

      {/* Current Streak */}
      <div className={`p-4 rounded-lg ${streakBg} border border-slate-700/50`}>
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-slate-400">Current Streak</span>
          <div className={streakColor}>{streakIcon}</div>
        </div>
        <div className="flex items-center justify-between">
          <span className={`text-3xl font-bold ${streakColor}`}>
            {data.current_streak_count}
          </span>
          <span className="text-sm text-slate-400">
            {isGreenStreak ? 'Green' : 'Red'} candles
          </span>
        </div>
      </div>

      {/* Trigger Countdown */}
      <div
        className={`p-4 rounded-lg border ${
          isTriggered
            ? 'bg-amber-500/10 border-amber-500/30'
            : isTriggerNear
            ? 'bg-orange-500/10 border-orange-500/30'
            : 'bg-slate-900/50 border-slate-700/50'
        }`}
      >
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-slate-400">Trigger Status</span>
          <TrendingUp
            className={`w-4 h-4 ${
              isTriggered
                ? 'text-amber-400'
                : isTriggerNear
                ? 'text-orange-400'
                : 'text-slate-500'
            }`}
          />
        </div>

        {isTriggered ? (
          <div>
            <div className="text-2xl font-bold text-amber-400 mb-1">TRIGGERED!</div>
            <p className="text-xs text-amber-400/70">
              Waiting for reversal signal...
            </p>
          </div>
        ) : (
          <div>
            <div className="flex items-baseline gap-2">
              <span
                className={`text-2xl font-bold ${
                  isTriggerNear ? 'text-orange-400' : 'text-slate-300'
                }`}
              >
                {candlesUntilTrigger}
              </span>
              <span className="text-sm text-slate-400">candles to trigger</span>
            </div>
            <div className="mt-2 h-2 bg-slate-800 rounded-full overflow-hidden">
              <div
                className={`h-full transition-all ${
                  data.current_streak_count >= 6
                    ? 'bg-amber-500'
                    : data.current_streak_count >= 4
                    ? 'bg-orange-500'
                    : 'bg-sky-500'
                }`}
                style={{ width: `${Math.min((data.current_streak_count / 8) * 100, 100)}%` }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Last Candles Sequence */}
      <div className="p-4 rounded-lg bg-slate-900/50 border border-slate-700/50">
        <p className="text-xs text-slate-400 mb-3">Recent Candles</p>
        <div className="flex gap-1.5">
          {Array.from({ length: Math.min(data.current_streak_count, 12) }).map((_, i) => (
            <div
              key={i}
              className={`flex-1 h-8 rounded ${
                isGreenStreak ? 'bg-emerald-500/30' : 'bg-red-500/30'
              } border ${isGreenStreak ? 'border-emerald-500/50' : 'border-red-500/50'}`}
              title={`Candle ${i + 1}`}
            />
          ))}
        </div>
        <p className="text-xs text-slate-500 mt-2 text-center">
          {data.current_streak_count > 12
            ? `+${data.current_streak_count - 12} more candles`
            : `${data.current_streak_count} consecutive ${isGreenStreak ? 'green' : 'red'}`}
        </p>
      </div>

      {/* Strategy Info */}
      <div className="p-3 rounded-lg bg-slate-800/30 border border-slate-700/30">
        <p className="text-xs text-slate-400 text-center">
          {isTriggered ? (
            <>
              <span className="text-amber-400 font-semibold">
                Ready to enter on next {isGreenStreak ? 'RED' : 'GREEN'} candle
              </span>
            </>
          ) : (
            <>
              Abutre waits for <span className="text-sky-400 font-semibold">8+</span> streak before
              entry
            </>
          )}
        </p>
      </div>
    </div>
  )
}
