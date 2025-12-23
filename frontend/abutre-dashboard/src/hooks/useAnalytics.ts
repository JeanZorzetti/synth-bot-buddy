/**
 * ABUTRE DASHBOARD - Analytics Hook
 *
 * Hook para consumir endpoints de análise estatística
 */

'use client'

import { useState, useCallback } from 'react'

const API_BASE_URL = process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:8000'

export interface SurvivalMetrics {
  max_level_reached: number
  max_level_frequency: number
  death_sequences: Array<{
    trade_id: string
    level: number
    stake: number
    time: string
    result: string
  }>
  recovery_factor: number
  critical_hours: number[]
}

export interface PerformanceMetrics {
  total_trades: number
  win_rate: number
  profit_factor: number
  total_profit: number
  max_drawdown: number
  avg_win: number
  avg_loss: number
  max_win_streak: number
  max_loss_streak: number
  sharpe_ratio: number | null
}

export interface HourlyStat {
  hour: number
  trade_count: number
  win_rate: number
  avg_profit: number
  risk_score: number
}

export interface HourlyAnalysis {
  hourly_stats: HourlyStat[]
  best_hour: HourlyStat | null
  worst_hour: HourlyStat | null
}

export interface EquityCurveData {
  equity_curve: Array<{
    time: string
    balance: number
    cumulative_profit: number
  }>
  peak_balance: number
  lowest_balance: number
}

export function useAnalytics() {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const getSurvivalMetrics = useCallback(async (dateFrom?: string, dateTo?: string): Promise<SurvivalMetrics | null> => {
    setIsLoading(true)
    setError(null)

    try {
      const params = new URLSearchParams()
      if (dateFrom) params.append('date_from', dateFrom)
      if (dateTo) params.append('date_to', dateTo)

      const response = await fetch(`${API_BASE_URL}/api/abutre/analytics/survival?${params}`)

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      return data
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error'
      setError(errorMessage)
      console.error('[useAnalytics] Error fetching survival metrics:', err)
      return null
    } finally {
      setIsLoading(false)
    }
  }, [])

  const getPerformanceMetrics = useCallback(async (dateFrom?: string, dateTo?: string): Promise<PerformanceMetrics | null> => {
    setIsLoading(true)
    setError(null)

    try {
      const params = new URLSearchParams()
      if (dateFrom) params.append('date_from', dateFrom)
      if (dateTo) params.append('date_to', dateTo)

      const response = await fetch(`${API_BASE_URL}/api/abutre/analytics/performance?${params}`)

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      return data
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error'
      setError(errorMessage)
      console.error('[useAnalytics] Error fetching performance metrics:', err)
      return null
    } finally {
      setIsLoading(false)
    }
  }, [])

  const getHourlyAnalysis = useCallback(async (dateFrom?: string, dateTo?: string): Promise<HourlyAnalysis | null> => {
    setIsLoading(true)
    setError(null)

    try {
      const params = new URLSearchParams()
      if (dateFrom) params.append('date_from', dateFrom)
      if (dateTo) params.append('date_to', dateTo)

      const response = await fetch(`${API_BASE_URL}/api/abutre/analytics/hourly?${params}`)

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const rawData: Array<{ hour: number; trades: number; win_rate: number; avg_profit: number; risk_score: number }> = await response.json()

      // Transform backend response to match component expectations
      const hourly_stats: HourlyStat[] = rawData.map(item => ({
        hour: item.hour,
        trade_count: item.trades,
        win_rate: item.win_rate / 100, // Backend returns percentage, component expects decimal
        avg_profit: item.avg_profit,
        risk_score: item.risk_score
      }))

      // Find best and worst hours
      const best_hour = hourly_stats.length > 0
        ? hourly_stats.reduce((best, current) => current.win_rate > best.win_rate ? current : best)
        : null

      const worst_hour = hourly_stats.length > 0
        ? hourly_stats.reduce((worst, current) => current.win_rate < worst.win_rate ? current : worst)
        : null

      return {
        hourly_stats,
        best_hour,
        worst_hour
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error'
      setError(errorMessage)
      console.error('[useAnalytics] Error fetching hourly analysis:', err)
      return null
    } finally {
      setIsLoading(false)
    }
  }, [])

  const getEquityCurve = useCallback(async (dateFrom?: string, dateTo?: string): Promise<EquityCurveData | null> => {
    setIsLoading(true)
    setError(null)

    try {
      const params = new URLSearchParams()
      if (dateFrom) params.append('date_from', dateFrom)
      if (dateTo) params.append('date_to', dateTo)

      const response = await fetch(`${API_BASE_URL}/api/abutre/analytics/equity-curve?${params}`)

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()

      // Transform backend response to match component expectations
      return {
        equity_curve: result.data.map((item: any) => ({
          time: item.timestamp,
          balance: item.balance,
          cumulative_profit: item.cumulative_profit
        })),
        peak_balance: result.summary.peak_balance,
        lowest_balance: result.summary.lowest_balance
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error'
      setError(errorMessage)
      console.error('[useAnalytics] Error fetching equity curve:', err)
      return null
    } finally {
      setIsLoading(false)
    }
  }, [])

  return {
    isLoading,
    error,
    getSurvivalMetrics,
    getPerformanceMetrics,
    getHourlyAnalysis,
    getEquityCurve
  }
}
