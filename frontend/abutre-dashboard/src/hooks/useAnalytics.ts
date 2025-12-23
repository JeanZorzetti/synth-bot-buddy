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

export interface HourlyAnalysis {
  hour: number
  trades: number
  win_rate: number
  avg_profit: number
  risk_score: number
}

export interface EquityCurve {
  data: Array<{
    timestamp: string
    balance: number
    cumulative_profit: number
    trade_id: string
  }>
  summary: {
    initial_balance: number
    final_balance: number
    total_profit: number
    peak_balance: number
    lowest_balance: number
  }
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

  const getHourlyAnalysis = useCallback(async (dateFrom?: string, dateTo?: string): Promise<HourlyAnalysis[] | null> => {
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

      const data = await response.json()
      return data
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error'
      setError(errorMessage)
      console.error('[useAnalytics] Error fetching hourly analysis:', err)
      return null
    } finally {
      setIsLoading(false)
    }
  }, [])

  const getEquityCurve = useCallback(async (dateFrom?: string, dateTo?: string): Promise<EquityCurve | null> => {
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
      return {
        data: result.data,
        summary: result.summary
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
