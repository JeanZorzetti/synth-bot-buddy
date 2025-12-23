/**
 * ABUTRE DASHBOARD - Historical Data Hook
 *
 * Hook para buscar e sincronizar dados históricos por período
 */

'use client'

import { useState, useCallback } from 'react'
import { useDashboard } from './useDashboard'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

interface SyncResult {
  success: boolean
  message: string
  trades_synced?: number
  trades_failed?: number
}

export function useHistoricalData() {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [syncResult, setSyncResult] = useState<SyncResult | null>(null)

  const { addTrade, updateTrade } = useDashboard()

  /**
   * Busca trades do banco de dados por período
   */
  const fetchTradesByPeriod = useCallback(async (dateFrom: string, dateTo: string) => {
    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/abutre/sync/trades?date_from=${dateFrom}&date_to=${dateTo}&limit=1000`
      )

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()

      if (data.status === 'success') {
        // Atualizar trades no store
        const trades = data.data.trades || []

        console.log(`[useHistoricalData] Loaded ${trades.length} trades from ${dateFrom} to ${dateTo}`)

        return {
          success: true,
          trades,
          count: trades.length
        }
      } else {
        throw new Error(data.message || 'Failed to fetch trades')
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error'
      setError(errorMessage)
      console.error('[useHistoricalData] Error fetching trades:', err)

      return {
        success: false,
        trades: [],
        count: 0,
        error: errorMessage
      }
    } finally {
      setIsLoading(false)
    }
  }, [])

  /**
   * Sincroniza trades da Deriv API para um período específico
   */
  const syncPeriod = useCallback(async (dateFrom: string, dateTo: string, force = false) => {
    setIsLoading(true)
    setError(null)
    setSyncResult(null)

    try {
      const response = await fetch(`${API_BASE_URL}/api/abutre/sync/trigger`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          date_from: `${dateFrom}T00:00:00`,
          date_to: `${dateTo}T23:59:59`,
          force
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()

      setSyncResult({
        success: data.status === 'success',
        message: data.message,
        trades_synced: data.trades_synced,
        trades_failed: data.trades_failed
      })

      console.log(`[useHistoricalData] Sync completed: ${data.trades_synced} trades`)

      return data
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error'
      setError(errorMessage)
      console.error('[useHistoricalData] Error syncing period:', err)

      setSyncResult({
        success: false,
        message: errorMessage
      })

      return {
        success: false,
        error: errorMessage
      }
    } finally {
      setIsLoading(false)
    }
  }, [])

  /**
   * Sincronização rápida dos últimos N dias
   */
  const quickSync = useCallback(async (days: number) => {
    setIsLoading(true)
    setError(null)
    setSyncResult(null)

    try {
      const response = await fetch(`${API_BASE_URL}/api/abutre/sync/quick/${days}`, {
        method: 'GET'
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()

      setSyncResult({
        success: data.status === 'success',
        message: data.message,
        trades_synced: data.trades_synced,
        trades_failed: data.trades_failed
      })

      console.log(`[useHistoricalData] Quick sync (${days}d): ${data.trades_synced} trades`)

      // Após sincronizar, buscar os trades
      if (data.status === 'success') {
        const dateTo = new Date().toISOString().split('T')[0]
        const dateFrom = new Date()
        dateFrom.setDate(dateFrom.getDate() - days)
        const dateFromStr = dateFrom.toISOString().split('T')[0]

        await fetchTradesByPeriod(dateFromStr, dateTo)
      }

      return data
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error'
      setError(errorMessage)
      console.error('[useHistoricalData] Error in quick sync:', err)

      setSyncResult({
        success: false,
        message: errorMessage
      })

      return {
        success: false,
        error: errorMessage
      }
    } finally {
      setIsLoading(false)
    }
  }, [fetchTradesByPeriod])

  return {
    isLoading,
    error,
    syncResult,
    fetchTradesByPeriod,
    syncPeriod,
    quickSync
  }
}
