/**
 * useAbutreEvents Hook
 *
 * Consome os novos endpoints de eventos do Abutre (XML-based)
 * e mantém estado sincronizado via WebSocket
 */

import { useEffect, useState } from 'react'
import { useWebSocket } from './useWebSocket'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://botderivapi.roilabs.com.br'

interface AbutreStats {
  total_trades: number
  wins: number
  losses: number
  win_rate_pct: number
  total_profit: number
  avg_win: number
  avg_loss: number
  max_level_used: number
  current_balance: number
  roi_pct: number
}

interface AbuttreBalance {
  timestamp: string
  balance: number
  peak_balance: number
  drawdown_pct: number
  total_trades: number
  wins: number
  losses: number
  roi_pct: number
}

interface AbutreTrade {
  id: number
  trade_id: string
  contract_id: string | null
  entry_time: string
  direction: string
  initial_stake: number
  max_level_reached: number
  total_staked: number
  exit_time: string | null
  result: string | null
  profit: number | null
  balance_after: number | null
  source: string
  created_at: string
  updated_at: string
}

export function useAbutreEvents() {
  const [stats, setStats] = useState<AbutreStats | null>(null)
  const [trades, setTrades] = useState<AbutreTrade[]>([])
  const [balanceHistory, setBalanceHistory] = useState<AbuttreBalance[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // WebSocket para atualizações em tempo real
  const { lastMessage } = useWebSocket()

  // Fetch initial data
  useEffect(() => {
    fetchStats()
    fetchTrades()
    fetchBalanceHistory()
  }, [])

  // Listen to WebSocket events
  useEffect(() => {
    if (!lastMessage) return

    try {
      const data = JSON.parse(lastMessage)

      // Handle different event types
      switch (data.event) {
        case 'trade_closed':
          // Refresh stats and trades
          fetchStats()
          fetchTrades()
          break

        case 'balance_update':
          // Refresh balance history
          fetchBalanceHistory()
          break

        case 'risk_stats':
          // Update stats from WebSocket
          if (data.data) {
            setStats((prev) => ({
              ...prev,
              ...data.data,
            } as AbutreStats))
          }
          break

        default:
          break
      }
    } catch (err) {
      console.error('Error parsing WebSocket message:', err)
    }
  }, [lastMessage])

  async function fetchStats() {
    try {
      const response = await fetch(`${API_BASE_URL}/api/abutre/events/stats`)
      const result = await response.json()

      if (result.status === 'success') {
        setStats(result.data)
      }
    } catch (err) {
      console.error('Error fetching stats:', err)
      setError('Failed to fetch stats')
    } finally {
      setLoading(false)
    }
  }

  async function fetchTrades(limit = 50) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/abutre/events/trades?limit=${limit}`)
      const result = await response.json()

      if (result.status === 'success') {
        setTrades(result.data)
      }
    } catch (err) {
      console.error('Error fetching trades:', err)
      setError('Failed to fetch trades')
    }
  }

  async function fetchBalanceHistory(limit = 1000) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/abutre/events/balance_history?limit=${limit}`)
      const result = await response.json()

      if (result.status === 'success') {
        setBalanceHistory(result.data)
      }
    } catch (err) {
      console.error('Error fetching balance history:', err)
      setError('Failed to fetch balance history')
    }
  }

  return {
    stats,
    trades,
    balanceHistory,
    loading,
    error,
    refetch: {
      stats: fetchStats,
      trades: fetchTrades,
      balanceHistory: fetchBalanceHistory,
    },
  }
}
