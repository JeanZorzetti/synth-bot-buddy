/**
 * ABUTRE DASHBOARD - WebSocket Hook
 *
 * React hook to manage WebSocket connection and sync with Zustand store
 */

'use client'

import { useEffect, useRef } from 'react'
import { useDashboard } from './useDashboard'
import { initializeWebSocket, getWebSocketClient, WebSocketClient } from '@/lib/websocket-client.ts'

export function useWebSocket() {
  const wsClientRef = useRef<WebSocketClient | null>(null)
  const isInitializedRef = useRef(false)

  // Zustand store actions
  const {
    setConnected,
    setBotStatus,
    updateBalance,
    updatePosition,
    updateMarketData,
    updateRiskStats,
    addTrade,
    updateTrade,
    addEvent,
  } = useDashboard()

  useEffect(() => {
    // Prevent double initialization in development (React StrictMode)
    if (isInitializedRef.current) return

    isInitializedRef.current = true

    // Initialize WebSocket connection
    const connectWebSocket = async () => {
      try {
        console.log('[useWebSocket] Initializing WebSocket connection...')

        const client = await initializeWebSocket({
          // Connection status
          onConnected: (connected) => {
            console.log('[useWebSocket] Connection status:', connected)
            setConnected(connected)
          },

          // Balance updates
          onBalanceUpdate: (data) => {
            console.log('[useWebSocket] Balance update:', data)
            updateBalance(data.balance)
          },

          // New candle
          onNewCandle: (data) => {
            console.log('[useWebSocket] New candle:', data)
            // Could add candle to a history buffer if needed
          },

          // Trigger detected
          onTriggerDetected: (data) => {
            console.log('[useWebSocket] Trigger detected:', data)
            addEvent({
              id: Date.now(),
              timestamp: new Date().toISOString(),
              event_type: 'TRIGGER_DETECTED',
              severity: 'INFO',
              message: `Trigger detected: ${data.streak_count} ${data.direction} candles`,
              context: JSON.stringify(data),
            })
          },

          // Trade opened
          onTradeOpened: (data) => {
            console.log('[useWebSocket] Trade opened:', data)
            addTrade(data)
            addEvent({
              id: Date.now(),
              timestamp: new Date().toISOString(),
              event_type: 'TRADE_OPENED',
              severity: 'INFO',
              message: `Trade opened: ${data.direction} at Level ${data.max_level_reached}`,
              context: JSON.stringify(data),
            })
          },

          // Trade closed
          onTradeClosed: (data) => {
            console.log('[useWebSocket] Trade closed:', data)
            updateTrade(data.trade_id, data)

            const severity = data.result === 'WIN' ? 'INFO' :
                           data.result === 'STOP_LOSS' ? 'WARNING' : 'ERROR'

            addEvent({
              id: Date.now(),
              timestamp: new Date().toISOString(),
              event_type: 'TRADE_CLOSED',
              severity,
              message: `Trade ${data.result}: ${data.profit >= 0 ? '+' : ''}$${data.profit.toFixed(2)}`,
              context: JSON.stringify(data),
            })
          },

          // Position update
          onPositionUpdate: (data) => {
            console.log('[useWebSocket] Position update:', data)
            updatePosition(data)
          },

          // System alert
          onSystemAlert: (data) => {
            console.log('[useWebSocket] System alert:', data)
            addEvent(data)
          },

          // Bot status
          onBotStatus: (data) => {
            console.log('[useWebSocket] Bot status:', data)
            setBotStatus(data.status)
            addEvent({
              id: Date.now(),
              timestamp: new Date().toISOString(),
              event_type: 'BOT_STATUS',
              severity: 'INFO',
              message: `Bot status: ${data.status.toUpperCase()} - ${data.message}`,
              context: JSON.stringify(data),
            })
          },

          // Market data
          onMarketData: (data) => {
            console.log('[useWebSocket] Market data:', data)
            updateMarketData(data)
          },

          // Risk stats
          onRiskStats: (data) => {
            console.log('[useWebSocket] Risk stats:', data)
            updateRiskStats(data)
          },
        })

        wsClientRef.current = client
        console.log('[useWebSocket] WebSocket initialized successfully')
      } catch (error) {
        console.error('[useWebSocket] Failed to initialize WebSocket:', error)
        setConnected(false)
      }
    }

    connectWebSocket()

    // Cleanup on unmount
    return () => {
      console.log('[useWebSocket] Cleaning up WebSocket connection...')
      if (wsClientRef.current) {
        wsClientRef.current.disconnect()
        wsClientRef.current = null
      }
      isInitializedRef.current = false
    }
  }, [
    setConnected,
    setBotStatus,
    updateBalance,
    updatePosition,
    updateMarketData,
    updateRiskStats,
    addTrade,
    updateTrade,
    addEvent,
  ])

  // Return WebSocket client for sending commands
  return {
    client: wsClientRef.current,
    isConnected: wsClientRef.current?.isConnected() ?? false,
  }
}
