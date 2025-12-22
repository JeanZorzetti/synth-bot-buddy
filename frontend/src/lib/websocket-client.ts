/**
 * ABUTRE DASHBOARD - WebSocket Client
 *
 * Real-time connection to backend using Socket.IO
 * Handles all WebSocket events and updates Zustand store
 */

import { io, Socket } from 'socket.io-client'
import type { WSEvent } from '@/types'

type EventCallback = (data: any) => void

export class WebSocketClient {
  private socket: Socket | null = null
  private url: string
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 1000
  private isConnecting = false
  private eventCallbacks: Map<string, Set<EventCallback>> = new Map()

  constructor(url: string = 'http://localhost:8000') {
    this.url = url
  }

  /**
   * Connect to WebSocket server
   */
  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.socket?.connected) {
        console.log('[WebSocket] Already connected')
        resolve()
        return
      }

      if (this.isConnecting) {
        console.log('[WebSocket] Connection already in progress')
        return
      }

      this.isConnecting = true

      console.log(`[WebSocket] Connecting to ${this.url}...`)

      this.socket = io(this.url, {
        transports: ['websocket', 'polling'],
        reconnection: true,
        reconnectionDelay: this.reconnectDelay,
        reconnectionDelayMax: 5000,
        reconnectionAttempts: this.maxReconnectAttempts,
        timeout: 10000,
      })

      // Connection success
      this.socket.on('connect', () => {
        console.log('[WebSocket] Connected successfully')
        this.isConnecting = false
        this.reconnectAttempts = 0
        this.emit('connected', { connected: true })
        resolve()
      })

      // Connection error
      this.socket.on('connect_error', (error) => {
        console.error('[WebSocket] Connection error:', error.message)
        this.isConnecting = false
        this.reconnectAttempts++

        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
          console.error('[WebSocket] Max reconnection attempts reached')
          this.emit('connected', { connected: false })
          reject(new Error('Failed to connect after maximum attempts'))
        }
      })

      // Disconnection
      this.socket.on('disconnect', (reason) => {
        console.log('[WebSocket] Disconnected:', reason)
        this.emit('connected', { connected: false })

        if (reason === 'io server disconnect') {
          // Server initiated disconnect, reconnect manually
          this.socket?.connect()
        }
      })

      // Reconnection attempt
      this.socket.on('reconnect_attempt', (attempt) => {
        console.log(`[WebSocket] Reconnection attempt ${attempt}`)
      })

      // Reconnection success
      this.socket.on('reconnect', (attempt) => {
        console.log(`[WebSocket] Reconnected after ${attempt} attempts`)
        this.reconnectAttempts = 0
        this.emit('connected', { connected: true })
      })

      // Register event listeners
      this.registerEventListeners()
    })
  }

  /**
   * Register all WebSocket event listeners
   */
  private registerEventListeners() {
    if (!this.socket) return

    // Balance update
    this.socket.on('balance_update', (data) => {
      console.log('[WebSocket] balance_update:', data)
      this.emit('balance_update', data)
    })

    // New candle
    this.socket.on('new_candle', (data) => {
      console.log('[WebSocket] new_candle:', data)
      this.emit('new_candle', data)
    })

    // Trigger detected
    this.socket.on('trigger_detected', (data) => {
      console.log('[WebSocket] trigger_detected:', data)
      this.emit('trigger_detected', data)
    })

    // Trade opened
    this.socket.on('trade_opened', (data) => {
      console.log('[WebSocket] trade_opened:', data)
      this.emit('trade_opened', data)
    })

    // Trade closed
    this.socket.on('trade_closed', (data) => {
      console.log('[WebSocket] trade_closed:', data)
      this.emit('trade_closed', data)
    })

    // Position update
    this.socket.on('position_update', (data) => {
      console.log('[WebSocket] position_update:', data)
      this.emit('position_update', data)
    })

    // System alert
    this.socket.on('system_alert', (data) => {
      console.log('[WebSocket] system_alert:', data)
      this.emit('system_alert', data)
    })

    // Bot status
    this.socket.on('bot_status', (data) => {
      console.log('[WebSocket] bot_status:', data)
      this.emit('bot_status', data)
    })

    // Market data update
    this.socket.on('market_data', (data) => {
      console.log('[WebSocket] market_data:', data)
      this.emit('market_data', data)
    })

    // Risk stats update
    this.socket.on('risk_stats', (data) => {
      console.log('[WebSocket] risk_stats:', data)
      this.emit('risk_stats', data)
    })
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect() {
    if (this.socket) {
      console.log('[WebSocket] Disconnecting...')
      this.socket.disconnect()
      this.socket = null
      this.isConnecting = false
      this.emit('connected', { connected: false })
    }
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.socket?.connected ?? false
  }

  /**
   * Subscribe to event
   */
  on(event: string, callback: EventCallback) {
    if (!this.eventCallbacks.has(event)) {
      this.eventCallbacks.set(event, new Set())
    }
    this.eventCallbacks.get(event)?.add(callback)
  }

  /**
   * Unsubscribe from event
   */
  off(event: string, callback: EventCallback) {
    this.eventCallbacks.get(event)?.delete(callback)
  }

  /**
   * Emit event to all subscribers
   */
  private emit(event: string, data: any) {
    const callbacks = this.eventCallbacks.get(event)
    if (callbacks) {
      callbacks.forEach((callback) => callback(data))
    }
  }

  /**
   * Send command to backend
   */
  send(event: string, data?: any) {
    if (!this.socket?.connected) {
      console.error('[WebSocket] Not connected, cannot send:', event)
      return
    }

    console.log(`[WebSocket] Sending ${event}:`, data)
    this.socket.emit(event, data)
  }

  /**
   * Request initial state from backend
   */
  requestInitialState() {
    this.send('request_state')
  }

  /**
   * Send bot command (start/stop/pause)
   */
  sendBotCommand(command: 'start' | 'stop' | 'pause') {
    this.send('bot_command', { command })
  }

  /**
   * Update bot settings
   */
  updateSettings(settings: any) {
    this.send('update_settings', settings)
  }
}

// Singleton instance
let wsClient: WebSocketClient | null = null

/**
 * Get WebSocket client instance
 */
export function getWebSocketClient(): WebSocketClient {
  if (!wsClient) {
    // Use VITE_WS_URL for production (same as main backend)
    const url = (import.meta.env.VITE_WS_URL || 'http://localhost:8000').replace(/^ws/, 'http')
    wsClient = new WebSocketClient(url)
  }
  return wsClient
}

/**
 * Initialize WebSocket connection and connect to Zustand store
 */
export async function initializeWebSocket(callbacks: {
  onConnected?: (connected: boolean) => void
  onBalanceUpdate?: (data: any) => void
  onNewCandle?: (data: any) => void
  onTriggerDetected?: (data: any) => void
  onTradeOpened?: (data: any) => void
  onTradeClosed?: (data: any) => void
  onPositionUpdate?: (data: any) => void
  onSystemAlert?: (data: any) => void
  onBotStatus?: (data: any) => void
  onMarketData?: (data: any) => void
  onRiskStats?: (data: any) => void
}): Promise<WebSocketClient> {
  const client = getWebSocketClient()

  // Register callbacks
  if (callbacks.onConnected) {
    client.on('connected', (data) => callbacks.onConnected!(data.connected))
  }
  if (callbacks.onBalanceUpdate) {
    client.on('balance_update', callbacks.onBalanceUpdate)
  }
  if (callbacks.onNewCandle) {
    client.on('new_candle', callbacks.onNewCandle)
  }
  if (callbacks.onTriggerDetected) {
    client.on('trigger_detected', callbacks.onTriggerDetected)
  }
  if (callbacks.onTradeOpened) {
    client.on('trade_opened', callbacks.onTradeOpened)
  }
  if (callbacks.onTradeClosed) {
    client.on('trade_closed', callbacks.onTradeClosed)
  }
  if (callbacks.onPositionUpdate) {
    client.on('position_update', callbacks.onPositionUpdate)
  }
  if (callbacks.onSystemAlert) {
    client.on('system_alert', callbacks.onSystemAlert)
  }
  if (callbacks.onBotStatus) {
    client.on('bot_status', callbacks.onBotStatus)
  }
  if (callbacks.onMarketData) {
    client.on('market_data', callbacks.onMarketData)
  }
  if (callbacks.onRiskStats) {
    client.on('risk_stats', callbacks.onRiskStats)
  }

  // Connect
  await client.connect()

  // Request initial state
  client.requestInitialState()

  return client
}
