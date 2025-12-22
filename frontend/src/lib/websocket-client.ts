/**
 * ABUTRE DASHBOARD - WebSocket Client
 *
 * Real-time connection to backend using Native WebSocket (FastAPI)
 * Handles all WebSocket events and updates Zustand store
 */

import type { WSEvent } from '@/types'

type EventCallback = (data: any) => void

// Simplified Socket interface for native WebSocket
interface SocketInterface {
  connected: boolean
  on: (event: string, callback: any) => void
  emit: (event: string, data?: any) => void
  disconnect: () => void
  connect: () => void
}

export class WebSocketClient {
  private socket: SocketInterface | null = null
  private url: string
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 1000
  private isConnecting = false
  private eventCallbacks: Map<string, Set<EventCallback>> = new Map()

  constructor(url: string = 'http://localhost:8000') {
    // Convert HTTP URL to WebSocket URL for native WebSocket (not Socket.IO)
    this.url = url.replace(/^http/, 'ws') + '/ws/abutre'
  }

  /**
   * Connect to WebSocket server (Native WebSocket, not Socket.IO)
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

      console.log(`[WebSocket] Connecting to ${this.url}... (Native WebSocket)`)

      // Use native WebSocket instead of Socket.IO
      const ws = new WebSocket(this.url)

      // Wrap native WebSocket to match Socket.IO interface
      this.socket = {
        connected: false,
        on: (event: string, callback: any) => {
          // Store callbacks for native WebSocket events
          if (!this.eventCallbacks.has(event)) {
            this.eventCallbacks.set(event, new Set())
          }
          this.eventCallbacks.get(event)?.add(callback)
        },
        emit: (event: string, data?: any) => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ event, data }))
          }
        },
        disconnect: () => ws.close(),
        connect: () => {
          // Native WebSocket doesn't have reconnect method
          console.warn('[WebSocket] Native WebSocket does not support manual reconnect')
        }
      } as any

      ws.onopen = () => {
        console.log('[WebSocket] Connected successfully')
        this.isConnecting = false
        this.reconnectAttempts = 0
        this.socket!.connected = true
        this.emit('connected', { connected: true })
        resolve()
      }

      ws.onerror = (error) => {
        console.error('[WebSocket] Connection error:', error)
        this.isConnecting = false
        this.reconnectAttempts++

        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
          console.error('[WebSocket] Max reconnection attempts reached')
          this.emit('connected', { connected: false })
          reject(new Error('Failed to connect after maximum attempts'))
        }
      }

      ws.onclose = (event) => {
        console.log('[WebSocket] Disconnected:', event.reason)
        this.socket!.connected = false
        this.emit('connected', { connected: false })

        // Auto-reconnect after delay
        if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
          setTimeout(() => {
            console.log('[WebSocket] Attempting to reconnect...')
            this.connect()
          }, this.reconnectDelay)
        }
      }

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)

          // Handle different message formats from FastAPI WebSocket
          if (message.event && message.data) {
            // Format: { event: "balance_update", data: {...} }
            this.emit(message.event, message.data)
          } else if (message.type) {
            // Format: { type: "balance_update", ...rest }
            const { type, ...data } = message
            this.emit(type, data)
          } else {
            // Direct data format
            this.emit('message', message)
          }
        } catch (error) {
          console.error('[WebSocket] Failed to parse message:', error)
        }
      }
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
