/**
 * ABUTRE DASHBOARD - TypeScript Types
 */

// Bot Status
export type BotStatus = 'running' | 'stopped' | 'paused' | 'error'

// Position State
export interface PositionState {
  in_position: boolean
  entry_time: string | null
  entry_streak_size: number
  direction: 'CALL' | 'PUT' | null
  current_level: number
  current_stake: number
  total_loss: number
  contract_id: string | null
}

// Trading Signal
export interface TradingSignal {
  action: 'ENTER' | 'LEVEL_UP' | 'CLOSE' | 'WAIT'
  direction: 'CALL' | 'PUT' | null
  stake: number | null
  level: number | null
  reason: string
  timestamp: string
}

// Trade
export interface Trade {
  id: number
  trade_id: number
  entry_time: string
  entry_candle_idx: number
  entry_streak_size: number
  direction: 'CALL' | 'PUT'
  initial_stake: number
  max_level_reached: number
  contract_id: string | null
  exit_time: string | null
  result: 'WIN' | 'LOSS' | 'STOP_LOSS' | null
  profit: number | null
  balance_before: number
  balance_after: number | null
}

// Candle
export interface Candle {
  id: number
  timestamp: string
  open: number
  high: number
  low: number
  close: number
  color: 1 | -1 | 0  // 1=green, -1=red, 0=doji
  ticks_count: number
}

// Balance Snapshot
export interface BalanceSnapshot {
  id: number
  timestamp: string
  balance: number
  peak_balance: number
  drawdown_pct: number
  total_trades: number
  wins: number
  losses: number
}

// System Event
export interface SystemEvent {
  id: number
  timestamp: string
  event_type: string
  severity: 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL'
  message: string
  context: string | null
}

// Risk Stats
export interface RiskStats {
  initial_balance: number
  current_balance: number
  peak_balance: number
  current_drawdown_pct: number
  max_drawdown_pct: number
  daily_loss: number
  total_trades: number
  wins: number
  losses: number
  win_rate_pct: number
  roi_pct: number
  emergency_stop: boolean
  emergency_reason: string
}

// Market Monitor Data
export interface MarketData {
  symbol: string
  current_price: number
  last_candle_color: 1 | -1 | 0
  current_streak_count: number
  current_streak_direction: 1 | -1
  countdown_to_trigger: number
}

// WebSocket Events
export type WSEvent =
  | { type: 'balance_update'; data: { balance: number } }
  | { type: 'new_candle'; data: Candle }
  | { type: 'trigger_detected'; data: { streak_count: number; direction: string } }
  | { type: 'trade_opened'; data: Trade }
  | { type: 'trade_closed'; data: Trade }
  | { type: 'position_update'; data: PositionState }
  | { type: 'system_alert'; data: SystemEvent }
  | { type: 'bot_status'; data: { status: BotStatus; message: string } }

// Dashboard State (Zustand)
export interface DashboardState {
  // Connection
  isConnected: boolean
  botStatus: BotStatus

  // Real-time data
  currentBalance: number
  position: PositionState | null
  marketData: MarketData | null
  riskStats: RiskStats | null

  // Historical data
  trades: Trade[]
  balanceHistory: BalanceSnapshot[]
  recentEvents: SystemEvent[]

  // Actions
  setConnected: (connected: boolean) => void
  setBotStatus: (status: BotStatus) => void
  updateBalance: (balance: number) => void
  updatePosition: (position: PositionState) => void
  updateMarketData: (data: MarketData) => void
  updateRiskStats: (stats: RiskStats) => void
  addTrade: (trade: Trade) => void
  updateTrade: (tradeId: number, updates: Partial<Trade>) => void
  addBalanceSnapshot: (snapshot: BalanceSnapshot) => void
  addEvent: (event: SystemEvent) => void
}
