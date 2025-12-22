/**
 * ABUTRE DASHBOARD - Zustand Store
 *
 * Global state management for dashboard data
 * Handles real-time updates from WebSocket and UI state
 */

import { create } from 'zustand'
import type { DashboardState } from '@/types'

export const useDashboard = create<DashboardState>((set) => ({
  // Connection State
  isConnected: false,
  botStatus: 'stopped',

  // Real-time Data
  currentBalance: 0,
  position: null,
  marketData: null,
  riskStats: null,

  // Historical Data
  trades: [],
  balanceHistory: [],
  recentEvents: [],

  // Actions
  setConnected: (connected) => set({ isConnected: connected }),

  setBotStatus: (status) => set({ botStatus: status }),

  updateBalance: (balance) =>
    set((state) => {
      // Update balance history
      const now = new Date().toISOString()
      const lastSnapshot = state.balanceHistory[state.balanceHistory.length - 1]

      // Calculate peak and drawdown
      const peak = Math.max(
        balance,
        lastSnapshot?.peak_balance || balance
      )
      const drawdown = (peak - balance) / peak

      const newSnapshot = {
        id: state.balanceHistory.length + 1,
        timestamp: now,
        balance,
        peak_balance: peak,
        drawdown_pct: drawdown,
        total_trades: state.trades.length,
        wins: state.trades.filter((t) => t.result === 'WIN').length,
        losses: state.trades.filter((t) => t.result === 'LOSS').length,
      }

      return {
        currentBalance: balance,
        balanceHistory: [...state.balanceHistory, newSnapshot],
      }
    }),

  updatePosition: (position) => set({ position }),

  updateMarketData: (data) => set({ marketData: data }),

  updateRiskStats: (stats) => set({ riskStats: stats }),

  addTrade: (trade) =>
    set((state) => ({
      trades: [...state.trades, trade],
    })),

  updateTrade: (tradeId, updates) =>
    set((state) => ({
      trades: state.trades.map((trade) =>
        trade.trade_id === tradeId ? { ...trade, ...updates } : trade
      ),
    })),

  addBalanceSnapshot: (snapshot) =>
    set((state) => ({
      balanceHistory: [...state.balanceHistory, snapshot],
    })),

  addEvent: (event) =>
    set((state) => ({
      recentEvents: [event, ...state.recentEvents].slice(0, 50), // Keep last 50 events
    })),
}))
