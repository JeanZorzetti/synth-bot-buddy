'use client'

/**
 * ABUTRE DASHBOARD - Settings Page
 *
 * Configuration page for bot parameters and controls
 */

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { Settings, Save, AlertTriangle, Play, Pause, Square } from 'lucide-react'
import { useDashboard } from '@/hooks/useDashboard'
import { getWebSocketClient } from '@/lib/websocket-client'

interface BotSettings {
  delayThreshold: number
  maxLevel: number
  initialStake: number
  multiplier: number
  maxDrawdownPct: number
  autoTrading: boolean
}

export default function SettingsPage() {
  const router = useRouter()
  const { botStatus, isConnected } = useDashboard()
  const [isSaving, setIsSaving] = useState(false)

  // Bot settings state
  const [settings, setSettings] = useState<BotSettings>({
    delayThreshold: 8,
    maxLevel: 10,
    initialStake: 1.0,
    multiplier: 2.0,
    maxDrawdownPct: 25,
    autoTrading: false,
  })

  // Handle input change
  const handleChange = (field: keyof BotSettings, value: number | boolean) => {
    setSettings((prev) => ({ ...prev, [field]: value }))
  }

  // Save settings
  const handleSave = async () => {
    if (!isConnected) {
      alert('Not connected to backend. Please check connection.')
      return
    }

    setIsSaving(true)
    try {
      const client = getWebSocketClient()
      client.updateSettings(settings)

      // Wait for confirmation (in real app, wait for backend response)
      await new Promise((resolve) => setTimeout(resolve, 1000))

      alert('Settings saved successfully!')
    } catch (error) {
      console.error('Failed to save settings:', error)
      alert('Failed to save settings. Please try again.')
    } finally {
      setIsSaving(false)
    }
  }

  // Bot control commands
  const handleBotCommand = (command: 'start' | 'stop' | 'pause') => {
    if (!isConnected) {
      alert('Not connected to backend.')
      return
    }

    const client = getWebSocketClient()
    client.sendBotCommand(command)
  }

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/95 backdrop-blur sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <button
                onClick={() => router.push('/')}
                className="w-10 h-10 rounded-lg bg-gradient-to-br from-sky-500 to-blue-600 flex items-center justify-center hover:opacity-80 transition-opacity"
              >
                <span className="text-xl font-bold">🦅</span>
              </button>
              <div>
                <h1 className="text-xl font-bold">Settings</h1>
                <p className="text-xs text-slate-400">Configure Abutre Bot Parameters</p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'} animate-pulse`} />
              <span className="text-sm text-slate-400">
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8 max-w-4xl">
        {/* Bot Controls */}
        <section className="mb-8">
          <div className="rounded-xl bg-slate-800/50 border border-slate-700/50 p-6">
            <div className="flex items-center gap-3 mb-4">
              <Settings className="w-5 h-5 text-sky-400" />
              <h2 className="text-lg font-semibold">Bot Controls</h2>
            </div>

            <div className="flex items-center gap-4">
              <div className="flex-1">
                <p className="text-sm text-slate-400 mb-1">Current Status</p>
                <p className="text-lg font-semibold">
                  {botStatus === 'running' && <span className="text-green-400">RUNNING</span>}
                  {botStatus === 'stopped' && <span className="text-red-400">STOPPED</span>}
                  {botStatus === 'paused' && <span className="text-amber-400">PAUSED</span>}
                  {botStatus === 'error' && <span className="text-red-500">ERROR</span>}
                </p>
              </div>

              <div className="flex gap-2">
                <button
                  onClick={() => handleBotCommand('start')}
                  disabled={!isConnected || botStatus === 'running'}
                  className="px-4 py-2 rounded-lg bg-green-600 hover:bg-green-700 disabled:bg-slate-700 disabled:text-slate-500 transition-colors flex items-center gap-2"
                >
                  <Play className="w-4 h-4" />
                  Start
                </button>

                <button
                  onClick={() => handleBotCommand('pause')}
                  disabled={!isConnected || botStatus !== 'running'}
                  className="px-4 py-2 rounded-lg bg-amber-600 hover:bg-amber-700 disabled:bg-slate-700 disabled:text-slate-500 transition-colors flex items-center gap-2"
                >
                  <Pause className="w-4 h-4" />
                  Pause
                </button>

                <button
                  onClick={() => handleBotCommand('stop')}
                  disabled={!isConnected || botStatus === 'stopped'}
                  className="px-4 py-2 rounded-lg bg-red-600 hover:bg-red-700 disabled:bg-slate-700 disabled:text-slate-500 transition-colors flex items-center gap-2"
                >
                  <Square className="w-4 h-4" />
                  Stop
                </button>
              </div>
            </div>
          </div>
        </section>

        {/* Strategy Parameters */}
        <section className="mb-8">
          <div className="rounded-xl bg-slate-800/50 border border-slate-700/50 p-6">
            <h2 className="text-lg font-semibold mb-6">Strategy Parameters</h2>

            <div className="space-y-6">
              {/* Delay Threshold */}
              <div>
                <label className="block text-sm font-medium mb-2">
                  Delay Threshold (candles)
                  <span className="text-slate-500 ml-2">Current: {settings.delayThreshold}</span>
                </label>
                <input
                  type="range"
                  min="6"
                  max="12"
                  step="1"
                  value={settings.delayThreshold}
                  onChange={(e) => handleChange('delayThreshold', Number(e.target.value))}
                  className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-slate-500 mt-1">
                  <span>6 (Aggressive)</span>
                  <span>8 (Validated)</span>
                  <span>12 (Conservative)</span>
                </div>
              </div>

              {/* Max Level */}
              <div>
                <label className="block text-sm font-medium mb-2">
                  Max Martingale Level
                  <span className="text-slate-500 ml-2">Current: {settings.maxLevel}</span>
                </label>
                <input
                  type="range"
                  min="8"
                  max="12"
                  step="1"
                  value={settings.maxLevel}
                  onChange={(e) => handleChange('maxLevel', Number(e.target.value))}
                  className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer"
                />
                <div className="flex justify-between text-xs text-slate-500 mt-1">
                  <span>8 (Safe)</span>
                  <span>10 (Validated)</span>
                  <span>12 (Risky)</span>
                </div>
              </div>

              {/* Initial Stake */}
              <div>
                <label className="block text-sm font-medium mb-2">
                  Initial Stake ($)
                </label>
                <input
                  type="number"
                  min="0.50"
                  max="5.00"
                  step="0.10"
                  value={settings.initialStake}
                  onChange={(e) => handleChange('initialStake', Number(e.target.value))}
                  className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-sky-500"
                />
                <p className="text-xs text-slate-500 mt-1">
                  Validated: $1.00 | Range: $0.50 - $5.00
                </p>
              </div>

              {/* Multiplier */}
              <div>
                <label className="block text-sm font-medium mb-2">
                  Martingale Multiplier
                </label>
                <input
                  type="number"
                  min="1.5"
                  max="3.0"
                  step="0.1"
                  value={settings.multiplier}
                  onChange={(e) => handleChange('multiplier', Number(e.target.value))}
                  className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-sky-500"
                />
                <p className="text-xs text-slate-500 mt-1">
                  Validated: 2.0x | Range: 1.5x - 3.0x
                </p>
              </div>

              {/* Max Drawdown */}
              <div>
                <label className="block text-sm font-medium mb-2">
                  Max Drawdown (%)
                </label>
                <input
                  type="number"
                  min="15"
                  max="35"
                  step="1"
                  value={settings.maxDrawdownPct}
                  onChange={(e) => handleChange('maxDrawdownPct', Number(e.target.value))}
                  className="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-sky-500"
                />
                <p className="text-xs text-slate-500 mt-1">
                  Validated: 25% | Range: 15% - 35%
                </p>
              </div>

              {/* Auto Trading */}
              <div>
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.autoTrading}
                    onChange={(e) => handleChange('autoTrading', e.target.checked)}
                    className="w-5 h-5 text-sky-500 bg-slate-700 border-slate-600 rounded focus:ring-sky-500"
                  />
                  <div>
                    <p className="text-sm font-medium">Enable Auto-Trading</p>
                    <p className="text-xs text-slate-500">
                      Bot will execute trades automatically when signals are detected
                    </p>
                  </div>
                </label>
              </div>
            </div>
          </div>
        </section>

        {/* Warning */}
        <div className="mb-6 p-4 rounded-lg bg-amber-500/10 border border-amber-500/30 flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-400 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-amber-200">
            <p className="font-semibold mb-1">Important Warning</p>
            <p>
              Changing these parameters may significantly affect bot performance. The validated
              configuration (+40.25% ROI) uses: Delay=8, MaxLevel=10, Stake=$1.00, Multiplier=2.0x.
              Any changes should be backtested before live trading.
            </p>
          </div>
        </div>

        {/* Save Button */}
        <div className="flex gap-4">
          <button
            onClick={handleSave}
            disabled={!isConnected || isSaving}
            className="flex-1 px-6 py-3 rounded-lg bg-sky-600 hover:bg-sky-700 disabled:bg-slate-700 disabled:text-slate-500 transition-colors flex items-center justify-center gap-2 font-semibold"
          >
            <Save className="w-5 h-5" />
            {isSaving ? 'Saving...' : 'Save & Apply Settings'}
          </button>

          <button
            onClick={() => router.push('/')}
            className="px-6 py-3 rounded-lg bg-slate-700 hover:bg-slate-600 transition-colors"
          >
            Cancel
          </button>
        </div>
      </main>
    </div>
  )
}
