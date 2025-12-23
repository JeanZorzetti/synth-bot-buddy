'use client'

import { useState } from 'react'
import { Calendar, Loader2 } from 'lucide-react'

interface PeriodSelectorProps {
  onPeriodChange: (dateFrom: string, dateTo: string) => void
  onSync: (days?: number) => Promise<void>
  isLoading?: boolean
}

export default function PeriodSelector({ onPeriodChange, onSync, isLoading = false }: PeriodSelectorProps) {
  const [selectedPreset, setSelectedPreset] = useState<number | null>(null)
  const [customDateFrom, setCustomDateFrom] = useState('')
  const [customDateTo, setCustomDateTo] = useState('')
  const [showCustom, setShowCustom] = useState(false)

  const presets = [
    { label: 'Última Semana', days: 7 },
    { label: 'Último Mês', days: 30 },
    { label: 'Últimos 3 Meses', days: 90 },
  ]

  const handlePresetClick = async (days: number) => {
    setSelectedPreset(days)
    setShowCustom(false)

    // Calcular datas
    const dateTo = new Date()
    const dateFrom = new Date()
    dateFrom.setDate(dateFrom.getDate() - days)

    const dateToStr = dateTo.toISOString().split('T')[0]
    const dateFromStr = dateFrom.toISOString().split('T')[0]

    onPeriodChange(dateFromStr, dateToStr)

    // Sincronizar automaticamente
    await onSync(days)
  }

  const handleCustomPeriod = () => {
    if (!customDateFrom || !customDateTo) {
      alert('Por favor, selecione ambas as datas')
      return
    }

    const dateFrom = new Date(customDateFrom)
    const dateTo = new Date(customDateTo)

    if (dateFrom > dateTo) {
      alert('Data inicial deve ser anterior à data final')
      return
    }

    const diffDays = Math.ceil((dateTo.getTime() - dateFrom.getTime()) / (1000 * 60 * 60 * 24))

    if (diffDays > 90) {
      alert('Período máximo permitido: 90 dias')
      return
    }

    setSelectedPreset(null)
    onPeriodChange(customDateFrom, customDateTo)
  }

  return (
    <div className="rounded-xl bg-slate-800/50 border border-slate-700/50 p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Calendar className="w-5 h-5 text-sky-500" />
          <h3 className="text-lg font-semibold">Período de Análise</h3>
        </div>
        <button
          onClick={() => setShowCustom(!showCustom)}
          className="text-sm text-sky-400 hover:text-sky-300 transition-colors"
        >
          {showCustom ? 'Presets' : 'Período Customizado'}
        </button>
      </div>

      {!showCustom ? (
        /* Presets */
        <div className="grid grid-cols-3 gap-3">
          {presets.map((preset) => (
            <button
              key={preset.days}
              onClick={() => handlePresetClick(preset.days)}
              disabled={isLoading}
              className={`
                px-4 py-3 rounded-lg font-medium transition-all
                ${selectedPreset === preset.days
                  ? 'bg-sky-500 text-white shadow-lg shadow-sky-500/20'
                  : 'bg-slate-700/50 text-slate-300 hover:bg-slate-700 hover:text-white'
                }
                ${isLoading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
              `}
            >
              {isLoading && selectedPreset === preset.days ? (
                <div className="flex items-center justify-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Sincronizando...</span>
                </div>
              ) : (
                preset.label
              )}
            </button>
          ))}
        </div>
      ) : (
        /* Custom Period */
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-slate-400 mb-2">Data Inicial</label>
              <input
                type="date"
                value={customDateFrom}
                onChange={(e) => setCustomDateFrom(e.target.value)}
                max={new Date().toISOString().split('T')[0]}
                className="w-full px-3 py-2 rounded-lg bg-slate-700 border border-slate-600 text-slate-100 focus:outline-none focus:border-sky-500"
              />
            </div>
            <div>
              <label className="block text-sm text-slate-400 mb-2">Data Final</label>
              <input
                type="date"
                value={customDateTo}
                onChange={(e) => setCustomDateTo(e.target.value)}
                max={new Date().toISOString().split('T')[0]}
                className="w-full px-3 py-2 rounded-lg bg-slate-700 border border-slate-600 text-slate-100 focus:outline-none focus:border-sky-500"
              />
            </div>
          </div>
          <button
            onClick={handleCustomPeriod}
            disabled={isLoading || !customDateFrom || !customDateTo}
            className={`
              w-full px-4 py-3 rounded-lg font-medium transition-all
              ${isLoading || !customDateFrom || !customDateTo
                ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
                : 'bg-sky-500 text-white hover:bg-sky-600 shadow-lg shadow-sky-500/20'
              }
            `}
          >
            {isLoading ? (
              <div className="flex items-center justify-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Buscando Trades...</span>
              </div>
            ) : (
              'Buscar Período'
            )}
          </button>
        </div>
      )}

      {/* Info */}
      <div className="mt-4 p-3 rounded-lg bg-slate-900/50 border border-slate-700/30">
        <p className="text-xs text-slate-400">
          💡 <strong>Dica:</strong> Use os presets para sincronizar rapidamente, ou selecione um período customizado (máximo 90 dias).
        </p>
      </div>
    </div>
  )
}
