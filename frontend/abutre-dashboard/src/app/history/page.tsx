'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { ArrowLeft, Download, RefreshCw } from 'lucide-react'
import PeriodSelector from '@/components/PeriodSelector'
import SyncStatus from '@/components/SyncStatus'
import TradesTable from '@/components/TradesTable'
import { useHistoricalData } from '@/hooks/useHistoricalData'

export default function HistoryPage() {
  const router = useRouter()
  const { isLoading, syncResult, fetchTradesByPeriod, quickSync } = useHistoricalData()

  const [trades, setTrades] = useState<any[]>([])
  const [period, setPeriod] = useState<{ from: string; to: string } | null>(null)
  const [totalTrades, setTotalTrades] = useState(0)

  const handlePeriodChange = async (dateFrom: string, dateTo: string) => {
    setPeriod({ from: dateFrom, to: dateTo })

    // Buscar trades do período
    const result = await fetchTradesByPeriod(dateFrom, dateTo)

    if (result.success) {
      setTrades(result.trades)
      setTotalTrades(result.count)
    }
  }

  const handleSync = async (days?: number) => {
    if (days) {
      // Sincronização rápida
      await quickSync(days)
    } else if (period) {
      // Sincronização customizada (não implementada ainda - requer endpoint adicional)
      console.log('Custom sync not implemented yet')
    }
  }

  const exportToCSV = () => {
    if (trades.length === 0) {
      alert('Nenhum trade para exportar')
      return
    }

    const headers = ['ID', 'Data/Hora', 'Direção', 'Stake', 'Resultado', 'Profit', 'Level']
    const rows = trades.map(trade => [
      trade.trade_id,
      new Date(trade.entry_time).toLocaleString('pt-BR'),
      trade.direction,
      trade.initial_stake,
      trade.result || 'OPEN',
      trade.profit || '0',
      trade.max_level_reached
    ])

    const csvContent = [
      headers.join(','),
      ...rows.map(row => row.join(','))
    ].join('\n')

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
    const link = document.createElement('a')
    const url = URL.createObjectURL(blob)

    link.setAttribute('href', url)
    link.setAttribute('download', `abutre_trades_${period?.from}_${period?.to}.csv`)
    link.style.visibility = 'hidden'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100">
      {/* Header */}
      <header className="border-b border-slate-800 bg-slate-900/95 backdrop-blur sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <button
                onClick={() => router.push('/')}
                className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 transition-colors"
              >
                <ArrowLeft className="w-5 h-5 text-slate-400" />
              </button>
              <div>
                <h1 className="text-xl font-bold">Histórico de Trades</h1>
                <p className="text-xs text-slate-400">Visualize e sincronize trades por período</p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              {totalTrades > 0 && (
                <>
                  <div className="px-4 py-2 rounded-lg bg-slate-800 text-sm">
                    <span className="text-slate-400">Total: </span>
                    <span className="font-bold text-sky-400">{totalTrades} trades</span>
                  </div>

                  <button
                    onClick={exportToCSV}
                    className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/20 transition-colors"
                  >
                    <Download className="w-4 h-4" />
                    Exportar CSV
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        {/* Period Selector */}
        <div className="mb-6">
          <PeriodSelector
            onPeriodChange={handlePeriodChange}
            onSync={handleSync}
            isLoading={isLoading}
          />
        </div>

        {/* Sync Status */}
        {syncResult && (
          <div className="mb-6">
            <SyncStatus result={syncResult} />
          </div>
        )}

        {/* Trades Table */}
        {period && (
          <div className="rounded-xl bg-slate-800/50 border border-slate-700/50 p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-lg font-semibold">Trades do Período</h2>
                <p className="text-sm text-slate-400">
                  {period.from} até {period.to}
                </p>
              </div>

              {trades.length > 0 && (
                <button
                  onClick={() => handlePeriodChange(period.from, period.to)}
                  disabled={isLoading}
                  className="flex items-center gap-2 px-3 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 transition-colors text-sm"
                >
                  <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
                  Atualizar
                </button>
              )}
            </div>

            {isLoading ? (
              <div className="flex items-center justify-center py-12">
                <div className="text-center">
                  <RefreshCw className="w-8 h-8 animate-spin text-sky-500 mx-auto mb-2" />
                  <p className="text-slate-400">Carregando trades...</p>
                </div>
              </div>
            ) : trades.length > 0 ? (
              <TradesTable trades={trades} maxRows={50} />
            ) : (
              <div className="text-center py-12">
                <p className="text-slate-400 mb-2">Nenhum trade encontrado neste período</p>
                <p className="text-xs text-slate-500">
                  Tente sincronizar primeiro usando os botões acima
                </p>
              </div>
            )}
          </div>
        )}

        {!period && (
          <div className="rounded-xl bg-slate-800/30 border border-slate-700/30 p-12 text-center">
            <p className="text-slate-400 mb-2">Selecione um período acima para visualizar os trades</p>
            <p className="text-xs text-slate-500">
              Use os presets rápidos ou escolha um período customizado (máximo 90 dias)
            </p>
          </div>
        )}
      </main>
    </div>
  )
}
