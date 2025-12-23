import { useEffect, useState } from 'react'
import { useAbutreEvents } from '@/hooks/useAbutreEvents'
import { RefreshCw } from 'lucide-react'

export default function AbutreDashboard() {
  const [mounted, setMounted] = useState(false)
  const { trades, loading, refetch } = useAbutreEvents()

  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return null
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold">Abutre Bot - Histórico de Trades</h1>
            <p className="text-gray-400 mt-2">Todas as operações executadas</p>
          </div>
          <button
            onClick={() => refetch.trades()}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            Atualizar
          </button>
        </div>

        {/* Trades Table */}
        <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-800">
                <tr>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-300">ID</th>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-300">Data/Hora</th>
                  <th className="px-6 py-4 text-left text-sm font-semibold text-gray-300">Direção</th>
                  <th className="px-6 py-4 text-right text-sm font-semibold text-gray-300">Stake</th>
                  <th className="px-6 py-4 text-center text-sm font-semibold text-gray-300">Level</th>
                  <th className="px-6 py-4 text-center text-sm font-semibold text-gray-300">Resultado</th>
                  <th className="px-6 py-4 text-right text-sm font-semibold text-gray-300">Profit</th>
                  <th className="px-6 py-4 text-right text-sm font-semibold text-gray-300">Balance</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800">
                {loading && (
                  <tr>
                    <td colSpan={8} className="px-6 py-12 text-center text-gray-500">
                      Carregando trades...
                    </td>
                  </tr>
                )}

                {!loading && trades.length === 0 && (
                  <tr>
                    <td colSpan={8} className="px-6 py-12 text-center text-gray-500">
                      Nenhum trade encontrado
                    </td>
                  </tr>
                )}

                {!loading && trades.map((trade) => (
                  <tr key={trade.id} className="hover:bg-gray-800/50 transition-colors">
                    <td className="px-6 py-4 text-sm text-gray-400">
                      #{trade.id}
                    </td>
                    <td className="px-6 py-4 text-sm text-gray-300">
                      {new Date(trade.entry_time).toLocaleString('pt-BR')}
                    </td>
                    <td className="px-6 py-4">
                      <span
                        className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium ${
                          trade.direction === 'CALL'
                            ? 'bg-green-900/30 text-green-400 border border-green-800'
                            : 'bg-red-900/30 text-red-400 border border-red-800'
                        }`}
                      >
                        {trade.direction}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-right text-sm text-gray-300">
                      ${trade.initial_stake.toFixed(2)}
                    </td>
                    <td className="px-6 py-4 text-center text-sm text-gray-300">
                      {trade.max_level_reached}
                    </td>
                    <td className="px-6 py-4 text-center">
                      {trade.result ? (
                        <span
                          className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium ${
                            trade.result === 'WIN'
                              ? 'bg-green-900/30 text-green-400 border border-green-800'
                              : 'bg-red-900/30 text-red-400 border border-red-800'
                          }`}
                        >
                          {trade.result}
                        </span>
                      ) : (
                        <span className="text-gray-500 text-xs">ABERTO</span>
                      )}
                    </td>
                    <td className="px-6 py-4 text-right text-sm">
                      {trade.profit !== null ? (
                        <span className={trade.profit >= 0 ? 'text-green-400' : 'text-red-400'}>
                          {trade.profit >= 0 ? '+' : ''}${trade.profit.toFixed(2)}
                        </span>
                      ) : (
                        <span className="text-gray-500">-</span>
                      )}
                    </td>
                    <td className="px-6 py-4 text-right text-sm text-gray-300">
                      {trade.balance_after !== null ? (
                        `$${trade.balance_after.toFixed(2)}`
                      ) : (
                        <span className="text-gray-500">-</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Footer Info */}
        {!loading && trades.length > 0 && (
          <div className="mt-4 text-sm text-gray-500 text-center">
            Mostrando {trades.length} trades
          </div>
        )}
      </div>
    </div>
  )
}
