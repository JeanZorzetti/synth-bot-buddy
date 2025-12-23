'use client'

import { CheckCircle2, XCircle, AlertCircle } from 'lucide-react'

interface SyncStatusProps {
  result: {
    success: boolean
    message: string
    trades_synced?: number
    trades_failed?: number
  } | null
  onClose?: () => void
}

export default function SyncStatus({ result, onClose }: SyncStatusProps) {
  if (!result) return null

  const { success, message, trades_synced, trades_failed } = result

  return (
    <div className={`
      rounded-lg p-4 border
      ${success
        ? 'bg-emerald-500/10 border-emerald-500/30'
        : 'bg-red-500/10 border-red-500/30'
      }
    `}>
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0 mt-0.5">
          {success ? (
            <CheckCircle2 className="w-5 h-5 text-emerald-500" />
          ) : (
            <XCircle className="w-5 h-5 text-red-500" />
          )}
        </div>

        <div className="flex-1">
          <h4 className={`font-medium mb-1 ${success ? 'text-emerald-400' : 'text-red-400'}`}>
            {success ? 'Sincronização Concluída' : 'Erro na Sincronização'}
          </h4>

          <p className="text-sm text-slate-300 mb-2">{message}</p>

          {trades_synced !== undefined && (
            <div className="flex gap-4 text-xs text-slate-400">
              <span>✅ Sincronizados: <strong className="text-emerald-400">{trades_synced}</strong></span>
              {trades_failed !== undefined && trades_failed > 0 && (
                <span>❌ Falhas: <strong className="text-red-400">{trades_failed}</strong></span>
              )}
            </div>
          )}
        </div>

        {onClose && (
          <button
            onClick={onClose}
            className="flex-shrink-0 text-slate-400 hover:text-slate-200 transition-colors"
          >
            ✕
          </button>
        )}
      </div>
    </div>
  )
}
