'use client'

import { Skull, Shield, TrendingUp, Clock, AlertTriangle } from 'lucide-react'
import type { SurvivalMetrics } from '@/hooks/useAnalytics'

interface SurvivalCardProps {
  metrics: SurvivalMetrics | null
  isLoading?: boolean
}

export default function SurvivalCard({ metrics, isLoading = false }: SurvivalCardProps) {
  if (isLoading) {
    return (
      <div className="rounded-xl bg-slate-800/50 border border-slate-700/50 p-6">
        <div className="animate-pulse">
          <div className="h-6 bg-slate-700 rounded w-1/3 mb-4"></div>
          <div className="space-y-3">
            <div className="h-4 bg-slate-700 rounded w-full"></div>
            <div className="h-4 bg-slate-700 rounded w-5/6"></div>
            <div className="h-4 bg-slate-700 rounded w-4/6"></div>
          </div>
        </div>
      </div>
    )
  }

  if (!metrics) {
    return (
      <div className="rounded-xl bg-slate-800/50 border border-slate-700/50 p-6">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-2 rounded-lg bg-slate-700/50">
            <Shield className="w-5 h-5 text-slate-400" />
          </div>
          <h3 className="text-lg font-semibold text-slate-300">Análise de Sobrevivência</h3>
        </div>
        <p className="text-sm text-slate-400">Nenhum dado disponível</p>
      </div>
    )
  }

  const { max_level_reached, max_level_frequency, death_sequences, recovery_factor, critical_hours } = metrics

  // Color coding based on max level reached
  const getLevelColor = (level: number) => {
    if (level >= 10) return 'text-red-500'
    if (level >= 7) return 'text-orange-500'
    if (level >= 5) return 'text-yellow-500'
    return 'text-green-500'
  }

  const getLevelBgColor = (level: number) => {
    if (level >= 10) return 'bg-red-500/10 border-red-500/30'
    if (level >= 7) return 'bg-orange-500/10 border-orange-500/30'
    if (level >= 5) return 'bg-yellow-500/10 border-yellow-500/30'
    return 'bg-green-500/10 border-green-500/30'
  }

  // Recovery factor interpretation
  const getRecoveryStatus = (factor: number) => {
    if (factor >= 2.0) return { label: 'Excelente', color: 'text-green-500', icon: TrendingUp }
    if (factor >= 1.0) return { label: 'Bom', color: 'text-blue-500', icon: Shield }
    if (factor >= 0.5) return { label: 'Regular', color: 'text-yellow-500', icon: AlertTriangle }
    return { label: 'Crítico', color: 'text-red-500', icon: Skull }
  }

  const recoveryStatus = getRecoveryStatus(recovery_factor)
  const RecoveryIcon = recoveryStatus.icon

  return (
    <div className="rounded-xl bg-slate-800/50 border border-slate-700/50 p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg ${getLevelBgColor(max_level_reached)}`}>
            <Shield className={`w-5 h-5 ${getLevelColor(max_level_reached)}`} />
          </div>
          <h3 className="text-lg font-semibold text-slate-300">Análise de Sobrevivência</h3>
        </div>
      </div>

      {/* Max Level Reached */}
      <div className={`p-4 rounded-lg border ${getLevelBgColor(max_level_reached)}`}>
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-slate-300">Nível Máximo Atingido</span>
          <span className={`text-2xl font-bold ${getLevelColor(max_level_reached)}`}>
            Lv {max_level_reached}
          </span>
        </div>
        <p className="text-xs text-slate-400">
          Atingido {max_level_frequency}x no período analisado
        </p>
      </div>

      {/* Recovery Factor */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-slate-300">Fator de Recuperação</span>
          <div className="flex items-center gap-2">
            <RecoveryIcon className={`w-4 h-4 ${recoveryStatus.color}`} />
            <span className={`text-sm font-semibold ${recoveryStatus.color}`}>
              {recoveryStatus.label}
            </span>
          </div>
        </div>

        {/* Progress bar */}
        <div className="relative h-2 bg-slate-700 rounded-full overflow-hidden">
          <div
            className={`absolute left-0 top-0 h-full transition-all duration-500 ${
              recovery_factor >= 2.0 ? 'bg-green-500' :
              recovery_factor >= 1.0 ? 'bg-blue-500' :
              recovery_factor >= 0.5 ? 'bg-yellow-500' : 'bg-red-500'
            }`}
            style={{ width: `${Math.min(recovery_factor * 50, 100)}%` }}
          />
        </div>

        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-400">
            Lucro / Risco Máximo: {recovery_factor.toFixed(2)}x
          </span>
          <span className={recoveryStatus.color}>
            {recovery_factor >= 1.0 ? 'Positivo' : 'Negativo'}
          </span>
        </div>
      </div>

      {/* Critical Hours */}
      {critical_hours.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Clock className="w-4 h-4 text-orange-500" />
            <span className="text-sm font-medium text-slate-300">Horários Críticos</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {critical_hours.map(hour => (
              <div
                key={hour}
                className="px-3 py-1.5 rounded-lg bg-orange-500/10 border border-orange-500/30"
              >
                <span className="text-sm font-medium text-orange-400">
                  {hour.toString().padStart(2, '0')}:00h
                </span>
              </div>
            ))}
          </div>
          <p className="text-xs text-slate-400">
            Horários com maior risco de atingir níveis altos
          </p>
        </div>
      )}

      {/* Death Sequences (Near-death experiences) */}
      {death_sequences.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Skull className="w-4 h-4 text-red-500" />
            <span className="text-sm font-medium text-slate-300">
              Sequências Críticas ({death_sequences.length})
            </span>
          </div>

          <div className="max-h-64 overflow-y-auto space-y-2 custom-scrollbar">
            {death_sequences.map((seq, idx) => (
              <div
                key={idx}
                className="p-3 rounded-lg bg-red-500/5 border border-red-500/20 hover:bg-red-500/10 transition-colors"
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs font-mono text-slate-400">
                    {seq.trade_id}
                  </span>
                  <span className={`text-sm font-bold ${getLevelColor(seq.level)}`}>
                    Lv {seq.level}
                  </span>
                </div>

                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-400">
                    Stake: ${seq.stake.toFixed(2)}
                  </span>
                  <span className="text-slate-400">
                    {new Date(seq.time).toLocaleString('pt-BR', {
                      day: '2-digit',
                      month: '2-digit',
                      hour: '2-digit',
                      minute: '2-digit'
                    })}
                  </span>
                </div>

                <div className="mt-1">
                  <span className={`text-xs font-medium ${
                    seq.result === 'win' ? 'text-green-500' : 'text-red-500'
                  }`}>
                    {seq.result === 'win' ? '✓ Recuperou' : '✗ Perdeu'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Warning if max level is critical */}
      {max_level_reached >= 7 && (
        <div className="p-4 rounded-lg bg-orange-500/10 border border-orange-500/30">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-orange-500 flex-shrink-0 mt-0.5" />
            <div className="space-y-1">
              <p className="text-sm font-medium text-orange-400">
                Alerta de Risco Alto
              </p>
              <p className="text-xs text-slate-400">
                O bot atingiu níveis críticos de Martingale. Considere reduzir o stake inicial
                ou ajustar os limites de segurança.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
