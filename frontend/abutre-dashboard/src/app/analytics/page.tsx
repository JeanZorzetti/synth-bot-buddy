'use client'

import { useState, useEffect } from 'react'
import { ArrowLeft, RefreshCw, Calendar, BarChart3 } from 'lucide-react'
import Link from 'next/link'
import {
  useAnalytics,
  SurvivalMetrics,
  PerformanceMetrics as PerformanceMetricsType,
  HourlyAnalysis,
  EquityCurveData
} from '@/hooks/useAnalytics'
import PerformanceMetrics from '@/components/PerformanceMetrics'
import SurvivalCard from '@/components/SurvivalCard'
import EquityCurveChart from '@/components/EquityCurveChart'
import HourlyHeatmap from '@/components/HourlyHeatmap'
import PeriodSelector from '@/components/PeriodSelector'

export default function AnalyticsPage() {
  const {
    isLoading,
    error,
    getSurvivalMetrics,
    getPerformanceMetrics,
    getHourlyAnalysis,
    getEquityCurve
  } = useAnalytics()

  const [survivalData, setSurvivalData] = useState<SurvivalMetrics | null>(null)
  const [performanceData, setPerformanceData] = useState<PerformanceMetricsType | null>(null)
  const [hourlyData, setHourlyData] = useState<HourlyAnalysis | null>(null)
  const [equityData, setEquityData] = useState<EquityCurveData | null>(null)

  const [period, setPeriod] = useState<{ from: string; to: string }>({
    from: '',
    to: ''
  })

  const [isRefreshing, setIsRefreshing] = useState(false)

  // Load all analytics data
  const loadAnalytics = async (dateFrom?: string, dateTo?: string) => {
    setIsRefreshing(true)

    try {
      const [survival, performance, hourly, equity] = await Promise.all([
        getSurvivalMetrics(dateFrom, dateTo),
        getPerformanceMetrics(dateFrom, dateTo),
        getHourlyAnalysis(dateFrom, dateTo),
        getEquityCurve(dateFrom, dateTo)
      ])

      setSurvivalData(survival)
      setPerformanceData(performance)
      setHourlyData(hourly)
      setEquityData(equity)
    } catch (err) {
      console.error('Error loading analytics:', err)
    } finally {
      setIsRefreshing(false)
    }
  }

  // Initial load
  useEffect(() => {
    loadAnalytics()
  }, [])

  // Handle period change
  const handlePeriodChange = (dateFrom: string, dateTo: string) => {
    setPeriod({ from: dateFrom, to: dateTo })
    loadAnalytics(dateFrom, dateTo)
  }

  // Handle refresh
  const handleRefresh = () => {
    if (period.from && period.to) {
      loadAnalytics(period.from, period.to)
    } else {
      loadAnalytics()
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <header className="border-b border-slate-700/50 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link
                href="/abutre"
                className="p-2 rounded-lg hover:bg-slate-800 transition-colors"
              >
                <ArrowLeft className="w-5 h-5 text-slate-400" />
              </Link>

              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-blue-500/10">
                  <BarChart3 className="w-6 h-6 text-blue-500" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-slate-100">
                    Abutre Analytics
                  </h1>
                  <p className="text-sm text-slate-400">
                    Análise histórica completa do trading bot
                  </p>
                </div>
              </div>
            </div>

            <button
              onClick={handleRefresh}
              disabled={isRefreshing}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 disabled:cursor-not-allowed text-white rounded-lg transition-colors"
            >
              <RefreshCw className={`w-4 h-4 ${isRefreshing ? 'animate-spin' : ''}`} />
              {isRefreshing ? 'Atualizando...' : 'Atualizar'}
            </button>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-6 py-8 space-y-6">
        {/* Period Selector */}
        <div className="rounded-xl bg-slate-800/50 border border-slate-700/50 p-6">
          <div className="flex items-center gap-2 mb-4">
            <Calendar className="w-5 h-5 text-slate-400" />
            <h2 className="text-lg font-semibold text-slate-300">
              Selecionar Período
            </h2>
          </div>

          <PeriodSelector
            onPeriodChange={handlePeriodChange}
            onSync={async () => {
              // This will be called by period selector's sync button
              // For analytics, we just reload the data
              await loadAnalytics(period.from, period.to)
            }}
          />
        </div>

        {/* Error Display */}
        {error && (
          <div className="rounded-xl bg-red-500/10 border border-red-500/30 p-4">
            <p className="text-sm text-red-400">{error}</p>
          </div>
        )}

        {/* Performance Metrics */}
        <div>
          <h2 className="text-xl font-bold text-slate-200 mb-4">
            Performance Overview
          </h2>
          <PerformanceMetrics
            metrics={performanceData}
            isLoading={isLoading || isRefreshing}
          />
        </div>

        {/* Survival Analysis */}
        <div>
          <h2 className="text-xl font-bold text-slate-200 mb-4">
            Análise de Sobrevivência
          </h2>
          <SurvivalCard
            metrics={survivalData}
            isLoading={isLoading || isRefreshing}
          />
        </div>

        {/* Equity Curve */}
        <div>
          <h2 className="text-xl font-bold text-slate-200 mb-4">
            Evolução do Capital
          </h2>
          <EquityCurveChart
            data={equityData}
            isLoading={isLoading || isRefreshing}
          />
        </div>

        {/* Hourly Heatmap */}
        <div>
          <h2 className="text-xl font-bold text-slate-200 mb-4">
            Análise por Horário
          </h2>
          <HourlyHeatmap
            data={hourlyData}
            isLoading={isLoading || isRefreshing}
          />
        </div>

        {/* Footer Info */}
        <div className="rounded-xl bg-blue-500/5 border border-blue-500/20 p-6">
          <div className="flex items-start gap-3">
            <BarChart3 className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
            <div className="space-y-2">
              <p className="text-sm font-medium text-blue-400">
                Sobre este Dashboard
              </p>
              <ul className="text-xs text-slate-400 space-y-1 list-disc list-inside">
                <li>
                  <strong className="text-slate-300">Performance:</strong> Win rate, profit factor, drawdown e streaks
                </li>
                <li>
                  <strong className="text-slate-300">Sobrevivência:</strong> Análise de níveis críticos de Martingale e recuperação
                </li>
                <li>
                  <strong className="text-slate-300">Equity Curve:</strong> Evolução do capital ao longo do tempo
                </li>
                <li>
                  <strong className="text-slate-300">Heatmap Horário:</strong> Padrões de risco por hora do dia
                </li>
              </ul>
              <p className="text-xs text-slate-500 pt-2">
                💡 Use o seletor de período para analisar intervalos específicos ou visualizar todo o histórico
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
