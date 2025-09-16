import React, { useState, useEffect } from 'react';
import { BarChart3, TrendingUp, Zap, Target, RefreshCw, AlertTriangle } from 'lucide-react';
import { apiService } from '../services/api';

interface ProposalsStatsProps {
  className?: string;
}

interface ProposalsStats {
  total_requests: number;
  cache_hits: number;
  cache_misses: number;
  validation_errors: number;
  api_errors: number;
  cache: {
    size: number;
    max_size: number;
    hit_rate: number;
    oldest_entry?: number;
  };
  price_cache_symbols: number;
  hit_rate: number;
  error_rate: number;
}

export const ProposalsStatsPanel: React.FC<ProposalsStatsProps> = ({
  className = ""
}) => {
  const [stats, setStats] = useState<ProposalsStats | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [engineRunning, setEngineRunning] = useState(false);

  const loadStats = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const result = await apiService.getProposalsStats();

      if (result.status === 'success' && result.stats) {
        setStats(result.stats);
        setEngineRunning(result.engine_running);
      } else {
        setError(result.message || 'Erro ao carregar estatísticas');
      }
    } catch (err: any) {
      setError(err.message || 'Erro ao carregar estatísticas');
    } finally {
      setIsLoading(false);
    }
  };

  const resetStats = async () => {
    try {
      const result = await apiService.resetProposalsStats();
      if (result.status === 'success') {
        await loadStats(); // Reload stats after reset
      }
    } catch (err: any) {
      setError(err.message || 'Erro ao resetar estatísticas');
    }
  };

  useEffect(() => {
    loadStats();

    // Auto-refresh every 10 seconds
    const interval = setInterval(loadStats, 10000);

    return () => clearInterval(interval);
  }, []);

  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  const formatNumber = (value: number) => {
    return value.toLocaleString();
  };

  if (error) {
    return (
      <div className={`bg-red-50 border border-red-200 rounded-lg p-4 ${className}`}>
        <div className="flex items-center space-x-2">
          <AlertTriangle className="w-5 h-5 text-red-500" />
          <span className="text-red-700 font-medium">Erro nas Estatísticas</span>
        </div>
        <p className="text-red-600 text-sm mt-2">{error}</p>
        <button
          onClick={loadStats}
          className="mt-3 text-red-600 hover:text-red-800 text-sm underline"
        >
          Tentar novamente
        </button>
      </div>
    );
  }

  return (
    <div className={`bg-white rounded-lg shadow-sm border border-gray-200 p-6 ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-blue-100 rounded-lg">
            <BarChart3 className="w-6 h-6 text-blue-600" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-900">
              Proposals Engine Stats
            </h3>
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${engineRunning ? 'bg-green-500' : 'bg-red-500'}`}></div>
              <span className="text-sm text-gray-600">
                {engineRunning ? 'Ativo' : 'Inativo'}
              </span>
            </div>
          </div>
        </div>
        <div className="flex space-x-2">
          <button
            onClick={loadStats}
            disabled={isLoading}
            className="p-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition-colors"
            title="Atualizar"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
          <button
            onClick={resetStats}
            disabled={isLoading}
            className="p-2 text-orange-600 hover:text-orange-800 hover:bg-orange-100 rounded-lg transition-colors"
            title="Resetar Estatísticas"
          >
            <Target className="w-4 h-4" />
          </button>
        </div>
      </div>

      {isLoading && !stats ? (
        <div className="text-center py-8">
          <RefreshCw className="w-8 h-8 animate-spin text-blue-600 mx-auto mb-2" />
          <p className="text-gray-600">Carregando estatísticas...</p>
        </div>
      ) : stats ? (
        <div className="space-y-6">
          {/* Performance Metrics */}
          <div>
            <h4 className="text-sm font-semibold text-gray-700 mb-3">Performance</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-green-50 rounded-lg p-3">
                <div className="flex items-center space-x-2 mb-1">
                  <TrendingUp className="w-4 h-4 text-green-600" />
                  <span className="text-xs text-green-700 font-medium">Hit Rate</span>
                </div>
                <span className="text-lg font-bold text-green-800">
                  {formatPercentage(stats.hit_rate)}
                </span>
              </div>

              <div className="bg-blue-50 rounded-lg p-3">
                <div className="flex items-center space-x-2 mb-1">
                  <Zap className="w-4 h-4 text-blue-600" />
                  <span className="text-xs text-blue-700 font-medium">Total Requests</span>
                </div>
                <span className="text-lg font-bold text-blue-800">
                  {formatNumber(stats.total_requests)}
                </span>
              </div>

              <div className="bg-purple-50 rounded-lg p-3">
                <div className="flex items-center space-x-2 mb-1">
                  <Target className="w-4 h-4 text-purple-600" />
                  <span className="text-xs text-purple-700 font-medium">Cache Hits</span>
                </div>
                <span className="text-lg font-bold text-purple-800">
                  {formatNumber(stats.cache_hits)}
                </span>
              </div>

              <div className="bg-orange-50 rounded-lg p-3">
                <div className="flex items-center space-x-2 mb-1">
                  <AlertTriangle className="w-4 h-4 text-orange-600" />
                  <span className="text-xs text-orange-700 font-medium">Error Rate</span>
                </div>
                <span className="text-lg font-bold text-orange-800">
                  {formatPercentage(stats.error_rate)}
                </span>
              </div>
            </div>
          </div>

          {/* Detailed Stats */}
          <div>
            <h4 className="text-sm font-semibold text-gray-700 mb-3">Detalhes</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Cache Hits:</span>
                  <span className="font-medium">{formatNumber(stats.cache_hits)}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Cache Misses:</span>
                  <span className="font-medium">{formatNumber(stats.cache_misses)}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Validation Errors:</span>
                  <span className="font-medium text-red-600">{formatNumber(stats.validation_errors)}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">API Errors:</span>
                  <span className="font-medium text-red-600">{formatNumber(stats.api_errors)}</span>
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Cache Size:</span>
                  <span className="font-medium">
                    {stats.cache.size} / {stats.cache.max_size}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Cache Hit Rate:</span>
                  <span className="font-medium">{formatPercentage(stats.cache.hit_rate)}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Price Cache Symbols:</span>
                  <span className="font-medium">{formatNumber(stats.price_cache_symbols)}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Progress Bars */}
          <div>
            <h4 className="text-sm font-semibold text-gray-700 mb-3">Cache Usage</h4>
            <div className="space-y-2">
              <div>
                <div className="flex justify-between text-xs text-gray-600 mb-1">
                  <span>Cache Utilization</span>
                  <span>{((stats.cache.size / stats.cache.max_size) * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${(stats.cache.size / stats.cache.max_size) * 100}%` }}
                  ></div>
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs text-gray-600 mb-1">
                  <span>Hit Rate</span>
                  <span>{formatPercentage(stats.hit_rate)}</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-green-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${stats.hit_rate * 100}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="text-center py-8 text-gray-500">
          Nenhuma estatística disponível
        </div>
      )}
    </div>
  );
};

export default ProposalsStatsPanel;