import React from 'react';
import { CheckCircle, AlertCircle, TrendingUp, DollarSign, Clock, X, BarChart3 } from 'lucide-react';

interface TradeResult {
  status: string;
  message: string;
  contract: {
    contract_id: number;
    buy_price: number;
    payout: number;
    longcode: string;
    symbol: string;
    contract_type: string;
    duration: string;
    stake_amount: number;
  };
  balance_after: number;
  timestamp: number;
  risk_info?: {
    risk_level: string;
    risk_percentage: number;
    recommended_amount: number;
    is_martingale: boolean;
  };
}

interface TradeResultModalProps {
  result: TradeResult | null;
  error: string | null;
  isOpen: boolean;
  onClose: () => void;
}

export const TradeResultModal: React.FC<TradeResultModalProps> = ({
  result,
  error,
  isOpen,
  onClose
}) => {
  if (!isOpen) return null;

  const getContractTypeLabel = (type: string) => {
    const labels: Record<string, string> = {
      'CALL': 'Rise (Call)',
      'PUT': 'Fall (Put)',
      'DIGITEVEN': 'Even',
      'DIGITODD': 'Odd',
      'DIGITOVER': 'Over',
      'DIGITUNDER': 'Under'
    };
    return labels[type] || type;
  };

  const getRiskLevelColor = (level: string) => {
    switch (level) {
      case 'LOW': return 'text-green-600 bg-green-100';
      case 'MEDIUM': return 'text-yellow-600 bg-yellow-100';
      case 'HIGH': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const formatDateTime = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleString('pt-BR');
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center space-x-3">
            {error ? (
              <AlertCircle className="w-6 h-6 text-red-600" />
            ) : (
              <CheckCircle className="w-6 h-6 text-green-600" />
            )}
            <h3 className="text-lg font-semibold text-gray-900">
              {error ? 'Erro na Operação' : 'Contrato Comprado'}
            </h3>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          {error ? (
            /* Error Display */
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex items-center space-x-2 mb-2">
                <AlertCircle className="w-5 h-5 text-red-600" />
                <span className="font-medium text-red-800">Erro</span>
              </div>
              <p className="text-red-700">{error}</p>
            </div>
          ) : result ? (
            /* Success Display */
            <div className="space-y-6">
              {/* Success Message */}
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <div className="flex items-center space-x-2 mb-2">
                  <CheckCircle className="w-5 h-5 text-green-600" />
                  <span className="font-medium text-green-800">Sucesso!</span>
                </div>
                <p className="text-green-700">{result.message}</p>
              </div>

              {/* Contract Details */}
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-medium text-gray-900 mb-3 flex items-center">
                  <BarChart3 className="w-4 h-4 mr-2" />
                  Detalhes do Contrato
                </h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">ID:</span>
                    <span className="font-mono text-gray-900">{result.contract.contract_id}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Tipo:</span>
                    <span className="font-medium">{getContractTypeLabel(result.contract.contract_type)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Símbolo:</span>
                    <span className="font-medium">{result.contract.symbol}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Duração:</span>
                    <span className="font-medium">{result.contract.duration}</span>
                  </div>
                </div>
              </div>

              {/* Financial Details */}
              <div className="bg-blue-50 rounded-lg p-4">
                <h4 className="font-medium text-gray-900 mb-3 flex items-center">
                  <DollarSign className="w-4 h-4 mr-2" />
                  Informações Financeiras
                </h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Stake Pago:</span>
                    <span className="font-medium text-red-600">-${result.contract.buy_price.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Payout Potencial:</span>
                    <span className="font-medium text-green-600">+${result.contract.payout.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Lucro Potencial:</span>
                    <span className="font-medium text-green-600">
                      +${(result.contract.payout - result.contract.buy_price).toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between border-t border-blue-200 pt-2 mt-2">
                    <span className="text-gray-600">Saldo Atual:</span>
                    <span className="font-semibold text-blue-900">${result.balance_after.toFixed(2)}</span>
                  </div>
                </div>
              </div>

              {/* Risk Information */}
              {result.risk_info && (
                <div className="bg-yellow-50 rounded-lg p-4">
                  <h4 className="font-medium text-gray-900 mb-3 flex items-center">
                    <TrendingUp className="w-4 h-4 mr-2" />
                    Gestão de Risco
                  </h4>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Nível de Risco:</span>
                      <span className={`px-2 py-1 rounded text-xs font-medium ${getRiskLevelColor(result.risk_info.risk_level)}`}>
                        {result.risk_info.risk_level}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">% do Capital:</span>
                      <span className="font-medium">{result.risk_info.risk_percentage.toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Amount Recomendado:</span>
                      <span className="font-medium">${result.risk_info.recommended_amount.toFixed(2)}</span>
                    </div>
                    {result.risk_info.is_martingale && (
                      <div className="flex justify-between">
                        <span className="text-gray-600">Estratégia:</span>
                        <span className="font-medium text-orange-600">Martingale Ativo</span>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Contract Description */}
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="font-medium text-gray-900 mb-2">Descrição do Contrato</h4>
                <p className="text-sm text-gray-700 leading-relaxed">
                  {result.contract.longcode}
                </p>
              </div>

              {/* Timestamp */}
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-600 flex items-center">
                    <Clock className="w-4 h-4 mr-1" />
                    Executado em:
                  </span>
                  <span className="font-medium">{formatDateTime(result.timestamp)}</span>
                </div>
              </div>
            </div>
          ) : null}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-gray-200">
          <button
            onClick={onClose}
            className="w-full bg-gray-600 text-white py-2 px-4 rounded-lg hover:bg-gray-700 transition-colors"
          >
            Fechar
          </button>
        </div>
      </div>
    </div>
  );
};

export default TradeResultModal;