import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, AlertCircle, Loader2, Target, DollarSign, Clock, BarChart3 } from 'lucide-react';
import { apiService } from '../services/api';

interface ContractType {
  value: string;
  label: string;
  description: string;
  icon: React.ElementType;
  color: string;
  needsBarrier?: boolean;
}

interface TradingBuyFormProps {
  symbol?: string;
  onTradeSuccess?: (result: any) => void;
  onTradeError?: (error: string) => void;
  disabled?: boolean;
}

const CONTRACT_TYPES: ContractType[] = [
  {
    value: 'CALL',
    label: 'Rise (Call)',
    description: 'Preço final será maior que preço de entrada',
    icon: TrendingUp,
    color: 'text-green-600'
  },
  {
    value: 'PUT',
    label: 'Fall (Put)',
    description: 'Preço final será menor que preço de entrada',
    icon: TrendingDown,
    color: 'text-red-600'
  },
  {
    value: 'DIGITEVEN',
    label: 'Even',
    description: 'Último dígito será par (0,2,4,6,8)',
    icon: Target,
    color: 'text-blue-600'
  },
  {
    value: 'DIGITODD',
    label: 'Odd',
    description: 'Último dígito será ímpar (1,3,5,7,9)',
    icon: Target,
    color: 'text-purple-600'
  },
  {
    value: 'DIGITOVER',
    label: 'Over',
    description: 'Último dígito será maior que barreira',
    icon: BarChart3,
    color: 'text-orange-600',
    needsBarrier: true
  },
  {
    value: 'DIGITUNDER',
    label: 'Under',
    description: 'Último dígito será menor que barreira',
    icon: BarChart3,
    color: 'text-indigo-600',
    needsBarrier: true
  }
];

const SYMBOLS = [
  { value: 'R_10', label: 'Volatility 10 Index', volatility: '~10%' },
  { value: 'R_25', label: 'Volatility 25 Index', volatility: '~25%' },
  { value: 'R_50', label: 'Volatility 50 Index', volatility: '~50%' },
  { value: 'R_75', label: 'Volatility 75 Index', volatility: '~75%' },
  { value: 'R_100', label: 'Volatility 100 Index', volatility: '~100%' },
];

const DURATIONS = [
  { value: 1, label: '1 minuto', unit: 'm' },
  { value: 2, label: '2 minutos', unit: 'm' },
  { value: 3, label: '3 minutos', unit: 'm' },
  { value: 5, label: '5 minutos', unit: 'm' },
  { value: 10, label: '10 minutos', unit: 'm' },
  { value: 15, label: '15 minutos', unit: 'm' },
  { value: 30, label: '30 minutos', unit: 'm' },
  { value: 60, label: '1 hora', unit: 'm' },
];

export const TradingBuyForm: React.FC<TradingBuyFormProps> = ({
  symbol: initialSymbol = 'R_100',
  onTradeSuccess,
  onTradeError,
  disabled = false
}) => {
  const [contractType, setContractType] = useState('CALL');
  const [symbol, setSymbol] = useState(initialSymbol);
  const [amount, setAmount] = useState(1);
  const [duration, setDuration] = useState(5);
  const [durationUnit, setDurationUnit] = useState('m');
  const [barrier, setBarrier] = useState('5');
  const [isLoading, setIsLoading] = useState(false);
  const [balance, setBalance] = useState<number | null>(null);
  const [lastTick, setLastTick] = useState<any>(null);
  const [currentProposal, setCurrentProposal] = useState<any>(null);
  const [proposalLoading, setProposalLoading] = useState(false);
  const [proposalError, setProposalError] = useState<string | null>(null);
  const [proposalUpdateInterval, setProposalUpdateInterval] = useState<NodeJS.Timeout | null>(null);

  // Get current contract type details
  const currentContractType = CONTRACT_TYPES.find(ct => ct.value === contractType);
  const IconComponent = currentContractType?.icon || TrendingUp;

  // Load balance and last tick on mount
  useEffect(() => {
    loadBalance();
    loadLastTick();
  }, [symbol]);

  // Load proposal when parameters change
  useEffect(() => {
    if (amount > 0 && duration > 0) {
      // Clear existing interval
      if (proposalUpdateInterval) {
        clearInterval(proposalUpdateInterval);
      }

      // Load initial proposal
      loadProposal(false);

      // Set up real-time updates every 3 seconds
      const interval = setInterval(() => {
        loadProposal(true);
      }, 3000);

      setProposalUpdateInterval(interval);

      // Cleanup on unmount or parameter change
      return () => {
        clearInterval(interval);
      };
    }
  }, [contractType, symbol, amount, duration, durationUnit, barrier]);

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (proposalUpdateInterval) {
        clearInterval(proposalUpdateInterval);
      }
    };
  }, []);

  const loadBalance = async () => {
    try {
      const result = await apiService.derivGetBalance();
      setBalance(result.balance);
    } catch (error) {
      console.warn('Could not load balance:', error);
    }
  };

  const loadLastTick = async () => {
    try {
      const result = await apiService.derivGetLastTick(symbol);
      setLastTick(result.tick);
    } catch (error) {
      console.warn('Could not load last tick:', error);
    }
  };

  // Load real-time proposal
  const loadProposal = async (realtime: boolean = false) => {
    if (!amount || amount <= 0 || !duration || !contractType) return;

    setProposalLoading(true);
    setProposalError(null);

    try {
      const params = {
        contract_type: contractType,
        symbol,
        amount,
        duration,
        duration_unit: durationUnit,
        barrier: currentContractType?.needsBarrier ? barrier : undefined,
        basis: 'stake',
        currency: 'USD'
      };

      const result = realtime
        ? await apiService.getRealtimeProposal(params)
        : await apiService.getProposal(params);

      if (result.status === 'success' && result.proposal) {
        setCurrentProposal(result.proposal);
        setProposalError(null);
      } else {
        setProposalError(result.message || 'Erro ao obter cotação');
        setCurrentProposal(null);
      }
    } catch (error: any) {
      console.error('Erro ao carregar proposal:', error);
      setProposalError(error.message || 'Erro ao obter cotação');
      setCurrentProposal(null);
    } finally {
      setProposalLoading(false);
    }
  };

  // Validate barrier for current contract type
  const validateBarrier = (contractType: string, barrier: string): { isValid: boolean; error?: string } => {
    if (contractType === 'DIGITOVER' || contractType === 'DIGITUNDER') {
      const barrierNum = parseInt(barrier);
      if (isNaN(barrierNum) || barrierNum < 0 || barrierNum > 9) {
        return { isValid: false, error: 'Barrier deve ser um dígito entre 0 e 9' };
      }
    }
    return { isValid: true };
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (isLoading || disabled) return;

    setIsLoading(true);

    try {
      // Validate amount
      if (amount <= 0) {
        throw new Error('Amount deve ser maior que zero');
      }

      if (balance !== null && amount > balance) {
        throw new Error(`Saldo insuficiente. Saldo atual: $${balance.toFixed(2)}`);
      }

      // Validate barrier if needed
      if (currentContractType?.needsBarrier) {
        const barrierValidation = validateBarrier(contractType, barrier);
        if (!barrierValidation.isValid) {
          throw new Error(barrierValidation.error || 'Barrier inválida');
        }
      }

      // Prepare contract parameters
      const contractParams = {
        contract_type: contractType,
        symbol,
        amount,
        duration,
        duration_unit: durationUnit,
        ...(currentContractType?.needsBarrier && barrier ? { barrier } : {}),
        basis: 'stake',
        currency: 'USD'
      };

      const result = await apiService.derivBuyContract(
        contractParams.contract_type,
        contractParams.symbol,
        contractParams.amount,
        contractParams.duration,
        contractParams.duration_unit,
        contractParams.barrier,
        contractParams.basis,
        contractParams.currency
      );

      // Update balance
      if (result.balance_after !== undefined) {
        setBalance(result.balance_after);
      }

      onTradeSuccess?.(result);

    } catch (error: any) {
      console.error('Error buying contract:', error);
      const errorMessage = error.message || 'Erro ao comprar contrato';
      onTradeError?.(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const isFormValid = () => {
    return amount > 0 &&
           duration > 0 &&
           symbol &&
           contractType &&
           (!currentContractType?.needsBarrier || barrier);
  };

  const getPotentialPayout = () => {
    // Use real proposal data if available, otherwise fallback to estimation
    if (currentProposal?.payout) {
      return currentProposal.payout;
    }

    // Simple estimation - real calculation would depend on market conditions
    const baseMultiplier = contractType.includes('DIGIT') ? 9.5 : 1.85;
    return amount * baseMultiplier;
  };

  const getCurrentAskPrice = () => {
    // Use real proposal data if available, otherwise fallback to amount
    if (currentProposal?.ask_price) {
      return currentProposal.ask_price;
    }
    return amount;
  };

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <div className="flex items-center space-x-3 mb-6">
        <div className="p-2 bg-green-100 rounded-lg">
          <DollarSign className="w-6 h-6 text-green-600" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-gray-900">Comprar Contrato</h3>
          <p className="text-sm text-gray-600">
            Configure e execute sua operação
          </p>
        </div>
      </div>

      {/* Balance Display */}
      {balance !== null && (
        <div className="bg-gray-50 rounded-lg p-3 mb-4">
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-600">Saldo Disponível:</span>
            <span className="text-lg font-semibold text-gray-900">${balance.toFixed(2)}</span>
          </div>
        </div>
      )}

      {/* Last Tick Display */}
      {lastTick && (
        <div className="bg-blue-50 rounded-lg p-3 mb-4">
          <div className="flex justify-between items-center">
            <span className="text-sm text-blue-700">Último preço ({symbol}):</span>
            <span className="text-lg font-semibold text-blue-900">{lastTick.price}</span>
          </div>
        </div>
      )}

      {/* Real-time Proposal Display */}
      <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-lg p-4 mb-4 border border-green-200">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${proposalLoading ? 'bg-yellow-500 animate-pulse' : currentProposal ? 'bg-green-500' : 'bg-gray-400'}`}></div>
            <span className="text-sm font-medium text-gray-700">
              Cotação em Tempo Real
            </span>
          </div>
          {proposalLoading && (
            <Loader2 className="w-4 h-4 animate-spin text-blue-600" />
          )}
        </div>

        {proposalError && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-3">
            <div className="flex items-center space-x-2">
              <AlertCircle className="w-4 h-4 text-red-500" />
              <span className="text-sm text-red-700">{proposalError}</span>
            </div>
          </div>
        )}

        {currentProposal && !proposalError && (
          <div className="grid grid-cols-2 gap-4">
            <div>
              <span className="text-xs text-gray-600 block">Preço de Compra</span>
              <span className="text-lg font-bold text-green-600">
                ${currentProposal.ask_price?.toFixed(2) || '0.00'}
              </span>
            </div>
            <div>
              <span className="text-xs text-gray-600 block">Pagamento Potencial</span>
              <span className="text-lg font-bold text-blue-600">
                ${currentProposal.payout?.toFixed(2) || '0.00'}
              </span>
            </div>
            {currentProposal.spot && (
              <div className="col-span-2">
                <span className="text-xs text-gray-600 block">Preço Atual (Spot)</span>
                <span className="text-sm font-medium text-gray-800">
                  {currentProposal.spot}
                </span>
              </div>
            )}
            <div className="col-span-2">
              <span className="text-xs text-gray-600 block">Lucro Potencial</span>
              <span className="text-sm font-semibold text-purple-600">
                ${((currentProposal.payout || 0) - (currentProposal.ask_price || 0)).toFixed(2)}
                {' '}({(((currentProposal.payout || 0) - (currentProposal.ask_price || 0)) / (currentProposal.ask_price || 1) * 100).toFixed(1)}%)
              </span>
            </div>
          </div>
        )}

        {!currentProposal && !proposalError && !proposalLoading && (
          <div className="text-center text-gray-500 text-sm py-2">
            Configure os parâmetros para ver a cotação
          </div>
        )}
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Contract Type Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-3">
            Tipo de Contrato
          </label>
          <div className="grid grid-cols-2 gap-3">
            {CONTRACT_TYPES.map((type) => {
              const TypeIcon = type.icon;
              return (
                <button
                  key={type.value}
                  type="button"
                  onClick={() => setContractType(type.value)}
                  className={`p-3 border rounded-lg text-left transition-colors ${
                    contractType === type.value
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:bg-gray-50'
                  }`}
                >
                  <div className="flex items-center space-x-2 mb-1">
                    <TypeIcon className={`w-4 h-4 ${type.color}`} />
                    <span className="font-medium text-sm">{type.label}</span>
                  </div>
                  <p className="text-xs text-gray-600">{type.description}</p>
                </button>
              );
            })}
          </div>
        </div>

        {/* Barrier (if needed) */}
        {currentContractType?.needsBarrier && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Barreira (0-9)
            </label>
            <select
              value={barrier}
              onChange={(e) => setBarrier(e.target.value)}
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              {[0,1,2,3,4,5,6,7,8,9].map(num => (
                <option key={num} value={num.toString()}>{num}</option>
              ))}
            </select>
          </div>
        )}

        {/* Symbol Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Símbolo
          </label>
          <select
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          >
            {SYMBOLS.map(sym => (
              <option key={sym.value} value={sym.value}>
                {sym.label} ({sym.volatility})
              </option>
            ))}
          </select>
        </div>

        {/* Amount */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Valor (USD)
          </label>
          <input
            type="number"
            min="0.35"
            max="50000"
            step="0.01"
            value={amount}
            onChange={(e) => setAmount(parseFloat(e.target.value) || 0)}
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            placeholder="1.00"
          />
          <p className="text-xs text-gray-500 mt-1">
            Mínimo: $0.35 | Máximo: $50,000
          </p>
        </div>

        {/* Duration */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Duração
          </label>
          <select
            value={`${duration}_${durationUnit}`}
            onChange={(e) => {
              const [dur, unit] = e.target.value.split('_');
              setDuration(parseInt(dur));
              setDurationUnit(unit);
            }}
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          >
            {DURATIONS.map(dur => (
              <option key={`${dur.value}_${dur.unit}`} value={`${dur.value}_${dur.unit}`}>
                {dur.label}
              </option>
            ))}
          </select>
        </div>

        {/* Trade Summary */}
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <h4 className="font-medium text-gray-900">Resumo da Operação</h4>
            {currentProposal && (
              <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded-full">
                Dados Reais
              </span>
            )}
          </div>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600">Tipo:</span>
              <span className="font-medium">{currentContractType?.label}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Símbolo:</span>
              <span className="font-medium">{symbol}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Stake:</span>
              <span className="font-medium">${amount.toFixed(2)}</span>
            </div>
            {currentProposal && (
              <div className="flex justify-between">
                <span className="text-gray-600">Preço Real:</span>
                <span className="font-medium text-blue-600">${getCurrentAskPrice().toFixed(2)}</span>
              </div>
            )}
            <div className="flex justify-between">
              <span className="text-gray-600">Duração:</span>
              <span className="font-medium">{duration}{durationUnit}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">
                {currentProposal ? 'Payout Real:' : 'Payout Estimado:'}
              </span>
              <span className={`font-medium ${currentProposal ? 'text-green-700' : 'text-green-600'}`}>
                ${getPotentialPayout().toFixed(2)}
              </span>
            </div>
            {currentProposal && (
              <div className="flex justify-between border-t pt-2 mt-2">
                <span className="text-gray-600">Lucro Potencial:</span>
                <span className="font-semibold text-purple-600">
                  ${(getPotentialPayout() - getCurrentAskPrice()).toFixed(2)}
                  {' '}({((getPotentialPayout() - getCurrentAskPrice()) / getCurrentAskPrice() * 100).toFixed(1)}%)
                </span>
              </div>
            )}
            {currentContractType?.needsBarrier && (
              <div className="flex justify-between">
                <span className="text-gray-600">Barreira:</span>
                <span className="font-medium">{barrier}</span>
              </div>
            )}
          </div>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={!isFormValid() || isLoading || disabled}
          className="w-full flex items-center justify-center space-x-2 bg-green-600 text-white py-3 px-4 rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Processando...</span>
            </>
          ) : (
            <>
              <IconComponent className="w-5 h-5" />
              <span>Comprar Contrato</span>
            </>
          )}
        </button>
      </form>
    </div>
  );
};

export default TradingBuyForm;