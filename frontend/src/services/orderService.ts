/**
 * Order Service - Serviço para execução de ordens na Deriv API
 * Objetivo 1: Fazer a aplicação executar uma ordem
 */

export interface OrderParams {
  token: string;
  contractType: 'CALL' | 'PUT';
  symbol: string;
  amount: number;
  duration: number;
  durationUnit?: 'm' | 'h' | 'd';
}

export interface OrderResult {
  success: boolean;
  contract_id?: number;
  buy_price?: number;
  payout?: number;
  longcode?: string;
  error?: string;
}

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * Executa uma ordem na Deriv API através do backend
 */
export const executeOrder = async (params: OrderParams): Promise<OrderResult> => {
  try {
    const response = await fetch(`${API_URL}/api/order/execute`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        token: params.token,
        contract_type: params.contractType,
        symbol: params.symbol,
        amount: params.amount,
        duration: params.duration,
        duration_unit: params.durationUnit || 'm'
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP Error: ${response.status}`);
    }

    const result: OrderResult = await response.json();
    return result;

  } catch (error) {
    console.error('Error executing order:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Erro desconhecido ao executar ordem'
    };
  }
};

/**
 * Valida o token da Deriv
 */
export const validateToken = (token: string): boolean => {
  // Token deve ter pelo menos 10 caracteres
  return token && token.trim().length >= 10;
};

/**
 * Formata o valor da ordem para display
 */
export const formatOrderAmount = (amount: number): string => {
  return `$${amount.toFixed(2)}`;
};

/**
 * Calcula o lucro potencial
 */
export const calculatePotentialProfit = (buyPrice: number, payout: number): number => {
  return payout - buyPrice;
};
