const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface BotStatus {
  is_running: boolean;
  connection_status: string;
  balance: number;
  last_tick: {
    symbol: string;
    price: number;
    timestamp: number;
  } | null;
  session_pnl: number;
  trades_count: number;
  capital_management: {
    next_amount: number;
    current_sequence: number;
    is_in_loss_sequence: boolean;
    accumulated_profit: number;
    risk_level: string;
  };
}

export interface CapitalStats {
  capital_info: {
    initial_capital: number;
    current_capital: number;
    accumulated_profit: number;
    last_profit: number;
    is_in_loss_sequence: boolean;
    loss_sequence_count: number;
    win_streak: number;
  };
  session_stats: {
    total_trades: number;
    wins: number;
    losses: number;
    win_rate: number;
    total_invested: number;
    total_returned: number;
    net_profit: number;
    max_drawdown: number;
    current_drawdown: number;
    max_sequence_length: number;
    current_sequence_length: number;
  };
  recent_trades: any[];
}

export interface RiskAssessment {
  next_amount: number;
  risk_percentage: number;
  risk_level: string;
  recommendations: string[];
}

export interface ApiResponse<T = any> {
  status: string;
  message?: string;
  data?: T;
}

class ApiService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = API_BASE_URL;
  }

  private async request<T = any>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error(`API Error [${endpoint}]:`, error);
      throw error;
    }
  }

  async get<T = any>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, { method: 'GET' });
  }

  async post<T = any>(endpoint: string, data?: any): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
    });
  }

  // Health check
  async getHealth(): Promise<{ status: string; message: string; version: string }> {
    return this.get('/health');
  }

  // Bot status
  async getBotStatus(): Promise<BotStatus> {
    return this.get('/status');
  }

  // Connection management
  async connectToApi(): Promise<ApiResponse> {
    return this.post('/connect');
  }

  async disconnectFromApi(): Promise<ApiResponse> {
    return this.post('/disconnect');
  }

  // Bot control
  async startBot(): Promise<ApiResponse> {
    return this.post('/start');
  }

  async stopBot(): Promise<ApiResponse> {
    return this.post('/stop');
  }

  // Trading
  async buyContract(params: {
    contract_type: string;
    amount?: number;
    duration?: number;
    symbol?: string;
  }): Promise<ApiResponse> {
    return this.post('/buy', params);
  }

  // Capital Management
  async getCapitalStats(): Promise<CapitalStats> {
    return this.get('/capital/stats');
  }

  async getRiskAssessment(): Promise<RiskAssessment> {
    return this.get('/capital/risk');
  }

  async getNextAmount(): Promise<{
    next_amount: number;
    risk_level: string;
    risk_percentage: number;
    is_in_loss_sequence: boolean;
    recommendations: string[];
  }> {
    return this.get('/capital/next-amount');
  }

  async resetCapitalSession(): Promise<ApiResponse> {
    return this.post('/capital/reset');
  }

  async simulateSequence(results: string[]): Promise<{
    simulation: any;
    summary: {
      total_trades: number;
      wins: number;
      losses: number;
      final_profit_loss: number;
    };
  }> {
    return this.post('/capital/simulate', results);
  }

  async getTradeHistory(): Promise<{
    trade_history: any[];
    total_trades: number;
    current_stats: CapitalStats;
  }> {
    return this.get('/capital/history');
  }

  // Utility method to check if API is available
  async isAvailable(): Promise<boolean> {
    try {
      await this.getHealth();
      return true;
    } catch {
      return false;
    }
  }
}

export const apiService = new ApiService();
export default apiService;