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
    return this.get('/');
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
    amount: number;
    duration?: number;
    symbol?: string;
  }): Promise<ApiResponse> {
    return this.post('/buy', params);
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