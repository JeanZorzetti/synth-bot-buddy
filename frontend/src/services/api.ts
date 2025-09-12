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

// OAuth interfaces
export interface OAuthStartResponse {
  status: string;
  authorization_url: string;
  state: string;
  scopes: string[];
  redirect_uri: string;
  message: string;
}

export interface OAuthCallbackResponse {
  status: string;
  message: string;
  token_info: {
    token_type: string;
    expires_in: number;
    scope: string;
    created_at: string;
  };
  user_info: {
    email?: string;
    user_id?: string;
    account_id?: string;
    [key: string]: any;
  };
  encrypted_token: string;
}

export interface OAuthTokenValidation {
  valid: boolean;
  status: string;
  message: string;
  user_info?: {
    email?: string;
    user_id?: string;
    account_id?: string;
    [key: string]: any;
  };
  token_info?: {
    valid: boolean;
    account_info: any;
    scopes: string[];
    expires_at: string;
    token_type: string;
  };
  error_type?: string;
}

export interface OAuthScopes {
  available_scopes: string[];
  default_scopes: string[];
  scope_descriptions: Record<string, string>;
  recommended_scopes: string[];
  minimal_scopes: string[];
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
    // Normalize URL to prevent double slashes
    const baseUrl = this.baseUrl.endsWith('/') ? this.baseUrl.slice(0, -1) : this.baseUrl;
    const path = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
    const url = `${baseUrl}${path}`;
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
  async connectToApi(apiToken: string): Promise<ApiResponse> {
    return this.post('/connect', { api_token: apiToken });
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

  // Token validation
  async validateToken(apiToken: string): Promise<{
    valid: boolean;
    error?: string;
    message?: string;
    state: string;
  }> {
    return this.post('/validate-token', { api_token: apiToken });
  }

  // Settings management
  async getSettings(): Promise<{
    settings: {
      stop_loss: number;
      take_profit: number;
      stake_amount: number;
      aggressiveness: string;
      indicators: {
        use_rsi: boolean;
        use_moving_averages: boolean;
        use_bollinger: boolean;
      };
      selected_assets: Record<string, boolean>;
    }
  }> {
    return this.get('/settings');
  }

  async updateSettings(settings: {
    stop_loss: number;
    take_profit: number;
    stake_amount: number;
    aggressiveness: string;
    indicators: Record<string, boolean>;
    selected_assets: Record<string, boolean>;
  }): Promise<{
    status: string;
    message: string;
    settings: any;
  }> {
    return this.post('/settings', settings);
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

  // --- OAuth 2.0 Methods ---

  // Start OAuth flow
  async startOAuthFlow(scopes?: string[], redirectUri?: string): Promise<OAuthStartResponse> {
    const requestBody: any = {};
    if (scopes) requestBody.scopes = scopes;
    if (redirectUri) requestBody.redirect_uri = redirectUri;
    
    return this.post('/oauth/start', requestBody);
  }

  // Handle OAuth callback
  async handleOAuthCallback(code: string, state: string): Promise<OAuthCallbackResponse> {
    return this.post('/oauth/callback', { code, state });
  }

  // Refresh OAuth token
  async refreshOAuthToken(refreshToken: string): Promise<{
    status: string;
    message: string;
    token_info: {
      token_type: string;
      expires_in: number;
      scope: string;
      created_at: string;
    };
    encrypted_token: string;
  }> {
    return this.post('/oauth/refresh', { refresh_token: refreshToken });
  }

  // Validate OAuth token (enhanced)
  async validateOAuthToken(accessToken: string): Promise<OAuthTokenValidation> {
    return this.post('/oauth/validate', { api_token: accessToken });
  }

  // Revoke OAuth token
  async revokeOAuthToken(accessToken: string): Promise<ApiResponse> {
    return this.post('/oauth/revoke', { api_token: accessToken });
  }

  // Get available OAuth scopes
  async getOAuthScopes(): Promise<OAuthScopes> {
    return this.get('/oauth/scopes');
  }

  // Get OAuth configuration
  async getOAuthConfig(): Promise<{
    oauth_endpoints: {
      authorize_url: string;
      token_url: string;
      user_info_url: string;
    };
    client_configuration: {
      client_id: string;
      has_client_secret: boolean;
      default_redirect_uri: string;
    };
    security_features: {
      pkce_enabled: boolean;
      state_parameter: boolean;
      token_encryption: boolean;
    };
    active_states_count: number;
  }> {
    return this.get('/oauth/config');
  }

  // Connect using OAuth encrypted token
  async connectWithOAuth(encryptedToken: string): Promise<{
    status: string;
    message: string;
    auth_method: string;
    scopes: string;
    token_expires_at: string;
  }> {
    return this.post('/connect/oauth', { encrypted_token: encryptedToken });
  }

  // --- OAuth Utility Methods ---

  // Generate OAuth authorization URL (convenience method)
  async getOAuthAuthorizationUrl(options?: {
    scopes?: string[];
    redirectUri?: string;
  }): Promise<string> {
    const response = await this.startOAuthFlow(options?.scopes, options?.redirectUri);
    return response.authorization_url;
  }

  // Check if user has valid OAuth session
  async hasValidOAuthSession(): Promise<boolean> {
    try {
      const encryptedToken = localStorage.getItem('deriv_oauth_token');
      if (!encryptedToken) return false;

      // Note: We can't validate encrypted token directly from frontend
      // This would need to be done by trying to connect with it
      return true;
    } catch {
      return false;
    }
  }

  // Store OAuth token securely in localStorage
  storeOAuthToken(encryptedToken: string, tokenInfo: any, userInfo: any): void {
    localStorage.setItem('deriv_oauth_token', encryptedToken);
    localStorage.setItem('deriv_token_info', JSON.stringify(tokenInfo));
    localStorage.setItem('deriv_user_info', JSON.stringify(userInfo));
  }

  // Retrieve stored OAuth data
  getStoredOAuthData(): {
    encryptedToken: string | null;
    tokenInfo: any | null;
    userInfo: any | null;
  } {
    return {
      encryptedToken: localStorage.getItem('deriv_oauth_token'),
      tokenInfo: JSON.parse(localStorage.getItem('deriv_token_info') || 'null'),
      userInfo: JSON.parse(localStorage.getItem('deriv_user_info') || 'null')
    };
  }

  // Clear OAuth session
  clearOAuthSession(): void {
    localStorage.removeItem('deriv_oauth_token');
    localStorage.removeItem('deriv_token_info');
    localStorage.removeItem('deriv_user_info');
  }

  // Check if OAuth token is expired
  isOAuthTokenExpired(): boolean {
    try {
      const tokenInfo = JSON.parse(localStorage.getItem('deriv_token_info') || '{}');
      if (!tokenInfo.created_at || !tokenInfo.expires_in) return true;

      const createdAt = new Date(tokenInfo.created_at);
      const expiresAt = new Date(createdAt.getTime() + (tokenInfo.expires_in * 1000));
      const now = new Date();

      // Add 5 minute buffer
      const bufferTime = 5 * 60 * 1000; // 5 minutes in milliseconds
      return now.getTime() > (expiresAt.getTime() - bufferTime);
    } catch {
      return true;
    }
  }

  // =====================================================
  // DERIV API METHODS - 16 FUNCIONALIDADES REAIS
  // =====================================================

  // Verificar se endpoints Deriv estão disponíveis
  async checkDerivEndpointsAvailable(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/routes`);
      if (response.ok) {
        const routes = await response.json();
        // Verificar se existe pelo menos um endpoint /deriv/*
        return routes.some((route: any) => 
          route.path && route.path.startsWith('/deriv')
        );
      }
      return false;
    } catch (error) {
      console.warn('Could not check Deriv endpoints availability');
      return false;
    }
  }

  // Wrapper para métodos Deriv com fallback gracioso
  private async derivApiCall<T>(
    endpoint: string, 
    method: 'GET' | 'POST' = 'GET', 
    data?: any,
    fallbackValue?: T
  ): Promise<T> {
    try {
      if (method === 'GET') {
        return await this.derivApiCallSilent<T>(endpoint);
      } else {
        return await this.derivApiCallSilent<T>(endpoint, data);
      }
    } catch (error: any) {
      if (error.message?.includes('Not Found') || error.message?.includes('404')) {
        if (fallbackValue !== undefined) {
          return fallbackValue;
        }
        // Retornar um erro mais amigável para endpoints não disponíveis
        throw new Error('DERIV_ENDPOINTS_NOT_AVAILABLE');
      }
      throw error;
    }
  }

  // Request silencioso para endpoints Deriv (sem logs de erro para 404)
  private async derivApiCallSilent<T>(endpoint: string, data?: any): Promise<T> {
    const baseUrl = this.baseUrl.endsWith('/') ? this.baseUrl.slice(0, -1) : this.baseUrl;
    const path = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
    const url = `${baseUrl}${path}`;
    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
      },
      method: data ? 'POST' : 'GET',
      body: data ? JSON.stringify(data) : undefined,
    };

    const response = await fetch(url, config);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }
    
    return await response.json();
  }

  // 1. Conectar à API Deriv
  async derivConnect(apiToken: string, demo: boolean = true): Promise<{
    status: string;
    message: string;
    connection_info: any;
  }> {
    return this.derivApiCall('/deriv/connect', 'POST', { 
      api_token: apiToken, 
      demo 
    });
  }

  // 2. Desconectar da API Deriv
  async derivDisconnect(): Promise<{
    status: string;
    message: string;
  }> {
    return this.post('/deriv/disconnect');
  }

  // 3. Obter status da conexão Deriv
  async derivGetStatus(): Promise<{
    status: string;
    connection_info: any;
    api_status: string;
    subscribed_symbols: string[];
  }> {
    return this.derivApiCall('/deriv/status', 'GET', undefined, {
      status: 'development',
      connection_info: {
        is_connected: false,
        is_authenticated: false,
        balance: 0,
        api_status: 'endpoints_not_available'
      },
      api_status: 'endpoints_not_available',
      subscribed_symbols: []
    });
  }

  // 4. Health check da API Deriv
  async derivHealthCheck(): Promise<{
    status: string;
    health: any;
    is_healthy: boolean;
  }> {
    return this.get('/deriv/health');
  }

  // 5. Obter saldo da conta
  async derivGetBalance(): Promise<{
    status: string;
    balance: number;
    currency: string;
    loginid: string;
  }> {
    return this.get('/deriv/balance');
  }

  // 6. Obter portfólio de contratos abertos
  async derivGetPortfolio(): Promise<{
    status: string;
    contracts: any[];
    count: number;
  }> {
    return this.get('/deriv/portfolio');
  }

  // 7. Obter histórico de trades
  async derivGetHistory(limit: number = 50): Promise<{
    status: string;
    transactions: any[];
    count: number;
  }> {
    return this.get(`/deriv/history?limit=${limit}`);
  }

  // 8. Obter símbolos disponíveis
  async derivGetSymbols(): Promise<{
    status: string;
    symbols: string[];
    count: number;
  }> {
    return this.derivApiCall('/deriv/symbols', 'GET', undefined, {
      status: 'development',
      symbols: ['R_10', 'R_25', 'R_50', 'R_75', 'R_100'],
      count: 5
    });
  }

  // 9. Obter informações de um símbolo
  async derivGetSymbolInfo(symbol: string): Promise<{
    status: string;
    symbol: string;
    info: any;
  }> {
    return this.get(`/deriv/symbols/${symbol}/info`);
  }

  // 10. Subscrever a ticks de um símbolo
  async derivSubscribeTicks(symbol: string): Promise<{
    status: string;
    message: string;
    symbol: string;
  }> {
    return this.post(`/deriv/subscribe/ticks/${symbol}`);
  }

  // 11. Obter último tick de um símbolo
  async derivGetLastTick(symbol: string): Promise<{
    status: string;
    symbol: string;
    tick: {
      price: number;
      timestamp: number;
      epoch: number;
    };
  }> {
    return this.get(`/deriv/ticks/${symbol}/last`);
  }

  // 12. Comprar contrato
  async derivBuyContract(
    contractType: string,
    symbol: string,
    amount: number,
    duration: number,
    durationUnit: string = 'm',
    barrier?: string
  ): Promise<{
    status: string;
    message: string;
    contract: {
      contract_id: number;
      buy_price: number;
      payout: number;
      longcode: string;
    };
  }> {
    return this.post('/deriv/buy', {
      contract_type: contractType,
      symbol,
      amount,
      duration,
      duration_unit: durationUnit,
      barrier
    });
  }

  // 13. Vender contrato
  async derivSellContract(
    contractId: number,
    price?: number
  ): Promise<{
    status: string;
    message: string;
    sale: {
      sold_for: number;
      transaction_id: number;
    };
  }> {
    return this.post('/deriv/sell', {
      contract_id: contractId,
      price
    });
  }
}

export const apiService = new ApiService();
export default apiService;