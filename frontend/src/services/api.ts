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
  private derivEndpointsAvailable: boolean | null = null;
  private derivEndpointsCheckTime: number = 0;
  private readonly CACHE_DURATION = 60000; // 1 minuto

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

  // =============================================
  // CONTRACT PROPOSALS ENGINE METHODS
  // =============================================

  // Get single proposal with cache
  async getProposal(params: {
    contract_type: string;
    symbol: string;
    amount: number;
    duration: number;
    duration_unit?: string;
    barrier?: string;
    basis?: string;
    currency?: string;
  }): Promise<{
    status: string;
    proposal?: {
      id: string;
      ask_price: number;
      payout: number;
      spot?: number;
      barrier?: string;
      contract_type: string;
      symbol: string;
      display_value: string;
      timestamp: number;
      valid_until?: number;
    };
    request_params?: any;
    error_code?: string;
    message?: string;
  }> {
    return this.derivApiCall('/deriv/proposal', params);
  }

  // Get real-time proposal (no cache)
  async getRealtimeProposal(params: {
    contract_type: string;
    symbol: string;
    amount: number;
    duration: number;
    duration_unit?: string;
    barrier?: string;
    basis?: string;
    currency?: string;
  }): Promise<{
    status: string;
    proposal?: {
      id: string;
      ask_price: number;
      payout: number;
      spot?: number;
      barrier?: string;
      contract_type: string;
      symbol: string;
      display_value: string;
      timestamp: number;
      valid_until?: number;
    };
    realtime: boolean;
    cache_bypassed: boolean;
    error_code?: string;
    message?: string;
  }> {
    return this.derivApiCall('/deriv/proposal/realtime', params);
  }

  // Get multiple proposals in batch
  async getBatchProposals(params: {
    proposals: Array<{
      contract_type: string;
      symbol: string;
      amount: number;
      duration: number;
      duration_unit?: string;
      barrier?: string;
      basis?: string;
      currency?: string;
    }>;
    realtime?: boolean;
  }): Promise<{
    status: string;
    proposals?: Array<{
      status: string;
      proposal?: any;
      request_index: number;
      message?: string;
    }>;
    total_requests: number;
    successful_requests: number;
    failed_requests: number;
    realtime_mode: boolean;
    error_code?: string;
    message?: string;
  }> {
    return this.derivApiCall('/deriv/proposals/batch', params);
  }

  // Get proposals engine statistics
  async getProposalsStats(): Promise<{
    status: string;
    stats?: {
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
    };
    engine_running: boolean;
    message?: string;
  }> {
    return this.derivApiCall('/deriv/proposals/stats');
  }

  // Reset proposals engine statistics
  async resetProposalsStats(): Promise<{
    status: string;
    message: string;
  }> {
    return this.derivApiCall('/deriv/proposals/reset-stats', {}, 'POST');
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

  // Verificar se endpoints Deriv estão disponíveis (com cache)
  async checkDerivEndpointsAvailable(): Promise<boolean> {
    // Para desenvolvimento local, sempre retornar true
    if (this.baseUrl.includes('localhost')) {
      console.log('🚀 Backend local detectado - Deriv endpoints ativados');
      return true;
    }

    // Para produção roilabs.com.br, sempre retornar true (já sabemos que existe)
    if (this.baseUrl.includes('roilabs.com.br')) {
      console.log('🌐 Backend de produção detectado - Deriv endpoints ativados');
      return true;
    }

    const now = Date.now();
    
    // Usar cache se ainda válido
    if (this.derivEndpointsAvailable !== null && 
        (now - this.derivEndpointsCheckTime) < this.CACHE_DURATION) {
      return this.derivEndpointsAvailable;
    }

    try {
      const response = await fetch(`${this.baseUrl}/routes`);
      if (response.ok) {
        const routes = await response.json();
        // Verificar se existe pelo menos um endpoint /deriv/*
        this.derivEndpointsAvailable = routes.some((route: any) => 
          route.path && route.path.startsWith('/deriv')
        );
        console.log('✅ Deriv endpoints verificados:', this.derivEndpointsAvailable);
      } else {
        this.derivEndpointsAvailable = false;
      }
    } catch (error) {
      // Não mostrar warning repetidamente
      if (this.derivEndpointsAvailable === null) {
        console.warn('Could not check Deriv endpoints availability');
      }
      this.derivEndpointsAvailable = false;
    }
    
    this.derivEndpointsCheckTime = now;
    return this.derivEndpointsAvailable;
  }

  // Wrapper para métodos Deriv com fallback gracioso
  private async derivApiCall<T>(
    endpoint: string, 
    method: 'GET' | 'POST' = 'GET', 
    data?: any,
    fallbackValue?: T
  ): Promise<T> {
    // Verificar disponibilidade antes de fazer request
    const isAvailable = await this.checkDerivEndpointsAvailable();
    if (!isAvailable) {
      if (fallbackValue !== undefined) {
        return fallbackValue;
      }
      throw new Error('DERIV_ENDPOINTS_NOT_AVAILABLE');
    }

    try {
      if (method === 'GET') {
        return await this.derivApiCallSilent<T>(endpoint);
      } else {
        return await this.derivApiCallSilent<T>(endpoint, data);
      }
    } catch (error: any) {
      if (error.message?.includes('Not Found') || error.message?.includes('404')) {
        // Invalidar cache se ainda não funciona
        this.derivEndpointsAvailable = false;
        
        if (fallbackValue !== undefined) {
          return fallbackValue;
        }
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
    barrier?: string,
    basis: string = 'stake',
    currency: string = 'USD'
  ): Promise<{
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
  }> {
    return this.post('/deriv/buy', {
      contract_type: contractType,
      symbol,
      amount,
      duration,
      duration_unit: durationUnit,
      barrier,
      basis,
      currency
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

  // =====================================================
  // DERIV OAUTH METHODS
  // =====================================================

  // Iniciar fluxo OAuth da Deriv
  async startDerivOAuthFlow(options?: {
    appId?: string;
    redirectUri?: string;
    affiliateToken?: string;
    utmCampaign?: string;
  }): Promise<{
    status: string;
    authorization_url: string;
    app_id: string;
    redirect_uri: string;
    message: string;
  }> {
    return this.post('/deriv/oauth/start', {
      app_id: options?.appId || '99188',
      redirect_uri: options?.redirectUri || 'https://botderiv.roilabs.com.br/auth',
      affiliate_token: options?.affiliateToken,
      utm_campaign: options?.utmCampaign
    });
  }

  // Processar callback OAuth da Deriv
  async handleDerivOAuthCallback(params: {
    accounts: string[];
    token1: string;
    token2?: string;
    token3?: string;
    cur1?: string;
    cur2?: string;
    cur3?: string;
  }): Promise<{
    status: string;
    message: string;
    session_data: {
      accounts: string[];
      tokens: string[];
      currencies: string[];
    };
    primary_token: string;
    primary_account: string;
  }> {
    return this.post('/deriv/oauth/callback', {
      accounts: params.accounts,
      token1: params.token1,
      token2: params.token2,
      token3: params.token3,
      cur1: params.cur1 || 'USD',
      cur2: params.cur2,
      cur3: params.cur3
    });
  }

  // Conectar usando token OAuth da Deriv
  async connectWithDerivOAuth(token: string, demo: boolean = true): Promise<{
    status: string;
    message: string;
    connection_info: any;
    auth_method: string;
    demo_mode: boolean;
  }> {
    return this.post('/deriv/oauth/connect', { token, demo });
  }

  // Obter URL de autorização OAuth da Deriv
  getDerivOAuthUrl(options?: {
    appId?: string;
    affiliateToken?: string;
    utmCampaign?: string;
  }): string {
    const appId = options?.appId || '99188';
    let url = `https://oauth.deriv.com/oauth2/authorize?app_id=${appId}`;

    if (options?.affiliateToken) {
      url += `&affiliate_token=${options.affiliateToken}`;
    }

    if (options?.utmCampaign) {
      url += `&utm_campaign=${options.utmCampaign}`;
    }

    return url;
  }

  // Armazenar dados OAuth da Deriv
  storeDerivOAuthData(sessionData: any, primaryToken: string, primaryAccount: string): void {
    localStorage.setItem('deriv_oauth_session', JSON.stringify(sessionData));
    localStorage.setItem('deriv_primary_token', primaryToken);
    localStorage.setItem('deriv_primary_account', primaryAccount);
    localStorage.setItem('deriv_oauth_timestamp', Date.now().toString());
  }

  // Recuperar dados OAuth da Deriv
  getStoredDerivOAuthData(): {
    sessionData: any | null;
    primaryToken: string | null;
    primaryAccount: string | null;
    timestamp: number | null;
  } {
    return {
      sessionData: JSON.parse(localStorage.getItem('deriv_oauth_session') || 'null'),
      primaryToken: localStorage.getItem('deriv_primary_token'),
      primaryAccount: localStorage.getItem('deriv_primary_account'),
      timestamp: localStorage.getItem('deriv_oauth_timestamp') ? parseInt(localStorage.getItem('deriv_oauth_timestamp')!) : null
    };
  }

  // Limpar sessão OAuth da Deriv
  clearDerivOAuthSession(): void {
    localStorage.removeItem('deriv_oauth_session');
    localStorage.removeItem('deriv_primary_token');
    localStorage.removeItem('deriv_primary_account');
    localStorage.removeItem('deriv_oauth_timestamp');
  }

  // Verificar se há sessão OAuth da Deriv válida
  hasValidDerivOAuthSession(): boolean {
    const { primaryToken, timestamp } = this.getStoredDerivOAuthData();

    if (!primaryToken || !timestamp) {
      return false;
    }

    // Verificar se a sessão não é muito antiga (24 horas)
    const now = Date.now();
    const sessionAge = now - timestamp;
    const maxAge = 24 * 60 * 60 * 1000; // 24 horas em milliseconds

    return sessionAge < maxAge;
  }

  // Processar parâmetros da URL do callback OAuth
  parseOAuthCallbackParams(url: string): {
    accounts: string[];
    token1: string;
    token2?: string;
    token3?: string;
    cur1?: string;
    cur2?: string;
    cur3?: string;
  } | null {
    try {
      const urlObj = new URL(url);
      const params = urlObj.searchParams;

      const accounts = params.getAll('acct1');
      const token1 = params.get('token1');

      if (!token1 || accounts.length === 0) {
        return null;
      }

      return {
        accounts,
        token1,
        token2: params.get('token2') || undefined,
        token3: params.get('token3') || undefined,
        cur1: params.get('cur1') || 'USD',
        cur2: params.get('cur2') || undefined,
        cur3: params.get('cur3') || undefined
      };
    } catch {
      return null;
    }
  }
}

export const apiService = new ApiService();
export default apiService;

// Utility function to handle Deriv OAuth redirect
export const handleDerivOAuthRedirect = async () => {
  const currentUrl = window.location.href;
  const oauthParams = apiService.parseOAuthCallbackParams(currentUrl);

  if (oauthParams) {
    try {
      const result = await apiService.handleDerivOAuthCallback(oauthParams);
      apiService.storeDerivOAuthData(
        result.session_data,
        result.primary_token,
        result.primary_account
      );
      return result;
    } catch (error) {
      console.error('Erro ao processar callback OAuth da Deriv:', error);
      throw error;
    }
  }

  return null;
};

// ==========================================
// TRADES HISTORY API (FASE 7)
// ==========================================

// Helper function for authentication headers
const getAuthHeaders = (): HeadersInit => {
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
  };

  // Try to get token from localStorage
  const token = localStorage.getItem('deriv_oauth_token') ||
                localStorage.getItem('deriv_primary_token') ||
                localStorage.getItem('token');

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  return headers;
};

export interface Trade {
  id: number;
  timestamp: string;
  symbol: string;
  trade_type: string;
  entry_price: number;
  exit_price?: number;
  stake: number;
  profit_loss?: number;
  result: 'win' | 'loss' | 'pending';
  confidence?: number;
  strategy?: string;
  indicators_used?: any;
  ml_prediction?: number;
  order_flow_signal?: string;
  stop_loss?: number;
  take_profit?: number;
  exit_reason?: string;
  notes?: string;
  created_at: string;
  updated_at: string;
}

export interface TradesHistoryResponse {
  status: string;
  data: {
    trades: Trade[];
    pagination: {
      page: number;
      limit: number;
      total: number;
      total_pages: number;
      has_next: boolean;
      has_prev: boolean;
    };
  };
}

export interface TradeStatsResponse {
  status: string;
  data: {
    overall: {
      total_trades: number;
      wins: number;
      losses: number;
      pending: number;
      total_pnl: number;
      avg_pnl: number;
      max_profit: number;
      max_loss: number;
      avg_confidence: number;
      win_rate: number;
    };
    by_symbol: Array<{
      symbol: string;
      trades: number;
      wins: number;
      pnl: number;
    }>;
    by_strategy: Array<{
      strategy: string;
      trades: number;
      wins: number;
      pnl: number;
      avg_confidence: number;
    }>;
    recent_performance: Array<{
      date: string;
      trades: number;
      pnl: number;
    }>;
  };
}

export const tradesApi = {
  /**
   * Get trades history with filters and pagination
   */
  getHistory: async (params?: {
    page?: number;
    limit?: number;
    symbol?: string;
    trade_type?: string;
    result?: string;
    start_date?: string;
    end_date?: string;
    strategy?: string;
    sort_by?: string;
    sort_order?: string;
  }): Promise<TradesHistoryResponse> => {
    const queryParams = new URLSearchParams();

    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined && value !== null && value !== '') {
          queryParams.append(key, value.toString());
        }
      });
    }

    const response = await fetch(
      `${API_BASE_URL}/api/trades/history?${queryParams.toString()}`,
      {
        headers: getAuthHeaders(),
      }
    );

    if (!response.ok) {
      throw new Error('Failed to fetch trades history');
    }

    return response.json();
  },

  /**
   * Get overall trading statistics
   */
  getStats: async (): Promise<TradeStatsResponse> => {
    const response = await fetch(`${API_BASE_URL}/api/trades/stats`, {
      headers: getAuthHeaders(),
    });

    if (!response.ok) {
      throw new Error('Failed to fetch trade stats');
    }

    return response.json();
  },

  /**
   * Get a specific trade by ID
   */
  getTrade: async (tradeId: number): Promise<{ status: string; data: Trade }> => {
    const response = await fetch(`${API_BASE_URL}/api/trades/${tradeId}`, {
      headers: getAuthHeaders(),
    });

    if (!response.ok) {
      throw new Error('Failed to fetch trade');
    }

    return response.json();
  },

  /**
   * Record a new trade
   */
  recordTrade: async (tradeData: Partial<Trade>): Promise<{ status: string; message: string; trade_id: number }> => {
    const response = await fetch(`${API_BASE_URL}/api/trades/record`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify(tradeData),
    });

    if (!response.ok) {
      throw new Error('Failed to record trade');
    }

    return response.json();
  },

  /**
   * Update a trade
   */
  updateTrade: async (tradeId: number, updateData: Partial<Trade>): Promise<{ status: string; message: string }> => {
    const response = await fetch(`${API_BASE_URL}/api/trades/${tradeId}`, {
      method: 'PUT',
      headers: getAuthHeaders(),
      body: JSON.stringify(updateData),
    });

    if (!response.ok) {
      throw new Error('Failed to update trade');
    }

    return response.json();
  },

  /**
   * Delete a trade
   */
  deleteTrade: async (tradeId: number): Promise<{ status: string; message: string }> => {
    const response = await fetch(`${API_BASE_URL}/api/trades/${tradeId}`, {
      method: 'DELETE',
      headers: getAuthHeaders(),
    });

    if (!response.ok) {
      throw new Error('Failed to delete trade');
    }

    return response.json();
  },
};

// Backtesting API interfaces
export interface EquityPoint {
  date: string;
  full_date: string;
  capital: number;
  window: string;
  window_profit: number;
  total_return_pct: number;
}

export interface EquityCurveSummary {
  initial_capital: number;
  final_capital: number;
  total_return_pct: number;
  max_drawdown_pct: number;
  n_windows: number;
  period: string;
}

export interface EquityCurveResponse {
  equity_points: EquityPoint[];
  summary: EquityCurveSummary;
  notes: string;
}

export interface BacktestWindow {
  window: string;
  start_date: string;
  end_date: string;
  train_size: number;
  test_size: number;
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1: number;
  };
  trading: {
    total_signals: number;
    total_trades: number;
    winning_trades: number;
    losing_trades: number;
    win_rate: number;
    total_profit: number;
    avg_profit_per_trade: number;
    max_drawdown: number;
    sharpe_ratio: number;
  };
}

export interface BacktestWindowsResponse {
  windows: BacktestWindow[];
  summary: {
    total_windows: number;
    avg_metrics: {
      accuracy: number;
      precision: number;
      recall: number;
      f1: number;
    };
    avg_trading: {
      win_rate: number;
      total_profit: number;
      sharpe_ratio: number;
      max_drawdown: number;
    };
    period: string;
  };
  notes: string;
}

export const backtestingApi = {
  /**
   * Get equity curve from backtesting results
   */
  getEquityCurve: async (): Promise<EquityCurveResponse> => {
    const response = await fetch(`${API_BASE_URL}/api/ml/backtesting/equity-curve`, {
      headers: getAuthHeaders(),
    });

    if (!response.ok) {
      throw new Error('Failed to fetch equity curve');
    }

    return response.json();
  },

  /**
   * Get detailed results for each backtesting window
   */
  getWindows: async (): Promise<BacktestWindowsResponse> => {
    const response = await fetch(`${API_BASE_URL}/api/ml/backtesting/windows`, {
      headers: getAuthHeaders(),
    });

    if (!response.ok) {
      throw new Error('Failed to fetch backtest windows');
    }

    return response.json();
  },
};

// Alerts API interfaces
export interface AlertConfig {
  discord: {
    enabled: boolean;
    webhook_configured: boolean;
    webhook_url?: string;
  };
  telegram: {
    enabled: boolean;
    bot_configured: boolean;
    chat_id?: string;
    bot_token?: string;
  };
  email: {
    enabled: boolean;
    smtp_server?: string;
    smtp_port?: number;
    smtp_username?: string;
    smtp_password?: string;
    email_from?: string;
    email_to?: string[];
  };
  settings: {
    enabled_channels: string[];
    min_level: string;
  };
}

export interface AlertConfigResponse {
  status: string;
  config: AlertConfig;
}

export interface AlertHistoryItem {
  timestamp: string;
  title: string;
  message: string;
  level: string;
  channels: string[];
}

export interface AlertHistoryResponse {
  status: string;
  history: AlertHistoryItem[];
  total: number;
}

export const alertsApi = {
  /**
   * Get current alerts configuration
   */
  getConfig: async (): Promise<AlertConfigResponse> => {
    const response = await fetch(`${API_BASE_URL}/api/alerts/config`, {
      headers: getAuthHeaders(),
    });

    if (!response.ok) {
      throw new Error('Failed to fetch alerts config');
    }

    return response.json();
  },

  /**
   * Update alerts configuration
   */
  updateConfig: async (config: Partial<AlertConfig>): Promise<{ status: string; message: string }> => {
    const response = await fetch(`${API_BASE_URL}/api/alerts/config`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify(config),
    });

    if (!response.ok) {
      throw new Error('Failed to update alerts config');
    }

    return response.json();
  },

  /**
   * Send test alert
   */
  sendTest: async (testData: {
    channel: string;
    title?: string;
    message?: string;
    level?: string;
  }): Promise<{ status: string; message: string; channel: string; title: string }> => {
    const response = await fetch(`${API_BASE_URL}/api/alerts/test`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify(testData),
    });

    if (!response.ok) {
      throw new Error('Failed to send test alert');
    }

    return response.json();
  },

  /**
   * Get alerts history
   */
  getHistory: async (): Promise<AlertHistoryResponse> => {
    const response = await fetch(`${API_BASE_URL}/api/alerts/history`, {
      headers: getAuthHeaders(),
    });

    if (!response.ok) {
      throw new Error('Failed to fetch alerts history');
    }

    return response.json();
  },
};