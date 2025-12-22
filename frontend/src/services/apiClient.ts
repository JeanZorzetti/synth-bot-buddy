/**
 * API Client - Cliente unificado para todas as APIs do sistema
 * Integração completa com backend das Fases 6-10
 */

import axios, { AxiosInstance, AxiosResponse } from 'axios';

// Dashboard real-time interfaces
export interface AIMetrics {
  accuracy: number;
  confidence_avg: number;
  signals_generated: number;
  patterns_detected: number;
  model_version: string;
  last_prediction?: {
    direction: 'UP' | 'DOWN';
    confidence: number;
    symbol: string;
    timestamp: string;
  };
}

export interface TradingMetrics {
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  total_pnl: number;
  session_pnl: number;
  sharpe_ratio: number;
  max_drawdown: number;
  current_balance: number;
}

export interface SystemMetrics {
  uptime_hours: number;
  ticks_processed: number;
  processing_speed: number;
  api_latency: number;
  websocket_status: 'connected' | 'disconnected';
  deriv_api_status: 'connected' | 'disconnected';
}

export interface SystemLog {
  id: number;
  type: 'ai' | 'trade' | 'system';
  message: string;
  time: string;
  timestamp: string;
}

// Types para Phase 6-10 Integration
export interface RealTickData {
  timestamp: string;
  symbol: string;
  price: number;
  volume: number;
  bid?: number;
  ask?: number;
}

export interface ProcessedFeatures {
  symbol: string;
  features: Record<string, number>;
  timestamp: string;
  confidence: number;
}

export interface EnsemblePrediction {
  final_prediction: number;
  confidence: number;
  individual_predictions: ModelPrediction[];
  weight_distribution: Record<string, number>;
  consensus_level: number;
  prediction_type: string;
  timestamp: string;
}

export interface ModelPrediction {
  model_type: string;
  prediction_type: string;
  value: number;
  confidence: number;
  timestamp: string;
}

export interface TradingPosition {
  contract_id: string;
  symbol: string;
  trade_type: string;
  amount: number;
  entry_price: number;
  current_price?: number;
  unrealized_pnl?: number;
  entry_time: string;
  stop_loss?: number;
  take_profit?: number;
}

export interface MultiAssetSignal {
  primary_symbol: string;
  supporting_symbols: string[];
  signal_direction: number;
  confidence: number;
  cross_asset_strength: number;
  regime_consistency: number;
  timestamp: string;
}

export interface PortfolioMetrics {
  total_value: number;
  unrealized_pnl: number;
  realized_pnl: number;
  positions_count: number;
  cash_balance: number;
  allocation: Record<string, number>;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
}

export interface User {
  user_id: string;
  username: string;
  email: string;
  role: string;
  subscription_tier: string;
  permissions: string[];
  last_login?: string;
  organization_id?: string;
}

export interface Strategy {
  strategy_id: string;
  name: string;
  description: string;
  category: string;
  creator_id: string;
  pricing_model: string;
  price: number;
  rating: number;
  downloads: number;
  backtest_results?: any;
  supported_symbols: string[];
  tags: string[];
}

export interface APIKey {
  key_id: string;
  api_key: string;
  secret_key?: string;
  name: string;
  permissions: string[];
  rate_limit: number;
  created_at: string;
  last_used?: string;
}

class APIClient {
  private api: AxiosInstance;
  private wsConnections: Map<string, WebSocket> = new Map();
  private token: string | null = null;
  private apiKey: string | null = null;

  constructor() {
    this.api = axios.create({
      baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor para adicionar auth
    this.api.interceptors.request.use((config) => {
      if (this.token) {
        config.headers.Authorization = `Bearer ${this.token}`;
      }
      if (this.apiKey) {
        config.headers['X-API-Key'] = this.apiKey;
      }
      return config;
    });

    // Response interceptor para tratamento de erros
    this.api.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          this.clearAuth();
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  // Authentication
  async login(username: string, password: string): Promise<{ user: User; token: string }> {
    const response = await this.api.post('/auth/login', { username, password });
    this.token = response.data.access_token;
    localStorage.setItem('token', this.token);
    return response.data;
  }

  async logout(): Promise<void> {
    this.clearAuth();
    localStorage.removeItem('token');
  }

  setToken(token: string): void {
    this.token = token;
  }

  clearAuth(): void {
    this.token = null;
    this.apiKey = null;
  }

  // API Keys Management
  async createAPIKey(name: string, permissions: string[]): Promise<APIKey> {
    const response = await this.api.post('/api-keys', { name, permissions });
    return response.data;
  }

  async getAPIKeys(): Promise<APIKey[]> {
    const response = await this.api.get('/api-keys');
    return response.data;
  }

  // Dashboard Real Data Methods
  async getAIStatus(): Promise<AIMetrics> {
    const response = await this.api.get('/dashboard/ai-metrics');
    return response.data;
  }

  async getTradingMetrics(): Promise<TradingMetrics> {
    const response = await this.api.get('/dashboard/trading-metrics');
    return response.data;
  }

  async getSystemMetrics(): Promise<SystemMetrics> {
    const response = await this.api.get('/dashboard/system-metrics');
    return response.data;
  }

  async getSystemLogs(): Promise<SystemLog[]> {
    const response = await this.api.get('/dashboard/logs');
    return response.data;
  }

  // Trading Real Data Methods
  async getTradingAIStatus(): Promise<any> {
    const response = await this.api.get('/trading/ai-status');
    return response.data;
  }

  async getRiskMetrics(): Promise<any> {
    const response = await this.api.get('/trading/risk-metrics');
    return response.data;
  }

  async getRecentTradingDecisions(): Promise<any[]> {
    const response = await this.api.get('/trading/recent-decisions');
    return response.data;
  }

  async startAutonomousTrading(): Promise<void> {
    await this.api.post('/trading/start-autonomous');
  }

  async stopAutonomousTrading(): Promise<void> {
    await this.api.post('/trading/stop-autonomous');
  }

  async emergencyStopTrading(): Promise<void> {
    await this.api.post('/trading/emergency-stop');
  }

  // Training Real Data Methods
  async startDataCollection(): Promise<void> {
    await this.api.post('/training/start-collection');
  }

  async getDataCollectionProgress(): Promise<{ progress: number }> {
    const response = await this.api.get('/training/collection-progress');
    return response.data;
  }

  async getTrainingDatasets(): Promise<any[]> {
    const response = await this.api.get('/training/datasets');
    return response.data;
  }

  async startTrainingSession(config: any): Promise<any> {
    const response = await this.api.post('/training/start-session', config);
    return response.data;
  }

  async getTrainingSession(sessionId: string): Promise<any> {
    const response = await this.api.get(`/training/session/${sessionId}`);
    return response.data;
  }

  async stopTrainingSession(sessionId: string): Promise<void> {
    await this.api.post(`/training/session/${sessionId}/stop`);
  }

  async getTrainingHistory(): Promise<any[]> {
    const response = await this.api.get('/training/history');
    return response.data;
  }

  // Real-Time Data Methods
  async getDataQualityMetrics(symbols: string[]): Promise<any[]> {
    const response = await this.api.post('/realtime/data-quality', { symbols });
    return response.data;
  }

  async getRealTimeSystemStatus(): Promise<any> {
    const response = await this.api.get('/realtime/system-status');
    return response.data;
  }

  // AI Control Center Methods
  async getAIControlStatus(): Promise<any> {
    const response = await this.api.get('/ai/control-status');
    return response.data;
  }

  // Enterprise Platform Methods
  async getUserActivities(userIds: string[]): Promise<any[]> {
    const response = await this.api.post('/enterprise/user-activities', { user_ids: userIds });
    return response.data;
  }

  async getOrganizationStats(): Promise<any> {
    const response = await this.api.get('/enterprise/organization-stats');
    return response.data;
  }

  async getEnterpriseSystemMetrics(): Promise<any> {
    const response = await this.api.get('/enterprise/system-metrics');
    return response.data;
  }

  async getApiUsageData(): Promise<any[]> {
    const response = await this.api.get('/enterprise/api-usage');
    return response.data;
  }

  // Strategy Marketplace Methods
  async getUserStrategiesWithSales(): Promise<any[]> {
    const response = await this.api.get('/marketplace/user-strategies-sales');
    return response.data;
  }

  async getStrategyBacktestMetrics(strategyId: string): Promise<any> {
    const response = await this.api.get(`/marketplace/strategy/${strategyId}/backtest`);
    return response.data;
  }

  async getStrategyPerformanceChart(strategyId: string): Promise<any[]> {
    const response = await this.api.get(`/marketplace/strategy/${strategyId}/performance-chart`);
    return response.data;
  }

  async getStrategyReviews(strategyId: string): Promise<any[]> {
    const response = await this.api.get(`/marketplace/strategy/${strategyId}/reviews`);
    return response.data;
  }

  async deleteAPIKey(keyId: string): Promise<void> {
    await this.api.delete(`/api-keys/${keyId}`);
  }

  // Real Market Data (Phase 6)
  async getMarketData(symbols: string[], timeframe: string = '1m', limit: number = 100): Promise<any[]> {
    const response = await this.api.post('/v1/market-data', {
      symbols,
      timeframe,
      limit
    });
    return response.data;
  }

  async getRealTickData(symbol: string): Promise<RealTickData[]> {
    const response = await this.api.get(`/v1/market-data/ticks/${symbol}`);
    return response.data;
  }

  async getProcessedFeatures(symbol: string): Promise<ProcessedFeatures> {
    const response = await this.api.get(`/v1/features/${symbol}`);
    return response.data;
  }

  async getTimeSeriesData(symbol: string, start: string, end: string): Promise<any[]> {
    const response = await this.api.get(`/v1/timeseries/${symbol}`, {
      params: { start, end }
    });
    return response.data;
  }

  // AI & ML (Phase 8)
  async getEnsemblePrediction(symbol: string, features: number[][]): Promise<EnsemblePrediction> {
    const response = await this.api.post('/v1/ai/predict', {
      symbol,
      features
    });
    return response.data;
  }

  async getModelPerformance(): Promise<any> {
    const response = await this.api.get('/v1/ai/performance');
    return response.data;
  }

  async triggerModelRetrain(modelType?: string): Promise<void> {
    await this.api.post('/v1/ai/retrain', { model_type: modelType });
  }

  async getFeatureImportance(symbol: string): Promise<Record<string, number>> {
    const response = await this.api.get(`/v1/ai/features/importance/${symbol}`);
    return response.data;
  }

  async getLearningInsights(): Promise<any> {
    const response = await this.api.get('/v1/ai/learning/insights');
    return response.data;
  }

  // Trading Execution (Phase 7)
  async createTradingSignal(signal: {
    symbol: string;
    action: string;
    confidence: number;
    position_size?: number;
  }): Promise<any> {
    const response = await this.api.post('/v1/trading/signals', signal);
    return response.data;
  }

  async getTradingPositions(): Promise<TradingPosition[]> {
    const response = await this.api.get('/v1/trading/positions');
    return response.data;
  }

  async closePosition(contractId: string): Promise<void> {
    await this.api.post(`/v1/trading/positions/${contractId}/close`);
  }

  async getTradingHistory(limit: number = 100): Promise<any[]> {
    const response = await this.api.get('/v1/trading/history', {
      params: { limit }
    });
    return response.data;
  }

  async getTradingStatus(): Promise<any> {
    const response = await this.api.get('/v1/trading/status');
    return response.data;
  }

  async setTradingParameters(params: any): Promise<void> {
    await this.api.post('/v1/trading/parameters', params);
  }

  // Portfolio & Risk Management
  async getPortfolioSummary(): Promise<PortfolioMetrics> {
    const response = await this.api.get('/v1/portfolio/summary');
    return response.data;
  }

  async getPortfolioAllocation(): Promise<Record<string, number>> {
    const response = await this.api.get('/v1/portfolio/allocation');
    return response.data;
  }

  async optimizePortfolio(method: string, constraints?: any): Promise<any> {
    const response = await this.api.post('/v1/portfolio/optimize', {
      method,
      constraints
    });
    return response.data;
  }

  async getVaRAnalysis(): Promise<any> {
    const response = await this.api.get('/v1/risk/var');
    return response.data;
  }

  // Multi-Asset Management (Phase 9)
  async getMultiAssetStatus(): Promise<any> {
    const response = await this.api.get('/v1/multi-asset/status');
    return response.data;
  }

  async getCorrelationMatrix(): Promise<any> {
    const response = await this.api.get('/v1/multi-asset/correlations');
    return response.data;
  }

  async getAssetClusters(): Promise<any> {
    const response = await this.api.get('/v1/multi-asset/clusters');
    return response.data;
  }

  async getCrossAssetSignals(): Promise<MultiAssetSignal[]> {
    const response = await this.api.get('/v1/multi-asset/signals');
    return response.data;
  }

  async addSymbolToPortfolio(symbol: string, assetClass: string): Promise<void> {
    await this.api.post('/v1/multi-asset/symbols', {
      symbol,
      asset_class: assetClass
    });
  }

  async removeSymbolFromPortfolio(symbol: string): Promise<void> {
    await this.api.delete(`/v1/multi-asset/symbols/${symbol}`);
  }

  // Strategy Marketplace (Phase 10)
  async getStrategies(filters?: {
    category?: string;
    min_rating?: number;
    max_price?: number;
    sort_by?: string;
  }): Promise<Strategy[]> {
    const response = await this.api.get('/v1/marketplace/strategies', {
      params: filters
    });
    return response.data;
  }

  async getStrategy(strategyId: string): Promise<Strategy> {
    const response = await this.api.get(`/v1/marketplace/strategies/${strategyId}`);
    return response.data;
  }

  async purchaseStrategy(strategyId: string, licenseType: string = 'personal'): Promise<any> {
    const response = await this.api.post(`/v1/marketplace/strategies/${strategyId}/purchase`, {
      license_type: licenseType
    });
    return response.data;
  }

  async getUserStrategies(): Promise<Strategy[]> {
    const response = await this.api.get('/v1/marketplace/my-strategies');
    return response.data;
  }

  async submitStrategy(strategy: Partial<Strategy>): Promise<string> {
    const response = await this.api.post('/v1/marketplace/strategies', strategy);
    return response.data.strategy_id;
  }

  async getMarketplaceStats(): Promise<any> {
    const response = await this.api.get('/v1/marketplace/stats');
    return response.data;
  }

  // Performance & Analytics
  async getSystemPerformance(): Promise<any> {
    const response = await this.api.get('/v1/system/performance');
    return response.data;
  }

  async getInfrastructureStatus(): Promise<any> {
    const response = await this.api.get('/v1/infrastructure/status');
    return response.data;
  }

  async getOptimizationReport(): Promise<any> {
    const response = await this.api.get('/v1/optimization/report');
    return response.data;
  }

  async getAPIUsageStats(): Promise<any> {
    const response = await this.api.get('/v1/api/usage-stats');
    return response.data;
  }

  // User Management (Enterprise)
  async getCurrentUser(): Promise<User> {
    const response = await this.api.get('/v1/users/me');
    return response.data;
  }

  async updateUserProfile(updates: Partial<User>): Promise<User> {
    const response = await this.api.patch('/v1/users/me', updates);
    return response.data;
  }

  async getUserDashboard(): Promise<any> {
    const response = await this.api.get('/v1/users/dashboard');
    return response.data;
  }

  async getOrganizationUsers(): Promise<User[]> {
    const response = await this.api.get('/v1/organization/users');
    return response.data;
  }

  // WebSocket Connections
  connectWebSocket(endpoint: string, onMessage: (data: any) => void): WebSocket {
    const wsUrl = (import.meta.env.VITE_WS_URL || 'ws://localhost:8000') + endpoint;
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log(`WebSocket connected: ${endpoint}`);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('WebSocket message parse error:', error);
      }
    };

    ws.onclose = () => {
      console.log(`WebSocket disconnected: ${endpoint}`);
      this.wsConnections.delete(endpoint);
    };

    ws.onerror = (error) => {
      console.error(`WebSocket error on ${endpoint}:`, error);
    };

    this.wsConnections.set(endpoint, ws);
    return ws;
  }

  subscribeToMarketData(symbols: string[], onUpdate: (data: RealTickData) => void): WebSocket {
    return this.connectWebSocket('/ws/market-data', (data) => {
      if (data.type === 'market_update' && symbols.includes(data.symbol)) {
        onUpdate(data.data);
      }
    });
  }

  subscribeToTradingUpdates(onUpdate: (data: any) => void): WebSocket {
    return this.connectWebSocket('/ws/trading', (data) => {
      if (data.type === 'trading_update') {
        onUpdate(data.data);
      }
    });
  }

  subscribeToAIUpdates(onUpdate: (data: any) => void): WebSocket {
    return this.connectWebSocket('/ws/ai', (data) => {
      if (data.type === 'ai_update') {
        onUpdate(data.data);
      }
    });
  }

  disconnectWebSocket(endpoint: string): void {
    const ws = this.wsConnections.get(endpoint);
    if (ws) {
      ws.close();
      this.wsConnections.delete(endpoint);
    }
  }

  disconnectAllWebSockets(): void {
    this.wsConnections.forEach((ws) => ws.close());
    this.wsConnections.clear();
  }

  // Health Check
  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    const response = await this.api.get('/health');
    return response.data;
  }
}

// Export singleton instance
export const apiClient = new APIClient();
export default apiClient;