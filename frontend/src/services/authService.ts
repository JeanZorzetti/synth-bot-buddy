/**
 * 🔐 AUTHENTICATION SERVICE
 * Complete JWT authentication with real backend integration
 */

import { apiClient } from './apiClient';

export interface LoginCredentials {
  email: string;
  password: string;
  rememberMe?: boolean;
}

export interface RegisterData {
  email: string;
  username: string;
  password: string;
  confirmPassword: string;
  fullName: string;
  phone?: string;
  country?: string;
  acceptTerms: boolean;
  acceptMarketing?: boolean;
}

export interface User {
  id: string;
  email: string;
  username: string;
  fullName: string;
  phone?: string;
  country?: string;
  role: 'trial' | 'basic' | 'premium' | 'enterprise' | 'admin';
  subscriptionStatus: 'trial' | 'active' | 'expired' | 'cancelled';
  subscriptionPlan: string;
  subscriptionExpires?: string;
  isActive: boolean;
  isVerified: boolean;
  kycStatus: 'pending' | 'verified' | 'rejected';
  totalTrades: number;
  totalProfitLoss: number;
  tradingSettings: {
    maxRiskPerTrade: number;
    maxDailyRisk: number;
    autoTrading: boolean;
    defaultPositionSize: number;
    stopLossPct: number;
    takeProfitPct: number;
  };
  riskProfile: {
    riskTolerance: 'low' | 'medium' | 'high';
    tradingExperience: 'beginner' | 'intermediate' | 'advanced';
    maxDrawdown: number;
    preferredAssets: string[];
    tradingHours: {
      start: string;
      end: string;
    };
  };
  preferences: {
    theme: 'light' | 'dark' | 'auto';
    language: string;
    timezone: string;
    emailNotifications: boolean;
    smsNotifications: boolean;
    pushNotifications: boolean;
  };
  createdAt: string;
  lastLogin?: string;
  avatar?: string;
}

export interface AuthTokens {
  accessToken: string;
  refreshToken: string;
  tokenType: string;
  expiresIn: number;
}

export interface AuthResponse {
  user: User;
  tokens: AuthTokens;
  session: {
    sessionId: string;
    expiresAt: string;
  };
}

class AuthService {
  private static instance: AuthService;
  private user: User | null = null;
  private tokens: AuthTokens | null = null;
  private refreshTimer: NodeJS.Timeout | null = null;

  constructor() {
    this.loadStoredAuth();
    this.setupTokenRefresh();
  }

  static getInstance(): AuthService {
    if (!AuthService.instance) {
      AuthService.instance = new AuthService();
    }
    return AuthService.instance;
  }

  /**
   * Login user with email and password
   */
  async login(credentials: LoginCredentials): Promise<AuthResponse> {
    try {
      const response = await apiClient.post<AuthResponse>('/auth/login', {
        email: credentials.email,
        password: credentials.password,
        rememberMe: credentials.rememberMe || false,
        userAgent: navigator.userAgent,
        ipAddress: await this.getClientIP()
      });

      const { user, tokens, session } = response;

      // Store authentication data
      this.user = user;
      this.tokens = tokens;
      this.storeAuthData({ user, tokens, session });

      // Setup token refresh
      this.setupTokenRefresh();

      // Track login event
      this.trackEvent('user_login', {
        userId: user.id,
        loginMethod: 'email_password',
        userAgent: navigator.userAgent
      });

      return response;
    } catch (error: any) {
      const errorMessage = error.response?.data?.message || 'Login failed';

      // Track failed login attempt
      this.trackEvent('login_failed', {
        email: credentials.email,
        reason: errorMessage,
        userAgent: navigator.userAgent
      });

      throw new Error(errorMessage);
    }
  }

  /**
   * Register new user
   */
  async register(registerData: RegisterData): Promise<AuthResponse> {
    try {
      // Validate passwords match
      if (registerData.password !== registerData.confirmPassword) {
        throw new Error('Passwords do not match');
      }

      const response = await apiClient.post<AuthResponse>('/auth/register', {
        email: registerData.email,
        username: registerData.username,
        password: registerData.password,
        fullName: registerData.fullName,
        phone: registerData.phone,
        country: registerData.country,
        acceptTerms: registerData.acceptTerms,
        acceptMarketing: registerData.acceptMarketing || false,
        userAgent: navigator.userAgent,
        ipAddress: await this.getClientIP()
      });

      const { user, tokens, session } = response;

      // Store authentication data
      this.user = user;
      this.tokens = tokens;
      this.storeAuthData({ user, tokens, session });

      // Setup token refresh
      this.setupTokenRefresh();

      // Track registration event
      this.trackEvent('user_register', {
        userId: user.id,
        registrationMethod: 'email_password',
        userAgent: navigator.userAgent
      });

      return response;
    } catch (error: any) {
      const errorMessage = error.response?.data?.message || 'Registration failed';

      // Track failed registration
      this.trackEvent('registration_failed', {
        email: registerData.email,
        reason: errorMessage,
        userAgent: navigator.userAgent
      });

      throw new Error(errorMessage);
    }
  }

  /**
   * Logout user
   */
  async logout(): Promise<void> {
    try {
      if (this.tokens?.accessToken) {
        // Notify backend about logout
        await apiClient.post('/auth/logout', {}, {
          headers: {
            Authorization: `Bearer ${this.tokens.accessToken}`
          }
        });
      }
    } catch (error) {
      console.warn('Logout API call failed:', error);
    } finally {
      // Track logout event
      if (this.user) {
        this.trackEvent('user_logout', {
          userId: this.user.id,
          userAgent: navigator.userAgent
        });
      }

      // Clear local data
      this.clearAuthData();
    }
  }

  /**
   * Refresh access token
   */
  async refreshToken(): Promise<AuthTokens> {
    try {
      if (!this.tokens?.refreshToken) {
        throw new Error('No refresh token available');
      }

      const response = await apiClient.post<AuthTokens>('/auth/refresh', {
        refreshToken: this.tokens.refreshToken
      });

      // Update stored tokens
      this.tokens = response;
      this.updateStoredTokens(response);

      // Setup next refresh
      this.setupTokenRefresh();

      return response;
    } catch (error) {
      console.error('Token refresh failed:', error);

      // If refresh fails, logout user
      await this.logout();

      throw new Error('Session expired. Please login again.');
    }
  }

  /**
   * Get current user
   */
  getCurrentUser(): User | null {
    return this.user;
  }

  /**
   * Get current tokens
   */
  getCurrentTokens(): AuthTokens | null {
    return this.tokens;
  }

  /**
   * Check if user is authenticated
   */
  isAuthenticated(): boolean {
    return !!(this.user && this.tokens?.accessToken);
  }

  /**
   * Check if user has specific role
   */
  hasRole(role: string): boolean {
    return this.user?.role === role;
  }

  /**
   * Check if user has any of the specified roles
   */
  hasAnyRole(roles: string[]): boolean {
    return roles.includes(this.user?.role || '');
  }

  /**
   * Update user profile
   */
  async updateProfile(updates: Partial<User>): Promise<User> {
    try {
      if (!this.user?.id) {
        throw new Error('User not authenticated');
      }

      const response = await apiClient.put<User>(`/users/${this.user.id}/profile`, updates);

      // Update local user data
      this.user = { ...this.user, ...response };
      this.updateStoredUser(this.user);

      // Track profile update
      this.trackEvent('profile_updated', {
        userId: this.user.id,
        updatedFields: Object.keys(updates)
      });

      return this.user;
    } catch (error: any) {
      const errorMessage = error.response?.data?.message || 'Profile update failed';
      throw new Error(errorMessage);
    }
  }

  /**
   * Change password
   */
  async changePassword(currentPassword: string, newPassword: string): Promise<void> {
    try {
      if (!this.user?.id) {
        throw new Error('User not authenticated');
      }

      await apiClient.post(`/users/${this.user.id}/change-password`, {
        currentPassword,
        newPassword
      });

      // Track password change
      this.trackEvent('password_changed', {
        userId: this.user.id
      });
    } catch (error: any) {
      const errorMessage = error.response?.data?.message || 'Password change failed';
      throw new Error(errorMessage);
    }
  }

  /**
   * Request password reset
   */
  async requestPasswordReset(email: string): Promise<void> {
    try {
      await apiClient.post('/auth/forgot-password', { email });

      // Track password reset request
      this.trackEvent('password_reset_requested', {
        email
      });
    } catch (error: any) {
      const errorMessage = error.response?.data?.message || 'Password reset request failed';
      throw new Error(errorMessage);
    }
  }

  /**
   * Reset password with token
   */
  async resetPassword(token: string, newPassword: string): Promise<void> {
    try {
      await apiClient.post('/auth/reset-password', {
        token,
        newPassword
      });

      // Track password reset completion
      this.trackEvent('password_reset_completed', {
        resetToken: token.substring(0, 8) + '...' // Only log partial token
      });
    } catch (error: any) {
      const errorMessage = error.response?.data?.message || 'Password reset failed';
      throw new Error(errorMessage);
    }
  }

  /**
   * Verify email
   */
  async verifyEmail(token: string): Promise<void> {
    try {
      await apiClient.post('/auth/verify-email', { token });

      // Update user verification status
      if (this.user) {
        this.user.isVerified = true;
        this.updateStoredUser(this.user);
      }

      // Track email verification
      this.trackEvent('email_verified', {
        userId: this.user?.id,
        verificationToken: token.substring(0, 8) + '...'
      });
    } catch (error: any) {
      const errorMessage = error.response?.data?.message || 'Email verification failed';
      throw new Error(errorMessage);
    }
  }

  /**
   * Resend verification email
   */
  async resendVerificationEmail(): Promise<void> {
    try {
      if (!this.user?.email) {
        throw new Error('User not authenticated');
      }

      await apiClient.post('/auth/resend-verification', {
        email: this.user.email
      });

      // Track verification email resend
      this.trackEvent('verification_email_resent', {
        userId: this.user.id,
        email: this.user.email
      });
    } catch (error: any) {
      const errorMessage = error.response?.data?.message || 'Failed to resend verification email';
      throw new Error(errorMessage);
    }
  }

  /**
   * Get user sessions
   */
  async getUserSessions(): Promise<any[]> {
    try {
      if (!this.user?.id) {
        throw new Error('User not authenticated');
      }

      const response = await apiClient.get<any[]>(`/users/${this.user.id}/sessions`);
      return response;
    } catch (error: any) {
      const errorMessage = error.response?.data?.message || 'Failed to get user sessions';
      throw new Error(errorMessage);
    }
  }

  /**
   * Revoke session
   */
  async revokeSession(sessionId: string): Promise<void> {
    try {
      if (!this.user?.id) {
        throw new Error('User not authenticated');
      }

      await apiClient.delete(`/users/${this.user.id}/sessions/${sessionId}`);

      // Track session revocation
      this.trackEvent('session_revoked', {
        userId: this.user.id,
        sessionId
      });
    } catch (error: any) {
      const errorMessage = error.response?.data?.message || 'Failed to revoke session';
      throw new Error(errorMessage);
    }
  }

  /**
   * Private methods
   */
  private loadStoredAuth(): void {
    try {
      const storedUser = localStorage.getItem('tradingbot_user');
      const storedTokens = localStorage.getItem('tradingbot_tokens');

      if (storedUser && storedTokens) {
        this.user = JSON.parse(storedUser);
        this.tokens = JSON.parse(storedTokens);
      }
    } catch (error) {
      console.error('Failed to load stored auth:', error);
      this.clearAuthData();
    }
  }

  private storeAuthData(authData: AuthResponse): void {
    try {
      localStorage.setItem('tradingbot_user', JSON.stringify(authData.user));
      localStorage.setItem('tradingbot_tokens', JSON.stringify(authData.tokens));
      localStorage.setItem('tradingbot_session', JSON.stringify(authData.session));
    } catch (error) {
      console.error('Failed to store auth data:', error);
    }
  }

  private updateStoredUser(user: User): void {
    try {
      localStorage.setItem('tradingbot_user', JSON.stringify(user));
    } catch (error) {
      console.error('Failed to update stored user:', error);
    }
  }

  private updateStoredTokens(tokens: AuthTokens): void {
    try {
      localStorage.setItem('tradingbot_tokens', JSON.stringify(tokens));
    } catch (error) {
      console.error('Failed to update stored tokens:', error);
    }
  }

  private clearAuthData(): void {
    this.user = null;
    this.tokens = null;

    if (this.refreshTimer) {
      clearTimeout(this.refreshTimer);
      this.refreshTimer = null;
    }

    // Clear storage
    localStorage.removeItem('tradingbot_user');
    localStorage.removeItem('tradingbot_tokens');
    localStorage.removeItem('tradingbot_session');
  }

  private setupTokenRefresh(): void {
    if (this.refreshTimer) {
      clearTimeout(this.refreshTimer);
    }

    if (!this.tokens?.expiresIn) {
      return;
    }

    // Refresh token 5 minutes before expiration
    const refreshIn = (this.tokens.expiresIn - 300) * 1000;

    if (refreshIn > 0) {
      this.refreshTimer = setTimeout(async () => {
        try {
          await this.refreshToken();
        } catch (error) {
          console.error('Auto token refresh failed:', error);
        }
      }, refreshIn);
    }
  }

  private async getClientIP(): Promise<string> {
    try {
      const response = await fetch('https://api.ipify.org?format=json');
      const data = await response.json();
      return data.ip;
    } catch (error) {
      return '0.0.0.0';
    }
  }

  private trackEvent(eventName: string, eventData: any): void {
    try {
      // Track user events for analytics
      apiClient.post('/analytics/track', {
        eventType: eventName,
        eventData,
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent,
        url: window.location.href
      }).catch(error => {
        console.warn('Failed to track event:', error);
      });
    } catch (error) {
      console.warn('Event tracking failed:', error);
    }
  }
}

// Export singleton instance
export const authService = AuthService.getInstance();