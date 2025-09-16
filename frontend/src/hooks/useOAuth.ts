import { useState, useEffect, useCallback } from 'react';
import { apiService, OAuthCallbackResponse, OAuthTokenValidation } from '../services/api';

export interface OAuthUser {
  email?: string;
  user_id?: string;
  account_id?: string;
  [key: string]: any;
}

export interface OAuthState {
  isAuthenticated: boolean;
  isLoading: boolean;
  user: OAuthUser | null;
  token: {
    encrypted: string;
    info: any;
  } | null;
  error: string | null;
  scopes: string[];
}

export interface UseOAuthReturn extends OAuthState {
  // Actions
  startOAuthFlow: (scopes?: string[]) => Promise<void>;
  handleCallback: (code: string, state: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshToken: () => Promise<void>;
  clearError: () => void;
  
  // Connection methods
  connectWithOAuth: () => Promise<boolean>;
  
  // Utility methods
  isTokenExpired: () => boolean;
  hasValidSession: () => boolean;
  getAvailableScopes: () => Promise<string[]>;
}

export function useOAuth(): UseOAuthReturn {
  const [state, setState] = useState<OAuthState>({
    isAuthenticated: false,
    isLoading: true,
    user: null,
    token: null,
    error: null,
    scopes: []
  });

  // Initialize OAuth state from localStorage
  useEffect(() => {
    const initializeAuth = async () => {
      setState(prev => ({ ...prev, isLoading: true }));

      try {
        const storedData = apiService.getStoredOAuthData();
        
        if (storedData.encryptedToken && storedData.tokenInfo && storedData.userInfo) {
          // Check if token is expired
          if (!apiService.isOAuthTokenExpired()) {
            setState(prev => ({
              ...prev,
              isAuthenticated: true,
              user: storedData.userInfo,
              token: {
                encrypted: storedData.encryptedToken,
                info: storedData.tokenInfo
              },
              scopes: storedData.tokenInfo.scope?.split(' ') || [],
              error: null
            }));
          } else {
            // Token expired, clear session
            apiService.clearOAuthSession();
          }
        }
      } catch (error) {
        console.error('Error initializing OAuth:', error);
        apiService.clearOAuthSession();
        setState(prev => ({
          ...prev,
          error: 'Failed to initialize authentication'
        }));
      } finally {
        setState(prev => ({ ...prev, isLoading: false }));
      }
    };

    initializeAuth();
  }, []);

  // Start OAuth flow
  const startOAuthFlow = useCallback(async (scopes?: string[]) => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      const defaultScopes = ['read', 'trade']; // Default scopes for trading
      const requestedScopes = scopes || defaultScopes;
      
      const response = await apiService.startOAuthFlow(requestedScopes);
      
      // Redirect to OAuth provider
      window.location.href = response.authorization_url;
    } catch (error: any) {
      setState(prev => ({
        ...prev,
        error: error.message || 'Failed to start OAuth flow',
        isLoading: false
      }));
    }
  }, []);

  // Handle OAuth callback
  const handleCallback = useCallback(async (code: string, state: string) => {
    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      const response: OAuthCallbackResponse = await apiService.handleOAuthCallback(code, state);
      
      // Store token data
      apiService.storeOAuthToken(
        response.encrypted_token,
        response.token_info,
        response.user_info
      );

      // Update state
      setState(prev => ({
        ...prev,
        isAuthenticated: true,
        user: response.user_info,
        token: {
          encrypted: response.encrypted_token,
          info: response.token_info
        },
        scopes: response.token_info.scope.split(' '),
        error: null,
        isLoading: false
      }));

      return response;
    } catch (error: any) {
      setState(prev => ({
        ...prev,
        error: error.message || 'OAuth callback failed',
        isLoading: false
      }));
      throw error;
    }
  }, []);

  // Logout
  const logout = useCallback(async () => {
    setState(prev => ({ ...prev, isLoading: true }));

    try {
      // Try to revoke token if we have one
      const storedData = apiService.getStoredOAuthData();
      if (storedData.tokenInfo) {
        try {
          // Note: We can't revoke encrypted token directly, 
          // but we can clear the session
          await apiService.revokeOAuthToken(storedData.tokenInfo.access_token || '');
        } catch (revokeError) {
          // Ignore revoke errors, still logout locally
          console.warn('Token revocation failed:', revokeError);
        }
      }

      // Clear local session
      apiService.clearOAuthSession();

      // Reset state
      setState({
        isAuthenticated: false,
        isLoading: false,
        user: null,
        token: null,
        error: null,
        scopes: []
      });
    } catch (error: any) {
      setState(prev => ({
        ...prev,
        error: error.message || 'Logout failed',
        isLoading: false
      }));
    }
  }, []);

  // Refresh token
  const refreshToken = useCallback(async () => {
    const storedData = apiService.getStoredOAuthData();
    if (!storedData.tokenInfo?.refresh_token) {
      throw new Error('No refresh token available');
    }

    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      const response = await apiService.refreshOAuthToken(storedData.tokenInfo.refresh_token);
      
      // Store new token data
      apiService.storeOAuthToken(
        response.encrypted_token,
        response.token_info,
        state.user // Keep existing user info
      );

      // Update state
      setState(prev => ({
        ...prev,
        token: {
          encrypted: response.encrypted_token,
          info: response.token_info
        },
        scopes: response.token_info.scope.split(' '),
        error: null,
        isLoading: false
      }));
    } catch (error: any) {
      // Refresh failed, probably need to re-authenticate
      await logout();
      setState(prev => ({
        ...prev,
        error: 'Session expired. Please login again.',
        isLoading: false
      }));
      throw error;
    }
  }, [state.user, logout]);

  // Connect with OAuth token
  const connectWithOAuth = useCallback(async (): Promise<boolean> => {
    if (!state.token?.encrypted) {
      throw new Error('No OAuth token available');
    }

    try {
      const response = await apiService.connectWithOAuth(state.token.encrypted);
      return response.status === 'connecting';
    } catch (error: any) {
      // If connection fails due to expired token, try to refresh
      if (error.message.includes('expired') || error.message.includes('invalid')) {
        try {
          await refreshToken();
          // Retry connection with new token
          const newData = apiService.getStoredOAuthData();
          if (newData.encryptedToken) {
            const retryResponse = await apiService.connectWithOAuth(newData.encryptedToken);
            return retryResponse.status === 'connecting';
          }
        } catch (refreshError) {
          // Refresh failed, need to re-authenticate
          await logout();
          throw new Error('Session expired. Please login again.');
        }
      }
      throw error;
    }
  }, [state.token?.encrypted, refreshToken, logout]);

  // Clear error
  const clearError = useCallback(() => {
    setState(prev => ({ ...prev, error: null }));
  }, []);

  // Check if token is expired
  const isTokenExpired = useCallback(() => {
    return apiService.isOAuthTokenExpired();
  }, []);

  // Check if has valid session
  const hasValidSession = useCallback(() => {
    return state.isAuthenticated && state.token && !isTokenExpired();
  }, [state.isAuthenticated, state.token, isTokenExpired]);

  // Get available scopes
  const getAvailableScopes = useCallback(async () => {
    try {
      const response = await apiService.getOAuthScopes();
      return response.available_scopes;
    } catch (error) {
      console.error('Failed to get available scopes:', error);
      return [];
    }
  }, []);

  return {
    ...state,
    startOAuthFlow,
    handleCallback,
    logout,
    refreshToken,
    clearError,
    connectWithOAuth,
    isTokenExpired,
    hasValidSession,
    getAvailableScopes
  };
}