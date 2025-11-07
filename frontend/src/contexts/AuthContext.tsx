/**
 * 🔐 AUTHENTICATION CONTEXT
 * React context for managing authentication state
 */

import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { authService, User, AuthTokens, LoginCredentials, RegisterData } from '../services/authService';

interface AuthContextType {
  // State
  user: User | null;
  tokens: AuthTokens | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;

  // Actions
  login: (credentials: LoginCredentials) => Promise<void>;
  register: (data: RegisterData) => Promise<void>;
  logout: () => Promise<void>;
  updateProfile: (updates: Partial<User>) => Promise<void>;
  changePassword: (currentPassword: string, newPassword: string) => Promise<void>;
  requestPasswordReset: (email: string) => Promise<void>;
  resetPassword: (token: string, newPassword: string) => Promise<void>;
  verifyEmail: (token: string) => Promise<void>;
  resendVerificationEmail: () => Promise<void>;
  clearError: () => void;

  // Utilities
  hasRole: (role: string) => boolean;
  hasAnyRole: (roles: string[]) => boolean;
  hasPermission: (permission: string) => boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null);
  const [tokens, setTokens] = useState<AuthTokens | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Initialize auth state
  useEffect(() => {
    initializeAuth();
  }, []);

  const initializeAuth = async () => {
    try {
      setIsLoading(true);

      // Get stored authentication data
      const storedUser = authService.getCurrentUser();
      const storedTokens = authService.getCurrentTokens();

      if (storedUser && storedTokens) {
        setUser(storedUser);
        setTokens(storedTokens);

        // Validate token by making a test API call
        try {
          await authService.getCurrentUser();
        } catch (error) {
          // Token is invalid, clear auth data
          await handleLogout();
        }
      }
    } catch (error) {
      console.error('Auth initialization failed:', error);
      setError('Failed to initialize authentication');
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogin = async (credentials: LoginCredentials) => {
    try {
      setIsLoading(true);
      setError(null);

      const authResponse = await authService.login(credentials);

      setUser(authResponse.user);
      setTokens(authResponse.tokens);

      // Track successful login
      trackEvent('login_success', {
        userId: authResponse.user.id,
        userRole: authResponse.user.role
      });
    } catch (error: any) {
      setError(error.message);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const handleRegister = async (data: RegisterData) => {
    try {
      setIsLoading(true);
      setError(null);

      const authResponse = await authService.register(data);

      setUser(authResponse.user);
      setTokens(authResponse.tokens);

      // Track successful registration
      trackEvent('registration_success', {
        userId: authResponse.user.id,
        userRole: authResponse.user.role
      });
    } catch (error: any) {
      setError(error.message);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = async () => {
    try {
      setIsLoading(true);

      await authService.logout();

      setUser(null);
      setTokens(null);
      setError(null);

      // Track logout
      trackEvent('logout_success', {});
    } catch (error: any) {
      console.error('Logout error:', error);
      // Still clear local state even if API call fails
      setUser(null);
      setTokens(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleUpdateProfile = async (updates: Partial<User>) => {
    try {
      setIsLoading(true);
      setError(null);

      const updatedUser = await authService.updateProfile(updates);
      setUser(updatedUser);

      // Track profile update
      trackEvent('profile_update_success', {
        userId: updatedUser.id,
        updatedFields: Object.keys(updates)
      });
    } catch (error: any) {
      setError(error.message);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const handleChangePassword = async (currentPassword: string, newPassword: string) => {
    try {
      setIsLoading(true);
      setError(null);

      await authService.changePassword(currentPassword, newPassword);

      // Track password change
      trackEvent('password_change_success', {
        userId: user?.id
      });
    } catch (error: any) {
      setError(error.message);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const handleRequestPasswordReset = async (email: string) => {
    try {
      setIsLoading(true);
      setError(null);

      await authService.requestPasswordReset(email);

      // Track password reset request
      trackEvent('password_reset_request', { email });
    } catch (error: any) {
      setError(error.message);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const handleResetPassword = async (token: string, newPassword: string) => {
    try {
      setIsLoading(true);
      setError(null);

      await authService.resetPassword(token, newPassword);

      // Track password reset success
      trackEvent('password_reset_success', {});
    } catch (error: any) {
      setError(error.message);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const handleVerifyEmail = async (token: string) => {
    try {
      setIsLoading(true);
      setError(null);

      await authService.verifyEmail(token);

      // Update user state
      if (user) {
        const updatedUser = { ...user, isVerified: true };
        setUser(updatedUser);
      }

      // Track email verification
      trackEvent('email_verification_success', {
        userId: user?.id
      });
    } catch (error: any) {
      setError(error.message);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const handleResendVerificationEmail = async () => {
    try {
      setIsLoading(true);
      setError(null);

      await authService.resendVerificationEmail();

      // Track verification email resend
      trackEvent('verification_email_resend', {
        userId: user?.id
      });
    } catch (error: any) {
      setError(error.message);
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const clearError = () => {
    setError(null);
  };

  // Utility functions
  const hasRole = (role: string): boolean => {
    return authService.hasRole(role);
  };

  const hasAnyRole = (roles: string[]): boolean => {
    return authService.hasAnyRole(roles);
  };

  const hasPermission = (permission: string): boolean => {
    if (!user) return false;

    // Define role-based permissions
    const rolePermissions: Record<string, string[]> = {
      trial: [
        'view_dashboard',
        'basic_trading',
        'view_basic_analytics'
      ],
      basic: [
        'view_dashboard',
        'basic_trading',
        'advanced_trading',
        'view_analytics',
        'export_data',
        'api_access_basic'
      ],
      premium: [
        'view_dashboard',
        'basic_trading',
        'advanced_trading',
        'custom_strategies',
        'view_analytics',
        'advanced_analytics',
        'export_data',
        'api_access_full',
        'priority_support',
        'custom_reports'
      ],
      enterprise: [
        'view_dashboard',
        'basic_trading',
        'advanced_trading',
        'custom_strategies',
        'view_analytics',
        'advanced_analytics',
        'export_data',
        'api_access_full',
        'priority_support',
        'custom_reports',
        'white_label',
        'dedicated_support',
        'custom_integrations'
      ],
      admin: [
        'all_permissions',
        'admin_panel',
        'user_management',
        'system_configuration',
        'view_all_data',
        'manage_billing',
        'support_management'
      ]
    };

    const userPermissions = rolePermissions[user.role] || [];

    return userPermissions.includes('all_permissions') || userPermissions.includes(permission);
  };

  // Track events for analytics
  const trackEvent = (eventType: string, eventData: any) => {
    // This would integrate with your analytics service
    console.log('Auth Event:', eventType, eventData);
  };

  const contextValue: AuthContextType = {
    // State
    user,
    tokens,
    isAuthenticated: !!user && !!tokens,
    isLoading,
    error,

    // Actions
    login: handleLogin,
    register: handleRegister,
    logout: handleLogout,
    updateProfile: handleUpdateProfile,
    changePassword: handleChangePassword,
    requestPasswordReset: handleRequestPasswordReset,
    resetPassword: handleResetPassword,
    verifyEmail: handleVerifyEmail,
    resendVerificationEmail: handleResendVerificationEmail,
    clearError,

    // Utilities
    hasRole,
    hasAnyRole,
    hasPermission
  };

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );
}

// Custom hook to use auth context
export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

// Higher-order component for protected routes
interface ProtectedRouteProps {
  children: ReactNode;
  requiredRole?: string;
  requiredRoles?: string[];
  requiredPermission?: string;
  fallback?: ReactNode;
}

export function ProtectedRoute({
  children,
  requiredRole,
  requiredRoles,
  requiredPermission,
  fallback = <div>Access denied</div>
}: ProtectedRouteProps) {
  const { isAuthenticated, user, hasRole, hasAnyRole, hasPermission } = useAuth();

  if (!isAuthenticated || !user) {
    return <div>Please log in to access this page</div>;
  }

  if (requiredRole && !hasRole(requiredRole)) {
    return fallback;
  }

  if (requiredRoles && !hasAnyRole(requiredRoles)) {
    return fallback;
  }

  if (requiredPermission && !hasPermission(requiredPermission)) {
    return fallback;
  }

  return <>{children}</>;
}

// Hook for checking authentication status
export function useAuthStatus() {
  const { isAuthenticated, isLoading, user } = useAuth();

  return {
    isAuthenticated,
    isLoading,
    user,
    isAdmin: user?.role === 'admin',
    isPremium: user?.role === 'premium' || user?.role === 'enterprise',
    isVerified: user?.isVerified || false,
    subscriptionStatus: user?.subscriptionStatus
  };
}