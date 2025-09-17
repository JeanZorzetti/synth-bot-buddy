/**
 * 🔐 PERMISSIONS HOOK
 * Role-based access control with granular permissions
 */

import { useAuth } from '../contexts/AuthContext';

interface PermissionConfig {
  roles: string[];
  permissions: string[];
  allRequired?: boolean;
}

export function usePermissions() {
  const { user, hasRole, hasAnyRole, hasPermission } = useAuth();

  /**
   * Check if user has required permissions
   */
  const checkPermissions = (config: PermissionConfig): boolean => {
    if (!user) return false;

    // Check roles
    if (config.roles.length > 0) {
      const hasRequiredRole = config.allRequired
        ? config.roles.every(role => hasRole(role))
        : hasAnyRole(config.roles);

      if (!hasRequiredRole) return false;
    }

    // Check permissions
    if (config.permissions.length > 0) {
      const hasRequiredPermission = config.allRequired
        ? config.permissions.every(permission => hasPermission(permission))
        : config.permissions.some(permission => hasPermission(permission));

      if (!hasRequiredPermission) return false;
    }

    return true;
  };

  /**
   * Predefined permission checks for common use cases
   */
  const permissions = {
    // Dashboard access
    canViewDashboard: () => hasPermission('view_dashboard'),
    canViewAnalytics: () => hasPermission('view_analytics'),
    canViewAdvancedAnalytics: () => hasPermission('advanced_analytics'),

    // Trading permissions
    canBasicTrading: () => hasPermission('basic_trading'),
    canAdvancedTrading: () => hasPermission('advanced_trading'),
    canCustomStrategies: () => hasPermission('custom_strategies'),

    // Data and exports
    canExportData: () => hasPermission('export_data'),
    canViewAllData: () => hasPermission('view_all_data'),
    canCustomReports: () => hasPermission('custom_reports'),

    // API access
    canApiBasic: () => hasPermission('api_access_basic'),
    canApiFull: () => hasPermission('api_access_full'),

    // Support and priority features
    canPrioritySupport: () => hasPermission('priority_support'),
    canDedicatedSupport: () => hasPermission('dedicated_support'),

    // Enterprise features
    canWhiteLabel: () => hasPermission('white_label'),
    canCustomIntegrations: () => hasPermission('custom_integrations'),

    // Admin permissions
    canAdminPanel: () => hasPermission('admin_panel'),
    canUserManagement: () => hasPermission('user_management'),
    canSystemConfiguration: () => hasPermission('system_configuration'),
    canManageBilling: () => hasPermission('manage_billing'),
    canSupportManagement: () => hasPermission('support_management'),

    // Role-based checks
    isTrialUser: () => hasRole('trial'),
    isBasicUser: () => hasRole('basic'),
    isPremiumUser: () => hasRole('premium') || hasRole('enterprise'),
    isEnterpriseUser: () => hasRole('enterprise'),
    isAdmin: () => hasRole('admin'),

    // Combined permission checks
    canCreateAdvancedReports: () => permissions.isPremiumUser() && hasPermission('custom_reports'),
    canAccessEnterprise: () => permissions.isEnterpriseUser() || permissions.isAdmin(),
    canManageUsers: () => permissions.isAdmin() && hasPermission('user_management'),
    canConfigureSystem: () => permissions.isAdmin() && hasPermission('system_configuration'),
  };

  /**
   * Get user's subscription tier info
   */
  const getSubscriptionInfo = () => {
    if (!user) return null;

    return {
      role: user.role,
      status: user.subscriptionStatus,
      plan: user.subscriptionPlan,
      expires: user.subscriptionExpires,
      isActive: user.subscriptionStatus === 'active',
      isExpired: user.subscriptionStatus === 'expired',
      isTrial: user.role === 'trial',
      canUpgrade: user.role !== 'enterprise' && user.role !== 'admin',
    };
  };

  /**
   * Get available features for current user
   */
  const getAvailableFeatures = () => {
    if (!user) return [];

    const features = [];

    // Basic features (all users)
    features.push('basic_dashboard', 'basic_trading', 'basic_analytics');

    // Role-specific features
    if (permissions.canAdvancedTrading()) {
      features.push('advanced_trading', 'position_management');
    }

    if (permissions.canCustomStrategies()) {
      features.push('strategy_builder', 'backtesting');
    }

    if (permissions.canViewAdvancedAnalytics()) {
      features.push('advanced_charts', 'risk_analytics', 'performance_tracking');
    }

    if (permissions.canExportData()) {
      features.push('data_export', 'csv_download', 'api_export');
    }

    if (permissions.canApiBasic()) {
      features.push('rest_api', 'webhooks');
    }

    if (permissions.canApiFull()) {
      features.push('websocket_api', 'real_time_data', 'advanced_endpoints');
    }

    if (permissions.canCustomReports()) {
      features.push('custom_reports', 'scheduled_reports', 'white_label_reports');
    }

    if (permissions.canPrioritySupport()) {
      features.push('priority_support', 'dedicated_account_manager');
    }

    if (permissions.canWhiteLabel()) {
      features.push('white_label', 'custom_branding', 'subdomain');
    }

    if (permissions.canCustomIntegrations()) {
      features.push('custom_integrations', 'enterprise_sso', 'ldap');
    }

    // Admin features
    if (permissions.isAdmin()) {
      features.push('admin_panel', 'user_management', 'system_config', 'billing_management');
    }

    return features;
  };

  /**
   * Get feature limits for current user
   */
  const getFeatureLimits = () => {
    if (!user) return null;

    const limits = {
      maxApiCallsPerMinute: 60,
      maxConcurrentStrategies: 1,
      maxBacktestDays: 30,
      maxReports: 5,
      maxAlerts: 10,
      maxPositions: 5,
      dataRetentionDays: 30,
    };

    // Adjust limits based on subscription
    switch (user.role) {
      case 'trial':
        limits.maxApiCallsPerMinute = 10;
        limits.maxConcurrentStrategies = 1;
        limits.maxBacktestDays = 7;
        limits.maxReports = 1;
        limits.maxAlerts = 3;
        limits.maxPositions = 1;
        limits.dataRetentionDays = 7;
        break;

      case 'basic':
        limits.maxApiCallsPerMinute = 60;
        limits.maxConcurrentStrategies = 3;
        limits.maxBacktestDays = 90;
        limits.maxReports = 10;
        limits.maxAlerts = 25;
        limits.maxPositions = 10;
        limits.dataRetentionDays = 90;
        break;

      case 'premium':
        limits.maxApiCallsPerMinute = 300;
        limits.maxConcurrentStrategies = 10;
        limits.maxBacktestDays = 365;
        limits.maxReports = 50;
        limits.maxAlerts = 100;
        limits.maxPositions = 50;
        limits.dataRetentionDays = 365;
        break;

      case 'enterprise':
      case 'admin':
        limits.maxApiCallsPerMinute = 1000;
        limits.maxConcurrentStrategies = -1; // Unlimited
        limits.maxBacktestDays = -1; // Unlimited
        limits.maxReports = -1; // Unlimited
        limits.maxAlerts = -1; // Unlimited
        limits.maxPositions = -1; // Unlimited
        limits.dataRetentionDays = -1; // Unlimited
        break;
    }

    return limits;
  };

  return {
    checkPermissions,
    permissions,
    getSubscriptionInfo,
    getAvailableFeatures,
    getFeatureLimits,
    user,
  };
}

/**
 * Higher-order component for permission-based rendering
 */
interface PermissionGuardProps {
  children: React.ReactNode;
  roles?: string[];
  permissions?: string[];
  allRequired?: boolean;
  fallback?: React.ReactNode;
}

export function PermissionGuard({
  children,
  roles = [],
  permissions = [],
  allRequired = false,
  fallback = null,
}: PermissionGuardProps) {
  const { checkPermissions } = usePermissions();

  const hasAccess = checkPermissions({
    roles,
    permissions,
    allRequired,
  });

  if (!hasAccess) {
    return <>{fallback}</>;
  }

  return <>{children}</>;
}

/**
 * Hook for subscription upgrade prompts
 */
export function useSubscriptionPrompts() {
  const { permissions, getSubscriptionInfo } = usePermissions();
  const subscriptionInfo = getSubscriptionInfo();

  const getUpgradePrompt = (requiredFeature: string) => {
    if (!subscriptionInfo) return null;

    const prompts = {
      trial: {
        title: 'Upgrade to Basic',
        message: 'Unlock advanced trading features and longer data retention.',
        features: ['Advanced Trading', 'Extended Backtesting', 'API Access'],
        buttonText: 'Upgrade to Basic',
        targetPlan: 'basic',
      },
      basic: {
        title: 'Upgrade to Premium',
        message: 'Access premium features and unlimited strategies.',
        features: ['Custom Strategies', 'Advanced Analytics', 'Priority Support'],
        buttonText: 'Upgrade to Premium',
        targetPlan: 'premium',
      },
      premium: {
        title: 'Upgrade to Enterprise',
        message: 'Enterprise-grade features with white-label options.',
        features: ['White Label', 'Custom Integrations', 'Dedicated Support'],
        buttonText: 'Contact Sales',
        targetPlan: 'enterprise',
      },
    };

    return prompts[subscriptionInfo.role as keyof typeof prompts] || null;
  };

  const shouldShowUpgradePrompt = (feature: string) => {
    const availableFeatures = permissions.getAvailableFeatures?.() || [];
    return !availableFeatures.includes(feature);
  };

  return {
    getUpgradePrompt,
    shouldShowUpgradePrompt,
    subscriptionInfo,
  };
}