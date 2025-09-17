/**
 * 🔐 PERMISSION GUARD COMPONENT
 * Flexible permission-based access control for UI components
 */

import React from 'react';
import { usePermissions, PermissionGuard as BasePermissionGuard } from '../../hooks/usePermissions';
import { Alert, AlertDescription } from '../ui/alert';
import { Button } from '../ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Shield, Lock, Crown, Star, Zap } from 'lucide-react';
import { useRouter } from 'next/router';

interface PermissionGuardProps {
  children: React.ReactNode;
  roles?: string[];
  permissions?: string[];
  allRequired?: boolean;
  showUpgrade?: boolean;
  customMessage?: string;
  fallback?: React.ReactNode;
}

export function PermissionGuard({
  children,
  roles = [],
  permissions = [],
  allRequired = false,
  showUpgrade = true,
  customMessage,
  fallback,
}: PermissionGuardProps) {
  const { checkPermissions, getSubscriptionInfo } = usePermissions();
  const router = useRouter();

  const hasAccess = checkPermissions({
    roles,
    permissions,
    allRequired,
  });

  if (hasAccess) {
    return <>{children}</>;
  }

  // Use custom fallback if provided
  if (fallback) {
    return <>{fallback}</>;
  }

  // Show upgrade prompt if enabled
  if (showUpgrade) {
    return <UpgradePrompt permissions={permissions} roles={roles} customMessage={customMessage} />;
  }

  // Default access denied message
  return (
    <Alert variant="destructive">
      <Shield className="h-4 w-4" />
      <AlertDescription>
        {customMessage || 'You do not have permission to access this feature.'}
      </AlertDescription>
    </Alert>
  );
}

interface UpgradePromptProps {
  permissions: string[];
  roles: string[];
  customMessage?: string;
}

function UpgradePrompt({ permissions, roles, customMessage }: UpgradePromptProps) {
  const { getSubscriptionInfo } = usePermissions();
  const router = useRouter();
  const subscriptionInfo = getSubscriptionInfo();

  const handleUpgrade = () => {
    router.push('/billing/upgrade');
  };

  const handleContactSales = () => {
    router.push('/contact/sales');
  };

  const getUpgradeInfo = () => {
    if (!subscriptionInfo) return null;

    switch (subscriptionInfo.role) {
      case 'trial':
        return {
          icon: <Star className="h-8 w-8 text-blue-500" />,
          title: 'Upgrade to Basic Plan',
          description: 'Unlock advanced trading features and extended data retention.',
          features: [
            '✅ Advanced Trading Algorithms',
            '✅ 90-day Data Retention',
            '✅ API Access',
            '✅ Email Support',
            '✅ Multiple Strategies',
          ],
          price: '$29/month',
          buttonText: 'Upgrade to Basic',
          action: handleUpgrade,
          color: 'blue',
        };

      case 'basic':
        return {
          icon: <Crown className="h-8 w-8 text-purple-500" />,
          title: 'Upgrade to Premium Plan',
          description: 'Access premium features with unlimited strategies and priority support.',
          features: [
            '✅ All Basic Features',
            '✅ Custom Strategy Builder',
            '✅ Advanced Analytics',
            '✅ Priority Support',
            '✅ Unlimited Backtesting',
            '✅ Real-time Alerts',
          ],
          price: '$99/month',
          buttonText: 'Upgrade to Premium',
          action: handleUpgrade,
          color: 'purple',
        };

      case 'premium':
        return {
          icon: <Zap className="h-8 w-8 text-orange-500" />,
          title: 'Upgrade to Enterprise',
          description: 'Enterprise-grade features with white-label options and dedicated support.',
          features: [
            '✅ All Premium Features',
            '✅ White Label Solution',
            '✅ Custom Integrations',
            '✅ Dedicated Account Manager',
            '✅ SLA Guarantee',
            '✅ Custom Development',
          ],
          price: 'Custom Pricing',
          buttonText: 'Contact Sales',
          action: handleContactSales,
          color: 'orange',
        };

      default:
        return null;
    }
  };

  const upgradeInfo = getUpgradeInfo();

  if (!upgradeInfo) {
    return (
      <Alert>
        <Lock className="h-4 w-4" />
        <AlertDescription>
          {customMessage || 'This feature requires additional permissions.'}
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <Card className="max-w-md mx-auto">
      <CardHeader className="text-center pb-4">
        <div className="mx-auto mb-4">
          {upgradeInfo.icon}
        </div>
        <CardTitle className="text-xl">{upgradeInfo.title}</CardTitle>
        <CardDescription className="text-sm">
          {upgradeInfo.description}
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Features List */}
        <div className="space-y-2">
          {upgradeInfo.features.map((feature, index) => (
            <div key={index} className="text-sm text-gray-600 flex items-center">
              <span className="text-green-500 mr-2 font-bold">✓</span>
              {feature.replace('✅ ', '')}
            </div>
          ))}
        </div>

        {/* Pricing */}
        <div className="text-center py-4 border-t border-gray-200">
          <div className="text-2xl font-bold text-gray-900">
            {upgradeInfo.price}
          </div>
          {upgradeInfo.price !== 'Custom Pricing' && (
            <div className="text-sm text-gray-500">
              Billed monthly • Cancel anytime
            </div>
          )}
        </div>

        {/* Action Button */}
        <Button
          onClick={upgradeInfo.action}
          className={`w-full bg-${upgradeInfo.color}-500 hover:bg-${upgradeInfo.color}-600`}
          size="lg"
        >
          {upgradeInfo.buttonText}
        </Button>

        {/* Support Link */}
        <div className="text-center">
          <button
            onClick={() => router.push('/support')}
            className="text-sm text-gray-500 hover:text-gray-700 underline"
          >
            Need help choosing a plan?
          </button>
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * Specialized permission guards for common use cases
 */

export function AdminGuard({ children, fallback }: { children: React.ReactNode; fallback?: React.ReactNode }) {
  return (
    <PermissionGuard
      roles={['admin']}
      showUpgrade={false}
      fallback={fallback}
      customMessage="This feature is only available to administrators."
    >
      {children}
    </PermissionGuard>
  );
}

export function PremiumGuard({ children, feature }: { children: React.ReactNode; feature?: string }) {
  return (
    <PermissionGuard
      roles={['premium', 'enterprise', 'admin']}
      showUpgrade={true}
      customMessage={feature ? `${feature} requires a Premium subscription or higher.` : undefined}
    >
      {children}
    </PermissionGuard>
  );
}

export function EnterpriseGuard({ children, feature }: { children: React.ReactNode; feature?: string }) {
  return (
    <PermissionGuard
      roles={['enterprise', 'admin']}
      showUpgrade={true}
      customMessage={feature ? `${feature} is only available in Enterprise plans.` : undefined}
    >
      {children}
    </PermissionGuard>
  );
}

export function FeatureGuard({
  children,
  permission,
  feature
}: {
  children: React.ReactNode;
  permission: string;
  feature?: string;
}) {
  return (
    <PermissionGuard
      permissions={[permission]}
      showUpgrade={true}
      customMessage={feature ? `${feature} requires additional permissions.` : undefined}
    >
      {children}
    </PermissionGuard>
  );
}

/**
 * Feature toggle component
 */
interface FeatureToggleProps {
  feature: string;
  children: React.ReactNode;
  fallback?: React.ReactNode;
  showUpgrade?: boolean;
}

export function FeatureToggle({
  feature,
  children,
  fallback,
  showUpgrade = true
}: FeatureToggleProps) {
  const { getAvailableFeatures } = usePermissions();
  const availableFeatures = getAvailableFeatures();

  const hasFeature = availableFeatures.includes(feature);

  if (hasFeature) {
    return <>{children}</>;
  }

  if (fallback) {
    return <>{fallback}</>;
  }

  if (showUpgrade) {
    return <UpgradePrompt permissions={[]} roles={[]} customMessage={`${feature} is not available in your current plan.`} />;
  }

  return null;
}

/**
 * Role badge component
 */
export function RoleBadge({ className = "" }: { className?: string }) {
  const { getSubscriptionInfo } = usePermissions();
  const subscriptionInfo = getSubscriptionInfo();

  if (!subscriptionInfo) return null;

  const getRoleStyle = () => {
    switch (subscriptionInfo.role) {
      case 'trial':
        return 'bg-gray-100 text-gray-800 border-gray-300';
      case 'basic':
        return 'bg-blue-100 text-blue-800 border-blue-300';
      case 'premium':
        return 'bg-purple-100 text-purple-800 border-purple-300';
      case 'enterprise':
        return 'bg-orange-100 text-orange-800 border-orange-300';
      case 'admin':
        return 'bg-red-100 text-red-800 border-red-300';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  };

  const getRoleIcon = () => {
    switch (subscriptionInfo.role) {
      case 'trial':
        return '🚀';
      case 'basic':
        return '⭐';
      case 'premium':
        return '👑';
      case 'enterprise':
        return '🏢';
      case 'admin':
        return '⚡';
      default:
        return '👤';
    }
  };

  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${getRoleStyle()} ${className}`}
    >
      <span className="mr-1">{getRoleIcon()}</span>
      {subscriptionInfo.role.charAt(0).toUpperCase() + subscriptionInfo.role.slice(1)}
    </span>
  );
}