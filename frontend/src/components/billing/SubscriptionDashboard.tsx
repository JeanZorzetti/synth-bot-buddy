/**
 * 📊 SUBSCRIPTION DASHBOARD COMPONENT
 * Complete subscription management with billing overview and controls
 */

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '../../contexts/AuthContext';
import { usePermissions } from '../../hooks/usePermissions';
import { apiClient } from '../../services/apiClient';
import { Button } from '../ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Alert, AlertDescription } from '../ui/alert';
import { Progress } from '../ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import {
  CreditCard,
  Calendar,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  Crown,
  Star,
  Building,
  Zap,
  RefreshCw,
  Download,
  Settings,
  Bell,
  DollarSign,
  Clock,
  BarChart3,
  Shield,
  ArrowUpRight,
  ArrowDownRight,
  Gift,
  Users,
  Globe,
  Smartphone,
  Mail,
  Phone,
  AlertCircle
} from 'lucide-react';

interface SubscriptionData {
  id: string;
  plan: string;
  status: 'active' | 'expired' | 'cancelled' | 'past_due';
  currentPeriodStart: string;
  currentPeriodEnd: string;
  nextBillingDate: string;
  amount: number;
  currency: string;
  billingCycle: 'monthly' | 'annually';
  autoRenew: boolean;
  trialEndsAt?: string;
  cancelAtPeriodEnd: boolean;
  pausedAt?: string;
}

interface UsageData {
  period: string;
  apiCalls: { used: number; limit: number };
  strategies: { used: number; limit: number };
  reports: { used: number; limit: number };
  dataStorage: { used: number; limit: number; unit: 'GB' };
  positions: { used: number; limit: number };
}

interface BillingHistory {
  id: string;
  date: string;
  amount: number;
  currency: string;
  status: 'paid' | 'pending' | 'failed' | 'refunded';
  invoiceUrl?: string;
  description: string;
  paymentMethod: string;
}

interface SubscriptionDashboardProps {
  className?: string;
}

export function SubscriptionDashboard({ className = "" }: SubscriptionDashboardProps) {
  const router = useRouter();
  const { user } = useAuth();
  const { getSubscriptionInfo, getFeatureLimits } = usePermissions();

  const [activeTab, setActiveTab] = useState('overview');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [subscriptionData, setSubscriptionData] = useState<SubscriptionData | null>(null);
  const [usageData, setUsageData] = useState<UsageData | null>(null);
  const [billingHistory, setBillingHistory] = useState<BillingHistory[]>([]);

  const subscriptionInfo = getSubscriptionInfo();
  const featureLimits = getFeatureLimits();

  useEffect(() => {
    loadSubscriptionData();
  }, []);

  const loadSubscriptionData = async () => {
    try {
      setIsLoading(true);
      setError(null);

      const [subscription, usage, billing] = await Promise.all([
        apiClient.get<SubscriptionData>('/billing/subscription'),
        apiClient.get<UsageData>('/billing/usage'),
        apiClient.get<BillingHistory[]>('/billing/history')
      ]);

      setSubscriptionData(subscription);
      setUsageData(usage);
      setBillingHistory(billing);
    } catch (error: any) {
      setError(error.message || 'Failed to load subscription data');
    } finally {
      setIsLoading(false);
    }
  };

  const handleCancelSubscription = async () => {
    if (!confirm('Are you sure you want to cancel your subscription? You\'ll retain access until the end of your current billing period.')) {
      return;
    }

    try {
      setIsLoading(true);
      await apiClient.post('/billing/cancel');
      await loadSubscriptionData();
    } catch (error: any) {
      setError(error.message || 'Failed to cancel subscription');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReactivateSubscription = async () => {
    try {
      setIsLoading(true);
      await apiClient.post('/billing/reactivate');
      await loadSubscriptionData();
    } catch (error: any) {
      setError(error.message || 'Failed to reactivate subscription');
    } finally {
      setIsLoading(false);
    }
  };

  const handlePauseSubscription = async () => {
    if (!confirm('Are you sure you want to pause your subscription? You can resume it at any time.')) {
      return;
    }

    try {
      setIsLoading(true);
      await apiClient.post('/billing/pause');
      await loadSubscriptionData();
    } catch (error: any) {
      setError(error.message || 'Failed to pause subscription');
    } finally {
      setIsLoading(false);
    }
  };

  const formatCurrency = (amount: number, currency: string = 'USD') => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency,
    }).format(amount);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    });
  };

  const formatRelativeDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = date.getTime() - now.getTime();
    const diffDays = Math.ceil(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays < 0) {
      return `${Math.abs(diffDays)} days ago`;
    } else if (diffDays === 0) {
      return 'Today';
    } else if (diffDays === 1) {
      return 'Tomorrow';
    } else {
      return `In ${diffDays} days`;
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'active':
        return <Badge className="bg-green-100 text-green-800">Active</Badge>;
      case 'expired':
        return <Badge className="bg-red-100 text-red-800">Expired</Badge>;
      case 'cancelled':
        return <Badge className="bg-gray-100 text-gray-800">Cancelled</Badge>;
      case 'past_due':
        return <Badge className="bg-orange-100 text-orange-800">Past Due</Badge>;
      default:
        return <Badge variant="secondary">{status}</Badge>;
    }
  };

  const getPlanIcon = (plan: string) => {
    switch (plan) {
      case 'trial':
        return <Star className="h-5 w-5 text-gray-500" />;
      case 'basic':
        return <TrendingUp className="h-5 w-5 text-blue-500" />;
      case 'premium':
        return <Crown className="h-5 w-5 text-purple-500" />;
      case 'enterprise':
        return <Building className="h-5 w-5 text-orange-500" />;
      default:
        return <Zap className="h-5 w-5 text-gray-500" />;
    }
  };

  const getUsagePercentage = (used: number, limit: number) => {
    if (limit === -1) return 0; // Unlimited
    return Math.min((used / limit) * 100, 100);
  };

  const getUsageColor = (percentage: number) => {
    if (percentage >= 90) return 'bg-red-500';
    if (percentage >= 75) return 'bg-orange-500';
    if (percentage >= 50) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  const renderOverview = () => (
    <div className="space-y-6">
      {/* Current Plan Overview */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                {getPlanIcon(subscriptionData?.plan || 'trial')}
                <div>
                  <CardTitle className="capitalize">
                    {subscriptionData?.plan || 'Trial'} Plan
                  </CardTitle>
                  <CardDescription>
                    Your current subscription plan
                  </CardDescription>
                </div>
              </div>
              {subscriptionData && getStatusBadge(subscriptionData.status)}
            </div>
          </CardHeader>

          <CardContent className="space-y-4">
            {subscriptionData && (
              <>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm text-gray-500">Current Period</p>
                    <p className="font-medium">
                      {formatDate(subscriptionData.currentPeriodStart)} - {formatDate(subscriptionData.currentPeriodEnd)}
                    </p>
                  </div>

                  <div>
                    <p className="text-sm text-gray-500">Next Billing</p>
                    <p className="font-medium">
                      {formatRelativeDate(subscriptionData.nextBillingDate)}
                    </p>
                  </div>

                  <div>
                    <p className="text-sm text-gray-500">Amount</p>
                    <p className="font-medium text-lg">
                      {formatCurrency(subscriptionData.amount, subscriptionData.currency)}
                      <span className="text-sm text-gray-500">
                        /{subscriptionData.billingCycle === 'monthly' ? 'month' : 'year'}
                      </span>
                    </p>
                  </div>

                  <div>
                    <p className="text-sm text-gray-500">Auto Renewal</p>
                    <div className="flex items-center space-x-2">
                      {subscriptionData.autoRenew ? (
                        <>
                          <CheckCircle className="h-4 w-4 text-green-500" />
                          <span className="text-sm">Enabled</span>
                        </>
                      ) : (
                        <>
                          <AlertTriangle className="h-4 w-4 text-orange-500" />
                          <span className="text-sm">Disabled</span>
                        </>
                      )}
                    </div>
                  </div>
                </div>

                {subscriptionData.trialEndsAt && (
                  <Alert>
                    <Clock className="h-4 w-4" />
                    <AlertDescription>
                      Your trial ends on {formatDate(subscriptionData.trialEndsAt)}.
                      <Button variant="link" className="p-0 h-auto ml-2">
                        Upgrade now
                      </Button>
                    </AlertDescription>
                  </Alert>
                )}

                {subscriptionData.cancelAtPeriodEnd && (
                  <Alert variant="destructive">
                    <AlertTriangle className="h-4 w-4" />
                    <AlertDescription>
                      Your subscription will be cancelled at the end of the current period ({formatDate(subscriptionData.currentPeriodEnd)}).
                      <Button variant="link" className="p-0 h-auto ml-2" onClick={handleReactivateSubscription}>
                        Reactivate
                      </Button>
                    </AlertDescription>
                  </Alert>
                )}
              </>
            )}
          </CardContent>
        </Card>

        {/* Quick Actions */}
        <Card>
          <CardHeader>
            <CardTitle>Quick Actions</CardTitle>
          </CardHeader>

          <CardContent className="space-y-3">
            <Button
              onClick={() => router.push('/billing/pricing')}
              className="w-full justify-start"
              variant="outline"
            >
              <ArrowUpRight className="h-4 w-4 mr-2" />
              Upgrade Plan
            </Button>

            <Button
              onClick={() => router.push('/billing/payment-methods')}
              className="w-full justify-start"
              variant="outline"
            >
              <CreditCard className="h-4 w-4 mr-2" />
              Payment Methods
            </Button>

            <Button
              onClick={() => router.push('/billing/invoices')}
              className="w-full justify-start"
              variant="outline"
            >
              <Download className="h-4 w-4 mr-2" />
              Download Invoices
            </Button>

            {subscriptionData?.status === 'active' && !subscriptionData.cancelAtPeriodEnd && (
              <Button
                onClick={handleCancelSubscription}
                className="w-full justify-start"
                variant="outline"
              >
                <AlertTriangle className="h-4 w-4 mr-2" />
                Cancel Subscription
              </Button>
            )}

            <Button
              onClick={() => router.push('/support')}
              className="w-full justify-start"
              variant="outline"
            >
              <Mail className="h-4 w-4 mr-2" />
              Contact Support
            </Button>
          </CardContent>
        </Card>
      </div>

      {/* Usage Statistics */}
      {usageData && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Usage Statistics</CardTitle>
                <CardDescription>
                  Current usage for {usageData.period}
                </CardDescription>
              </div>
              <Button variant="outline" size="sm" onClick={loadSubscriptionData}>
                <RefreshCw className="h-4 w-4 mr-2" />
                Refresh
              </Button>
            </div>
          </CardHeader>

          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {/* API Calls */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">API Calls</span>
                  <span className="text-sm text-gray-500">
                    {usageData.apiCalls.used.toLocaleString()}/{usageData.apiCalls.limit === -1 ? '∞' : usageData.apiCalls.limit.toLocaleString()}
                  </span>
                </div>
                <Progress
                  value={getUsagePercentage(usageData.apiCalls.used, usageData.apiCalls.limit)}
                  className="h-2"
                />
              </div>

              {/* Strategies */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Active Strategies</span>
                  <span className="text-sm text-gray-500">
                    {usageData.strategies.used}/{usageData.strategies.limit === -1 ? '∞' : usageData.strategies.limit}
                  </span>
                </div>
                <Progress
                  value={getUsagePercentage(usageData.strategies.used, usageData.strategies.limit)}
                  className="h-2"
                />
              </div>

              {/* Reports */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Reports Generated</span>
                  <span className="text-sm text-gray-500">
                    {usageData.reports.used}/{usageData.reports.limit === -1 ? '∞' : usageData.reports.limit}
                  </span>
                </div>
                <Progress
                  value={getUsagePercentage(usageData.reports.used, usageData.reports.limit)}
                  className="h-2"
                />
              </div>

              {/* Data Storage */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Data Storage</span>
                  <span className="text-sm text-gray-500">
                    {usageData.dataStorage.used.toFixed(1)}{usageData.dataStorage.unit}/
                    {usageData.dataStorage.limit === -1 ? '∞' : `${usageData.dataStorage.limit}${usageData.dataStorage.unit}`}
                  </span>
                </div>
                <Progress
                  value={getUsagePercentage(usageData.dataStorage.used, usageData.dataStorage.limit)}
                  className="h-2"
                />
              </div>

              {/* Positions */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Open Positions</span>
                  <span className="text-sm text-gray-500">
                    {usageData.positions.used}/{usageData.positions.limit === -1 ? '∞' : usageData.positions.limit}
                  </span>
                </div>
                <Progress
                  value={getUsagePercentage(usageData.positions.used, usageData.positions.limit)}
                  className="h-2"
                />
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Recent Billing */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Recent Billing</CardTitle>
              <CardDescription>
                Your latest billing transactions
              </CardDescription>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setActiveTab('billing')}
            >
              View All
            </Button>
          </div>
        </CardHeader>

        <CardContent>
          {billingHistory.length === 0 ? (
            <div className="text-center py-8">
              <CreditCard className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">No billing history available</p>
            </div>
          ) : (
            <div className="space-y-3">
              {billingHistory.slice(0, 5).map((transaction) => (
                <div key={transaction.id} className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className={`p-2 rounded-lg ${
                      transaction.status === 'paid' ? 'bg-green-100' :
                      transaction.status === 'pending' ? 'bg-yellow-100' :
                      transaction.status === 'failed' ? 'bg-red-100' :
                      'bg-gray-100'
                    }`}>
                      <CreditCard className={`h-4 w-4 ${
                        transaction.status === 'paid' ? 'text-green-600' :
                        transaction.status === 'pending' ? 'text-yellow-600' :
                        transaction.status === 'failed' ? 'text-red-600' :
                        'text-gray-600'
                      }`} />
                    </div>
                    <div>
                      <p className="font-medium">{transaction.description}</p>
                      <p className="text-sm text-gray-500">
                        {formatDate(transaction.date)} • {transaction.paymentMethod}
                      </p>
                    </div>
                  </div>

                  <div className="text-right">
                    <p className="font-medium">
                      {formatCurrency(transaction.amount, transaction.currency)}
                    </p>
                    {getStatusBadge(transaction.status)}
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );

  const renderBilling = () => (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Billing History</CardTitle>
          <CardDescription>
            Complete history of your billing transactions
          </CardDescription>
        </CardHeader>

        <CardContent>
          {billingHistory.length === 0 ? (
            <div className="text-center py-8">
              <CreditCard className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">No billing history available</p>
            </div>
          ) : (
            <div className="space-y-3">
              {billingHistory.map((transaction) => (
                <div key={transaction.id} className="flex items-center justify-between p-4 border rounded-lg">
                  <div className="flex items-center space-x-4">
                    <div className={`p-2 rounded-lg ${
                      transaction.status === 'paid' ? 'bg-green-100' :
                      transaction.status === 'pending' ? 'bg-yellow-100' :
                      transaction.status === 'failed' ? 'bg-red-100' :
                      'bg-gray-100'
                    }`}>
                      <CreditCard className={`h-5 w-5 ${
                        transaction.status === 'paid' ? 'text-green-600' :
                        transaction.status === 'pending' ? 'text-yellow-600' :
                        transaction.status === 'failed' ? 'text-red-600' :
                        'text-gray-600'
                      }`} />
                    </div>
                    <div>
                      <p className="font-medium">{transaction.description}</p>
                      <p className="text-sm text-gray-500">
                        {formatDate(transaction.date)} • {transaction.paymentMethod}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center space-x-4">
                    <div className="text-right">
                      <p className="font-medium">
                        {formatCurrency(transaction.amount, transaction.currency)}
                      </p>
                      {getStatusBadge(transaction.status)}
                    </div>

                    {transaction.invoiceUrl && (
                      <Button variant="outline" size="sm">
                        <Download className="h-4 w-4 mr-2" />
                        Invoice
                      </Button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );

  if (!user) {
    return (
      <Alert>
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          Please log in to view your subscription dashboard.
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className={`max-w-6xl mx-auto space-y-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Subscription Dashboard</h1>
          <p className="text-gray-600">Manage your subscription and billing</p>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Loading State */}
      {isLoading ? (
        <div className="space-y-4">
          {[1, 2, 3].map(i => (
            <Card key={i}>
              <CardContent className="p-6">
                <div className="animate-pulse space-y-4">
                  <div className="h-4 bg-gray-200 rounded w-1/4"></div>
                  <div className="h-8 bg-gray-200 rounded w-1/2"></div>
                  <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        /* Main Content */
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-2 max-w-md">
            <TabsTrigger value="overview" className="flex items-center space-x-2">
              <BarChart3 className="h-4 w-4" />
              <span>Overview</span>
            </TabsTrigger>
            <TabsTrigger value="billing" className="flex items-center space-x-2">
              <CreditCard className="h-4 w-4" />
              <span>Billing</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="overview">
            {renderOverview()}
          </TabsContent>

          <TabsContent value="billing">
            {renderBilling()}
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
}