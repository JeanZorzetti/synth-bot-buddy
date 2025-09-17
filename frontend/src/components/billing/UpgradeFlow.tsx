/**
 * 🚀 UPGRADE/DOWNGRADE FLOW COMPONENT
 * Complete subscription change flow with prorations and confirmations
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
import { Switch } from '../ui/switch';
import { Progress } from '../ui/progress';
import {
  ArrowRight,
  ArrowLeft,
  Crown,
  Star,
  Building,
  Zap,
  Check,
  X,
  DollarSign,
  Calendar,
  CreditCard,
  AlertTriangle,
  CheckCircle,
  RefreshCw,
  TrendingUp,
  TrendingDown,
  Shield,
  Clock,
  Calculator,
  Gift,
  Sparkles,
  Users,
  BarChart3,
  Settings,
  Globe,
  Phone,
  Mail
} from 'lucide-react';

interface Plan {
  id: string;
  name: string;
  description: string;
  price: { monthly: number; annually: number };
  features: { category: string; items: string[] }[];
  limits: { [key: string]: number | string };
  popular?: boolean;
  enterprise?: boolean;
}

interface CurrentSubscription {
  plan: string;
  status: string;
  billingCycle: 'monthly' | 'annually';
  currentPeriodEnd: string;
  amount: number;
  currency: string;
}

interface ProrationCalculation {
  currentPlan: string;
  newPlan: string;
  prorationAmount: number;
  creditAmount: number;
  immediateCharge: number;
  nextBillingAmount: number;
  nextBillingDate: string;
  currency: string;
  breakdown: {
    unusedCredit: number;
    newPlanCharge: number;
    prorationDays: number;
    totalDays: number;
  };
}

interface UpgradeFlowProps {
  targetPlan?: string;
  className?: string;
}

type FlowStep = 'comparison' | 'proration' | 'confirmation' | 'processing' | 'success';

export function UpgradeFlow({ targetPlan, className = "" }: UpgradeFlowProps) {
  const router = useRouter();
  const { user } = useAuth();
  const { getSubscriptionInfo } = usePermissions();

  const [currentStep, setCurrentStep] = useState<FlowStep>('comparison');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [selectedPlan, setSelectedPlan] = useState<string>(targetPlan || '');
  const [isAnnual, setIsAnnual] = useState(true);
  const [currentSubscription, setCurrentSubscription] = useState<CurrentSubscription | null>(null);
  const [prorationData, setProrationData] = useState<ProrationCalculation | null>(null);
  const [changeReason, setChangeReason] = useState('');

  const subscriptionInfo = getSubscriptionInfo();

  const plans: Plan[] = [
    {
      id: 'basic',
      name: 'Basic',
      description: 'Perfect for individual traders',
      price: { monthly: 29, annually: 290 },
      features: [
        {
          category: 'Trading',
          items: ['Real-time trading', 'Basic strategies', 'Position management']
        },
        {
          category: 'Data',
          items: ['Real-time data', 'Basic charts', 'CSV export']
        },
        {
          category: 'Support',
          items: ['Email support', 'Basic API access']
        }
      ],
      limits: {
        'Max Positions': 10,
        'API Calls/min': 60,
        'Data Retention': '90 days',
        'Reports/month': 10
      }
    },
    {
      id: 'premium',
      name: 'Premium',
      description: 'For professional traders',
      price: { monthly: 99, annually: 990 },
      popular: true,
      features: [
        {
          category: 'Trading',
          items: ['Advanced strategies', 'Custom indicators', 'Portfolio optimization']
        },
        {
          category: 'Data',
          items: ['Multi-asset data', 'Advanced analytics', 'Custom reports']
        },
        {
          category: 'Support',
          items: ['Priority support', 'Phone support', 'Account manager']
        }
      ],
      limits: {
        'Max Positions': 50,
        'API Calls/min': 300,
        'Data Retention': '1 year',
        'Reports/month': 50
      }
    },
    {
      id: 'enterprise',
      name: 'Enterprise',
      description: 'For institutions and teams',
      price: { monthly: 499, annually: 4990 },
      enterprise: true,
      features: [
        {
          category: 'Trading',
          items: ['White label solution', 'Custom development', 'Multi-user management']
        },
        {
          category: 'Data',
          items: ['Custom data sources', 'Real-time streaming', 'Data warehouse']
        },
        {
          category: 'Support',
          items: ['24/7 support', 'SLA guarantee', 'Dedicated team']
        }
      ],
      limits: {
        'Max Positions': 'Unlimited',
        'API Calls/min': 'Unlimited',
        'Data Retention': 'Unlimited',
        'Users': 'Unlimited'
      }
    }
  ];

  useEffect(() => {
    loadCurrentSubscription();
  }, []);

  useEffect(() => {
    if (selectedPlan && currentSubscription && currentStep === 'proration') {
      calculateProration();
    }
  }, [selectedPlan, isAnnual, currentSubscription, currentStep]);

  const loadCurrentSubscription = async () => {
    try {
      const subscription = await apiClient.get<CurrentSubscription>('/billing/subscription/current');
      setCurrentSubscription(subscription);
    } catch (error) {
      console.error('Failed to load current subscription:', error);
    }
  };

  const calculateProration = async () => {
    if (!selectedPlan || !currentSubscription) return;

    try {
      setIsLoading(true);
      const proration = await apiClient.post<ProrationCalculation>('/billing/calculate-proration', {
        currentPlan: currentSubscription.plan,
        newPlan: selectedPlan,
        billingCycle: isAnnual ? 'annually' : 'monthly'
      });

      setProrationData(proration);
    } catch (error: any) {
      setError(error.message || 'Failed to calculate proration');
    } finally {
      setIsLoading(false);
    }
  };

  const executeSubscriptionChange = async () => {
    if (!selectedPlan || !prorationData) return;

    try {
      setCurrentStep('processing');
      setIsLoading(true);

      await apiClient.post('/billing/change-subscription', {
        newPlan: selectedPlan,
        billingCycle: isAnnual ? 'annually' : 'monthly',
        reason: changeReason,
        prorationAmount: prorationData.immediateCharge
      });

      setCurrentStep('success');
    } catch (error: any) {
      setError(error.message || 'Failed to change subscription');
      setCurrentStep('confirmation');
    } finally {
      setIsLoading(false);
    }
  };

  const getCurrentPlan = () => {
    return plans.find(plan => plan.id === subscriptionInfo?.role) || plans[0];
  };

  const getSelectedPlanData = () => {
    return plans.find(plan => plan.id === selectedPlan);
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

  const isUpgrade = () => {
    const currentPlan = getCurrentPlan();
    const newPlan = getSelectedPlanData();
    if (!currentPlan || !newPlan) return false;

    const planOrder = { basic: 1, premium: 2, enterprise: 3 };
    return planOrder[newPlan.id as keyof typeof planOrder] > planOrder[currentPlan.id as keyof typeof planOrder];
  };

  const getPlanIcon = (planId: string) => {
    switch (planId) {
      case 'basic':
        return <Star className="h-6 w-6 text-blue-500" />;
      case 'premium':
        return <Crown className="h-6 w-6 text-purple-500" />;
      case 'enterprise':
        return <Building className="h-6 w-6 text-orange-500" />;
      default:
        return <Zap className="h-6 w-6 text-gray-500" />;
    }
  };

  const renderPlanComparison = () => (
    <div className="space-y-6">
      {/* Current Plan */}
      <Card className="border-green-200 bg-green-50">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              {getPlanIcon(subscriptionInfo?.role || 'basic')}
              <div>
                <CardTitle className="capitalize">Your Current Plan: {subscriptionInfo?.role || 'Basic'}</CardTitle>
                <CardDescription>
                  Active until {currentSubscription ? formatDate(currentSubscription.currentPeriodEnd) : 'N/A'}
                </CardDescription>
              </div>
            </div>
            <Badge className="bg-green-100 text-green-800">Current</Badge>
          </div>
        </CardHeader>
      </Card>

      {/* Billing Toggle */}
      <div className="flex items-center justify-center space-x-4">
        <span className={`text-sm ${!isAnnual ? 'font-medium' : 'text-gray-500'}`}>
          Monthly
        </span>
        <Switch
          checked={isAnnual}
          onCheckedChange={setIsAnnual}
          className="data-[state=checked]:bg-purple-600"
        />
        <span className={`text-sm ${isAnnual ? 'font-medium' : 'text-gray-500'}`}>
          Annual
        </span>
        {isAnnual && (
          <Badge className="bg-green-100 text-green-800">
            <Gift className="h-3 w-3 mr-1" />
            Save 17%
          </Badge>
        )}
      </div>

      {/* Available Plans */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {plans.map((plan) => {
          const isCurrentPlan = plan.id === subscriptionInfo?.role;
          const displayPrice = isAnnual ? plan.price.annually / 12 : plan.price.monthly;
          const savings = isAnnual ? plan.price.monthly * 12 - plan.price.annually : 0;

          return (
            <Card
              key={plan.id}
              className={`relative cursor-pointer transition-all ${
                selectedPlan === plan.id
                  ? 'border-purple-500 shadow-lg scale-105'
                  : isCurrentPlan
                  ? 'border-green-200 bg-green-50'
                  : 'hover:shadow-md hover:scale-102'
              } ${plan.popular ? 'border-purple-200' : ''}`}
              onClick={() => !isCurrentPlan && setSelectedPlan(plan.id)}
            >
              {plan.popular && (
                <div className="absolute top-0 left-0 right-0 bg-purple-500 text-white text-center py-1 text-sm">
                  <Sparkles className="inline h-4 w-4 mr-1" />
                  Most Popular
                </div>
              )}

              <CardHeader className={`text-center ${plan.popular ? 'pt-8' : ''}`}>
                <div className="mx-auto mb-4">
                  {getPlanIcon(plan.id)}
                </div>

                <CardTitle>{plan.name}</CardTitle>
                <CardDescription>{plan.description}</CardDescription>

                <div className="mt-4">
                  {plan.enterprise ? (
                    <div className="text-2xl font-bold">Custom</div>
                  ) : (
                    <div>
                      <div className="text-3xl font-bold">
                        ${displayPrice.toFixed(0)}
                        <span className="text-lg font-normal text-gray-500">/month</span>
                      </div>
                      {isAnnual && savings > 0 && (
                        <div className="text-sm text-green-600">
                          Save ${savings} annually
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </CardHeader>

              <CardContent className="space-y-4">
                {/* Plan Limits */}
                <div className="grid grid-cols-2 gap-2">
                  {Object.entries(plan.limits).map(([key, value]) => (
                    <div key={key} className="text-center p-2 bg-gray-50 rounded">
                      <div className="text-sm font-bold">{value}</div>
                      <div className="text-xs text-gray-500">{key}</div>
                    </div>
                  ))}
                </div>

                {/* Key Features */}
                <div className="space-y-2">
                  {plan.features.slice(0, 2).map((category) => (
                    <div key={category.category}>
                      <p className="text-sm font-medium text-gray-700">{category.category}:</p>
                      <ul className="text-xs text-gray-600 space-y-1">
                        {category.items.slice(0, 2).map((item, index) => (
                          <li key={index} className="flex items-center">
                            <Check className="h-3 w-3 text-green-500 mr-1" />
                            {item}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>

                {isCurrentPlan ? (
                  <Button disabled className="w-full">
                    Current Plan
                  </Button>
                ) : (
                  <Button
                    variant={selectedPlan === plan.id ? "default" : "outline"}
                    className="w-full"
                    onClick={() => setSelectedPlan(plan.id)}
                  >
                    {selectedPlan === plan.id ? 'Selected' : `Choose ${plan.name}`}
                  </Button>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Continue Button */}
      {selectedPlan && selectedPlan !== subscriptionInfo?.role && (
        <div className="text-center">
          <Button
            onClick={() => setCurrentStep('proration')}
            size="lg"
            className="min-w-[200px]"
          >
            Continue to Review
            <ArrowRight className="h-4 w-4 ml-2" />
          </Button>
        </div>
      )}
    </div>
  );

  const renderProration = () => {
    const selectedPlanData = getSelectedPlanData();
    if (!selectedPlanData || !prorationData) return null;

    return (
      <div className="max-w-2xl mx-auto space-y-6">
        <div className="text-center">
          <h2 className="text-2xl font-bold mb-2">
            {isUpgrade() ? 'Upgrade' : 'Change'} to {selectedPlanData.name}
          </h2>
          <p className="text-gray-600">
            Review the changes and billing details below
          </p>
        </div>

        {/* Plan Change Summary */}
        <Card>
          <CardHeader>
            <CardTitle>Plan Change Summary</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="text-gray-500">From:</div>
                {getPlanIcon(currentSubscription?.plan || 'basic')}
                <span className="font-medium capitalize">{currentSubscription?.plan}</span>
              </div>
              <ArrowRight className="h-5 w-5 text-gray-400" />
              <div className="flex items-center space-x-3">
                <div className="text-gray-500">To:</div>
                {getPlanIcon(selectedPlan)}
                <span className="font-medium">{selectedPlanData.name}</span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 pt-4 border-t">
              <div>
                <p className="text-sm text-gray-500">Billing Cycle</p>
                <p className="font-medium capitalize">{isAnnual ? 'Annual' : 'Monthly'}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Effective Date</p>
                <p className="font-medium">Immediately</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Billing Calculation */}
        <Card>
          <CardHeader>
            <CardTitle>Billing Details</CardTitle>
            <CardDescription>
              Prorated billing calculation for your plan change
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-3">
              <div className="flex justify-between">
                <span>Unused credit from current plan</span>
                <span className="text-green-600">
                  +{formatCurrency(prorationData.creditAmount, prorationData.currency)}
                </span>
              </div>

              <div className="flex justify-between">
                <span>New plan charge (prorated)</span>
                <span>
                  {formatCurrency(prorationData.breakdown.newPlanCharge, prorationData.currency)}
                </span>
              </div>

              <div className="border-t pt-3">
                <div className="flex justify-between font-medium">
                  <span>{prorationData.immediateCharge >= 0 ? 'Amount due today' : 'Credit applied'}</span>
                  <span className={prorationData.immediateCharge >= 0 ? 'text-red-600' : 'text-green-600'}>
                    {prorationData.immediateCharge >= 0 ? '' : '+'}
                    {formatCurrency(Math.abs(prorationData.immediateCharge), prorationData.currency)}
                  </span>
                </div>
              </div>

              <div className="bg-gray-50 p-3 rounded">
                <div className="flex justify-between">
                  <span>Next billing amount</span>
                  <span className="font-medium">
                    {formatCurrency(prorationData.nextBillingAmount, prorationData.currency)}
                  </span>
                </div>
                <div className="flex justify-between text-sm text-gray-600">
                  <span>Next billing date</span>
                  <span>{formatDate(prorationData.nextBillingDate)}</span>
                </div>
              </div>
            </div>

            <div className="text-xs text-gray-500">
              <p>
                Proration calculated for {prorationData.breakdown.prorationDays} days
                out of {prorationData.breakdown.totalDays} total days in billing period.
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Feature Comparison */}
        <Card>
          <CardHeader>
            <CardTitle>What You'll Get</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-6">
              {selectedPlanData.features.map((category) => (
                <div key={category.category}>
                  <h4 className="font-medium mb-2">{category.category}</h4>
                  <ul className="space-y-1">
                    {category.items.map((item, index) => (
                      <li key={index} className="flex items-center text-sm">
                        <Check className="h-4 w-4 text-green-500 mr-2" />
                        {item}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Actions */}
        <div className="flex space-x-4">
          <Button
            variant="outline"
            onClick={() => setCurrentStep('comparison')}
            className="flex-1"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Plans
          </Button>
          <Button
            onClick={() => setCurrentStep('confirmation')}
            className="flex-1"
          >
            Continue to Confirm
            <ArrowRight className="h-4 w-4 ml-2" />
          </Button>
        </div>
      </div>
    );
  };

  const renderConfirmation = () => {
    const selectedPlanData = getSelectedPlanData();
    if (!selectedPlanData || !prorationData) return null;

    return (
      <div className="max-w-lg mx-auto space-y-6">
        <div className="text-center">
          <div className="mx-auto w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mb-4">
            {isUpgrade() ? (
              <TrendingUp className="h-8 w-8 text-blue-600" />
            ) : (
              <TrendingDown className="h-8 w-8 text-blue-600" />
            )}
          </div>
          <h2 className="text-2xl font-bold mb-2">
            Confirm Plan {isUpgrade() ? 'Upgrade' : 'Change'}
          </h2>
          <p className="text-gray-600">
            You're about to {isUpgrade() ? 'upgrade' : 'change'} to {selectedPlanData.name}
          </p>
        </div>

        <Card>
          <CardContent className="p-6 space-y-4">
            <div className="text-center">
              <div className="text-lg font-semibold">
                {prorationData.immediateCharge >= 0 ? 'Amount Due Today' : 'Credit Applied'}
              </div>
              <div className={`text-3xl font-bold ${
                prorationData.immediateCharge >= 0 ? 'text-red-600' : 'text-green-600'
              }`}>
                {prorationData.immediateCharge >= 0 ? '' : '+'}
                {formatCurrency(Math.abs(prorationData.immediateCharge), prorationData.currency)}
              </div>
            </div>

            <div className="space-y-2 text-sm border-t pt-4">
              <div className="flex justify-between">
                <span>New Plan:</span>
                <span className="font-medium">{selectedPlanData.name}</span>
              </div>
              <div className="flex justify-between">
                <span>Billing Cycle:</span>
                <span className="font-medium capitalize">{isAnnual ? 'Annual' : 'Monthly'}</span>
              </div>
              <div className="flex justify-between">
                <span>Next Billing:</span>
                <span className="font-medium">{formatDate(prorationData.nextBillingDate)}</span>
              </div>
              <div className="flex justify-between">
                <span>Next Amount:</span>
                <span className="font-medium">
                  {formatCurrency(prorationData.nextBillingAmount, prorationData.currency)}
                </span>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">
              Reason for change (optional)
            </label>
            <textarea
              value={changeReason}
              onChange={(e) => setChangeReason(e.target.value)}
              placeholder="Help us improve by telling us why you're changing plans..."
              className="w-full p-3 border rounded-md resize-none"
              rows={3}
            />
          </div>

          <Alert>
            <Shield className="h-4 w-4" />
            <AlertDescription>
              Your plan change will take effect immediately. You can change your plan again at any time.
            </AlertDescription>
          </Alert>
        </div>

        <div className="space-y-3">
          <Button
            onClick={executeSubscriptionChange}
            disabled={isLoading}
            className="w-full"
            size="lg"
          >
            {isLoading ? (
              <>
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <CheckCircle className="h-4 w-4 mr-2" />
                Confirm {isUpgrade() ? 'Upgrade' : 'Change'}
              </>
            )}
          </Button>

          <Button
            variant="outline"
            onClick={() => setCurrentStep('proration')}
            disabled={isLoading}
            className="w-full"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Review
          </Button>
        </div>
      </div>
    );
  };

  const renderProcessing = () => (
    <div className="max-w-md mx-auto text-center space-y-6">
      <div className="mx-auto w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mb-4">
        <RefreshCw className="h-8 w-8 text-blue-600 animate-spin" />
      </div>

      <div>
        <h2 className="text-2xl font-bold mb-2">Processing Your Request</h2>
        <p className="text-gray-600">
          We're updating your subscription. This usually takes just a few seconds.
        </p>
      </div>

      <div className="space-y-2">
        <Progress value={66} className="w-full" />
        <p className="text-sm text-gray-500">Updating subscription...</p>
      </div>

      <Alert>
        <Clock className="h-4 w-4" />
        <AlertDescription>
          Please don't close this window while we process your request.
        </AlertDescription>
      </Alert>
    </div>
  );

  const renderSuccess = () => {
    const selectedPlanData = getSelectedPlanData();

    return (
      <div className="max-w-md mx-auto text-center space-y-6">
        <div className="mx-auto w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mb-4">
          <CheckCircle className="h-8 w-8 text-green-600" />
        </div>

        <div>
          <h2 className="text-2xl font-bold mb-2">
            {isUpgrade() ? 'Upgrade' : 'Plan Change'} Successful!
          </h2>
          <p className="text-gray-600">
            You're now on the {selectedPlanData?.name} plan
          </p>
        </div>

        <Card>
          <CardContent className="p-6">
            <div className="space-y-3 text-sm">
              <div className="flex items-center space-x-3 p-3 bg-green-50 rounded-lg">
                <CheckCircle className="h-5 w-5 text-green-600" />
                <span>Plan updated successfully</span>
              </div>
              <div className="flex items-center space-x-3 p-3 bg-green-50 rounded-lg">
                <CheckCircle className="h-5 w-5 text-green-600" />
                <span>New features are now available</span>
              </div>
              <div className="flex items-center space-x-3 p-3 bg-green-50 rounded-lg">
                <CheckCircle className="h-5 w-5 text-green-600" />
                <span>Billing updated automatically</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="space-y-3">
          <Button
            onClick={() => router.push('/dashboard')}
            className="w-full"
            size="lg"
          >
            Go to Dashboard
            <ArrowRight className="h-4 w-4 ml-2" />
          </Button>

          <Button
            variant="outline"
            onClick={() => router.push('/billing')}
            className="w-full"
          >
            View Billing Details
          </Button>
        </div>

        <p className="text-xs text-gray-500">
          You'll receive a confirmation email shortly with your updated plan details.
        </p>
      </div>
    );
  };

  return (
    <div className={`max-w-6xl mx-auto space-y-6 ${className}`}>
      {/* Progress Steps */}
      <div className="flex items-center justify-center space-x-4 mb-8">
        {[
          { step: 'comparison', label: 'Choose Plan', icon: BarChart3 },
          { step: 'proration', label: 'Review', icon: Calculator },
          { step: 'confirmation', label: 'Confirm', icon: CheckCircle },
          { step: 'success', label: 'Complete', icon: Star }
        ].map(({ step, label, icon: Icon }, index) => {
          const isActive = currentStep === step;
          const isCompleted = ['comparison', 'proration', 'confirmation'].indexOf(currentStep) > index;

          return (
            <div key={step} className="flex items-center">
              <div className={`flex items-center justify-center w-10 h-10 rounded-full ${
                isActive ? 'bg-blue-600 text-white' :
                isCompleted ? 'bg-green-600 text-white' :
                'bg-gray-200 text-gray-500'
              }`}>
                <Icon className="h-5 w-5" />
              </div>
              <span className={`ml-2 text-sm ${
                isActive ? 'font-medium' : 'text-gray-500'
              }`}>
                {label}
              </span>
              {index < 3 && (
                <ArrowRight className="h-4 w-4 text-gray-400 mx-4" />
              )}
            </div>
          );
        })}
      </div>

      {/* Error Alert */}
      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Current Step Content */}
      {currentStep === 'comparison' && renderPlanComparison()}
      {currentStep === 'proration' && renderProration()}
      {currentStep === 'confirmation' && renderConfirmation()}
      {currentStep === 'processing' && renderProcessing()}
      {currentStep === 'success' && renderSuccess()}
    </div>
  );
}