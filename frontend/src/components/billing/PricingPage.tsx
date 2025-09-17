/**
 * 💳 PRICING PAGE COMPONENT
 * Interactive pricing page with plan comparison and upgrade flows
 */

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '../../contexts/AuthContext';
import { usePermissions } from '../../hooks/usePermissions';
import { Button } from '../ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Switch } from '../ui/switch';
import { Alert, AlertDescription } from '../ui/alert';
import {
  Check,
  X,
  Star,
  Crown,
  Zap,
  Building,
  ArrowRight,
  TrendingUp,
  Shield,
  Headphones,
  Globe,
  Users,
  BarChart3,
  Settings,
  Sparkles,
  Clock,
  DollarSign,
  CreditCard,
  Gift,
  AlertCircle,
  CheckCircle
} from 'lucide-react';

interface PricingPlan {
  id: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  price: {
    monthly: number;
    annually: number;
  };
  popular?: boolean;
  enterprise?: boolean;
  features: {
    category: string;
    items: {
      name: string;
      included: boolean | string | number;
      highlight?: boolean;
    }[];
  }[];
  limitations?: {
    name: string;
    value: string | number;
  }[];
  cta: string;
  color: string;
}

interface PricingPageProps {
  className?: string;
  showCurrentPlan?: boolean;
}

export function PricingPage({ className = "", showCurrentPlan = true }: PricingPageProps) {
  const router = useRouter();
  const { user } = useAuth();
  const { getSubscriptionInfo } = usePermissions();

  const [isAnnual, setIsAnnual] = useState(true);
  const [selectedPlan, setSelectedPlan] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const subscriptionInfo = getSubscriptionInfo();

  const pricingPlans: PricingPlan[] = [
    {
      id: 'trial',
      name: 'Trial',
      description: 'Perfect for getting started',
      icon: <Star className="h-6 w-6" />,
      price: { monthly: 0, annually: 0 },
      color: 'gray',
      cta: 'Start Free Trial',
      features: [
        {
          category: 'Trading Features',
          items: [
            { name: 'Basic Trading Algorithms', included: true },
            { name: 'Paper Trading', included: true },
            { name: 'Basic Market Analysis', included: true },
            { name: 'Advanced Trading', included: false },
            { name: 'Custom Strategies', included: false },
          ]
        },
        {
          category: 'Data & Analytics',
          items: [
            { name: 'Real-time Data', included: '1 hour delay' },
            { name: 'Basic Charts', included: true },
            { name: 'Data Export', included: false },
            { name: 'Advanced Analytics', included: false },
            { name: 'Custom Reports', included: false },
          ]
        },
        {
          category: 'Support & Features',
          items: [
            { name: 'Email Support', included: 'Basic' },
            { name: 'API Access', included: false },
            { name: 'Portfolio Management', included: 'Basic' },
            { name: 'Risk Management', included: 'Basic' },
          ]
        }
      ],
      limitations: [
        { name: 'Trial Period', value: '14 days' },
        { name: 'Max Positions', value: 1 },
        { name: 'Data Retention', value: '7 days' },
        { name: 'API Calls/min', value: 10 },
      ]
    },
    {
      id: 'basic',
      name: 'Basic',
      description: 'For individual traders',
      icon: <TrendingUp className="h-6 w-6" />,
      price: { monthly: 29, annually: 290 },
      color: 'blue',
      cta: 'Upgrade to Basic',
      features: [
        {
          category: 'Trading Features',
          items: [
            { name: 'Advanced Trading Algorithms', included: true, highlight: true },
            { name: 'Live Trading', included: true },
            { name: 'Multiple Strategies', included: 3 },
            { name: 'Backtesting', included: '90 days' },
            { name: 'Position Management', included: true },
          ]
        },
        {
          category: 'Data & Analytics',
          items: [
            { name: 'Real-time Data', included: true, highlight: true },
            { name: 'Advanced Charts', included: true },
            { name: 'Technical Indicators', included: '25+' },
            { name: 'Data Export', included: 'CSV/JSON' },
            { name: 'Basic Reports', included: true },
          ]
        },
        {
          category: 'Support & Features',
          items: [
            { name: 'Email Support', included: 'Standard' },
            { name: 'API Access', included: 'Basic' },
            { name: 'Mobile App', included: true },
            { name: 'Webhook Support', included: true },
          ]
        }
      ],
      limitations: [
        { name: 'Max Positions', value: 10 },
        { name: 'Data Retention', value: '90 days' },
        { name: 'API Calls/min', value: 60 },
        { name: 'Reports/month', value: 10 },
      ]
    },
    {
      id: 'premium',
      name: 'Premium',
      description: 'For professional traders',
      icon: <Crown className="h-6 w-6" />,
      price: { monthly: 99, annually: 990 },
      popular: true,
      color: 'purple',
      cta: 'Upgrade to Premium',
      features: [
        {
          category: 'Trading Features',
          items: [
            { name: 'Custom Strategy Builder', included: true, highlight: true },
            { name: 'Unlimited Strategies', included: true },
            { name: 'Advanced Backtesting', included: 'Unlimited' },
            { name: 'Portfolio Optimization', included: true },
            { name: 'Risk Analytics', included: 'Advanced' },
          ]
        },
        {
          category: 'Data & Analytics',
          items: [
            { name: 'Multi-Asset Data', included: true, highlight: true },
            { name: 'Level 2 Market Data', included: true },
            { name: 'Custom Indicators', included: true },
            { name: 'Advanced Reports', included: 'Unlimited' },
            { name: 'Data API', included: 'Full Access' },
          ]
        },
        {
          category: 'Support & Features',
          items: [
            { name: 'Priority Support', included: true, highlight: true },
            { name: 'Phone Support', included: true },
            { name: 'Dedicated Account Manager', included: true },
            { name: 'Custom Integrations', included: 'Limited' },
          ]
        }
      ],
      limitations: [
        { name: 'Max Positions', value: 50 },
        { name: 'Data Retention', value: '1 year' },
        { name: 'API Calls/min', value: 300 },
        { name: 'Reports/month', value: 50 },
      ]
    },
    {
      id: 'enterprise',
      name: 'Enterprise',
      description: 'For institutions and teams',
      icon: <Building className="h-6 w-6" />,
      price: { monthly: 499, annually: 4990 },
      enterprise: true,
      color: 'orange',
      cta: 'Contact Sales',
      features: [
        {
          category: 'Trading Features',
          items: [
            { name: 'White Label Solution', included: true, highlight: true },
            { name: 'Multi-User Management', included: true },
            { name: 'Custom Development', included: true },
            { name: 'Advanced Risk Controls', included: true },
            { name: 'Institutional Grade', included: true },
          ]
        },
        {
          category: 'Data & Analytics',
          items: [
            { name: 'Custom Data Sources', included: true, highlight: true },
            { name: 'Real-time Streaming', included: 'Unlimited' },
            { name: 'Custom Dashboards', included: true },
            { name: 'Enterprise Reports', included: true },
            { name: 'Data Warehouse', included: true },
          ]
        },
        {
          category: 'Support & Features',
          items: [
            { name: 'Dedicated Support Team', included: true, highlight: true },
            { name: '24/7 Support', included: true },
            { name: 'SLA Guarantee', included: '99.9%' },
            { name: 'On-premise Deployment', included: true },
          ]
        }
      ],
      limitations: [
        { name: 'Max Positions', value: 'Unlimited' },
        { name: 'Data Retention', value: 'Unlimited' },
        { name: 'API Calls/min', value: 'Unlimited' },
        { name: 'Users', value: 'Unlimited' },
      ]
    }
  ];

  const getDiscountPercentage = () => {
    return Math.round(((12 - 10) / 12) * 100); // 17% discount for annual
  };

  const getCurrentPlan = () => {
    return pricingPlans.find(plan => plan.id === subscriptionInfo?.role) || pricingPlans[0];
  };

  const handlePlanSelect = async (planId: string) => {
    if (!user) {
      router.push('/auth/login');
      return;
    }

    setSelectedPlan(planId);
    setIsLoading(true);

    try {
      if (planId === 'enterprise') {
        router.push('/contact/sales?plan=enterprise');
      } else if (planId === 'trial') {
        router.push('/auth/register');
      } else {
        router.push(`/billing/checkout?plan=${planId}&billing=${isAnnual ? 'annual' : 'monthly'}`);
      }
    } catch (error) {
      console.error('Plan selection failed:', error);
    } finally {
      setIsLoading(false);
      setSelectedPlan(null);
    }
  };

  const renderFeatureValue = (item: any) => {
    if (typeof item.included === 'boolean') {
      return item.included ? (
        <Check className="h-4 w-4 text-green-500" />
      ) : (
        <X className="h-4 w-4 text-gray-300" />
      );
    }

    if (typeof item.included === 'string' || typeof item.included === 'number') {
      return (
        <span className="text-sm font-medium text-gray-900">
          {item.included}
        </span>
      );
    }

    return <X className="h-4 w-4 text-gray-300" />;
  };

  const renderPlanCard = (plan: PricingPlan) => {
    const isCurrentPlan = subscriptionInfo?.role === plan.id;
    const monthlyPrice = plan.price.monthly;
    const annualPrice = plan.price.annually;
    const displayPrice = isAnnual ? annualPrice / 12 : monthlyPrice;
    const savings = isAnnual && monthlyPrice > 0 ? monthlyPrice * 12 - annualPrice : 0;

    return (
      <Card
        key={plan.id}
        className={`relative overflow-hidden transition-all duration-300 ${
          plan.popular
            ? 'border-purple-200 shadow-lg scale-105 z-10'
            : isCurrentPlan
            ? 'border-green-200 bg-green-50'
            : 'hover:shadow-md hover:scale-102'
        }`}
      >
        {plan.popular && (
          <div className="absolute top-0 left-0 right-0 bg-gradient-to-r from-purple-500 to-pink-500 text-white text-center py-2 text-sm font-medium">
            <Sparkles className="inline h-4 w-4 mr-1" />
            Most Popular
          </div>
        )}

        {isCurrentPlan && (
          <div className="absolute top-0 left-0 right-0 bg-green-500 text-white text-center py-2 text-sm font-medium">
            <CheckCircle className="inline h-4 w-4 mr-1" />
            Current Plan
          </div>
        )}

        <CardHeader className={`text-center ${plan.popular || isCurrentPlan ? 'pt-12' : 'pt-6'}`}>
          <div className={`mx-auto w-12 h-12 bg-${plan.color}-100 rounded-full flex items-center justify-center mb-4`}>
            <div className={`text-${plan.color}-600`}>
              {plan.icon}
            </div>
          </div>

          <CardTitle className="text-xl font-bold">{plan.name}</CardTitle>
          <CardDescription className="text-gray-600">
            {plan.description}
          </CardDescription>

          <div className="mt-4">
            {plan.price.monthly === 0 ? (
              <div className="text-3xl font-bold">Free</div>
            ) : plan.enterprise ? (
              <div className="text-3xl font-bold">Custom</div>
            ) : (
              <div className="space-y-2">
                <div className="text-3xl font-bold">
                  ${displayPrice.toFixed(0)}
                  <span className="text-lg font-normal text-gray-500">/month</span>
                </div>
                {isAnnual && savings > 0 && (
                  <div className="text-sm text-green-600">
                    Save ${savings} annually ({getDiscountPercentage()}% off)
                  </div>
                )}
                {!isAnnual && plan.price.annually > 0 && (
                  <div className="text-sm text-gray-500">
                    ${plan.price.annually}/year if paid annually
                  </div>
                )}
              </div>
            )}
          </div>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* Plan Limitations/Highlights */}
          {plan.limitations && (
            <div className="grid grid-cols-2 gap-2">
              {plan.limitations.map((limit, index) => (
                <div key={index} className="text-center p-2 bg-gray-50 rounded-lg">
                  <div className="text-sm font-bold text-gray-900">{limit.value}</div>
                  <div className="text-xs text-gray-500">{limit.name}</div>
                </div>
              ))}
            </div>
          )}

          {/* Features by Category */}
          <div className="space-y-4">
            {plan.features.map((category, categoryIndex) => (
              <div key={categoryIndex}>
                <h4 className="font-medium text-gray-900 mb-2">{category.category}</h4>
                <div className="space-y-2">
                  {category.items.map((item, itemIndex) => (
                    <div
                      key={itemIndex}
                      className={`flex items-center justify-between p-2 rounded ${
                        item.highlight ? 'bg-blue-50 border border-blue-200' : ''
                      }`}
                    >
                      <span className={`text-sm ${item.highlight ? 'font-medium' : ''}`}>
                        {item.name}
                        {item.highlight && (
                          <Badge variant="outline" className="ml-2 text-xs">
                            New
                          </Badge>
                        )}
                      </span>
                      {renderFeatureValue(item)}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* CTA Button */}
          <Button
            onClick={() => handlePlanSelect(plan.id)}
            disabled={isLoading && selectedPlan === plan.id || isCurrentPlan}
            className={`w-full ${
              plan.popular
                ? 'bg-purple-600 hover:bg-purple-700'
                : plan.enterprise
                ? 'bg-orange-600 hover:bg-orange-700'
                : isCurrentPlan
                ? 'bg-green-600 hover:bg-green-700'
                : `bg-${plan.color}-600 hover:bg-${plan.color}-700`
            }`}
            size="lg"
          >
            {isLoading && selectedPlan === plan.id ? (
              <>
                <Clock className="h-4 w-4 mr-2 animate-spin" />
                Processing...
              </>
            ) : isCurrentPlan ? (
              'Current Plan'
            ) : (
              <>
                {plan.cta}
                <ArrowRight className="h-4 w-4 ml-2" />
              </>
            )}
          </Button>

          {plan.enterprise && (
            <p className="text-xs text-center text-gray-500">
              Custom pricing based on your requirements
            </p>
          )}
        </CardContent>
      </Card>
    );
  };

  return (
    <div className={`max-w-7xl mx-auto space-y-8 ${className}`}>
      {/* Header */}
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold">Choose Your Trading Plan</h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          Select the perfect plan for your trading needs. Upgrade or downgrade at any time.
        </p>

        {/* Billing Toggle */}
        <div className="flex items-center justify-center space-x-4 mt-8">
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
            <Badge className="bg-green-100 text-green-800 ml-2">
              <Gift className="h-3 w-3 mr-1" />
              Save {getDiscountPercentage()}%
            </Badge>
          )}
        </div>
      </div>

      {/* Current Plan Alert */}
      {showCurrentPlan && user && (
        <Alert className="max-w-2xl mx-auto">
          <CheckCircle className="h-4 w-4" />
          <AlertDescription>
            You're currently on the <strong>{getCurrentPlan().name}</strong> plan.
            {subscriptionInfo?.status === 'expired' && (
              <span className="text-red-600"> Your subscription has expired.</span>
            )}
          </AlertDescription>
        </Alert>
      )}

      {/* Pricing Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-12">
        {pricingPlans.map(plan => renderPlanCard(plan))}
      </div>

      {/* FAQ Section */}
      <div className="mt-16 max-w-4xl mx-auto">
        <h2 className="text-2xl font-bold text-center mb-8">Frequently Asked Questions</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card>
            <CardContent className="p-6">
              <h3 className="font-medium mb-2">Can I change plans anytime?</h3>
              <p className="text-sm text-gray-600">
                Yes, you can upgrade or downgrade your plan at any time. Changes take effect immediately,
                and you'll be charged/credited on a prorated basis.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <h3 className="font-medium mb-2">What payment methods do you accept?</h3>
              <p className="text-sm text-gray-600">
                We accept major credit cards, PayPal, and PIX (for Brazilian customers).
                All payments are processed securely through encrypted channels.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <h3 className="font-medium mb-2">Is there a free trial?</h3>
              <p className="text-sm text-gray-600">
                Yes! You can start with our 14-day free trial that includes access to basic trading
                features and paper trading to test our platform.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <h3 className="font-medium mb-2">Can I cancel anytime?</h3>
              <p className="text-sm text-gray-600">
                Absolutely. You can cancel your subscription at any time from your billing dashboard.
                You'll retain access until the end of your current billing period.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Enterprise CTA */}
      <div className="mt-16 bg-gradient-to-r from-orange-500 to-red-600 rounded-2xl p-8 text-white text-center">
        <Building className="h-12 w-12 mx-auto mb-4" />
        <h2 className="text-2xl font-bold mb-2">Need an Enterprise Solution?</h2>
        <p className="text-lg opacity-90 mb-6">
          Get custom pricing, dedicated support, and enterprise-grade features
        </p>
        <Button
          onClick={() => router.push('/contact/sales')}
          className="bg-white text-orange-600 hover:bg-gray-100"
          size="lg"
        >
          <Headphones className="h-5 w-5 mr-2" />
          Contact Sales Team
        </Button>
      </div>

      {/* Trust Indicators */}
      <div className="mt-16 text-center">
        <p className="text-gray-500 mb-6">Trusted by thousands of traders worldwide</p>
        <div className="flex items-center justify-center space-x-8 opacity-60">
          <div className="flex items-center space-x-2">
            <Shield className="h-5 w-5" />
            <span className="text-sm">Bank-level Security</span>
          </div>
          <div className="flex items-center space-x-2">
            <Globe className="h-5 w-5" />
            <span className="text-sm">Global Access</span>
          </div>
          <div className="flex items-center space-x-2">
            <Users className="h-5 w-5" />
            <span className="text-sm">10,000+ Users</span>
          </div>
          <div className="flex items-center space-x-2">
            <BarChart3 className="h-5 w-5" />
            <span className="text-sm">99.9% Uptime</span>
          </div>
        </div>
      </div>
    </div>
  );
}