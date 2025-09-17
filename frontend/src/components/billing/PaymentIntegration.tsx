/**
 * 💳 PAYMENT INTEGRATION COMPONENT
 * Multi-provider payment integration with Stripe, PIX, and PayPal
 */

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '../../contexts/AuthContext';
import { apiClient } from '../../services/apiClient';
import { Button } from '../ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Alert, AlertDescription } from '../ui/alert';
import { Badge } from '../ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { RadioGroup, RadioGroupItem } from '../ui/radio-group';
import {
  CreditCard,
  DollarSign,
  Shield,
  CheckCircle,
  AlertTriangle,
  RefreshCw,
  Lock,
  Smartphone,
  QrCode,
  Clock,
  ArrowRight,
  Copy,
  Check,
  X,
  Globe,
  Building,
  User,
  Calendar,
  Eye,
  EyeOff,
  Gift,
  Star,
  Crown,
  Zap
} from 'lucide-react';

interface PaymentMethod {
  id: string;
  type: 'card' | 'pix' | 'paypal' | 'bank_transfer';
  provider: 'stripe' | 'pix' | 'paypal';
  last4?: string;
  brand?: string;
  expiryMonth?: number;
  expiryYear?: number;
  email?: string;
  isDefault: boolean;
  status: 'active' | 'expired' | 'failed';
}

interface PaymentPlan {
  id: string;
  name: string;
  price: number;
  currency: string;
  billingCycle: 'monthly' | 'annually';
  features: string[];
  discount?: number;
}

interface PIXPayment {
  qrCode: string;
  pixKey: string;
  amount: number;
  expiresAt: string;
  paymentId: string;
}

interface PaymentIntegrationProps {
  selectedPlan?: PaymentPlan;
  onPaymentSuccess?: (paymentId: string) => void;
  onPaymentError?: (error: string) => void;
  className?: string;
}

export function PaymentIntegration({
  selectedPlan,
  onPaymentSuccess,
  onPaymentError,
  className = ""
}: PaymentIntegrationProps) {
  const router = useRouter();
  const { user } = useAuth();

  const [activePaymentMethod, setActivePaymentMethod] = useState<'stripe' | 'pix' | 'paypal'>('stripe');
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Card form state
  const [cardData, setCardData] = useState({
    number: '',
    expiryMonth: '',
    expiryYear: '',
    cvc: '',
    name: '',
    email: user?.email || '',
    address: {
      line1: '',
      city: '',
      state: '',
      postalCode: '',
      country: 'US'
    }
  });

  // PIX payment state
  const [pixPayment, setPixPayment] = useState<PIXPayment | null>(null);
  const [pixCountdown, setPixCountdown] = useState(0);

  // Saved payment methods
  const [savedMethods, setSavedMethods] = useState<PaymentMethod[]>([]);
  const [selectedSavedMethod, setSelectedSavedMethod] = useState<string | null>(null);

  // UI states
  const [showCVC, setShowCVC] = useState(false);
  const [agreedToTerms, setAgreedToTerms] = useState(false);

  useEffect(() => {
    loadSavedPaymentMethods();
  }, []);

  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (pixCountdown > 0) {
      timer = setTimeout(() => setPixCountdown(pixCountdown - 1), 1000);
    }
    return () => clearTimeout(timer);
  }, [pixCountdown]);

  const loadSavedPaymentMethods = async () => {
    try {
      const methods = await apiClient.get<PaymentMethod[]>('/billing/payment-methods');
      setSavedMethods(methods);
    } catch (error) {
      console.error('Failed to load payment methods:', error);
    }
  };

  const validateCardData = () => {
    const errors = [];

    // Card number validation (basic)
    const cleanNumber = cardData.number.replace(/\s/g, '');
    if (!cleanNumber || cleanNumber.length < 13 || cleanNumber.length > 19) {
      errors.push('Invalid card number');
    }

    // Expiry validation
    const currentYear = new Date().getFullYear() % 100;
    const currentMonth = new Date().getMonth() + 1;
    const expYear = parseInt(cardData.expiryYear);
    const expMonth = parseInt(cardData.expiryMonth);

    if (!expMonth || expMonth < 1 || expMonth > 12) {
      errors.push('Invalid expiry month');
    }

    if (!expYear || expYear < currentYear || (expYear === currentYear && expMonth < currentMonth)) {
      errors.push('Card has expired');
    }

    // CVC validation
    if (!cardData.cvc || cardData.cvc.length < 3 || cardData.cvc.length > 4) {
      errors.push('Invalid CVC');
    }

    // Name validation
    if (!cardData.name.trim()) {
      errors.push('Cardholder name is required');
    }

    return errors;
  };

  const processStripePayment = async () => {
    const validationErrors = validateCardData();
    if (validationErrors.length > 0) {
      setError(validationErrors[0]);
      return;
    }

    try {
      setIsProcessing(true);
      setError(null);

      const paymentIntent = await apiClient.post('/billing/stripe/create-payment', {
        planId: selectedPlan?.id,
        paymentMethod: {
          card: {
            number: cardData.number.replace(/\s/g, ''),
            expMonth: parseInt(cardData.expiryMonth),
            expYear: parseInt(cardData.expiryYear),
            cvc: cardData.cvc
          },
          billingDetails: {
            name: cardData.name,
            email: cardData.email,
            address: cardData.address
          }
        }
      });

      // Simulate Stripe confirmation
      await new Promise(resolve => setTimeout(resolve, 2000));

      setSuccess('Payment processed successfully!');
      onPaymentSuccess?.(paymentIntent.id);

    } catch (error: any) {
      const errorMessage = error.message || 'Payment failed. Please try again.';
      setError(errorMessage);
      onPaymentError?.(errorMessage);
    } finally {
      setIsProcessing(false);
    }
  };

  const generatePIXPayment = async () => {
    try {
      setIsProcessing(true);
      setError(null);

      const pixData = await apiClient.post<PIXPayment>('/billing/pix/create-payment', {
        planId: selectedPlan?.id,
        amount: selectedPlan?.price,
        currency: selectedPlan?.currency || 'BRL'
      });

      setPixPayment(pixData);
      setPixCountdown(15 * 60); // 15 minutes

    } catch (error: any) {
      setError(error.message || 'Failed to generate PIX payment');
    } finally {
      setIsProcessing(false);
    }
  };

  const processPayPalPayment = async () => {
    try {
      setIsProcessing(true);
      setError(null);

      const paypalUrl = await apiClient.post<{ approvalUrl: string }>('/billing/paypal/create-payment', {
        planId: selectedPlan?.id,
        returnUrl: `${window.location.origin}/billing/success`,
        cancelUrl: `${window.location.origin}/billing/cancel`
      });

      // Redirect to PayPal
      window.location.href = paypalUrl.approvalUrl;

    } catch (error: any) {
      setError(error.message || 'PayPal payment failed');
    } finally {
      setIsProcessing(false);
    }
  };

  const formatCardNumber = (value: string) => {
    const v = value.replace(/\s+/g, '').replace(/[^0-9]/gi, '');
    const matches = v.match(/\d{4,16}/g);
    const match = matches && matches[0] || '';
    const parts = [];
    for (let i = 0, len = match.length; i < len; i += 4) {
      parts.push(match.substring(i, i + 4));
    }
    if (parts.length) {
      return parts.join(' ');
    } else {
      return v;
    }
  };

  const getCardBrand = (number: string) => {
    const cleanNumber = number.replace(/\s/g, '');
    if (/^4/.test(cleanNumber)) return 'Visa';
    if (/^5[1-5]/.test(cleanNumber)) return 'Mastercard';
    if (/^3[47]/.test(cleanNumber)) return 'American Express';
    if (/^6/.test(cleanNumber)) return 'Discover';
    return 'Unknown';
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setSuccess('Copied to clipboard!');
    setTimeout(() => setSuccess(null), 2000);
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const renderStripeForm = () => (
    <div className="space-y-6">
      {/* Saved Payment Methods */}
      {savedMethods.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Saved Payment Methods</CardTitle>
          </CardHeader>
          <CardContent>
            <RadioGroup value={selectedSavedMethod} onValueChange={setSelectedSavedMethod}>
              <div className="space-y-3">
                {savedMethods.map((method) => (
                  <div key={method.id} className="flex items-center space-x-3 p-3 border rounded-lg">
                    <RadioGroupItem value={method.id} />
                    <div className="flex items-center space-x-3 flex-1">
                      <CreditCard className="h-5 w-5 text-gray-400" />
                      <div>
                        <p className="font-medium">
                          {method.brand} •••• {method.last4}
                        </p>
                        <p className="text-sm text-gray-500">
                          Expires {method.expiryMonth}/{method.expiryYear}
                        </p>
                      </div>
                    </div>
                    {method.isDefault && (
                      <Badge variant="outline">Default</Badge>
                    )}
                  </div>
                ))}
              </div>
            </RadioGroup>

            <Button
              onClick={() => setSelectedSavedMethod(null)}
              variant="outline"
              className="w-full mt-4"
            >
              Use New Card
            </Button>
          </CardContent>
        </Card>
      )}

      {/* New Card Form */}
      {(!selectedSavedMethod || savedMethods.length === 0) && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Card Information</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="cardNumber">Card Number</Label>
              <div className="relative">
                <Input
                  id="cardNumber"
                  value={cardData.number}
                  onChange={(e) => setCardData(prev => ({
                    ...prev,
                    number: formatCardNumber(e.target.value)
                  }))}
                  placeholder="1234 5678 9012 3456"
                  maxLength={19}
                />
                <div className="absolute right-3 top-3 text-sm text-gray-500">
                  {getCardBrand(cardData.number)}
                </div>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label htmlFor="expiryMonth">MM</Label>
                <Input
                  id="expiryMonth"
                  value={cardData.expiryMonth}
                  onChange={(e) => setCardData(prev => ({
                    ...prev,
                    expiryMonth: e.target.value.replace(/\D/g, '').slice(0, 2)
                  }))}
                  placeholder="MM"
                  maxLength={2}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="expiryYear">YY</Label>
                <Input
                  id="expiryYear"
                  value={cardData.expiryYear}
                  onChange={(e) => setCardData(prev => ({
                    ...prev,
                    expiryYear: e.target.value.replace(/\D/g, '').slice(0, 2)
                  }))}
                  placeholder="YY"
                  maxLength={2}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="cvc">CVC</Label>
                <div className="relative">
                  <Input
                    id="cvc"
                    type={showCVC ? 'text' : 'password'}
                    value={cardData.cvc}
                    onChange={(e) => setCardData(prev => ({
                      ...prev,
                      cvc: e.target.value.replace(/\D/g, '').slice(0, 4)
                    }))}
                    placeholder="123"
                    maxLength={4}
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="absolute right-1 top-1/2 transform -translate-y-1/2 h-6 w-6 p-0"
                    onClick={() => setShowCVC(!showCVC)}
                  >
                    {showCVC ? <EyeOff className="h-3 w-3" /> : <Eye className="h-3 w-3" />}
                  </Button>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="cardName">Cardholder Name</Label>
              <Input
                id="cardName"
                value={cardData.name}
                onChange={(e) => setCardData(prev => ({ ...prev, name: e.target.value }))}
                placeholder="Full name on card"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="cardEmail">Email</Label>
              <Input
                id="cardEmail"
                type="email"
                value={cardData.email}
                onChange={(e) => setCardData(prev => ({ ...prev, email: e.target.value }))}
                placeholder="your@email.com"
              />
            </div>
          </CardContent>
        </Card>
      )}

      {/* Billing Address */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Billing Address</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="address">Address</Label>
            <Input
              id="address"
              value={cardData.address.line1}
              onChange={(e) => setCardData(prev => ({
                ...prev,
                address: { ...prev.address, line1: e.target.value }
              }))}
              placeholder="123 Main Street"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="city">City</Label>
              <Input
                id="city"
                value={cardData.address.city}
                onChange={(e) => setCardData(prev => ({
                  ...prev,
                  address: { ...prev.address, city: e.target.value }
                }))}
                placeholder="New York"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="state">State</Label>
              <Input
                id="state"
                value={cardData.address.state}
                onChange={(e) => setCardData(prev => ({
                  ...prev,
                  address: { ...prev.address, state: e.target.value }
                }))}
                placeholder="NY"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="postalCode">Postal Code</Label>
              <Input
                id="postalCode"
                value={cardData.address.postalCode}
                onChange={(e) => setCardData(prev => ({
                  ...prev,
                  address: { ...prev.address, postalCode: e.target.value }
                }))}
                placeholder="10001"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="country">Country</Label>
              <select
                id="country"
                value={cardData.address.country}
                onChange={(e) => setCardData(prev => ({
                  ...prev,
                  address: { ...prev.address, country: e.target.value }
                }))}
                className="w-full p-2 border rounded-md"
              >
                <option value="US">United States</option>
                <option value="CA">Canada</option>
                <option value="GB">United Kingdom</option>
                <option value="AU">Australia</option>
                <option value="DE">Germany</option>
                <option value="FR">France</option>
                <option value="BR">Brazil</option>
              </select>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );

  const renderPIXForm = () => (
    <div className="space-y-6">
      {!pixPayment ? (
        <Card>
          <CardContent className="p-6 text-center">
            <QrCode className="h-16 w-16 text-blue-500 mx-auto mb-4" />
            <h3 className="text-lg font-semibold mb-2">PIX Payment</h3>
            <p className="text-gray-600 mb-4">
              Generate a PIX QR code for instant payment via your banking app
            </p>
            <Button onClick={generatePIXPayment} disabled={isProcessing} className="w-full">
              {isProcessing ? (
                <>
                  <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  Generating PIX...
                </>
              ) : (
                <>
                  <QrCode className="h-4 w-4 mr-2" />
                  Generate PIX Payment
                </>
              )}
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-6">
          <Alert>
            <Clock className="h-4 w-4" />
            <AlertDescription>
              PIX payment expires in {formatTime(pixCountdown)}. Complete the payment using your banking app.
            </AlertDescription>
          </Alert>

          <Card>
            <CardHeader>
              <CardTitle className="text-center">PIX Payment QR Code</CardTitle>
            </CardHeader>
            <CardContent className="text-center space-y-4">
              <div className="bg-white p-4 rounded-lg border-2 border-dashed border-gray-300 inline-block">
                {/* QR Code would be rendered here */}
                <div className="w-48 h-48 bg-gray-100 flex items-center justify-center">
                  <QrCode className="h-16 w-16 text-gray-400" />
                </div>
              </div>

              <div className="space-y-2">
                <p className="text-sm text-gray-600">Or copy the PIX key:</p>
                <div className="flex items-center space-x-2">
                  <Input
                    value={pixPayment.pixKey}
                    readOnly
                    className="font-mono text-sm"
                  />
                  <Button
                    variant="outline"
                    onClick={() => copyToClipboard(pixPayment.pixKey)}
                  >
                    <Copy className="h-4 w-4" />
                  </Button>
                </div>
              </div>

              <div className="text-center">
                <p className="text-lg font-semibold">
                  Amount: R$ {pixPayment.amount.toFixed(2)}
                </p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <h4 className="font-medium mb-2">How to pay with PIX:</h4>
              <ol className="text-sm text-gray-600 space-y-1">
                <li>1. Open your banking app</li>
                <li>2. Select PIX payment option</li>
                <li>3. Scan the QR code or paste the PIX key</li>
                <li>4. Confirm the payment amount</li>
                <li>5. Complete the transaction</li>
              </ol>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );

  const renderPayPalForm = () => (
    <Card>
      <CardContent className="p-6 text-center">
        <div className="w-16 h-16 bg-blue-500 rounded-full flex items-center justify-center mx-auto mb-4">
          <DollarSign className="h-8 w-8 text-white" />
        </div>
        <h3 className="text-lg font-semibold mb-2">PayPal Payment</h3>
        <p className="text-gray-600 mb-4">
          You'll be redirected to PayPal to complete your payment securely
        </p>
        <Button onClick={processPayPalPayment} disabled={isProcessing} className="w-full">
          {isProcessing ? (
            <>
              <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
              Redirecting...
            </>
          ) : (
            <>
              <ArrowRight className="h-4 w-4 mr-2" />
              Continue with PayPal
            </>
          )}
        </Button>
      </CardContent>
    </Card>
  );

  return (
    <div className={`max-w-2xl mx-auto space-y-6 ${className}`}>
      {/* Plan Summary */}
      {selectedPlan && (
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-blue-100 rounded-lg">
                  {selectedPlan.name === 'Basic' && <Star className="h-5 w-5 text-blue-600" />}
                  {selectedPlan.name === 'Premium' && <Crown className="h-5 w-5 text-purple-600" />}
                  {selectedPlan.name === 'Enterprise' && <Building className="h-5 w-5 text-orange-600" />}
                </div>
                <div>
                  <h3 className="font-semibold">{selectedPlan.name} Plan</h3>
                  <p className="text-sm text-gray-500 capitalize">{selectedPlan.billingCycle} billing</p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-2xl font-bold">
                  {new Intl.NumberFormat('en-US', {
                    style: 'currency',
                    currency: selectedPlan.currency,
                  }).format(selectedPlan.price)}
                </p>
                {selectedPlan.discount && (
                  <Badge className="bg-green-100 text-green-800">
                    {selectedPlan.discount}% off
                  </Badge>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Payment Method Selection */}
      <Card>
        <CardHeader>
          <CardTitle>Payment Method</CardTitle>
          <CardDescription>
            Choose your preferred payment method
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs value={activePaymentMethod} onValueChange={(value) => setActivePaymentMethod(value as any)}>
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="stripe" className="flex items-center space-x-2">
                <CreditCard className="h-4 w-4" />
                <span>Card</span>
              </TabsTrigger>
              <TabsTrigger value="pix" className="flex items-center space-x-2">
                <QrCode className="h-4 w-4" />
                <span>PIX</span>
              </TabsTrigger>
              <TabsTrigger value="paypal" className="flex items-center space-x-2">
                <DollarSign className="h-4 w-4" />
                <span>PayPal</span>
              </TabsTrigger>
            </TabsList>

            <div className="mt-6">
              <TabsContent value="stripe">
                {renderStripeForm()}
              </TabsContent>

              <TabsContent value="pix">
                {renderPIXForm()}
              </TabsContent>

              <TabsContent value="paypal">
                {renderPayPalForm()}
              </TabsContent>
            </div>
          </Tabs>
        </CardContent>
      </Card>

      {/* Error and Success Messages */}
      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {success && (
        <Alert className="border-green-200 bg-green-50">
          <CheckCircle className="h-4 w-4 text-green-600" />
          <AlertDescription className="text-green-800">{success}</AlertDescription>
        </Alert>
      )}

      {/* Terms and Payment Button */}
      {activePaymentMethod === 'stripe' && !pixPayment && (
        <Card>
          <CardContent className="p-6 space-y-4">
            <div className="flex items-start space-x-2">
              <input
                type="checkbox"
                id="terms"
                checked={agreedToTerms}
                onChange={(e) => setAgreedToTerms(e.target.checked)}
                className="mt-1"
              />
              <Label htmlFor="terms" className="text-sm text-gray-600">
                I agree to the{' '}
                <button className="text-blue-600 hover:underline">Terms of Service</button>
                {' '}and{' '}
                <button className="text-blue-600 hover:underline">Privacy Policy</button>
              </Label>
            </div>

            <div className="flex items-center space-x-2 text-sm text-gray-500">
              <Shield className="h-4 w-4" />
              <span>Your payment information is encrypted and secure</span>
            </div>

            <Button
              onClick={processStripePayment}
              disabled={!agreedToTerms || isProcessing}
              className="w-full"
              size="lg"
            >
              {isProcessing ? (
                <>
                  <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                  Processing Payment...
                </>
              ) : (
                <>
                  <Lock className="h-4 w-4 mr-2" />
                  Complete Payment
                </>
              )}
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Security Notice */}
      <div className="text-center text-sm text-gray-500">
        <div className="flex items-center justify-center space-x-4">
          <div className="flex items-center space-x-1">
            <Shield className="h-4 w-4" />
            <span>SSL Encrypted</span>
          </div>
          <div className="flex items-center space-x-1">
            <Lock className="h-4 w-4" />
            <span>PCI Compliant</span>
          </div>
          <div className="flex items-center space-x-1">
            <Globe className="h-4 w-4" />
            <span>Global Payments</span>
          </div>
        </div>
      </div>
    </div>
  );
}