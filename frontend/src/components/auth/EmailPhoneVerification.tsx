/**
 * ✅ EMAIL & PHONE VERIFICATION COMPONENT
 * Complete verification system with OTP codes and resend functionality
 */

import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { apiClient } from '../../services/apiClient';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Alert, AlertDescription } from '../ui/alert';
import { Badge } from '../ui/badge';
import { Label } from '../ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import {
  Mail,
  Phone,
  CheckCircle,
  AlertTriangle,
  RefreshCw,
  Clock,
  Send,
  Shield,
  MessageSquare,
  ArrowRight,
  ArrowLeft,
  Copy,
  Eye,
  EyeOff,
  Globe,
  AlertCircle
} from 'lucide-react';

interface VerificationProps {
  className?: string;
  onVerificationComplete?: () => void;
}

type VerificationType = 'email' | 'phone';
type VerificationStep = 'overview' | 'send' | 'verify' | 'success';

interface VerificationStatus {
  email: {
    isVerified: boolean;
    lastSentAt?: string;
    attemptsRemaining: number;
  };
  phone: {
    isVerified: boolean;
    lastSentAt?: string;
    attemptsRemaining: number;
  };
}

export function EmailPhoneVerification({ className = "", onVerificationComplete }: VerificationProps) {
  const { user, verifyEmail, resendVerificationEmail, updateProfile } = useAuth();

  const [activeTab, setActiveTab] = useState<VerificationType>('email');
  const [currentStep, setCurrentStep] = useState<VerificationStep>('overview');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const [verificationCode, setVerificationCode] = useState('');
  const [phoneNumber, setPhoneNumber] = useState(user?.phone || '');
  const [countryCode, setCountryCode] = useState('+1');
  const [countdown, setCountdown] = useState(0);

  const [verificationStatus, setVerificationStatus] = useState<VerificationStatus>({
    email: {
      isVerified: user?.isVerified || false,
      attemptsRemaining: 3
    },
    phone: {
      isVerified: user?.phone ? true : false, // Simplified for demo
      attemptsRemaining: 3
    }
  });

  // Countdown timer for resend button
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (countdown > 0) {
      timer = setTimeout(() => setCountdown(countdown - 1), 1000);
    }
    return () => clearTimeout(timer);
  }, [countdown]);

  // Auto-format verification code input
  useEffect(() => {
    setVerificationCode(verificationCode.replace(/\D/g, '').slice(0, 6));
  }, [verificationCode]);

  const formatPhoneNumber = (phone: string) => {
    const cleaned = phone.replace(/\D/g, '');
    const match = cleaned.match(/^(\d{3})(\d{3})(\d{4})$/);
    if (match) {
      return `(${match[1]}) ${match[2]}-${match[3]}`;
    }
    return phone;
  };

  const sendEmailVerification = async () => {
    try {
      setIsLoading(true);
      setError(null);

      await resendVerificationEmail();
      setCurrentStep('verify');
      setCountdown(60);
      setSuccess('Verification email sent successfully!');

      // Update status
      setVerificationStatus(prev => ({
        ...prev,
        email: {
          ...prev.email,
          lastSentAt: new Date().toISOString(),
          attemptsRemaining: prev.email.attemptsRemaining - 1
        }
      }));
    } catch (error: any) {
      setError(error.message || 'Failed to send verification email');
    } finally {
      setIsLoading(false);
    }
  };

  const sendPhoneVerification = async () => {
    const fullPhoneNumber = `${countryCode}${phoneNumber.replace(/\D/g, '')}`;

    if (!phoneNumber || phoneNumber.replace(/\D/g, '').length < 10) {
      setError('Please enter a valid phone number');
      return;
    }

    try {
      setIsLoading(true);
      setError(null);

      await apiClient.post('/auth/verify-phone/send', {
        phoneNumber: fullPhoneNumber
      });

      setCurrentStep('verify');
      setCountdown(60);
      setSuccess('Verification code sent to your phone!');

      // Update status
      setVerificationStatus(prev => ({
        ...prev,
        phone: {
          ...prev.phone,
          lastSentAt: new Date().toISOString(),
          attemptsRemaining: prev.phone.attemptsRemaining - 1
        }
      }));
    } catch (error: any) {
      setError(error.message || 'Failed to send verification code');
    } finally {
      setIsLoading(false);
    }
  };

  const verifyEmailCode = async () => {
    if (!verificationCode || verificationCode.length !== 6) {
      setError('Please enter a valid 6-digit code');
      return;
    }

    try {
      setIsLoading(true);
      setError(null);

      await verifyEmail(verificationCode);
      setCurrentStep('success');
      setSuccess('Email verified successfully!');

      // Update status
      setVerificationStatus(prev => ({
        ...prev,
        email: { ...prev.email, isVerified: true }
      }));

      onVerificationComplete?.();
    } catch (error: any) {
      setError(error.message || 'Invalid verification code');
    } finally {
      setIsLoading(false);
    }
  };

  const verifyPhoneCode = async () => {
    if (!verificationCode || verificationCode.length !== 6) {
      setError('Please enter a valid 6-digit code');
      return;
    }

    const fullPhoneNumber = `${countryCode}${phoneNumber.replace(/\D/g, '')}`;

    try {
      setIsLoading(true);
      setError(null);

      await apiClient.post('/auth/verify-phone/confirm', {
        phoneNumber: fullPhoneNumber,
        code: verificationCode
      });

      // Update user profile with verified phone
      await updateProfile({ phone: fullPhoneNumber });

      setCurrentStep('success');
      setSuccess('Phone number verified successfully!');

      // Update status
      setVerificationStatus(prev => ({
        ...prev,
        phone: { ...prev.phone, isVerified: true }
      }));

      onVerificationComplete?.();
    } catch (error: any) {
      setError(error.message || 'Invalid verification code');
    } finally {
      setIsLoading(false);
    }
  };

  const handleResend = async () => {
    if (countdown > 0) return;

    if (activeTab === 'email') {
      await sendEmailVerification();
    } else {
      await sendPhoneVerification();
    }
  };

  const formatLastSent = (dateString?: string) => {
    if (!dateString) return '';
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins} minutes ago`;
    return date.toLocaleTimeString();
  };

  const renderOverview = () => (
    <div className="space-y-6">
      <div className="text-center">
        <div className="mx-auto w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mb-4">
          <Shield className="h-8 w-8 text-blue-600" />
        </div>
        <h2 className="text-2xl font-bold mb-2">Verify Your Account</h2>
        <p className="text-gray-600">
          Verify your email and phone number to secure your account and enable all features
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Email Verification Card */}
        <Card className={`cursor-pointer transition-all ${verificationStatus.email.isVerified ? 'border-green-200 bg-green-50' : 'hover:border-blue-300'}`}>
          <CardContent className="p-6">
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center space-x-3">
                <div className={`p-2 rounded-lg ${verificationStatus.email.isVerified ? 'bg-green-100' : 'bg-blue-100'}`}>
                  <Mail className={`h-5 w-5 ${verificationStatus.email.isVerified ? 'text-green-600' : 'text-blue-600'}`} />
                </div>
                <div>
                  <h3 className="font-medium">Email Verification</h3>
                  <p className="text-sm text-gray-500">{user?.email}</p>
                </div>
              </div>

              {verificationStatus.email.isVerified ? (
                <Badge className="bg-green-100 text-green-800">
                  <CheckCircle className="h-3 w-3 mr-1" />
                  Verified
                </Badge>
              ) : (
                <Badge className="bg-orange-100 text-orange-800">
                  Pending
                </Badge>
              )}
            </div>

            {verificationStatus.email.isVerified ? (
              <p className="text-sm text-green-600">Your email is verified and secure!</p>
            ) : (
              <div className="space-y-3">
                <p className="text-sm text-gray-600">
                  Verify your email to secure your account and receive important notifications.
                </p>
                <Button
                  size="sm"
                  onClick={() => {
                    setActiveTab('email');
                    setCurrentStep('send');
                  }}
                  className="w-full"
                >
                  Verify Email
                  <ArrowRight className="h-4 w-4 ml-2" />
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Phone Verification Card */}
        <Card className={`cursor-pointer transition-all ${verificationStatus.phone.isVerified ? 'border-green-200 bg-green-50' : 'hover:border-blue-300'}`}>
          <CardContent className="p-6">
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center space-x-3">
                <div className={`p-2 rounded-lg ${verificationStatus.phone.isVerified ? 'bg-green-100' : 'bg-blue-100'}`}>
                  <Phone className={`h-5 w-5 ${verificationStatus.phone.isVerified ? 'text-green-600' : 'text-blue-600'}`} />
                </div>
                <div>
                  <h3 className="font-medium">Phone Verification</h3>
                  <p className="text-sm text-gray-500">
                    {user?.phone ? formatPhoneNumber(user.phone) : 'No phone number'}
                  </p>
                </div>
              </div>

              {verificationStatus.phone.isVerified ? (
                <Badge className="bg-green-100 text-green-800">
                  <CheckCircle className="h-3 w-3 mr-1" />
                  Verified
                </Badge>
              ) : (
                <Badge className="bg-orange-100 text-orange-800">
                  Pending
                </Badge>
              )}
            </div>

            {verificationStatus.phone.isVerified ? (
              <p className="text-sm text-green-600">Your phone number is verified!</p>
            ) : (
              <div className="space-y-3">
                <p className="text-sm text-gray-600">
                  Add and verify your phone number for enhanced security and SMS alerts.
                </p>
                <Button
                  size="sm"
                  onClick={() => {
                    setActiveTab('phone');
                    setCurrentStep('send');
                  }}
                  className="w-full"
                  variant="outline"
                >
                  Verify Phone
                  <ArrowRight className="h-4 w-4 ml-2" />
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Benefits */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Why Verify Your Account?</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="flex items-start space-x-3">
              <Shield className="h-5 w-5 text-blue-500 mt-0.5" />
              <div>
                <p className="font-medium text-sm">Enhanced Security</p>
                <p className="text-xs text-gray-500">Protect your account from unauthorized access</p>
              </div>
            </div>

            <div className="flex items-start space-x-3">
              <Bell className="h-5 w-5 text-green-500 mt-0.5" />
              <div>
                <p className="font-medium text-sm">Important Notifications</p>
                <p className="text-xs text-gray-500">Receive critical trading alerts and updates</p>
              </div>
            </div>

            <div className="flex items-start space-x-3">
              <CheckCircle className="h-5 w-5 text-purple-500 mt-0.5" />
              <div>
                <p className="font-medium text-sm">Account Recovery</p>
                <p className="text-xs text-gray-500">Easily recover your account if needed</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );

  const renderSend = () => (
    <Card className="max-w-md mx-auto">
      <CardHeader className="text-center">
        <div className="mx-auto w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mb-4">
          {activeTab === 'email' ? (
            <Mail className="h-8 w-8 text-blue-600" />
          ) : (
            <Phone className="h-8 w-8 text-blue-600" />
          )}
        </div>
        <CardTitle>
          {activeTab === 'email' ? 'Verify Email Address' : 'Verify Phone Number'}
        </CardTitle>
        <CardDescription>
          {activeTab === 'email'
            ? 'We\'ll send a verification code to your email'
            : 'We\'ll send a verification code via SMS'
          }
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        {activeTab === 'email' ? (
          <div className="space-y-2">
            <Label>Email Address</Label>
            <div className="flex items-center space-x-2 p-3 bg-gray-50 rounded-lg">
              <Mail className="h-4 w-4 text-gray-400" />
              <span className="font-medium">{user?.email}</span>
              {verificationStatus.email.isVerified && (
                <CheckCircle className="h-4 w-4 text-green-500" />
              )}
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Phone Number</Label>
              <div className="flex space-x-2">
                <select
                  value={countryCode}
                  onChange={(e) => setCountryCode(e.target.value)}
                  className="px-3 py-2 border rounded-md bg-white"
                >
                  <option value="+1">🇺🇸 +1</option>
                  <option value="+44">🇬🇧 +44</option>
                  <option value="+55">🇧🇷 +55</option>
                  <option value="+81">🇯🇵 +81</option>
                  <option value="+86">🇨🇳 +86</option>
                  <option value="+49">🇩🇪 +49</option>
                  <option value="+33">🇫🇷 +33</option>
                </select>
                <Input
                  value={phoneNumber}
                  onChange={(e) => setPhoneNumber(e.target.value)}
                  placeholder="(555) 123-4567"
                  className="flex-1"
                />
              </div>
            </div>

            <Alert>
              <MessageSquare className="h-4 w-4" />
              <AlertDescription>
                Standard SMS rates may apply. You'll receive a 6-digit verification code.
              </AlertDescription>
            </Alert>
          </div>
        )}

        <div className="space-y-3">
          <Button
            onClick={activeTab === 'email' ? sendEmailVerification : sendPhoneVerification}
            disabled={isLoading || (activeTab === 'phone' && !phoneNumber)}
            className="w-full"
          >
            {isLoading ? (
              <>
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                Sending...
              </>
            ) : (
              <>
                <Send className="h-4 w-4 mr-2" />
                Send Verification Code
              </>
            )}
          </Button>

          <Button
            variant="outline"
            onClick={() => setCurrentStep('overview')}
            className="w-full"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back
          </Button>
        </div>
      </CardContent>
    </Card>
  );

  const renderVerify = () => (
    <Card className="max-w-md mx-auto">
      <CardHeader className="text-center">
        <div className="mx-auto w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mb-4">
          <MessageSquare className="h-8 w-8 text-green-600" />
        </div>
        <CardTitle>Enter Verification Code</CardTitle>
        <CardDescription>
          {activeTab === 'email'
            ? `We sent a code to ${user?.email}`
            : `We sent a code to ${countryCode} ${formatPhoneNumber(phoneNumber)}`
          }
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        <div className="space-y-2">
          <Label htmlFor="verificationCode">Verification Code</Label>
          <Input
            id="verificationCode"
            value={verificationCode}
            onChange={(e) => setVerificationCode(e.target.value)}
            placeholder="000000"
            className="text-center text-lg font-mono tracking-widest"
            maxLength={6}
          />
          <p className="text-xs text-gray-500 text-center">
            Enter the 6-digit code we sent to your {activeTab}
          </p>
        </div>

        <div className="space-y-3">
          <Button
            onClick={activeTab === 'email' ? verifyEmailCode : verifyPhoneCode}
            disabled={isLoading || verificationCode.length !== 6}
            className="w-full"
          >
            {isLoading ? (
              <>
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                Verifying...
              </>
            ) : (
              <>
                <CheckCircle className="h-4 w-4 mr-2" />
                Verify Code
              </>
            )}
          </Button>

          <div className="text-center">
            <p className="text-sm text-gray-600 mb-2">Didn't receive the code?</p>
            <Button
              variant="outline"
              onClick={handleResend}
              disabled={countdown > 0 || isLoading}
              size="sm"
            >
              {countdown > 0 ? (
                <>
                  <Clock className="h-4 w-4 mr-2" />
                  Resend in {countdown}s
                </>
              ) : (
                <>
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Resend Code
                </>
              )}
            </Button>
          </div>

          <Button
            variant="outline"
            onClick={() => setCurrentStep('send')}
            className="w-full"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Change {activeTab === 'email' ? 'Email' : 'Phone'}
          </Button>
        </div>
      </CardContent>
    </Card>
  );

  const renderSuccess = () => (
    <Card className="max-w-md mx-auto">
      <CardHeader className="text-center">
        <div className="mx-auto w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mb-4">
          <CheckCircle className="h-8 w-8 text-green-600" />
        </div>
        <CardTitle>
          {activeTab === 'email' ? 'Email Verified!' : 'Phone Verified!'}
        </CardTitle>
        <CardDescription>
          Your {activeTab} has been successfully verified
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        <div className="space-y-3">
          <div className="flex items-center space-x-3 p-3 bg-green-50 rounded-lg">
            <CheckCircle className="h-5 w-5 text-green-600" />
            <span className="text-sm">
              {activeTab === 'email' ? 'Email address verified' : 'Phone number verified'}
            </span>
          </div>
          <div className="flex items-center space-x-3 p-3 bg-green-50 rounded-lg">
            <Shield className="h-5 w-5 text-green-600" />
            <span className="text-sm">Account security enhanced</span>
          </div>
          <div className="flex items-center space-x-3 p-3 bg-green-50 rounded-lg">
            <Bell className="h-5 w-5 text-green-600" />
            <span className="text-sm">Notifications enabled</span>
          </div>
        </div>

        <div className="space-y-3">
          <Button
            onClick={() => setCurrentStep('overview')}
            className="w-full"
          >
            Continue
            <ArrowRight className="h-4 w-4 ml-2" />
          </Button>

          {!verificationStatus[activeTab === 'email' ? 'phone' : 'email'].isVerified && (
            <Button
              variant="outline"
              onClick={() => {
                setActiveTab(activeTab === 'email' ? 'phone' : 'email');
                setCurrentStep('send');
                setVerificationCode('');
              }}
              className="w-full"
            >
              Verify {activeTab === 'email' ? 'Phone' : 'Email'} Too
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );

  if (!user) {
    return (
      <Alert>
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          Please log in to verify your account.
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Error and Success Alerts */}
      {error && (
        <Alert variant="destructive" className="max-w-2xl mx-auto">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {success && (
        <Alert className="max-w-2xl mx-auto border-green-200 bg-green-50">
          <CheckCircle className="h-4 w-4 text-green-600" />
          <AlertDescription className="text-green-800">{success}</AlertDescription>
        </Alert>
      )}

      {/* Render current step */}
      {currentStep === 'overview' && renderOverview()}
      {currentStep === 'send' && renderSend()}
      {currentStep === 'verify' && renderVerify()}
      {currentStep === 'success' && renderSuccess()}
    </div>
  );
}