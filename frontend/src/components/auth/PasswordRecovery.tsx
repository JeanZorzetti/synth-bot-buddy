/**
 * 🔐 PASSWORD RECOVERY SYSTEM
 * Complete password reset flow with email verification and security checks
 */

import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '../../contexts/AuthContext';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Alert, AlertDescription } from '../ui/alert';
import { Label } from '../ui/label';
import {
  Mail,
  Lock,
  Key,
  ArrowLeft,
  ArrowRight,
  CheckCircle,
  AlertTriangle,
  Eye,
  EyeOff,
  RefreshCw,
  Clock,
  Shield,
  AlertCircle
} from 'lucide-react';

interface PasswordRecoveryProps {
  className?: string;
  mode?: 'request' | 'reset';
  token?: string;
}

type RecoveryStep = 'email' | 'sent' | 'reset' | 'success';

interface PasswordStrength {
  score: number;
  feedback: string[];
  hasMinLength: boolean;
  hasUppercase: boolean;
  hasLowercase: boolean;
  hasNumbers: boolean;
  hasSpecialChars: boolean;
}

export function PasswordRecovery({ className = "", mode = 'request', token }: PasswordRecoveryProps) {
  const router = useRouter();
  const { requestPasswordReset, resetPassword } = useAuth();

  const [currentStep, setCurrentStep] = useState<RecoveryStep>('email');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const [email, setEmail] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const [countdown, setCountdown] = useState(0);
  const [passwordStrength, setPasswordStrength] = useState<PasswordStrength>({
    score: 0,
    feedback: [],
    hasMinLength: false,
    hasUppercase: false,
    hasLowercase: false,
    hasNumbers: false,
    hasSpecialChars: false
  });

  // Initialize based on mode and token
  useEffect(() => {
    if (mode === 'reset' && token) {
      setCurrentStep('reset');
    }
  }, [mode, token]);

  // Countdown timer for resend button
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (countdown > 0) {
      timer = setTimeout(() => setCountdown(countdown - 1), 1000);
    }
    return () => clearTimeout(timer);
  }, [countdown]);

  // Password strength calculation
  useEffect(() => {
    calculatePasswordStrength(newPassword);
  }, [newPassword]);

  const calculatePasswordStrength = (password: string) => {
    const hasMinLength = password.length >= 8;
    const hasUppercase = /[A-Z]/.test(password);
    const hasLowercase = /[a-z]/.test(password);
    const hasNumbers = /\d/.test(password);
    const hasSpecialChars = /[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password);

    let score = 0;
    const feedback = [];

    if (hasMinLength) score += 1;
    else feedback.push('Use at least 8 characters');

    if (hasUppercase) score += 1;
    else feedback.push('Add uppercase letters');

    if (hasLowercase) score += 1;
    else feedback.push('Add lowercase letters');

    if (hasNumbers) score += 1;
    else feedback.push('Add numbers');

    if (hasSpecialChars) score += 1;
    else feedback.push('Add special characters (!@#$%...)');

    // Additional checks
    if (password.length >= 12) score += 1;
    if (/(.)\1{2,}/.test(password)) {
      score -= 1;
      feedback.push('Avoid repeated characters');
    }

    // Common password patterns
    const commonPatterns = ['password', '123456', 'qwerty', 'admin', 'letmein'];
    if (commonPatterns.some(pattern => password.toLowerCase().includes(pattern))) {
      score -= 2;
      feedback.push('Avoid common password patterns');
    }

    setPasswordStrength({
      score: Math.max(0, Math.min(5, score)),
      feedback,
      hasMinLength,
      hasUppercase,
      hasLowercase,
      hasNumbers,
      hasSpecialChars
    });
  };

  const getPasswordStrengthColor = (score: number) => {
    if (score <= 1) return 'bg-red-500';
    if (score <= 2) return 'bg-orange-500';
    if (score <= 3) return 'bg-yellow-500';
    if (score <= 4) return 'bg-blue-500';
    return 'bg-green-500';
  };

  const getPasswordStrengthText = (score: number) => {
    if (score <= 1) return 'Very Weak';
    if (score <= 2) return 'Weak';
    if (score <= 3) return 'Fair';
    if (score <= 4) return 'Good';
    return 'Strong';
  };

  const handleRequestReset = async () => {
    if (!email) {
      setError('Please enter your email address');
      return;
    }

    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      setError('Please enter a valid email address');
      return;
    }

    try {
      setIsLoading(true);
      setError(null);

      await requestPasswordReset(email);
      setCurrentStep('sent');
      setCountdown(60); // 60 second cooldown for resend
      setSuccess('Password reset email sent successfully!');
    } catch (error: any) {
      setError(error.message || 'Failed to send password reset email');
    } finally {
      setIsLoading(false);
    }
  };

  const handlePasswordReset = async () => {
    // Validation
    if (!newPassword || !confirmPassword) {
      setError('Please fill in all fields');
      return;
    }

    if (newPassword !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    if (passwordStrength.score < 3) {
      setError('Password is too weak. Please choose a stronger password.');
      return;
    }

    if (!token) {
      setError('Invalid reset token');
      return;
    }

    try {
      setIsLoading(true);
      setError(null);

      await resetPassword(token, newPassword);
      setCurrentStep('success');
      setSuccess('Password reset successfully!');
    } catch (error: any) {
      setError(error.message || 'Failed to reset password');
    } finally {
      setIsLoading(false);
    }
  };

  const handleResendEmail = async () => {
    if (countdown > 0) return;

    try {
      setIsLoading(true);
      setError(null);

      await requestPasswordReset(email);
      setCountdown(60);
      setSuccess('Password reset email sent again!');
    } catch (error: any) {
      setError(error.message || 'Failed to resend email');
    } finally {
      setIsLoading(false);
    }
  };

  const renderEmailStep = () => (
    <Card className="max-w-md mx-auto">
      <CardHeader className="text-center">
        <div className="mx-auto w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mb-4">
          <Key className="h-8 w-8 text-blue-600" />
        </div>
        <CardTitle>Reset Your Password</CardTitle>
        <CardDescription>
          Enter your email address and we'll send you a link to reset your password
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        <div className="space-y-2">
          <Label htmlFor="email">Email Address</Label>
          <div className="relative">
            <Mail className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
            <Input
              id="email"
              type="email"
              placeholder="Enter your email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="pl-10"
              disabled={isLoading}
            />
          </div>
        </div>

        <Button
          onClick={handleRequestReset}
          disabled={isLoading || !email}
          className="w-full"
        >
          {isLoading ? (
            <>
              <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
              Sending...
            </>
          ) : (
            <>
              <Mail className="h-4 w-4 mr-2" />
              Send Reset Link
            </>
          )}
        </Button>

        <div className="text-center">
          <Button
            variant="ghost"
            onClick={() => router.push('/auth/login')}
            className="text-sm"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Login
          </Button>
        </div>
      </CardContent>
    </Card>
  );

  const renderSentStep = () => (
    <Card className="max-w-md mx-auto">
      <CardHeader className="text-center">
        <div className="mx-auto w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mb-4">
          <Mail className="h-8 w-8 text-green-600" />
        </div>
        <CardTitle>Check Your Email</CardTitle>
        <CardDescription>
          We've sent a password reset link to <strong>{email}</strong>
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        <Alert>
          <CheckCircle className="h-4 w-4" />
          <AlertDescription>
            Click the link in the email to reset your password. The link will expire in 1 hour.
          </AlertDescription>
        </Alert>

        <div className="space-y-4">
          <p className="text-sm text-gray-600 text-center">
            Didn't receive the email? Check your spam folder or try again.
          </p>

          <Button
            variant="outline"
            onClick={handleResendEmail}
            disabled={countdown > 0 || isLoading}
            className="w-full"
          >
            {countdown > 0 ? (
              <>
                <Clock className="h-4 w-4 mr-2" />
                Resend in {countdown}s
              </>
            ) : isLoading ? (
              <>
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                Sending...
              </>
            ) : (
              <>
                <RefreshCw className="h-4 w-4 mr-2" />
                Resend Email
              </>
            )}
          </Button>
        </div>

        <div className="text-center">
          <Button
            variant="ghost"
            onClick={() => setCurrentStep('email')}
            className="text-sm"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Use Different Email
          </Button>
        </div>
      </CardContent>
    </Card>
  );

  const renderResetStep = () => (
    <Card className="max-w-md mx-auto">
      <CardHeader className="text-center">
        <div className="mx-auto w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mb-4">
          <Lock className="h-8 w-8 text-purple-600" />
        </div>
        <CardTitle>Create New Password</CardTitle>
        <CardDescription>
          Choose a strong password for your account
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        <div className="space-y-2">
          <Label htmlFor="newPassword">New Password</Label>
          <div className="relative">
            <Lock className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
            <Input
              id="newPassword"
              type={showNewPassword ? 'text' : 'password'}
              placeholder="Enter new password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              className="pl-10 pr-10"
              disabled={isLoading}
            />
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="absolute right-2 top-1/2 transform -translate-y-1/2"
              onClick={() => setShowNewPassword(!showNewPassword)}
            >
              {showNewPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </Button>
          </div>

          {/* Password Strength Indicator */}
          {newPassword && (
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span>Password Strength:</span>
                <span className={`font-medium ${
                  passwordStrength.score <= 2 ? 'text-red-600' :
                  passwordStrength.score <= 3 ? 'text-yellow-600' :
                  'text-green-600'
                }`}>
                  {getPasswordStrengthText(passwordStrength.score)}
                </span>
              </div>

              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all duration-300 ${getPasswordStrengthColor(passwordStrength.score)}`}
                  style={{ width: `${(passwordStrength.score / 5) * 100}%` }}
                ></div>
              </div>

              {passwordStrength.feedback.length > 0 && (
                <div className="text-xs text-gray-600">
                  <p className="font-medium mb-1">Suggestions:</p>
                  <ul className="space-y-1">
                    {passwordStrength.feedback.map((tip, index) => (
                      <li key={index} className="flex items-start space-x-1">
                        <span>•</span>
                        <span>{tip}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="space-y-2">
          <Label htmlFor="confirmPassword">Confirm New Password</Label>
          <div className="relative">
            <Lock className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
            <Input
              id="confirmPassword"
              type={showConfirmPassword ? 'text' : 'password'}
              placeholder="Confirm new password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              className="pl-10 pr-10"
              disabled={isLoading}
            />
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="absolute right-2 top-1/2 transform -translate-y-1/2"
              onClick={() => setShowConfirmPassword(!showConfirmPassword)}
            >
              {showConfirmPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </Button>
          </div>

          {/* Password Match Indicator */}
          {confirmPassword && (
            <div className="flex items-center space-x-2 text-xs">
              {newPassword === confirmPassword ? (
                <>
                  <CheckCircle className="h-3 w-3 text-green-600" />
                  <span className="text-green-600">Passwords match</span>
                </>
              ) : (
                <>
                  <AlertCircle className="h-3 w-3 text-red-600" />
                  <span className="text-red-600">Passwords don't match</span>
                </>
              )}
            </div>
          )}
        </div>

        {/* Password Requirements */}
        <div className="space-y-2">
          <p className="text-xs font-medium text-gray-700">Password Requirements:</p>
          <div className="grid grid-cols-1 gap-1 text-xs">
            <div className={`flex items-center space-x-2 ${passwordStrength.hasMinLength ? 'text-green-600' : 'text-gray-500'}`}>
              <CheckCircle className={`h-3 w-3 ${passwordStrength.hasMinLength ? 'text-green-600' : 'text-gray-400'}`} />
              <span>At least 8 characters</span>
            </div>
            <div className={`flex items-center space-x-2 ${passwordStrength.hasUppercase ? 'text-green-600' : 'text-gray-500'}`}>
              <CheckCircle className={`h-3 w-3 ${passwordStrength.hasUppercase ? 'text-green-600' : 'text-gray-400'}`} />
              <span>Uppercase letter</span>
            </div>
            <div className={`flex items-center space-x-2 ${passwordStrength.hasLowercase ? 'text-green-600' : 'text-gray-500'}`}>
              <CheckCircle className={`h-3 w-3 ${passwordStrength.hasLowercase ? 'text-green-600' : 'text-gray-400'}`} />
              <span>Lowercase letter</span>
            </div>
            <div className={`flex items-center space-x-2 ${passwordStrength.hasNumbers ? 'text-green-600' : 'text-gray-500'}`}>
              <CheckCircle className={`h-3 w-3 ${passwordStrength.hasNumbers ? 'text-green-600' : 'text-gray-400'}`} />
              <span>Number</span>
            </div>
            <div className={`flex items-center space-x-2 ${passwordStrength.hasSpecialChars ? 'text-green-600' : 'text-gray-500'}`}>
              <CheckCircle className={`h-3 w-3 ${passwordStrength.hasSpecialChars ? 'text-green-600' : 'text-gray-400'}`} />
              <span>Special character</span>
            </div>
          </div>
        </div>

        <Button
          onClick={handlePasswordReset}
          disabled={isLoading || !newPassword || !confirmPassword || newPassword !== confirmPassword || passwordStrength.score < 3}
          className="w-full"
        >
          {isLoading ? (
            <>
              <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
              Resetting Password...
            </>
          ) : (
            <>
              <Lock className="h-4 w-4 mr-2" />
              Reset Password
            </>
          )}
        </Button>
      </CardContent>
    </Card>
  );

  const renderSuccessStep = () => (
    <Card className="max-w-md mx-auto">
      <CardHeader className="text-center">
        <div className="mx-auto w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mb-4">
          <CheckCircle className="h-8 w-8 text-green-600" />
        </div>
        <CardTitle>Password Reset Successful!</CardTitle>
        <CardDescription>
          Your password has been successfully updated
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        <Alert>
          <Shield className="h-4 w-4" />
          <AlertDescription>
            For security reasons, you've been logged out of all devices.
            Please log in again with your new password.
          </AlertDescription>
        </Alert>

        <div className="space-y-3">
          <div className="flex items-center space-x-3 p-3 bg-green-50 rounded-lg">
            <CheckCircle className="h-5 w-5 text-green-600" />
            <span className="text-sm">Password updated successfully</span>
          </div>
          <div className="flex items-center space-x-3 p-3 bg-green-50 rounded-lg">
            <CheckCircle className="h-5 w-5 text-green-600" />
            <span className="text-sm">All sessions cleared for security</span>
          </div>
          <div className="flex items-center space-x-3 p-3 bg-green-50 rounded-lg">
            <CheckCircle className="h-5 w-5 text-green-600" />
            <span className="text-sm">Account access restored</span>
          </div>
        </div>

        <Button
          onClick={() => router.push('/auth/login')}
          className="w-full"
        >
          Continue to Login
          <ArrowRight className="h-4 w-4 ml-2" />
        </Button>
      </CardContent>
    </Card>
  );

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Error and Success Alerts */}
      {error && (
        <Alert variant="destructive" className="max-w-md mx-auto">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {success && (
        <Alert className="max-w-md mx-auto border-green-200 bg-green-50">
          <CheckCircle className="h-4 w-4 text-green-600" />
          <AlertDescription className="text-green-800">{success}</AlertDescription>
        </Alert>
      )}

      {/* Render current step */}
      {currentStep === 'email' && renderEmailStep()}
      {currentStep === 'sent' && renderSentStep()}
      {currentStep === 'reset' && renderResetStep()}
      {currentStep === 'success' && renderSuccessStep()}
    </div>
  );
}