/**
 * 🔐 TWO-FACTOR AUTHENTICATION COMPONENT
 * Complete 2FA setup and management with QR codes and backup codes
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
import {
  Shield,
  Smartphone,
  Key,
  QrCode,
  Download,
  Copy,
  CheckCircle,
  AlertTriangle,
  RefreshCw,
  Lock,
  Unlock,
  Eye,
  EyeOff,
  Settings,
  ArrowRight,
  ArrowLeft,
  AlertCircle
} from 'lucide-react';
import QRCodeDisplay from 'qrcode-generator';

interface TwoFactorData {
  secret: string;
  qrCodeUrl: string;
  backupCodes: string[];
  isEnabled: boolean;
}

interface TwoFactorAuthProps {
  className?: string;
  onClose?: () => void;
}

type SetupStep = 'overview' | 'setup' | 'verify' | 'backup' | 'complete';

export function TwoFactorAuth({ className = "", onClose }: TwoFactorAuthProps) {
  const { user, updateProfile } = useAuth();
  const [currentStep, setCurrentStep] = useState<SetupStep>('overview');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const [twoFactorData, setTwoFactorData] = useState<TwoFactorData | null>(null);
  const [verificationCode, setVerificationCode] = useState('');
  const [showBackupCodes, setShowBackupCodes] = useState(false);
  const [savedBackupCodes, setSavedBackupCodes] = useState(false);
  const [qrCodeDataUrl, setQrCodeDataUrl] = useState<string>('');

  // Check if 2FA is already enabled
  const is2FAEnabled = user?.preferences?.twoFactorEnabled || false;

  useEffect(() => {
    if (is2FAEnabled) {
      setCurrentStep('overview');
    }
  }, [is2FAEnabled]);

  useEffect(() => {
    if (twoFactorData?.qrCodeUrl) {
      generateQRCode(twoFactorData.qrCodeUrl);
    }
  }, [twoFactorData]);

  const generateQRCode = (url: string) => {
    try {
      const qr = QRCodeDisplay(0, 'L');
      qr.addData(url);
      qr.make();

      const size = 6;
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      const moduleCount = qr.getModuleCount();

      canvas.width = moduleCount * size;
      canvas.height = moduleCount * size;

      if (ctx) {
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        ctx.fillStyle = '#000000';
        for (let row = 0; row < moduleCount; row++) {
          for (let col = 0; col < moduleCount; col++) {
            if (qr.isDark(row, col)) {
              ctx.fillRect(col * size, row * size, size, size);
            }
          }
        }

        setQrCodeDataUrl(canvas.toDataURL());
      }
    } catch (error) {
      console.error('QR Code generation failed:', error);
    }
  };

  const initiateTwoFactorSetup = async () => {
    try {
      setIsLoading(true);
      setError(null);

      const response = await apiClient.post<TwoFactorData>('/auth/2fa/setup');
      setTwoFactorData(response);
      setCurrentStep('setup');
    } catch (error: any) {
      setError(error.message || 'Failed to initiate 2FA setup');
    } finally {
      setIsLoading(false);
    }
  };

  const verifyTwoFactorSetup = async () => {
    if (!verificationCode || verificationCode.length !== 6) {
      setError('Please enter a valid 6-digit code');
      return;
    }

    try {
      setIsLoading(true);
      setError(null);

      await apiClient.post('/auth/2fa/verify', {
        token: verificationCode,
        secret: twoFactorData?.secret
      });

      setCurrentStep('backup');
    } catch (error: any) {
      setError(error.message || 'Invalid verification code');
    } finally {
      setIsLoading(false);
    }
  };

  const completeTwoFactorSetup = async () => {
    try {
      setIsLoading(true);
      setError(null);

      await apiClient.post('/auth/2fa/enable');

      // Update user profile to reflect 2FA enabled
      await updateProfile({
        preferences: {
          ...user?.preferences,
          twoFactorEnabled: true
        }
      });

      setSuccess('Two-factor authentication has been successfully enabled!');
      setCurrentStep('complete');
    } catch (error: any) {
      setError(error.message || 'Failed to enable 2FA');
    } finally {
      setIsLoading(false);
    }
  };

  const disableTwoFactor = async () => {
    if (!confirm('Are you sure you want to disable two-factor authentication? This will make your account less secure.')) {
      return;
    }

    try {
      setIsLoading(true);
      setError(null);

      await apiClient.post('/auth/2fa/disable');

      // Update user profile
      await updateProfile({
        preferences: {
          ...user?.preferences,
          twoFactorEnabled: false
        }
      });

      setSuccess('Two-factor authentication has been disabled.');
      setCurrentStep('overview');
    } catch (error: any) {
      setError(error.message || 'Failed to disable 2FA');
    } finally {
      setIsLoading(false);
    }
  };

  const regenerateBackupCodes = async () => {
    try {
      setIsLoading(true);
      setError(null);

      const response = await apiClient.post<{ backupCodes: string[] }>('/auth/2fa/regenerate-backup-codes');
      setTwoFactorData(prev => prev ? { ...prev, backupCodes: response.backupCodes } : null);
      setSuccess('New backup codes generated successfully!');
    } catch (error: any) {
      setError(error.message || 'Failed to regenerate backup codes');
    } finally {
      setIsLoading(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setSuccess('Copied to clipboard!');
    setTimeout(() => setSuccess(null), 2000);
  };

  const downloadBackupCodes = () => {
    if (!twoFactorData?.backupCodes) return;

    const content = `Two-Factor Authentication Backup Codes
Generated: ${new Date().toLocaleString()}
Account: ${user?.email}

IMPORTANT: Store these codes in a safe place. Each code can only be used once.

${twoFactorData.backupCodes.map((code, index) => `${index + 1}. ${code}`).join('\n')}

These backup codes can be used to access your account if you lose access to your authenticator app.
Keep them secure and do not share them with anyone.`;

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `2fa-backup-codes-${user?.username || 'account'}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const renderOverview = () => (
    <Card className="max-w-2xl mx-auto">
      <CardHeader>
        <div className="flex items-center space-x-3">
          <div className={`p-2 rounded-lg ${is2FAEnabled ? 'bg-green-100' : 'bg-orange-100'}`}>
            <Shield className={`h-6 w-6 ${is2FAEnabled ? 'text-green-600' : 'text-orange-600'}`} />
          </div>
          <div>
            <CardTitle>Two-Factor Authentication</CardTitle>
            <CardDescription>
              Add an extra layer of security to your account
            </CardDescription>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        <div className="flex items-center justify-between p-4 border rounded-lg">
          <div className="flex items-center space-x-3">
            {is2FAEnabled ? (
              <>
                <CheckCircle className="h-5 w-5 text-green-500" />
                <div>
                  <p className="font-medium">Two-Factor Authentication Enabled</p>
                  <p className="text-sm text-gray-500">Your account is protected with 2FA</p>
                </div>
              </>
            ) : (
              <>
                <AlertTriangle className="h-5 w-5 text-orange-500" />
                <div>
                  <p className="font-medium">Two-Factor Authentication Disabled</p>
                  <p className="text-sm text-gray-500">Enable 2FA to secure your account</p>
                </div>
              </>
            )}
          </div>

          {is2FAEnabled ? (
            <Badge className="bg-green-100 text-green-800">Enabled</Badge>
          ) : (
            <Badge className="bg-orange-100 text-orange-800">Disabled</Badge>
          )}
        </div>

        <div className="space-y-4">
          <h4 className="font-medium">How it works:</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-4 border rounded-lg">
              <Smartphone className="h-8 w-8 text-blue-500 mx-auto mb-2" />
              <p className="font-medium text-sm">Install Authenticator</p>
              <p className="text-xs text-gray-500">Use Google Authenticator or similar app</p>
            </div>

            <div className="text-center p-4 border rounded-lg">
              <QrCode className="h-8 w-8 text-green-500 mx-auto mb-2" />
              <p className="font-medium text-sm">Scan QR Code</p>
              <p className="text-xs text-gray-500">Link your account to the app</p>
            </div>

            <div className="text-center p-4 border rounded-lg">
              <Lock className="h-8 w-8 text-purple-500 mx-auto mb-2" />
              <p className="font-medium text-sm">Enter Code</p>
              <p className="text-xs text-gray-500">Use 6-digit code when logging in</p>
            </div>
          </div>
        </div>

        <div className="flex space-x-3">
          {is2FAEnabled ? (
            <>
              <Button variant="outline" onClick={regenerateBackupCodes} disabled={isLoading}>
                <RefreshCw className="h-4 w-4 mr-2" />
                Regenerate Backup Codes
              </Button>
              <Button variant="destructive" onClick={disableTwoFactor} disabled={isLoading}>
                <Unlock className="h-4 w-4 mr-2" />
                Disable 2FA
              </Button>
            </>
          ) : (
            <Button onClick={initiateTwoFactorSetup} disabled={isLoading}>
              <Shield className="h-4 w-4 mr-2" />
              {isLoading ? 'Setting up...' : 'Enable Two-Factor Authentication'}
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );

  const renderSetup = () => (
    <Card className="max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle>Step 1: Setup Authenticator App</CardTitle>
        <CardDescription>
          Scan the QR code with your authenticator app
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        <div className="text-center">
          <div className="bg-white p-6 rounded-lg border-2 border-dashed border-gray-300 inline-block">
            {qrCodeDataUrl ? (
              <img src={qrCodeDataUrl} alt="2FA QR Code" className="w-48 h-48" />
            ) : (
              <div className="w-48 h-48 flex items-center justify-center">
                <RefreshCw className="h-8 w-8 animate-spin text-gray-400" />
              </div>
            )}
          </div>
        </div>

        <div className="space-y-4">
          <h4 className="font-medium">Recommended Authenticator Apps:</h4>
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 border rounded-lg">
              <p className="font-medium text-sm">Google Authenticator</p>
              <p className="text-xs text-gray-500">iOS & Android</p>
            </div>
            <div className="p-3 border rounded-lg">
              <p className="font-medium text-sm">Microsoft Authenticator</p>
              <p className="text-xs text-gray-500">iOS & Android</p>
            </div>
            <div className="p-3 border rounded-lg">
              <p className="font-medium text-sm">Authy</p>
              <p className="text-xs text-gray-500">Cross-platform</p>
            </div>
            <div className="p-3 border rounded-lg">
              <p className="font-medium text-sm">1Password</p>
              <p className="text-xs text-gray-500">Premium</p>
            </div>
          </div>
        </div>

        <div className="space-y-2">
          <Label>Manual Entry Key (if QR scan doesn't work):</Label>
          <div className="flex space-x-2">
            <Input
              value={twoFactorData?.secret || ''}
              readOnly
              className="font-mono text-sm"
            />
            <Button
              variant="outline"
              onClick={() => copyToClipboard(twoFactorData?.secret || '')}
            >
              <Copy className="h-4 w-4" />
            </Button>
          </div>
        </div>

        <div className="flex space-x-3">
          <Button
            variant="outline"
            onClick={() => setCurrentStep('overview')}
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back
          </Button>
          <Button
            onClick={() => setCurrentStep('verify')}
            disabled={!twoFactorData}
          >
            Next: Verify Setup
            <ArrowRight className="h-4 w-4 ml-2" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );

  const renderVerify = () => (
    <Card className="max-w-md mx-auto">
      <CardHeader>
        <CardTitle>Step 2: Verify Setup</CardTitle>
        <CardDescription>
          Enter the 6-digit code from your authenticator app
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        <div className="space-y-2">
          <Label htmlFor="verificationCode">Verification Code</Label>
          <Input
            id="verificationCode"
            value={verificationCode}
            onChange={(e) => setVerificationCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
            placeholder="000000"
            className="text-center text-lg font-mono tracking-widest"
            maxLength={6}
          />
          <p className="text-xs text-gray-500 text-center">
            Enter the 6-digit code from your authenticator app
          </p>
        </div>

        <div className="flex space-x-3">
          <Button
            variant="outline"
            onClick={() => setCurrentStep('setup')}
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back
          </Button>
          <Button
            onClick={verifyTwoFactorSetup}
            disabled={isLoading || verificationCode.length !== 6}
            className="flex-1"
          >
            {isLoading ? (
              <>
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                Verifying...
              </>
            ) : (
              'Verify & Continue'
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );

  const renderBackup = () => (
    <Card className="max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle>Step 3: Save Backup Codes</CardTitle>
        <CardDescription>
          These codes can be used if you lose access to your authenticator app
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            <strong>Important:</strong> Save these backup codes in a secure location.
            Each code can only be used once and they're your only way to recover account access
            if you lose your authenticator device.
          </AlertDescription>
        </Alert>

        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="font-medium">Backup Codes</h4>
            <div className="flex space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowBackupCodes(!showBackupCodes)}
              >
                {showBackupCodes ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={downloadBackupCodes}
              >
                <Download className="h-4 w-4" />
              </Button>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 p-4 bg-gray-50 rounded-lg font-mono text-sm">
            {twoFactorData?.backupCodes.map((code, index) => (
              <div
                key={index}
                className="p-2 bg-white rounded border flex items-center justify-between"
              >
                <span>{showBackupCodes ? code : '••••••••'}</span>
                {showBackupCodes && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyToClipboard(code)}
                    className="h-6 w-6 p-0"
                  >
                    <Copy className="h-3 w-3" />
                  </Button>
                )}
              </div>
            ))}
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <input
            type="checkbox"
            id="saved-codes"
            checked={savedBackupCodes}
            onChange={(e) => setSavedBackupCodes(e.target.checked)}
            className="rounded"
          />
          <Label htmlFor="saved-codes" className="text-sm">
            I have saved these backup codes in a secure location
          </Label>
        </div>

        <div className="flex space-x-3">
          <Button
            variant="outline"
            onClick={() => setCurrentStep('verify')}
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back
          </Button>
          <Button
            onClick={completeTwoFactorSetup}
            disabled={!savedBackupCodes || isLoading}
            className="flex-1"
          >
            {isLoading ? (
              <>
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                Enabling 2FA...
              </>
            ) : (
              'Complete Setup'
            )}
          </Button>
        </div>
      </CardContent>
    </Card>
  );

  const renderComplete = () => (
    <Card className="max-w-md mx-auto">
      <CardHeader>
        <div className="text-center">
          <div className="mx-auto w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mb-4">
            <CheckCircle className="h-8 w-8 text-green-600" />
          </div>
          <CardTitle>2FA Successfully Enabled!</CardTitle>
          <CardDescription>
            Your account is now protected with two-factor authentication
          </CardDescription>
        </div>
      </CardHeader>

      <CardContent className="space-y-6">
        <div className="space-y-3">
          <div className="flex items-center space-x-3 p-3 bg-green-50 rounded-lg">
            <CheckCircle className="h-5 w-5 text-green-600" />
            <span className="text-sm">Two-factor authentication enabled</span>
          </div>
          <div className="flex items-center space-x-3 p-3 bg-green-50 rounded-lg">
            <CheckCircle className="h-5 w-5 text-green-600" />
            <span className="text-sm">Backup codes saved securely</span>
          </div>
          <div className="flex items-center space-x-3 p-3 bg-green-50 rounded-lg">
            <CheckCircle className="h-5 w-5 text-green-600" />
            <span className="text-sm">Account security enhanced</span>
          </div>
        </div>

        <Alert>
          <Shield className="h-4 w-4" />
          <AlertDescription>
            From now on, you'll need to enter a code from your authenticator app
            when logging in to your account.
          </AlertDescription>
        </Alert>

        <Button
          onClick={() => {
            setCurrentStep('overview');
            onClose?.();
          }}
          className="w-full"
        >
          Done
        </Button>
      </CardContent>
    </Card>
  );

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
      {currentStep === 'setup' && renderSetup()}
      {currentStep === 'verify' && renderVerify()}
      {currentStep === 'backup' && renderBackup()}
      {currentStep === 'complete' && renderComplete()}
    </div>
  );
}