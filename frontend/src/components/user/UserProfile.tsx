/**
 * 👤 USER PROFILE COMPONENT
 * Complete user profile dashboard with settings and management
 */

import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { usePermissions, RoleBadge } from '../auth/PermissionGuard';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Alert, AlertDescription } from '../ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { Label } from '../ui/label';
import { Switch } from '../ui/switch';
import { Badge } from '../ui/badge';
import {
  User,
  Settings,
  Shield,
  CreditCard,
  Activity,
  Bell,
  Eye,
  EyeOff,
  Save,
  Upload,
  Download,
  Trash2,
  AlertCircle,
  CheckCircle,
  TrendingUp,
  BarChart3,
  DollarSign,
  Clock,
  MapPin,
  Phone,
  Mail,
  Calendar,
  Building,
  Globe,
  Key,
  Lock,
  Smartphone,
  Monitor,
  RefreshCw
} from 'lucide-react';

interface UserProfileProps {
  className?: string;
}

export function UserProfile({ className = "" }: UserProfileProps) {
  const { user, updateProfile, changePassword, isLoading, error } = useAuth();
  const { getSubscriptionInfo, getFeatureLimits, getAvailableFeatures } = usePermissions();

  const [activeTab, setActiveTab] = useState('profile');
  const [isEditing, setIsEditing] = useState(false);
  const [isSaving, setSaving] = useState(false);
  const [showCurrentPassword, setShowCurrentPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);

  const [profileData, setProfileData] = useState({
    fullName: user?.fullName || '',
    username: user?.username || '',
    email: user?.email || '',
    phone: user?.phone || '',
    country: user?.country || '',
    avatar: user?.avatar || '',
  });

  const [passwordData, setPasswordData] = useState({
    currentPassword: '',
    newPassword: '',
    confirmPassword: '',
  });

  const [preferences, setPreferences] = useState({
    theme: user?.preferences?.theme || 'auto',
    language: user?.preferences?.language || 'en',
    timezone: user?.preferences?.timezone || 'UTC',
    emailNotifications: user?.preferences?.emailNotifications || true,
    smsNotifications: user?.preferences?.smsNotifications || false,
    pushNotifications: user?.preferences?.pushNotifications || true,
  });

  const [tradingSettings, setTradingSettings] = useState({
    maxRiskPerTrade: user?.tradingSettings?.maxRiskPerTrade || 2,
    maxDailyRisk: user?.tradingSettings?.maxDailyRisk || 10,
    autoTrading: user?.tradingSettings?.autoTrading || false,
    defaultPositionSize: user?.tradingSettings?.defaultPositionSize || 1000,
    stopLossPct: user?.tradingSettings?.stopLossPct || 2,
    takeProfitPct: user?.tradingSettings?.takeProfitPct || 4,
  });

  const subscriptionInfo = getSubscriptionInfo();
  const featureLimits = getFeatureLimits();
  const availableFeatures = getAvailableFeatures();

  useEffect(() => {
    if (user) {
      setProfileData({
        fullName: user.fullName,
        username: user.username,
        email: user.email,
        phone: user.phone || '',
        country: user.country || '',
        avatar: user.avatar || '',
      });
    }
  }, [user]);

  const handleProfileUpdate = async () => {
    try {
      setSaving(true);
      await updateProfile({
        fullName: profileData.fullName,
        phone: profileData.phone,
        country: profileData.country,
        preferences,
        tradingSettings,
      });
      setIsEditing(false);
    } catch (error) {
      console.error('Profile update failed:', error);
    } finally {
      setSaving(false);
    }
  };

  const handlePasswordChange = async () => {
    if (passwordData.newPassword !== passwordData.confirmPassword) {
      alert('New passwords do not match');
      return;
    }

    try {
      setSaving(true);
      await changePassword(passwordData.currentPassword, passwordData.newPassword);
      setPasswordData({ currentPassword: '', newPassword: '', confirmPassword: '' });
      alert('Password changed successfully');
    } catch (error) {
      console.error('Password change failed:', error);
    } finally {
      setSaving(false);
    }
  };

  const formatDate = (dateString?: string) => {
    if (!dateString) return 'Never';
    return new Date(dateString).toLocaleDateString();
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(amount);
  };

  if (!user) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <User className="h-16 w-16 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-500">Please log in to view your profile</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`max-w-6xl mx-auto space-y-6 ${className}`}>
      {/* Profile Header */}
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center space-x-6">
            <div className="relative">
              <div className="w-24 h-24 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center text-white text-3xl font-bold">
                {user.fullName.charAt(0).toUpperCase()}
              </div>
              <Button
                size="sm"
                variant="outline"
                className="absolute -bottom-2 -right-2 rounded-full w-8 h-8 p-0"
              >
                <Upload className="h-4 w-4" />
              </Button>
            </div>

            <div className="flex-1">
              <div className="flex items-center space-x-3 mb-2">
                <h1 className="text-3xl font-bold">{user.fullName}</h1>
                <RoleBadge />
                {user.isVerified && (
                  <Badge variant="outline" className="text-green-600 border-green-200">
                    <CheckCircle className="h-3 w-3 mr-1" />
                    Verified
                  </Badge>
                )}
              </div>

              <p className="text-gray-600 mb-2">@{user.username}</p>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div className="flex items-center space-x-2">
                  <Mail className="h-4 w-4 text-gray-400" />
                  <span>{user.email}</span>
                </div>

                <div className="flex items-center space-x-2">
                  <Calendar className="h-4 w-4 text-gray-400" />
                  <span>Joined {formatDate(user.createdAt)}</span>
                </div>

                <div className="flex items-center space-x-2">
                  <Clock className="h-4 w-4 text-gray-400" />
                  <span>Last login {formatDate(user.lastLogin)}</span>
                </div>

                <div className="flex items-center space-x-2">
                  <BarChart3 className="h-4 w-4 text-gray-400" />
                  <span>{user.totalTrades} trades</span>
                </div>
              </div>
            </div>

            <div className="text-right">
              <div className="text-2xl font-bold text-green-600">
                {formatCurrency(user.totalProfitLoss)}
              </div>
              <p className="text-sm text-gray-500">Total P&L</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Error Alert */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Main Content Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="profile" className="flex items-center space-x-2">
            <User className="h-4 w-4" />
            <span>Profile</span>
          </TabsTrigger>
          <TabsTrigger value="trading" className="flex items-center space-x-2">
            <TrendingUp className="h-4 w-4" />
            <span>Trading</span>
          </TabsTrigger>
          <TabsTrigger value="security" className="flex items-center space-x-2">
            <Shield className="h-4 w-4" />
            <span>Security</span>
          </TabsTrigger>
          <TabsTrigger value="subscription" className="flex items-center space-x-2">
            <CreditCard className="h-4 w-4" />
            <span>Billing</span>
          </TabsTrigger>
          <TabsTrigger value="preferences" className="flex items-center space-x-2">
            <Settings className="h-4 w-4" />
            <span>Settings</span>
          </TabsTrigger>
          <TabsTrigger value="activity" className="flex items-center space-x-2">
            <Activity className="h-4 w-4" />
            <span>Activity</span>
          </TabsTrigger>
        </TabsList>

        {/* Profile Tab */}
        <TabsContent value="profile">
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>Personal Information</CardTitle>
                  <CardDescription>
                    Manage your personal details and contact information
                  </CardDescription>
                </div>
                <Button
                  variant={isEditing ? "outline" : "default"}
                  onClick={() => setIsEditing(!isEditing)}
                >
                  {isEditing ? 'Cancel' : 'Edit Profile'}
                </Button>
              </div>
            </CardHeader>

            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <Label htmlFor="fullName">Full Name</Label>
                  <Input
                    id="fullName"
                    value={profileData.fullName}
                    onChange={(e) => setProfileData(prev => ({ ...prev, fullName: e.target.value }))}
                    disabled={!isEditing}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="username">Username</Label>
                  <Input
                    id="username"
                    value={profileData.username}
                    disabled={true}
                  />
                  <p className="text-xs text-gray-500">Username cannot be changed</p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="email">Email Address</Label>
                  <Input
                    id="email"
                    type="email"
                    value={profileData.email}
                    disabled={true}
                  />
                  <p className="text-xs text-gray-500">Email changes require verification</p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="phone">Phone Number</Label>
                  <Input
                    id="phone"
                    value={profileData.phone}
                    onChange={(e) => setProfileData(prev => ({ ...prev, phone: e.target.value }))}
                    disabled={!isEditing}
                    placeholder="+1 (555) 123-4567"
                  />
                </div>

                <div className="space-y-2 md:col-span-2">
                  <Label htmlFor="country">Country</Label>
                  <Input
                    id="country"
                    value={profileData.country}
                    onChange={(e) => setProfileData(prev => ({ ...prev, country: e.target.value }))}
                    disabled={!isEditing}
                    placeholder="United States"
                  />
                </div>
              </div>

              {isEditing && (
                <div className="flex space-x-3">
                  <Button onClick={handleProfileUpdate} disabled={isSaving}>
                    <Save className="h-4 w-4 mr-2" />
                    {isSaving ? 'Saving...' : 'Save Changes'}
                  </Button>
                  <Button variant="outline" onClick={() => setIsEditing(false)}>
                    Cancel
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Trading Tab */}
        <TabsContent value="trading">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Trading Settings</CardTitle>
                <CardDescription>
                  Configure your trading preferences and risk parameters
                </CardDescription>
              </CardHeader>

              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <Label>Max Risk Per Trade (%)</Label>
                  <Input
                    type="number"
                    value={tradingSettings.maxRiskPerTrade}
                    onChange={(e) => setTradingSettings(prev => ({
                      ...prev,
                      maxRiskPerTrade: parseFloat(e.target.value)
                    }))}
                    min="0.1"
                    max="10"
                    step="0.1"
                  />
                </div>

                <div className="space-y-2">
                  <Label>Max Daily Risk (%)</Label>
                  <Input
                    type="number"
                    value={tradingSettings.maxDailyRisk}
                    onChange={(e) => setTradingSettings(prev => ({
                      ...prev,
                      maxDailyRisk: parseFloat(e.target.value)
                    }))}
                    min="1"
                    max="50"
                    step="1"
                  />
                </div>

                <div className="space-y-2">
                  <Label>Default Position Size ($)</Label>
                  <Input
                    type="number"
                    value={tradingSettings.defaultPositionSize}
                    onChange={(e) => setTradingSettings(prev => ({
                      ...prev,
                      defaultPositionSize: parseFloat(e.target.value)
                    }))}
                    min="100"
                    step="100"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Auto Trading</Label>
                    <p className="text-sm text-gray-500">Enable automated trading execution</p>
                  </div>
                  <Switch
                    checked={tradingSettings.autoTrading}
                    onCheckedChange={(checked) => setTradingSettings(prev => ({
                      ...prev,
                      autoTrading: checked
                    }))}
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Trading Statistics</CardTitle>
                <CardDescription>
                  Your trading performance overview
                </CardDescription>
              </CardHeader>

              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center p-4 bg-gray-50 rounded-lg">
                    <div className="text-2xl font-bold text-blue-600">{user.totalTrades}</div>
                    <div className="text-sm text-gray-500">Total Trades</div>
                  </div>

                  <div className="text-center p-4 bg-gray-50 rounded-lg">
                    <div className={`text-2xl font-bold ${user.totalProfitLoss >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {formatCurrency(user.totalProfitLoss)}
                    </div>
                    <div className="text-sm text-gray-500">Total P&L</div>
                  </div>
                </div>

                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-sm">Risk Profile:</span>
                    <Badge variant="outline">
                      {user.riskProfile.riskTolerance.toUpperCase()}
                    </Badge>
                  </div>

                  <div className="flex justify-between">
                    <span className="text-sm">Experience:</span>
                    <span className="text-sm font-medium">
                      {user.riskProfile.tradingExperience}
                    </span>
                  </div>

                  <div className="flex justify-between">
                    <span className="text-sm">Max Drawdown:</span>
                    <span className="text-sm font-medium">
                      {user.riskProfile.maxDrawdown}%
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Security Tab */}
        <TabsContent value="security">
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Change Password</CardTitle>
                <CardDescription>
                  Keep your account secure with a strong password
                </CardDescription>
              </CardHeader>

              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="currentPassword">Current Password</Label>
                  <div className="relative">
                    <Input
                      id="currentPassword"
                      type={showCurrentPassword ? 'text' : 'password'}
                      value={passwordData.currentPassword}
                      onChange={(e) => setPasswordData(prev => ({
                        ...prev,
                        currentPassword: e.target.value
                      }))}
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="absolute right-2 top-1/2 transform -translate-y-1/2"
                      onClick={() => setShowCurrentPassword(!showCurrentPassword)}
                    >
                      {showCurrentPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </Button>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="newPassword">New Password</Label>
                  <div className="relative">
                    <Input
                      id="newPassword"
                      type={showNewPassword ? 'text' : 'password'}
                      value={passwordData.newPassword}
                      onChange={(e) => setPasswordData(prev => ({
                        ...prev,
                        newPassword: e.target.value
                      }))}
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
                </div>

                <div className="space-y-2">
                  <Label htmlFor="confirmPassword">Confirm New Password</Label>
                  <Input
                    id="confirmPassword"
                    type="password"
                    value={passwordData.confirmPassword}
                    onChange={(e) => setPasswordData(prev => ({
                      ...prev,
                      confirmPassword: e.target.value
                    }))}
                  />
                </div>

                <Button onClick={handlePasswordChange} disabled={isSaving}>
                  <Lock className="h-4 w-4 mr-2" />
                  {isSaving ? 'Updating...' : 'Update Password'}
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Two-Factor Authentication</CardTitle>
                <CardDescription>
                  Add an extra layer of security to your account
                </CardDescription>
              </CardHeader>

              <CardContent>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <Smartphone className="h-5 w-5" />
                    <div>
                      <p className="font-medium">Authenticator App</p>
                      <p className="text-sm text-gray-500">Use an app like Google Authenticator</p>
                    </div>
                  </div>
                  <Button variant="outline">
                    Enable 2FA
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Subscription Tab */}
        <TabsContent value="subscription">
          <Card>
            <CardHeader>
              <CardTitle>Subscription Details</CardTitle>
              <CardDescription>
                Manage your subscription and billing information
              </CardDescription>
            </CardHeader>

            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="space-y-2">
                  <Label>Current Plan</Label>
                  <div className="flex items-center space-x-2">
                    <RoleBadge />
                    <Badge variant={subscriptionInfo?.isActive ? "default" : "destructive"}>
                      {subscriptionInfo?.status}
                    </Badge>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>Plan</Label>
                  <p className="font-medium">{user.subscriptionPlan}</p>
                </div>

                <div className="space-y-2">
                  <Label>Expires</Label>
                  <p className="font-medium">{formatDate(user.subscriptionExpires)}</p>
                </div>
              </div>

              {featureLimits && (
                <div className="space-y-4">
                  <h4 className="font-medium">Usage Limits</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="text-center p-3 bg-gray-50 rounded-lg">
                      <div className="text-lg font-bold">
                        {featureLimits.maxApiCallsPerMinute === -1 ? '∞' : featureLimits.maxApiCallsPerMinute}
                      </div>
                      <div className="text-xs text-gray-500">API Calls/min</div>
                    </div>

                    <div className="text-center p-3 bg-gray-50 rounded-lg">
                      <div className="text-lg font-bold">
                        {featureLimits.maxConcurrentStrategies === -1 ? '∞' : featureLimits.maxConcurrentStrategies}
                      </div>
                      <div className="text-xs text-gray-500">Strategies</div>
                    </div>

                    <div className="text-center p-3 bg-gray-50 rounded-lg">
                      <div className="text-lg font-bold">
                        {featureLimits.maxReports === -1 ? '∞' : featureLimits.maxReports}
                      </div>
                      <div className="text-xs text-gray-500">Reports</div>
                    </div>

                    <div className="text-center p-3 bg-gray-50 rounded-lg">
                      <div className="text-lg font-bold">
                        {featureLimits.dataRetentionDays === -1 ? '∞' : `${featureLimits.dataRetentionDays}d`}
                      </div>
                      <div className="text-xs text-gray-500">Data Retention</div>
                    </div>
                  </div>
                </div>
              )}

              <div className="flex space-x-3">
                <Button>
                  <CreditCard className="h-4 w-4 mr-2" />
                  Manage Billing
                </Button>
                <Button variant="outline">
                  <Download className="h-4 w-4 mr-2" />
                  Download Invoice
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Preferences Tab */}
        <TabsContent value="preferences">
          <Card>
            <CardHeader>
              <CardTitle>Preferences</CardTitle>
              <CardDescription>
                Customize your experience and notification settings
              </CardDescription>
            </CardHeader>

            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h4 className="font-medium">Appearance</h4>

                  <div className="space-y-2">
                    <Label>Theme</Label>
                    <select
                      value={preferences.theme}
                      onChange={(e) => setPreferences(prev => ({ ...prev, theme: e.target.value }))}
                      className="w-full p-2 border rounded-md"
                    >
                      <option value="light">Light</option>
                      <option value="dark">Dark</option>
                      <option value="auto">Auto</option>
                    </select>
                  </div>

                  <div className="space-y-2">
                    <Label>Language</Label>
                    <select
                      value={preferences.language}
                      onChange={(e) => setPreferences(prev => ({ ...prev, language: e.target.value }))}
                      className="w-full p-2 border rounded-md"
                    >
                      <option value="en">English</option>
                      <option value="pt">Português</option>
                      <option value="es">Español</option>
                    </select>
                  </div>

                  <div className="space-y-2">
                    <Label>Timezone</Label>
                    <select
                      value={preferences.timezone}
                      onChange={(e) => setPreferences(prev => ({ ...prev, timezone: e.target.value }))}
                      className="w-full p-2 border rounded-md"
                    >
                      <option value="UTC">UTC</option>
                      <option value="America/New_York">Eastern Time</option>
                      <option value="America/Chicago">Central Time</option>
                      <option value="America/Denver">Mountain Time</option>
                      <option value="America/Los_Angeles">Pacific Time</option>
                    </select>
                  </div>
                </div>

                <div className="space-y-4">
                  <h4 className="font-medium">Notifications</h4>

                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Email Notifications</Label>
                        <p className="text-sm text-gray-500">Receive updates via email</p>
                      </div>
                      <Switch
                        checked={preferences.emailNotifications}
                        onCheckedChange={(checked) => setPreferences(prev => ({
                          ...prev,
                          emailNotifications: checked
                        }))}
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>SMS Notifications</Label>
                        <p className="text-sm text-gray-500">Receive alerts via SMS</p>
                      </div>
                      <Switch
                        checked={preferences.smsNotifications}
                        onCheckedChange={(checked) => setPreferences(prev => ({
                          ...prev,
                          smsNotifications: checked
                        }))}
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Push Notifications</Label>
                        <p className="text-sm text-gray-500">Browser push notifications</p>
                      </div>
                      <Switch
                        checked={preferences.pushNotifications}
                        onCheckedChange={(checked) => setPreferences(prev => ({
                          ...prev,
                          pushNotifications: checked
                        }))}
                      />
                    </div>
                  </div>
                </div>
              </div>

              <Button onClick={handleProfileUpdate} disabled={isSaving}>
                <Save className="h-4 w-4 mr-2" />
                {isSaving ? 'Saving...' : 'Save Preferences'}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Activity Tab */}
        <TabsContent value="activity">
          <Card>
            <CardHeader>
              <CardTitle>Recent Activity</CardTitle>
              <CardDescription>
                Your recent account and trading activity
              </CardDescription>
            </CardHeader>

            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center space-x-4 p-3 border rounded-lg">
                  <Monitor className="h-5 w-5 text-blue-500" />
                  <div className="flex-1">
                    <p className="font-medium">Login from new device</p>
                    <p className="text-sm text-gray-500">Chrome on Windows • Just now</p>
                  </div>
                </div>

                <div className="flex items-center space-x-4 p-3 border rounded-lg">
                  <TrendingUp className="h-5 w-5 text-green-500" />
                  <div className="flex-1">
                    <p className="font-medium">Successful trade executed</p>
                    <p className="text-sm text-gray-500">EUR/USD • +$125.50 • 2 hours ago</p>
                  </div>
                </div>

                <div className="flex items-center space-x-4 p-3 border rounded-lg">
                  <Settings className="h-5 w-5 text-orange-500" />
                  <div className="flex-1">
                    <p className="font-medium">Trading settings updated</p>
                    <p className="text-sm text-gray-500">Risk parameters modified • 1 day ago</p>
                  </div>
                </div>

                <div className="flex items-center space-x-4 p-3 border rounded-lg">
                  <CreditCard className="h-5 w-5 text-purple-500" />
                  <div className="flex-1">
                    <p className="font-medium">Subscription renewed</p>
                    <p className="text-sm text-gray-500">Premium plan • 3 days ago</p>
                  </div>
                </div>
              </div>

              <div className="mt-6 text-center">
                <Button variant="outline">
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Load More Activity
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}