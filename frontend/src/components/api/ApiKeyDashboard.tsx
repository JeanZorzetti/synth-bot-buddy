/**
 * 🔑 API KEY MANAGEMENT DASHBOARD
 * Complete API key management with permissions, monitoring, and security
 */

import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { usePermissions } from '../../hooks/usePermissions';
import { apiClient } from '../../services/apiClient';
import { Button } from '../ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Badge } from '../ui/badge';
import { Alert, AlertDescription } from '../ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { Switch } from '../ui/switch';
import { Progress } from '../ui/progress';
import {
  Key,
  Plus,
  Copy,
  Edit,
  Trash2,
  Eye,
  EyeOff,
  Shield,
  Globe,
  Activity,
  Clock,
  AlertTriangle,
  CheckCircle,
  RefreshCw,
  Settings,
  BarChart3,
  Lock,
  Unlock,
  Filter,
  Search,
  Download,
  Upload,
  Code,
  FileText,
  Zap,
  Users,
  Database,
  Server,
  Terminal,
  BookOpen,
  Play,
  Pause,
  RotateCcw,
  Calendar,
  MapPin,
  TrendingUp,
  AlertCircle
} from 'lucide-react';

interface ApiKey {
  id: string;
  name: string;
  key: string;
  description: string;
  permissions: string[];
  ipWhitelist: string[];
  rateLimit: {
    requestsPerMinute: number;
    requestsPerHour: number;
    requestsPerDay: number;
  };
  usage: {
    requestsThisMonth: number;
    requestsToday: number;
    lastUsed: string;
  };
  status: 'active' | 'inactive' | 'expired' | 'revoked';
  createdAt: string;
  expiresAt?: string;
  lastRotated?: string;
  environment: 'production' | 'sandbox' | 'development';
  scopes: ApiScope[];
}

interface ApiScope {
  id: string;
  name: string;
  description: string;
  category: 'trading' | 'data' | 'user' | 'billing' | 'admin';
  methods: string[];
  enabled: boolean;
}

interface ApiUsageStats {
  totalRequests: number;
  requestsToday: number;
  requestsThisMonth: number;
  averageResponseTime: number;
  errorRate: number;
  topEndpoints: {
    endpoint: string;
    requests: number;
    percentage: number;
  }[];
  usageByHour: {
    hour: string;
    requests: number;
  }[];
}

interface ApiKeyDashboardProps {
  className?: string;
}

export function ApiKeyDashboard({ className = "" }: ApiKeyDashboardProps) {
  const { user } = useAuth();
  const { permissions } = usePermissions();

  const [activeTab, setActiveTab] = useState('keys');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [apiKeys, setApiKeys] = useState<ApiKey[]>([]);
  const [selectedKey, setSelectedKey] = useState<ApiKey | null>(null);
  const [usageStats, setUsageStats] = useState<ApiUsageStats | null>(null);
  const [availableScopes, setAvailableScopes] = useState<ApiScope[]>([]);

  // Create/Edit key form state
  const [isCreating, setIsCreating] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    environment: 'production' as const,
    expiresIn: '365', // days
    ipWhitelist: '',
    rateLimit: {
      requestsPerMinute: 100,
      requestsPerHour: 1000,
      requestsPerDay: 10000,
    },
    scopes: [] as string[],
  });

  // Filters and search
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [environmentFilter, setEnvironmentFilter] = useState<string>('all');

  // UI states
  const [showKeyValues, setShowKeyValues] = useState<{ [keyId: string]: boolean }>({});
  const [copyingKey, setCopyingKey] = useState<string | null>(null);

  useEffect(() => {
    loadApiKeys();
    loadUsageStats();
    loadAvailableScopes();
  }, []);

  const loadApiKeys = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const keys = await apiClient.get<ApiKey[]>('/api-keys');
      setApiKeys(keys);
    } catch (error: any) {
      setError(error.message || 'Failed to load API keys');
    } finally {
      setIsLoading(false);
    }
  };

  const loadUsageStats = async () => {
    try {
      const stats = await apiClient.get<ApiUsageStats>('/api-keys/usage-stats');
      setUsageStats(stats);
    } catch (error) {
      console.error('Failed to load usage stats:', error);
    }
  };

  const loadAvailableScopes = async () => {
    try {
      const scopes = await apiClient.get<ApiScope[]>('/api-keys/scopes');
      setAvailableScopes(scopes);
    } catch (error) {
      console.error('Failed to load scopes:', error);
    }
  };

  const createApiKey = async () => {
    try {
      setIsLoading(true);
      setError(null);

      const ipList = formData.ipWhitelist
        .split(',')
        .map(ip => ip.trim())
        .filter(ip => ip.length > 0);

      const newKey = await apiClient.post<ApiKey>('/api-keys', {
        name: formData.name,
        description: formData.description,
        environment: formData.environment,
        expiresAt: formData.expiresIn === 'never' ? null :
          new Date(Date.now() + parseInt(formData.expiresIn) * 24 * 60 * 60 * 1000).toISOString(),
        ipWhitelist: ipList,
        rateLimit: formData.rateLimit,
        scopes: formData.scopes,
      });

      setApiKeys(prev => [newKey, ...prev]);
      setIsCreating(false);
      resetForm();
    } catch (error: any) {
      setError(error.message || 'Failed to create API key');
    } finally {
      setIsLoading(false);
    }
  };

  const updateApiKey = async (keyId: string, updates: Partial<ApiKey>) => {
    try {
      setIsLoading(true);
      const updatedKey = await apiClient.put<ApiKey>(`/api-keys/${keyId}`, updates);
      setApiKeys(prev => prev.map(key => key.id === keyId ? updatedKey : key));
    } catch (error: any) {
      setError(error.message || 'Failed to update API key');
    } finally {
      setIsLoading(false);
    }
  };

  const deleteApiKey = async (keyId: string) => {
    if (!confirm('Are you sure you want to delete this API key? This action cannot be undone.')) {
      return;
    }

    try {
      setIsLoading(true);
      await apiClient.delete(`/api-keys/${keyId}`);
      setApiKeys(prev => prev.filter(key => key.id !== keyId));
    } catch (error: any) {
      setError(error.message || 'Failed to delete API key');
    } finally {
      setIsLoading(false);
    }
  };

  const rotateApiKey = async (keyId: string) => {
    if (!confirm('Are you sure you want to rotate this API key? The old key will be invalidated immediately.')) {
      return;
    }

    try {
      setIsLoading(true);
      const rotatedKey = await apiClient.post<ApiKey>(`/api-keys/${keyId}/rotate`);
      setApiKeys(prev => prev.map(key => key.id === keyId ? rotatedKey : key));
    } catch (error: any) {
      setError(error.message || 'Failed to rotate API key');
    } finally {
      setIsLoading(false);
    }
  };

  const toggleKeyStatus = async (keyId: string, status: 'active' | 'inactive') => {
    try {
      await updateApiKey(keyId, { status });
    } catch (error) {
      console.error('Failed to toggle key status:', error);
    }
  };

  const copyToClipboard = async (text: string, keyId: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopyingKey(keyId);
      setTimeout(() => setCopyingKey(null), 2000);
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
    }
  };

  const resetForm = () => {
    setFormData({
      name: '',
      description: '',
      environment: 'production',
      expiresIn: '365',
      ipWhitelist: '',
      rateLimit: {
        requestsPerMinute: 100,
        requestsPerHour: 1000,
        requestsPerDay: 10000,
      },
      scopes: [],
    });
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getStatusBadge = (status: string) => {
    const variants = {
      active: { className: 'bg-green-100 text-green-800', icon: CheckCircle },
      inactive: { className: 'bg-gray-100 text-gray-800', icon: Pause },
      expired: { className: 'bg-red-100 text-red-800', icon: Clock },
      revoked: { className: 'bg-red-100 text-red-800', icon: Lock },
    };

    const variant = variants[status as keyof typeof variants] || variants.active;
    const Icon = variant.icon;

    return (
      <Badge className={variant.className}>
        <Icon className="h-3 w-3 mr-1" />
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </Badge>
    );
  };

  const getEnvironmentBadge = (environment: string) => {
    const colors = {
      production: 'bg-red-100 text-red-800',
      sandbox: 'bg-yellow-100 text-yellow-800',
      development: 'bg-blue-100 text-blue-800',
    };

    return (
      <Badge className={colors[environment as keyof typeof colors] || colors.development}>
        {environment.charAt(0).toUpperCase() + environment.slice(1)}
      </Badge>
    );
  };

  const filterApiKeys = () => {
    let filtered = [...apiKeys];

    if (searchTerm) {
      filtered = filtered.filter(key =>
        key.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        key.description.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    if (statusFilter !== 'all') {
      filtered = filtered.filter(key => key.status === statusFilter);
    }

    if (environmentFilter !== 'all') {
      filtered = filtered.filter(key => key.environment === environmentFilter);
    }

    return filtered.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());
  };

  const renderOverview = () => (
    <div className="space-y-6">
      {/* Usage Statistics */}
      {usageStats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-blue-100 rounded-lg">
                  <BarChart3 className="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <p className="text-2xl font-bold">{usageStats.totalRequests.toLocaleString()}</p>
                  <p className="text-sm text-gray-500">Total Requests</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-green-100 rounded-lg">
                  <TrendingUp className="h-5 w-5 text-green-600" />
                </div>
                <div>
                  <p className="text-2xl font-bold">{usageStats.requestsToday.toLocaleString()}</p>
                  <p className="text-sm text-gray-500">Requests Today</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-purple-100 rounded-lg">
                  <Clock className="h-5 w-5 text-purple-600" />
                </div>
                <div>
                  <p className="text-2xl font-bold">{usageStats.averageResponseTime}ms</p>
                  <p className="text-sm text-gray-500">Avg Response</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-orange-100 rounded-lg">
                  <AlertTriangle className="h-5 w-5 text-orange-600" />
                </div>
                <div>
                  <p className="text-2xl font-bold">{(usageStats.errorRate * 100).toFixed(2)}%</p>
                  <p className="text-sm text-gray-500">Error Rate</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <Button
              onClick={() => setIsCreating(true)}
              className="h-16 flex-col space-y-2"
            >
              <Plus className="h-6 w-6" />
              <span>Create API Key</span>
            </Button>

            <Button
              variant="outline"
              onClick={() => setActiveTab('docs')}
              className="h-16 flex-col space-y-2"
            >
              <BookOpen className="h-6 w-6" />
              <span>View Docs</span>
            </Button>

            <Button
              variant="outline"
              onClick={() => setActiveTab('testing')}
              className="h-16 flex-col space-y-2"
            >
              <Terminal className="h-6 w-6" />
              <span>Test API</span>
            </Button>

            <Button
              variant="outline"
              onClick={loadUsageStats}
              className="h-16 flex-col space-y-2"
            >
              <RefreshCw className="h-6 w-6" />
              <span>Refresh Stats</span>
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Recent API Keys */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Recent API Keys</CardTitle>
            <Button
              variant="outline"
              onClick={() => setActiveTab('keys')}
            >
              View All
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {apiKeys.slice(0, 5).map((key) => (
              <div key={key.id} className="flex items-center justify-between p-3 border rounded-lg">
                <div className="flex items-center space-x-3">
                  <Key className="h-5 w-5 text-gray-400" />
                  <div>
                    <p className="font-medium">{key.name}</p>
                    <p className="text-sm text-gray-500">{key.description}</p>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  {getEnvironmentBadge(key.environment)}
                  {getStatusBadge(key.status)}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );

  const renderApiKeys = () => (
    <div className="space-y-6">
      {/* Filters and Actions */}
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-4">
              <div className="relative">
                <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                <Input
                  placeholder="Search API keys..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 w-64"
                />
              </div>

              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="px-3 py-2 border rounded-md bg-white"
              >
                <option value="all">All Status</option>
                <option value="active">Active</option>
                <option value="inactive">Inactive</option>
                <option value="expired">Expired</option>
                <option value="revoked">Revoked</option>
              </select>

              <select
                value={environmentFilter}
                onChange={(e) => setEnvironmentFilter(e.target.value)}
                className="px-3 py-2 border rounded-md bg-white"
              >
                <option value="all">All Environments</option>
                <option value="production">Production</option>
                <option value="sandbox">Sandbox</option>
                <option value="development">Development</option>
              </select>
            </div>

            <Button onClick={() => setIsCreating(true)}>
              <Plus className="h-4 w-4 mr-2" />
              Create API Key
            </Button>
          </div>

          <div className="text-sm text-gray-500">
            Showing {filterApiKeys().length} of {apiKeys.length} API keys
          </div>
        </CardContent>
      </Card>

      {/* API Keys List */}
      <div className="space-y-4">
        {filterApiKeys().map((key) => (
          <Card key={key.id} className="hover:shadow-md transition-shadow">
            <CardContent className="p-6">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-3 mb-2">
                    <h3 className="font-semibold text-lg">{key.name}</h3>
                    {getStatusBadge(key.status)}
                    {getEnvironmentBadge(key.environment)}
                  </div>

                  <p className="text-gray-600 mb-4">{key.description}</p>

                  {/* API Key Value */}
                  <div className="mb-4">
                    <Label className="text-sm font-medium">API Key</Label>
                    <div className="flex items-center space-x-2 mt-1">
                      <div className="flex-1 p-2 bg-gray-50 rounded border font-mono text-sm">
                        {showKeyValues[key.id] ? key.key : '••••••••••••••••••••••••••••••••'}
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setShowKeyValues(prev => ({
                          ...prev,
                          [key.id]: !prev[key.id]
                        }))}
                      >
                        {showKeyValues[key.id] ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => copyToClipboard(key.key, key.id)}
                      >
                        {copyingKey === key.id ? (
                          <CheckCircle className="h-4 w-4 text-green-500" />
                        ) : (
                          <Copy className="h-4 w-4" />
                        )}
                      </Button>
                    </div>
                  </div>

                  {/* Key Details */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <p className="text-gray-500">Created</p>
                      <p className="font-medium">{formatDate(key.createdAt)}</p>
                    </div>

                    <div>
                      <p className="text-gray-500">Last Used</p>
                      <p className="font-medium">
                        {key.usage.lastUsed ? formatDate(key.usage.lastUsed) : 'Never'}
                      </p>
                    </div>

                    <div>
                      <p className="text-gray-500">This Month</p>
                      <p className="font-medium">{key.usage.requestsThisMonth.toLocaleString()} requests</p>
                    </div>

                    <div>
                      <p className="text-gray-500">Rate Limit</p>
                      <p className="font-medium">{key.rateLimit.requestsPerMinute}/min</p>
                    </div>
                  </div>

                  {/* IP Whitelist */}
                  {key.ipWhitelist.length > 0 && (
                    <div className="mt-4">
                      <p className="text-sm font-medium text-gray-700 mb-2">IP Whitelist:</p>
                      <div className="flex flex-wrap gap-2">
                        {key.ipWhitelist.map((ip, index) => (
                          <Badge key={index} variant="outline" className="text-xs">
                            <Globe className="h-3 w-3 mr-1" />
                            {ip}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Permissions Preview */}
                  <div className="mt-4">
                    <p className="text-sm font-medium text-gray-700 mb-2">Permissions:</p>
                    <div className="flex flex-wrap gap-2">
                      {key.permissions.slice(0, 5).map((permission, index) => (
                        <Badge key={index} variant="outline" className="text-xs">
                          {permission}
                        </Badge>
                      ))}
                      {key.permissions.length > 5 && (
                        <Badge variant="outline" className="text-xs">
                          +{key.permissions.length - 5} more
                        </Badge>
                      )}
                    </div>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex flex-col space-y-2 ml-4">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      setSelectedKey(key);
                      setIsEditing(true);
                    }}
                  >
                    <Edit className="h-4 w-4 mr-2" />
                    Edit
                  </Button>

                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => rotateApiKey(key.id)}
                  >
                    <RotateCcw className="h-4 w-4 mr-2" />
                    Rotate
                  </Button>

                  <Switch
                    checked={key.status === 'active'}
                    onCheckedChange={(checked) =>
                      toggleKeyStatus(key.id, checked ? 'active' : 'inactive')
                    }
                  />

                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => deleteApiKey(key.id)}
                    className="text-red-600 hover:text-red-700"
                  >
                    <Trash2 className="h-4 w-4 mr-2" />
                    Delete
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}

        {filterApiKeys().length === 0 && (
          <Card>
            <CardContent className="p-8 text-center">
              <Key className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">No API keys found</p>
              <Button onClick={() => setIsCreating(true)} className="mt-4">
                Create Your First API Key
              </Button>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );

  const renderCreateForm = () => (
    <Card className="max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle>Create New API Key</CardTitle>
        <CardDescription>
          Generate a new API key with custom permissions and settings
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-6">
        {/* Basic Information */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="name">Key Name</Label>
            <Input
              id="name"
              value={formData.name}
              onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
              placeholder="My Trading Bot API Key"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="environment">Environment</Label>
            <select
              id="environment"
              value={formData.environment}
              onChange={(e) => setFormData(prev => ({ ...prev, environment: e.target.value as any }))}
              className="w-full p-2 border rounded-md"
            >
              <option value="production">Production</option>
              <option value="sandbox">Sandbox</option>
              <option value="development">Development</option>
            </select>
          </div>
        </div>

        <div className="space-y-2">
          <Label htmlFor="description">Description</Label>
          <textarea
            id="description"
            value={formData.description}
            onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
            placeholder="Describe what this API key will be used for..."
            className="w-full p-2 border rounded-md resize-none"
            rows={3}
          />
        </div>

        {/* Expiration */}
        <div className="space-y-2">
          <Label htmlFor="expiresIn">Expiration</Label>
          <select
            id="expiresIn"
            value={formData.expiresIn}
            onChange={(e) => setFormData(prev => ({ ...prev, expiresIn: e.target.value }))}
            className="w-full p-2 border rounded-md"
          >
            <option value="30">30 days</option>
            <option value="90">90 days</option>
            <option value="180">6 months</option>
            <option value="365">1 year</option>
            <option value="never">Never expires</option>
          </select>
        </div>

        {/* IP Whitelist */}
        <div className="space-y-2">
          <Label htmlFor="ipWhitelist">IP Whitelist (optional)</Label>
          <Input
            id="ipWhitelist"
            value={formData.ipWhitelist}
            onChange={(e) => setFormData(prev => ({ ...prev, ipWhitelist: e.target.value }))}
            placeholder="192.168.1.1, 10.0.0.0/24"
          />
          <p className="text-xs text-gray-500">
            Comma-separated list of IP addresses or CIDR ranges. Leave empty to allow all IPs.
          </p>
        </div>

        {/* Rate Limits */}
        <div className="space-y-4">
          <Label>Rate Limits</Label>
          <div className="grid grid-cols-3 gap-4">
            <div className="space-y-2">
              <Label className="text-sm">Per Minute</Label>
              <Input
                type="number"
                value={formData.rateLimit.requestsPerMinute}
                onChange={(e) => setFormData(prev => ({
                  ...prev,
                  rateLimit: { ...prev.rateLimit, requestsPerMinute: parseInt(e.target.value) || 0 }
                }))}
                min="1"
                max="10000"
              />
            </div>

            <div className="space-y-2">
              <Label className="text-sm">Per Hour</Label>
              <Input
                type="number"
                value={formData.rateLimit.requestsPerHour}
                onChange={(e) => setFormData(prev => ({
                  ...prev,
                  rateLimit: { ...prev.rateLimit, requestsPerHour: parseInt(e.target.value) || 0 }
                }))}
                min="1"
                max="100000"
              />
            </div>

            <div className="space-y-2">
              <Label className="text-sm">Per Day</Label>
              <Input
                type="number"
                value={formData.rateLimit.requestsPerDay}
                onChange={(e) => setFormData(prev => ({
                  ...prev,
                  rateLimit: { ...prev.rateLimit, requestsPerDay: parseInt(e.target.value) || 0 }
                }))}
                min="1"
                max="1000000"
              />
            </div>
          </div>
        </div>

        {/* Permissions */}
        <div className="space-y-4">
          <Label>Permissions</Label>
          <div className="grid grid-cols-1 gap-4">
            {['trading', 'data', 'user', 'billing'].map((category) => {
              const categoryScopes = availableScopes.filter(scope => scope.category === category);
              return (
                <div key={category} className="border rounded-lg p-4">
                  <h4 className="font-medium mb-3 capitalize">{category} Permissions</h4>
                  <div className="space-y-2">
                    {categoryScopes.map((scope) => (
                      <div key={scope.id} className="flex items-center justify-between">
                        <div>
                          <p className="font-medium text-sm">{scope.name}</p>
                          <p className="text-xs text-gray-500">{scope.description}</p>
                        </div>
                        <Switch
                          checked={formData.scopes.includes(scope.id)}
                          onCheckedChange={(checked) => {
                            if (checked) {
                              setFormData(prev => ({
                                ...prev,
                                scopes: [...prev.scopes, scope.id]
                              }));
                            } else {
                              setFormData(prev => ({
                                ...prev,
                                scopes: prev.scopes.filter(id => id !== scope.id)
                              }));
                            }
                          }}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Actions */}
        <div className="flex space-x-4">
          <Button
            onClick={createApiKey}
            disabled={!formData.name || formData.scopes.length === 0 || isLoading}
            className="flex-1"
          >
            {isLoading ? (
              <>
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                Creating...
              </>
            ) : (
              <>
                <Key className="h-4 w-4 mr-2" />
                Create API Key
              </>
            )}
          </Button>

          <Button
            variant="outline"
            onClick={() => {
              setIsCreating(false);
              resetForm();
            }}
            className="flex-1"
          >
            Cancel
          </Button>
        </div>
      </CardContent>
    </Card>
  );

  if (!user) {
    return (
      <Alert>
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          Please log in to manage your API keys.
        </AlertDescription>
      </Alert>
    );
  }

  if (isCreating) {
    return <div className={className}>{renderCreateForm()}</div>;
  }

  return (
    <div className={`max-w-6xl mx-auto space-y-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">API Key Management</h1>
          <p className="text-gray-600">Manage your API keys, permissions, and usage</p>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Main Content */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4 max-w-lg">
          <TabsTrigger value="overview" className="flex items-center space-x-2">
            <BarChart3 className="h-4 w-4" />
            <span>Overview</span>
          </TabsTrigger>
          <TabsTrigger value="keys" className="flex items-center space-x-2">
            <Key className="h-4 w-4" />
            <span>API Keys</span>
          </TabsTrigger>
          <TabsTrigger value="docs" className="flex items-center space-x-2">
            <BookOpen className="h-4 w-4" />
            <span>Docs</span>
          </TabsTrigger>
          <TabsTrigger value="testing" className="flex items-center space-x-2">
            <Terminal className="h-4 w-4" />
            <span>Testing</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          {renderOverview()}
        </TabsContent>

        <TabsContent value="keys">
          {renderApiKeys()}
        </TabsContent>

        <TabsContent value="docs">
          <Card>
            <CardContent className="p-8 text-center">
              <BookOpen className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">API Documentation coming soon...</p>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="testing">
          <Card>
            <CardContent className="p-8 text-center">
              <Terminal className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">API Testing Interface coming soon...</p>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}