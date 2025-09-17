/**
 * Enterprise Platform - Phase 10 Integration
 * Interface completa para plataforma empresarial multi-usuário
 */

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';
import { Switch } from '@/components/ui/switch';
import { Textarea } from '@/components/ui/textarea';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow
} from '@/components/ui/table';
import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell
} from 'recharts';
import {
  Users, Shield, Key, Settings, Activity, BarChart3,
  UserPlus, UserMinus, Crown, Building, Globe,
  TrendingUp, Clock, RefreshCw, Download, Upload,
  CheckCircle, XCircle, AlertTriangle, Eye, EyeOff
} from 'lucide-react';
import { apiClient, User, APIKey } from '@/services/apiClient';

interface OrganizationStats {
  total_users: number;
  active_users: number;
  api_calls_today: number;
  storage_used: number;
  bandwidth_used: number;
  subscription_tier: string;
  billing_status: string;
}

interface UserActivity {
  user_id: string;
  username: string;
  last_activity: string;
  api_calls: number;
  strategies_used: number;
  trading_volume: number;
  status: 'active' | 'inactive' | 'suspended';
}

interface SystemMetrics {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  api_response_time: number;
  uptime: number;
  error_rate: number;
}

interface AuditLog {
  id: string;
  user_id: string;
  action: string;
  resource: string;
  timestamp: string;
  ip_address: string;
  status: 'success' | 'failed';
  details: string;
}

const EnterprisePlatform: React.FC = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [organizationUsers, setOrganizationUsers] = useState<User[]>([]);
  const [userActivities, setUserActivities] = useState<UserActivity[]>([]);
  const [apiKeys, setApiKeys] = useState<APIKey[]>([]);
  const [organizationStats, setOrganizationStats] = useState<OrganizationStats | null>(null);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [auditLogs, setAuditLogs] = useState<AuditLog[]>([]);

  const [newUserDialog, setNewUserDialog] = useState(false);
  const [newApiKeyDialog, setNewApiKeyDialog] = useState(false);
  const [newUserData, setNewUserData] = useState({
    username: '',
    email: '',
    role: '',
    subscription_tier: ''
  });
  const [newApiKeyData, setNewApiKeyData] = useState({
    name: '',
    permissions: [] as string[]
  });

  const [selectedTimeframe, setSelectedTimeframe] = useState('24h');
  const [showApiKeySecret, setShowApiKeySecret] = useState<Record<string, boolean>>({});

  const loadEnterpriseData = useCallback(async () => {
    setIsLoading(true);
    try {
      const [user, orgUsers, apiKeysData] = await Promise.all([
        apiClient.getCurrentUser(),
        apiClient.getOrganizationUsers(),
        apiClient.getAPIKeys()
      ]);

      setCurrentUser(user);
      setOrganizationUsers(orgUsers);
      setApiKeys(apiKeysData);

      // Mock data for demonstration - would come from backend
      setOrganizationStats({
        total_users: orgUsers.length,
        active_users: orgUsers.filter(u => u.last_login).length,
        api_calls_today: 15420,
        storage_used: 2.4,
        bandwidth_used: 8.7,
        subscription_tier: 'Enterprise',
        billing_status: 'active'
      });

      setSystemMetrics({
        cpu_usage: 45.2,
        memory_usage: 67.8,
        disk_usage: 23.4,
        api_response_time: 125,
        uptime: 99.98,
        error_rate: 0.02
      });

      // Load real user activities
      const realActivities = await apiClient.getUserActivities(orgUsers.map(u => u.user_id));
      setUserActivities(realActivities.map(activity => ({
        user_id: activity.user_id,
        username: activity.username,
        last_activity: activity.last_activity || new Date().toISOString(),
        api_calls: activity.api_calls || 0,
        strategies_used: activity.strategies_used || 0,
        trading_volume: activity.trading_volume || 0,
        status: activity.status || 'inactive'
      })));

    } catch (error) {
      console.error('Erro ao carregar dados empresariais:', error);
    }
    setIsLoading(false);
  }, []);

  useEffect(() => {
    loadEnterpriseData();
    const interval = setInterval(loadEnterpriseData, 30000);
    return () => clearInterval(interval);
  }, [loadEnterpriseData]);

  const handleCreateUser = async () => {
    try {
      // Implementation would call backend to create user
      console.log('Creating user:', newUserData);
      setNewUserDialog(false);
      setNewUserData({ username: '', email: '', role: '', subscription_tier: '' });
      await loadEnterpriseData();
    } catch (error) {
      console.error('Erro ao criar usuário:', error);
    }
  };

  const handleCreateAPIKey = async () => {
    try {
      await apiClient.createAPIKey(newApiKeyData.name, newApiKeyData.permissions);
      setNewApiKeyDialog(false);
      setNewApiKeyData({ name: '', permissions: [] });
      await loadEnterpriseData();
    } catch (error) {
      console.error('Erro ao criar API key:', error);
    }
  };

  const handleDeleteAPIKey = async (keyId: string) => {
    try {
      await apiClient.deleteAPIKey(keyId);
      await loadEnterpriseData();
    } catch (error) {
      console.error('Erro ao deletar API key:', error);
    }
  };

  const toggleApiKeyVisibility = (keyId: string) => {
    setShowApiKeySecret(prev => ({
      ...prev,
      [keyId]: !prev[keyId]
    }));
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'bg-green-500';
      case 'inactive': return 'bg-yellow-500';
      case 'suspended': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getRoleColor = (role: string) => {
    switch (role.toLowerCase()) {
      case 'admin': return 'bg-red-500';
      case 'manager': return 'bg-blue-500';
      case 'trader': return 'bg-green-500';
      case 'viewer': return 'bg-gray-500';
      default: return 'bg-purple-500';
    }
  };

  const mockApiUsageData = [
    { time: '00:00', calls: 234 },
    { time: '04:00', calls: 156 },
    { time: '08:00', calls: 789 },
    { time: '12:00', calls: 1234 },
    { time: '16:00', calls: 987 },
    { time: '20:00', calls: 456 }
  ];

  const roleDistributionData = organizationUsers.reduce((acc, user) => {
    const role = user.role || 'undefined';
    acc[role] = (acc[role] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const roleChartData = Object.entries(roleDistributionData).map(([role, count]) => ({
    name: role,
    value: count,
    color: getRoleColor(role).replace('bg-', '#')
  }));

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Enterprise Platform</h1>
          <p className="text-muted-foreground">
            Gerenciamento empresarial e administração multi-usuário
          </p>
        </div>
        <div className="flex gap-2">
          <Button onClick={loadEnterpriseData} disabled={isLoading}>
            <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Atualizar
          </Button>
          <Button variant="outline">
            <Download className="h-4 w-4 mr-2" />
            Exportar Relatório
          </Button>
        </div>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-6 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Total Usuários</p>
                <p className="text-2xl font-bold">
                  {organizationStats?.total_users || 0}
                </p>
              </div>
              <Users className="h-8 w-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Usuários Ativos</p>
                <p className="text-2xl font-bold">
                  {organizationStats?.active_users || 0}
                </p>
              </div>
              <Activity className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">API Calls Hoje</p>
                <p className="text-2xl font-bold">
                  {organizationStats?.api_calls_today?.toLocaleString() || '0'}
                </p>
              </div>
              <BarChart3 className="h-8 w-8 text-purple-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Storage (GB)</p>
                <p className="text-2xl font-bold">
                  {organizationStats?.storage_used?.toFixed(1) || '0.0'}
                </p>
              </div>
              <Building className="h-8 w-8 text-orange-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">Uptime</p>
                <p className="text-2xl font-bold">
                  {systemMetrics?.uptime?.toFixed(2) || '0.00'}%
                </p>
              </div>
              <TrendingUp className="h-8 w-8 text-green-600" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">API Keys</p>
                <p className="text-2xl font-bold">
                  {apiKeys.length}
                </p>
              </div>
              <Key className="h-8 w-8 text-yellow-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="users" className="space-y-4">
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="users">Usuários</TabsTrigger>
          <TabsTrigger value="apikeys">API Keys</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
          <TabsTrigger value="system">Sistema</TabsTrigger>
          <TabsTrigger value="security">Segurança</TabsTrigger>
          <TabsTrigger value="billing">Billing</TabsTrigger>
        </TabsList>

        <TabsContent value="users" className="space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold">Gerenciamento de Usuários</h3>
            <Dialog open={newUserDialog} onOpenChange={setNewUserDialog}>
              <DialogTrigger asChild>
                <Button>
                  <UserPlus className="h-4 w-4 mr-2" />
                  Novo Usuário
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Criar Novo Usuário</DialogTitle>
                  <DialogDescription>
                    Adicione um novo usuário à organização
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4">
                  <div>
                    <Label>Nome de Usuário</Label>
                    <Input
                      value={newUserData.username}
                      onChange={(e) => setNewUserData({...newUserData, username: e.target.value})}
                    />
                  </div>
                  <div>
                    <Label>Email</Label>
                    <Input
                      type="email"
                      value={newUserData.email}
                      onChange={(e) => setNewUserData({...newUserData, email: e.target.value})}
                    />
                  </div>
                  <div>
                    <Label>Role</Label>
                    <Select
                      value={newUserData.role}
                      onValueChange={(value) => setNewUserData({...newUserData, role: value})}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Selecione o role" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="admin">Admin</SelectItem>
                        <SelectItem value="manager">Manager</SelectItem>
                        <SelectItem value="trader">Trader</SelectItem>
                        <SelectItem value="viewer">Viewer</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label>Subscription Tier</Label>
                    <Select
                      value={newUserData.subscription_tier}
                      onValueChange={(value) => setNewUserData({...newUserData, subscription_tier: value})}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Selecione o plano" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="basic">Basic</SelectItem>
                        <SelectItem value="professional">Professional</SelectItem>
                        <SelectItem value="enterprise">Enterprise</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <Button onClick={handleCreateUser} className="w-full">
                    Criar Usuário
                  </Button>
                </div>
              </DialogContent>
            </Dialog>
          </div>

          <Card>
            <CardContent className="p-0">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Usuário</TableHead>
                    <TableHead>Email</TableHead>
                    <TableHead>Role</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Último Login</TableHead>
                    <TableHead>Ações</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {organizationUsers.map((user) => {
                    const activity = userActivities.find(a => a.user_id === user.user_id);
                    return (
                      <TableRow key={user.user_id}>
                        <TableCell className="font-medium">
                          <div className="flex items-center space-x-2">
                            <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white text-sm">
                              {user.username.charAt(0).toUpperCase()}
                            </div>
                            <span>{user.username}</span>
                          </div>
                        </TableCell>
                        <TableCell>{user.email}</TableCell>
                        <TableCell>
                          <Badge className={`text-white ${getRoleColor(user.role)}`}>
                            {user.role}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <div className="flex items-center space-x-2">
                            <div className={`w-2 h-2 rounded-full ${getStatusColor(activity?.status || 'inactive')}`}></div>
                            <span className="capitalize">{activity?.status || 'inactive'}</span>
                          </div>
                        </TableCell>
                        <TableCell>
                          {user.last_login ? new Date(user.last_login).toLocaleDateString() : 'Nunca'}
                        </TableCell>
                        <TableCell>
                          <div className="flex space-x-2">
                            <Button variant="outline" size="sm">
                              <Settings className="h-4 w-4" />
                            </Button>
                            <Button variant="outline" size="sm">
                              <UserMinus className="h-4 w-4" />
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="apikeys" className="space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold">Gerenciamento de API Keys</h3>
            <Dialog open={newApiKeyDialog} onOpenChange={setNewApiKeyDialog}>
              <DialogTrigger asChild>
                <Button>
                  <Key className="h-4 w-4 mr-2" />
                  Nova API Key
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Criar Nova API Key</DialogTitle>
                  <DialogDescription>
                    Crie uma nova chave de API com permissões específicas
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4">
                  <div>
                    <Label>Nome da API Key</Label>
                    <Input
                      value={newApiKeyData.name}
                      onChange={(e) => setNewApiKeyData({...newApiKeyData, name: e.target.value})}
                      placeholder="Ex: Production API Key"
                    />
                  </div>
                  <div>
                    <Label>Permissões</Label>
                    <div className="space-y-2 mt-2">
                      {['read', 'write', 'trading', 'admin'].map((permission) => (
                        <div key={permission} className="flex items-center space-x-2">
                          <Switch
                            checked={newApiKeyData.permissions.includes(permission)}
                            onCheckedChange={(checked) => {
                              if (checked) {
                                setNewApiKeyData({
                                  ...newApiKeyData,
                                  permissions: [...newApiKeyData.permissions, permission]
                                });
                              } else {
                                setNewApiKeyData({
                                  ...newApiKeyData,
                                  permissions: newApiKeyData.permissions.filter(p => p !== permission)
                                });
                              }
                            }}
                          />
                          <Label className="capitalize">{permission}</Label>
                        </div>
                      ))}
                    </div>
                  </div>
                  <Button onClick={handleCreateAPIKey} className="w-full">
                    Criar API Key
                  </Button>
                </div>
              </DialogContent>
            </Dialog>
          </div>

          <div className="grid gap-4">
            {apiKeys.map((apiKey) => (
              <Card key={apiKey.key_id}>
                <CardContent className="p-4">
                  <div className="flex justify-between items-start">
                    <div className="space-y-2">
                      <div className="flex items-center space-x-2">
                        <h4 className="font-semibold">{apiKey.name}</h4>
                        <Badge variant="outline">
                          {apiKey.permissions.length} permissões
                        </Badge>
                      </div>

                      <div className="flex items-center space-x-2">
                        <code className="bg-muted px-2 py-1 rounded text-sm">
                          {showApiKeySecret[apiKey.key_id]
                            ? apiKey.api_key
                            : apiKey.api_key.substring(0, 8) + '...' + apiKey.api_key.slice(-4)
                          }
                        </code>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => toggleApiKeyVisibility(apiKey.key_id)}
                        >
                          {showApiKeySecret[apiKey.key_id] ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                        </Button>
                      </div>

                      <div className="flex space-x-4 text-sm text-muted-foreground">
                        <span>Criada: {new Date(apiKey.created_at).toLocaleDateString()}</span>
                        <span>Último uso: {apiKey.last_used ? new Date(apiKey.last_used).toLocaleDateString() : 'Nunca'}</span>
                        <span>Rate limit: {apiKey.rate_limit}/min</span>
                      </div>

                      <div className="flex flex-wrap gap-1">
                        {apiKey.permissions.map((permission) => (
                          <Badge key={permission} variant="secondary" className="text-xs">
                            {permission}
                          </Badge>
                        ))}
                      </div>
                    </div>

                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={() => handleDeleteAPIKey(apiKey.key_id)}
                    >
                      Deletar
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Uso da API (24h)</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={mockApiUsageData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <Tooltip />
                    <Area type="monotone" dataKey="calls" stroke="#3B82F6" fill="#3B82F6" fillOpacity={0.3} />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Distribuição por Role</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={roleChartData}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={120}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {roleChartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Atividade de Usuários</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Usuário</TableHead>
                    <TableHead>API Calls</TableHead>
                    <TableHead>Estratégias</TableHead>
                    <TableHead>Volume Trading</TableHead>
                    <TableHead>Última Atividade</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {userActivities.slice(0, 10).map((activity) => (
                    <TableRow key={activity.user_id}>
                      <TableCell>{activity.username}</TableCell>
                      <TableCell>{activity.api_calls.toLocaleString()}</TableCell>
                      <TableCell>{activity.strategies_used}</TableCell>
                      <TableCell>${activity.trading_volume.toFixed(2)}</TableCell>
                      <TableCell>{new Date(activity.last_activity).toLocaleString()}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="system" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>CPU Usage</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">{systemMetrics?.cpu_usage?.toFixed(1) || '0.0'}%</div>
                <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full"
                    style={{ width: `${systemMetrics?.cpu_usage || 0}%` }}
                  ></div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Memory Usage</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">{systemMetrics?.memory_usage?.toFixed(1) || '0.0'}%</div>
                <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                  <div
                    className="bg-green-600 h-2 rounded-full"
                    style={{ width: `${systemMetrics?.memory_usage || 0}%` }}
                  ></div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Disk Usage</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">{systemMetrics?.disk_usage?.toFixed(1) || '0.0'}%</div>
                <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                  <div
                    className="bg-yellow-600 h-2 rounded-full"
                    style={{ width: `${systemMetrics?.disk_usage || 0}%` }}
                  ></div>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">API Response Time</p>
                    <p className="text-2xl font-bold">{systemMetrics?.api_response_time || 0}ms</p>
                  </div>
                  <Clock className="h-8 w-8 text-blue-500" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Uptime</p>
                    <p className="text-2xl font-bold">{systemMetrics?.uptime?.toFixed(2) || '0.00'}%</p>
                  </div>
                  <TrendingUp className="h-8 w-8 text-green-500" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Error Rate</p>
                    <p className="text-2xl font-bold">{systemMetrics?.error_rate?.toFixed(2) || '0.00'}%</p>
                  </div>
                  <AlertTriangle className="h-8 w-8 text-red-500" />
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="security" className="space-y-4">
          <Alert>
            <Shield className="h-4 w-4" />
            <AlertDescription>
              Configurações de segurança e auditoria estão em desenvolvimento
            </AlertDescription>
          </Alert>
        </TabsContent>

        <TabsContent value="billing" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Informações de Cobrança</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex justify-between">
                  <span>Plano Atual:</span>
                  <Badge className="bg-purple-500 text-white">
                    {organizationStats?.subscription_tier || 'Basic'}
                  </Badge>
                </div>
                <div className="flex justify-between">
                  <span>Status:</span>
                  <Badge className="bg-green-500 text-white">
                    {organizationStats?.billing_status || 'Active'}
                  </Badge>
                </div>
                <div className="flex justify-between">
                  <span>Storage Usado:</span>
                  <span>{organizationStats?.storage_used?.toFixed(1) || '0.0'} GB</span>
                </div>
                <div className="flex justify-between">
                  <span>Bandwidth Usado:</span>
                  <span>{organizationStats?.bandwidth_used?.toFixed(1) || '0.0'} GB</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default EnterprisePlatform;