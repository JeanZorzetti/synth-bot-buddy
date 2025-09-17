/**
 * 📊 API ACTIVITY LOGS VIEWER
 * Real-time API activity monitoring with detailed logs and analytics
 */

import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { apiClient } from '../../services/apiClient';
import { Button } from '../ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Input } from '../ui/input';
import { Badge } from '../ui/badge';
import { Alert, AlertDescription } from '../ui/alert';
import { Progress } from '../ui/progress';
import {
  Search,
  Filter,
  Download,
  RefreshCw,
  Eye,
  Calendar,
  Clock,
  Globe,
  Server,
  AlertTriangle,
  CheckCircle,
  X,
  BarChart3,
  TrendingUp,
  TrendingDown,
  Activity,
  MapPin,
  User,
  Code,
  Database,
  Zap,
  Shield,
  ExternalLink,
  PlayCircle,
  PauseCircle,
  FileText,
  Terminal,
  Hash,
  Timer,
  AlertCircle
} from 'lucide-react';

interface ApiLogEntry {
  id: string;
  timestamp: string;
  method: string;
  endpoint: string;
  statusCode: number;
  responseTime: number;
  userAgent: string;
  ipAddress: string;
  location?: {
    country: string;
    city: string;
    region: string;
  };
  apiKey: {
    id: string;
    name: string;
    environment: string;
  };
  requestSize: number;
  responseSize: number;
  errorMessage?: string;
  requestId: string;
  userId?: string;
  rateLimited: boolean;
  cached: boolean;
}

interface ApiMetrics {
  totalRequests: number;
  successRate: number;
  averageResponseTime: number;
  peakRps: number;
  topEndpoints: {
    endpoint: string;
    count: number;
    percentage: number;
  }[];
  statusCodes: {
    code: number;
    count: number;
    percentage: number;
  }[];
  responseTimeDistribution: {
    range: string;
    count: number;
    percentage: number;
  }[];
  hourlyStats: {
    hour: string;
    requests: number;
    errors: number;
    avgResponseTime: number;
  }[];
}

interface ApiActivityLogsProps {
  className?: string;
  apiKeyId?: string;
}

export function ApiActivityLogs({ className = "", apiKeyId }: ApiActivityLogsProps) {
  const { user } = useAuth();

  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isRealTime, setIsRealTime] = useState(false);

  const [logs, setLogs] = useState<ApiLogEntry[]>([]);
  const [metrics, setMetrics] = useState<ApiMetrics | null>(null);
  const [selectedLog, setSelectedLog] = useState<ApiLogEntry | null>(null);

  // Filters
  const [filters, setFilters] = useState({
    search: '',
    method: 'all',
    status: 'all',
    apiKey: apiKeyId || 'all',
    timeRange: '24h',
    endpoint: '',
    ipAddress: '',
    minResponseTime: '',
    maxResponseTime: '',
  });

  // Pagination
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize] = useState(50);
  const [totalLogs, setTotalLogs] = useState(0);

  useEffect(() => {
    loadLogs();
    loadMetrics();
  }, [filters, currentPage]);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isRealTime) {
      interval = setInterval(() => {
        loadLogs(false);
      }, 5000);
    }
    return () => clearInterval(interval);
  }, [isRealTime, filters]);

  const loadLogs = async (showLoading = true) => {
    try {
      if (showLoading) setIsLoading(true);
      setError(null);

      const queryParams = new URLSearchParams({
        page: currentPage.toString(),
        limit: pageSize.toString(),
        ...Object.entries(filters).reduce((acc, [key, value]) => {
          if (value && value !== 'all') {
            acc[key] = value;
          }
          return acc;
        }, {} as Record<string, string>)
      });

      const response = await apiClient.get<{
        logs: ApiLogEntry[];
        total: number;
        page: number;
        totalPages: number;
      }>(`/api-keys/logs?${queryParams}`);

      setLogs(response.logs);
      setTotalLogs(response.total);
    } catch (error: any) {
      setError(error.message || 'Failed to load API logs');
    } finally {
      setIsLoading(false);
    }
  };

  const loadMetrics = async () => {
    try {
      const queryParams = new URLSearchParams({
        timeRange: filters.timeRange,
        ...(filters.apiKey !== 'all' && { apiKey: filters.apiKey })
      });

      const metricsData = await apiClient.get<ApiMetrics>(`/api-keys/metrics?${queryParams}`);
      setMetrics(metricsData);
    } catch (error) {
      console.error('Failed to load metrics:', error);
    }
  };

  const exportLogs = async () => {
    try {
      const queryParams = new URLSearchParams({
        format: 'csv',
        ...Object.entries(filters).reduce((acc, [key, value]) => {
          if (value && value !== 'all') {
            acc[key] = value;
          }
          return acc;
        }, {} as Record<string, string>)
      });

      const response = await apiClient.get(`/api-keys/logs/export?${queryParams}`, {
        responseType: 'blob'
      });

      const url = window.URL.createObjectURL(new Blob([response]));
      const link = document.createElement('a');
      link.href = url;
      link.download = `api-logs-${new Date().toISOString().split('T')[0]}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error: any) {
      setError(error.message || 'Failed to export logs');
    }
  };

  const clearFilters = () => {
    setFilters({
      search: '',
      method: 'all',
      status: 'all',
      apiKey: apiKeyId || 'all',
      timeRange: '24h',
      endpoint: '',
      ipAddress: '',
      minResponseTime: '',
      maxResponseTime: '',
    });
    setCurrentPage(1);
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getStatusBadge = (statusCode: number) => {
    let variant = 'bg-gray-100 text-gray-800';
    let icon = CheckCircle;

    if (statusCode >= 200 && statusCode < 300) {
      variant = 'bg-green-100 text-green-800';
      icon = CheckCircle;
    } else if (statusCode >= 300 && statusCode < 400) {
      variant = 'bg-blue-100 text-blue-800';
      icon = ExternalLink;
    } else if (statusCode >= 400 && statusCode < 500) {
      variant = 'bg-orange-100 text-orange-800';
      icon = AlertTriangle;
    } else if (statusCode >= 500) {
      variant = 'bg-red-100 text-red-800';
      icon = X;
    }

    const Icon = icon;

    return (
      <Badge className={variant}>
        <Icon className="h-3 w-3 mr-1" />
        {statusCode}
      </Badge>
    );
  };

  const getMethodBadge = (method: string) => {
    const colors = {
      GET: 'bg-blue-100 text-blue-800',
      POST: 'bg-green-100 text-green-800',
      PUT: 'bg-yellow-100 text-yellow-800',
      DELETE: 'bg-red-100 text-red-800',
      PATCH: 'bg-purple-100 text-purple-800',
    };

    return (
      <Badge className={colors[method as keyof typeof colors] || 'bg-gray-100 text-gray-800'}>
        {method}
      </Badge>
    );
  };

  const getResponseTimeColor = (responseTime: number) => {
    if (responseTime < 100) return 'text-green-600';
    if (responseTime < 500) return 'text-yellow-600';
    if (responseTime < 1000) return 'text-orange-600';
    return 'text-red-600';
  };

  const renderMetrics = () => {
    if (!metrics) return null;

    return (
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-blue-100 rounded-lg">
                <BarChart3 className="h-5 w-5 text-blue-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">{metrics.totalRequests.toLocaleString()}</p>
                <p className="text-sm text-gray-500">Total Requests</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-green-100 rounded-lg">
                <CheckCircle className="h-5 w-5 text-green-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">{(metrics.successRate * 100).toFixed(1)}%</p>
                <p className="text-sm text-gray-500">Success Rate</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-purple-100 rounded-lg">
                <Timer className="h-5 w-5 text-purple-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">{metrics.averageResponseTime}ms</p>
                <p className="text-sm text-gray-500">Avg Response</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-orange-100 rounded-lg">
                <Zap className="h-5 w-5 text-orange-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">{metrics.peakRps}</p>
                <p className="text-sm text-gray-500">Peak RPS</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  };

  const renderFilters = () => (
    <Card className="mb-6">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">Filters</CardTitle>
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsRealTime(!isRealTime)}
              className={isRealTime ? 'bg-green-50 border-green-200' : ''}
            >
              {isRealTime ? (
                <>
                  <PauseCircle className="h-4 w-4 mr-2" />
                  Stop Real-time
                </>
              ) : (
                <>
                  <PlayCircle className="h-4 w-4 mr-2" />
                  Real-time
                </>
              )}
            </Button>
            <Button variant="outline" size="sm" onClick={clearFilters}>
              Clear Filters
            </Button>
            <Button variant="outline" size="sm" onClick={exportLogs}>
              <Download className="h-4 w-4 mr-2" />
              Export
            </Button>
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="relative">
            <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
            <Input
              placeholder="Search logs..."
              value={filters.search}
              onChange={(e) => setFilters(prev => ({ ...prev, search: e.target.value }))}
              className="pl-10"
            />
          </div>

          <select
            value={filters.method}
            onChange={(e) => setFilters(prev => ({ ...prev, method: e.target.value }))}
            className="px-3 py-2 border rounded-md bg-white"
          >
            <option value="all">All Methods</option>
            <option value="GET">GET</option>
            <option value="POST">POST</option>
            <option value="PUT">PUT</option>
            <option value="DELETE">DELETE</option>
            <option value="PATCH">PATCH</option>
          </select>

          <select
            value={filters.status}
            onChange={(e) => setFilters(prev => ({ ...prev, status: e.target.value }))}
            className="px-3 py-2 border rounded-md bg-white"
          >
            <option value="all">All Status</option>
            <option value="2xx">2xx Success</option>
            <option value="3xx">3xx Redirect</option>
            <option value="4xx">4xx Client Error</option>
            <option value="5xx">5xx Server Error</option>
          </select>

          <select
            value={filters.timeRange}
            onChange={(e) => setFilters(prev => ({ ...prev, timeRange: e.target.value }))}
            className="px-3 py-2 border rounded-md bg-white"
          >
            <option value="1h">Last Hour</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
          </select>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Input
            placeholder="Filter by endpoint..."
            value={filters.endpoint}
            onChange={(e) => setFilters(prev => ({ ...prev, endpoint: e.target.value }))}
          />

          <Input
            placeholder="Filter by IP address..."
            value={filters.ipAddress}
            onChange={(e) => setFilters(prev => ({ ...prev, ipAddress: e.target.value }))}
          />

          <div className="grid grid-cols-2 gap-2">
            <Input
              placeholder="Min response time (ms)"
              value={filters.minResponseTime}
              onChange={(e) => setFilters(prev => ({ ...prev, minResponseTime: e.target.value }))}
              type="number"
            />
            <Input
              placeholder="Max response time (ms)"
              value={filters.maxResponseTime}
              onChange={(e) => setFilters(prev => ({ ...prev, maxResponseTime: e.target.value }))}
              type="number"
            />
          </div>
        </div>

        <div className="flex items-center justify-between text-sm text-gray-500">
          <span>
            Showing {logs.length} of {totalLogs.toLocaleString()} logs
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={() => loadLogs()}
            disabled={isLoading}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </CardContent>
    </Card>
  );

  const renderLogEntry = (log: ApiLogEntry) => (
    <Card key={log.id} className="hover:shadow-md transition-shadow cursor-pointer">
      <CardContent className="p-4" onClick={() => setSelectedLog(log)}>
        <div className="flex items-start justify-between">
          <div className="flex-1 space-y-2">
            <div className="flex items-center space-x-3">
              <span className="text-sm text-gray-500 font-mono">
                {formatTimestamp(log.timestamp)}
              </span>
              {getMethodBadge(log.method)}
              {getStatusBadge(log.statusCode)}

              {log.rateLimited && (
                <Badge className="bg-red-100 text-red-800">
                  <Shield className="h-3 w-3 mr-1" />
                  Rate Limited
                </Badge>
              )}

              {log.cached && (
                <Badge className="bg-blue-100 text-blue-800">
                  <Database className="h-3 w-3 mr-1" />
                  Cached
                </Badge>
              )}
            </div>

            <div className="font-mono text-sm">
              <span className="font-medium">{log.method}</span> {log.endpoint}
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div className="flex items-center space-x-2">
                <Timer className="h-4 w-4 text-gray-400" />
                <span className={getResponseTimeColor(log.responseTime)}>
                  {log.responseTime}ms
                </span>
              </div>

              <div className="flex items-center space-x-2">
                <Globe className="h-4 w-4 text-gray-400" />
                <span className="truncate">{log.ipAddress}</span>
              </div>

              <div className="flex items-center space-x-2">
                <Hash className="h-4 w-4 text-gray-400" />
                <span className="truncate">{log.apiKey.name}</span>
              </div>

              <div className="flex items-center space-x-2">
                <Server className="h-4 w-4 text-gray-400" />
                <span>{formatBytes(log.responseSize)}</span>
              </div>
            </div>

            {log.errorMessage && (
              <div className="text-sm text-red-600 bg-red-50 p-2 rounded">
                <AlertTriangle className="h-4 w-4 inline mr-2" />
                {log.errorMessage}
              </div>
            )}
          </div>

          <Button variant="ghost" size="sm">
            <Eye className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  );

  const renderLogDetail = () => {
    if (!selectedLog) return null;

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <Card className="max-w-4xl w-full mx-4 max-h-[90vh] overflow-y-auto">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Request Details</CardTitle>
              <Button variant="ghost" onClick={() => setSelectedLog(null)}>
                <X className="h-4 w-4" />
              </Button>
            </div>
          </CardHeader>

          <CardContent className="space-y-6">
            {/* Request Overview */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p className="text-sm text-gray-500">Method</p>
                <p className="font-mono">{selectedLog.method}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Status</p>
                <div>{getStatusBadge(selectedLog.statusCode)}</div>
              </div>
              <div>
                <p className="text-sm text-gray-500">Response Time</p>
                <p className={`font-mono ${getResponseTimeColor(selectedLog.responseTime)}`}>
                  {selectedLog.responseTime}ms
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Request ID</p>
                <p className="font-mono text-xs">{selectedLog.requestId}</p>
              </div>
            </div>

            {/* Request Details */}
            <div className="space-y-4">
              <div>
                <p className="text-sm font-medium text-gray-700 mb-2">Endpoint</p>
                <div className="p-3 bg-gray-50 rounded font-mono text-sm">
                  {selectedLog.method} {selectedLog.endpoint}
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <p className="text-sm font-medium text-gray-700 mb-2">Client Information</p>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>IP Address:</span>
                      <span className="font-mono">{selectedLog.ipAddress}</span>
                    </div>
                    {selectedLog.location && (
                      <div className="flex justify-between">
                        <span>Location:</span>
                        <span>{selectedLog.location.city}, {selectedLog.location.country}</span>
                      </div>
                    )}
                    <div className="flex justify-between">
                      <span>User Agent:</span>
                      <span className="font-mono text-xs truncate max-w-48" title={selectedLog.userAgent}>
                        {selectedLog.userAgent}
                      </span>
                    </div>
                  </div>
                </div>

                <div>
                  <p className="text-sm font-medium text-gray-700 mb-2">API Key Information</p>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span>Key Name:</span>
                      <span>{selectedLog.apiKey.name}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Environment:</span>
                      <span className="capitalize">{selectedLog.apiKey.environment}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Key ID:</span>
                      <span className="font-mono text-xs">{selectedLog.apiKey.id}</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <p className="text-sm font-medium text-gray-700 mb-2">Request Size</p>
                  <p className="text-lg font-semibold">{formatBytes(selectedLog.requestSize)}</p>
                </div>

                <div>
                  <p className="text-sm font-medium text-gray-700 mb-2">Response Size</p>
                  <p className="text-lg font-semibold">{formatBytes(selectedLog.responseSize)}</p>
                </div>
              </div>

              {selectedLog.errorMessage && (
                <div>
                  <p className="text-sm font-medium text-gray-700 mb-2">Error Details</p>
                  <div className="p-3 bg-red-50 border border-red-200 rounded text-red-800">
                    {selectedLog.errorMessage}
                  </div>
                </div>
              )}

              <div className="flex space-x-2">
                <Badge variant="outline" className={selectedLog.rateLimited ? 'border-red-200 text-red-800' : 'text-gray-600'}>
                  {selectedLog.rateLimited ? 'Rate Limited' : 'Not Rate Limited'}
                </Badge>
                <Badge variant="outline" className={selectedLog.cached ? 'border-blue-200 text-blue-800' : 'text-gray-600'}>
                  {selectedLog.cached ? 'Cached Response' : 'Fresh Response'}
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  };

  if (!user) {
    return (
      <Alert>
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          Please log in to view API activity logs.
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className={`max-w-7xl mx-auto space-y-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">API Activity Logs</h1>
          <p className="text-gray-600">Monitor real-time API usage and performance</p>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Real-time Status */}
      {isRealTime && (
        <Alert className="border-green-200 bg-green-50">
          <Activity className="h-4 w-4 text-green-600" />
          <AlertDescription className="text-green-800">
            Real-time monitoring is active. Logs are updated every 5 seconds.
          </AlertDescription>
        </Alert>
      )}

      {/* Metrics */}
      {renderMetrics()}

      {/* Filters */}
      {renderFilters()}

      {/* Loading State */}
      {isLoading && logs.length === 0 ? (
        <div className="space-y-4">
          {[1, 2, 3, 4, 5].map(i => (
            <Card key={i}>
              <CardContent className="p-4">
                <div className="animate-pulse space-y-3">
                  <div className="h-4 bg-gray-200 rounded w-1/4"></div>
                  <div className="h-6 bg-gray-200 rounded w-3/4"></div>
                  <div className="h-4 bg-gray-200 rounded w-1/2"></div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      ) : (
        /* Logs List */
        <div className="space-y-4">
          {logs.map(renderLogEntry)}

          {logs.length === 0 && (
            <Card>
              <CardContent className="p-8 text-center">
                <Activity className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-500">No API logs found for the selected filters</p>
              </CardContent>
            </Card>
          )}

          {/* Pagination */}
          {totalLogs > pageSize && (
            <div className="flex items-center justify-between">
              <p className="text-sm text-gray-500">
                Showing {((currentPage - 1) * pageSize) + 1} to {Math.min(currentPage * pageSize, totalLogs)} of {totalLogs.toLocaleString()} logs
              </p>

              <div className="flex space-x-2">
                <Button
                  variant="outline"
                  onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                  disabled={currentPage === 1}
                >
                  Previous
                </Button>
                <Button
                  variant="outline"
                  onClick={() => setCurrentPage(prev => prev + 1)}
                  disabled={currentPage * pageSize >= totalLogs}
                >
                  Next
                </Button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Log Detail Modal */}
      {renderLogDetail()}
    </div>
  );
}