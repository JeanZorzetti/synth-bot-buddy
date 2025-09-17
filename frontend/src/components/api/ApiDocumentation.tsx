/**
 * 📚 INTERACTIVE API DOCUMENTATION
 * Complete API documentation with live testing and code examples
 */

import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { apiClient } from '../../services/apiClient';
import { Button } from '../ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Input } from '../ui/input';
import { Label } from '../ui/label';
import { Badge } from '../ui/badge';
import { Alert, AlertDescription } from '../ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { Switch } from '../ui/switch';
import {
  BookOpen,
  Code,
  Play,
  Copy,
  Download,
  Search,
  Filter,
  ExternalLink,
  CheckCircle,
  AlertTriangle,
  Info,
  Zap,
  Database,
  Globe,
  Lock,
  Key,
  Users,
  BarChart3,
  Settings,
  FileText,
  Terminal,
  Lightbulb,
  ArrowRight,
  ChevronDown,
  ChevronRight,
  Hash,
  Type,
  List,
  Server,
  Shield,
  Clock,
  RefreshCw,
  Eye,
  EyeOff,
  AlertCircle
} from 'lucide-react';

interface ApiEndpoint {
  id: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH';
  path: string;
  summary: string;
  description: string;
  category: string;
  tags: string[];
  deprecated?: boolean;
  authentication: {
    required: boolean;
    type: 'api_key' | 'bearer' | 'basic';
    scopes?: string[];
  };
  parameters: ApiParameter[];
  requestBody?: {
    required: boolean;
    contentType: string;
    schema: any;
    examples: { [key: string]: any };
  };
  responses: {
    [statusCode: string]: {
      description: string;
      schema?: any;
      examples?: { [key: string]: any };
    };
  };
  rateLimit?: {
    requests: number;
    window: string;
  };
  examples: {
    curl: string;
    javascript: string;
    python: string;
    php: string;
    java: string;
  };
}

interface ApiParameter {
  name: string;
  in: 'query' | 'path' | 'header' | 'body';
  type: string;
  required: boolean;
  description: string;
  format?: string;
  example?: any;
  enum?: string[];
  default?: any;
  minimum?: number;
  maximum?: number;
}

interface ApiCategory {
  id: string;
  name: string;
  description: string;
  endpoints: string[];
}

interface TestRequest {
  endpoint: string;
  method: string;
  parameters: { [key: string]: any };
  headers: { [key: string]: string };
  body?: string;
}

interface TestResponse {
  status: number;
  statusText: string;
  headers: { [key: string]: string };
  data: any;
  duration: number;
  size: number;
}

interface ApiDocumentationProps {
  className?: string;
}

export function ApiDocumentation({ className = "" }: ApiDocumentationProps) {
  const { user } = useAuth();

  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [endpoints, setEndpoints] = useState<ApiEndpoint[]>([]);
  const [categories, setCategories] = useState<ApiCategory[]>([]);
  const [selectedEndpoint, setSelectedEndpoint] = useState<ApiEndpoint | null>(null);
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set());

  // Search and filters
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedMethod, setSelectedMethod] = useState<string>('all');
  const [showDeprecated, setShowDeprecated] = useState(false);

  // Testing interface
  const [isTestingMode, setIsTestingMode] = useState(false);
  const [testRequest, setTestRequest] = useState<TestRequest>({
    endpoint: '',
    method: 'GET',
    parameters: {},
    headers: { 'Authorization': 'Bearer YOUR_API_KEY' },
  });
  const [testResponse, setTestResponse] = useState<TestResponse | null>(null);
  const [isTesting, setIsTesting] = useState(false);

  // Code examples
  const [selectedLanguage, setSelectedLanguage] = useState<'curl' | 'javascript' | 'python' | 'php' | 'java'>('curl');
  const [showApiKey, setShowApiKey] = useState(false);

  useEffect(() => {
    loadApiDocumentation();
  }, []);

  useEffect(() => {
    if (endpoints.length > 0 && !selectedEndpoint) {
      setSelectedEndpoint(endpoints[0]);
    }
  }, [endpoints]);

  useEffect(() => {
    if (selectedEndpoint) {
      updateTestRequest();
    }
  }, [selectedEndpoint]);

  const loadApiDocumentation = async () => {
    try {
      setIsLoading(true);
      setError(null);

      const [endpointsData, categoriesData] = await Promise.all([
        apiClient.get<ApiEndpoint[]>('/api-docs/endpoints'),
        apiClient.get<ApiCategory[]>('/api-docs/categories')
      ]);

      setEndpoints(endpointsData);
      setCategories(categoriesData);

      // Expand first category by default
      if (categoriesData.length > 0) {
        setExpandedCategories(new Set([categoriesData[0].id]));
      }
    } catch (error: any) {
      setError(error.message || 'Failed to load API documentation');
    } finally {
      setIsLoading(false);
    }
  };

  const updateTestRequest = () => {
    if (!selectedEndpoint) return;

    const pathParams = selectedEndpoint.parameters
      .filter(p => p.in === 'path')
      .reduce((acc, param) => {
        acc[param.name] = param.example || '';
        return acc;
      }, {} as { [key: string]: any });

    const queryParams = selectedEndpoint.parameters
      .filter(p => p.in === 'query')
      .reduce((acc, param) => {
        if (param.required) {
          acc[param.name] = param.example || param.default || '';
        }
        return acc;
      }, {} as { [key: string]: any });

    setTestRequest({
      endpoint: selectedEndpoint.path,
      method: selectedEndpoint.method,
      parameters: { ...pathParams, ...queryParams },
      headers: { 'Authorization': 'Bearer YOUR_API_KEY' },
      body: selectedEndpoint.requestBody?.examples?.default ?
        JSON.stringify(selectedEndpoint.requestBody.examples.default, null, 2) : undefined,
    });
  };

  const executeTest = async () => {
    if (!selectedEndpoint) return;

    try {
      setIsTesting(true);
      setError(null);

      const startTime = Date.now();

      // Build the URL with path and query parameters
      let url = testRequest.endpoint;

      // Replace path parameters
      Object.entries(testRequest.parameters).forEach(([key, value]) => {
        if (url.includes(`{${key}}`)) {
          url = url.replace(`{${key}}`, encodeURIComponent(value));
        }
      });

      // Add query parameters
      const queryParams = new URLSearchParams();
      Object.entries(testRequest.parameters).forEach(([key, value]) => {
        if (!testRequest.endpoint.includes(`{${key}}`) && value !== '') {
          queryParams.append(key, value);
        }
      });

      if (queryParams.toString()) {
        url += '?' + queryParams.toString();
      }

      const requestOptions: any = {
        method: testRequest.method,
        headers: testRequest.headers,
      };

      if (testRequest.body && ['POST', 'PUT', 'PATCH'].includes(testRequest.method)) {
        requestOptions.body = testRequest.body;
        requestOptions.headers['Content-Type'] = 'application/json';
      }

      const response = await fetch(url, requestOptions);
      const endTime = Date.now();

      const responseData = await response.text();
      let parsedData;
      try {
        parsedData = JSON.parse(responseData);
      } catch {
        parsedData = responseData;
      }

      const responseHeaders: { [key: string]: string } = {};
      response.headers.forEach((value, key) => {
        responseHeaders[key] = value;
      });

      setTestResponse({
        status: response.status,
        statusText: response.statusText,
        headers: responseHeaders,
        data: parsedData,
        duration: endTime - startTime,
        size: new Blob([responseData]).size,
      });

    } catch (error: any) {
      setError(error.message || 'Failed to execute test request');
    } finally {
      setIsTesting(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const generateCodeExample = (endpoint: ApiEndpoint, language: string) => {
    const apiKey = showApiKey ? 'your_actual_api_key_here' : 'YOUR_API_KEY';
    const examples = endpoint.examples;

    if (language === 'curl') {
      return examples.curl.replace('YOUR_API_KEY', apiKey);
    } else if (language === 'javascript') {
      return examples.javascript.replace('YOUR_API_KEY', apiKey);
    } else if (language === 'python') {
      return examples.python.replace('YOUR_API_KEY', apiKey);
    } else if (language === 'php') {
      return examples.php.replace('YOUR_API_KEY', apiKey);
    } else if (language === 'java') {
      return examples.java.replace('YOUR_API_KEY', apiKey);
    }

    return examples.curl;
  };

  const filterEndpoints = () => {
    let filtered = [...endpoints];

    if (searchTerm) {
      filtered = filtered.filter(endpoint =>
        endpoint.summary.toLowerCase().includes(searchTerm.toLowerCase()) ||
        endpoint.path.toLowerCase().includes(searchTerm.toLowerCase()) ||
        endpoint.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
        endpoint.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()))
      );
    }

    if (selectedCategory !== 'all') {
      filtered = filtered.filter(endpoint => endpoint.category === selectedCategory);
    }

    if (selectedMethod !== 'all') {
      filtered = filtered.filter(endpoint => endpoint.method === selectedMethod);
    }

    if (!showDeprecated) {
      filtered = filtered.filter(endpoint => !endpoint.deprecated);
    }

    return filtered;
  };

  const getMethodColor = (method: string) => {
    const colors = {
      GET: 'bg-blue-100 text-blue-800',
      POST: 'bg-green-100 text-green-800',
      PUT: 'bg-yellow-100 text-yellow-800',
      DELETE: 'bg-red-100 text-red-800',
      PATCH: 'bg-purple-100 text-purple-800',
    };
    return colors[method as keyof typeof colors] || 'bg-gray-100 text-gray-800';
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const renderSidebar = () => (
    <div className="w-80 border-r bg-gray-50 overflow-y-auto">
      {/* Search and Filters */}
      <div className="p-4 border-b space-y-4">
        <div className="relative">
          <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
          <Input
            placeholder="Search endpoints..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10"
          />
        </div>

        <div className="grid grid-cols-2 gap-2">
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="px-2 py-1 text-sm border rounded bg-white"
          >
            <option value="all">All Categories</option>
            {categories.map(category => (
              <option key={category.id} value={category.id}>
                {category.name}
              </option>
            ))}
          </select>

          <select
            value={selectedMethod}
            onChange={(e) => setSelectedMethod(e.target.value)}
            className="px-2 py-1 text-sm border rounded bg-white"
          >
            <option value="all">All Methods</option>
            <option value="GET">GET</option>
            <option value="POST">POST</option>
            <option value="PUT">PUT</option>
            <option value="DELETE">DELETE</option>
            <option value="PATCH">PATCH</option>
          </select>
        </div>

        <div className="flex items-center space-x-2">
          <Switch
            checked={showDeprecated}
            onCheckedChange={setShowDeprecated}
            id="show-deprecated"
          />
          <Label htmlFor="show-deprecated" className="text-sm">
            Show deprecated
          </Label>
        </div>
      </div>

      {/* Endpoints by Category */}
      <div className="p-4">
        {categories.map(category => {
          const categoryEndpoints = filterEndpoints().filter(e => e.category === category.id);
          if (categoryEndpoints.length === 0) return null;

          return (
            <div key={category.id} className="mb-4">
              <button
                onClick={() => {
                  const newExpanded = new Set(expandedCategories);
                  if (newExpanded.has(category.id)) {
                    newExpanded.delete(category.id);
                  } else {
                    newExpanded.add(category.id);
                  }
                  setExpandedCategories(newExpanded);
                }}
                className="flex items-center justify-between w-full p-2 text-left text-sm font-medium text-gray-700 hover:bg-gray-100 rounded"
              >
                <span>{category.name}</span>
                {expandedCategories.has(category.id) ? (
                  <ChevronDown className="h-4 w-4" />
                ) : (
                  <ChevronRight className="h-4 w-4" />
                )}
              </button>

              {expandedCategories.has(category.id) && (
                <div className="ml-4 space-y-1">
                  {categoryEndpoints.map(endpoint => (
                    <button
                      key={endpoint.id}
                      onClick={() => setSelectedEndpoint(endpoint)}
                      className={`w-full text-left p-2 rounded text-sm transition-colors ${
                        selectedEndpoint?.id === endpoint.id
                          ? 'bg-blue-100 border border-blue-200'
                          : 'hover:bg-gray-100'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <Badge className={getMethodColor(endpoint.method)}>
                          {endpoint.method}
                        </Badge>
                        {endpoint.deprecated && (
                          <Badge variant="outline" className="text-xs text-orange-600">
                            Deprecated
                          </Badge>
                        )}
                      </div>
                      <div className="font-mono text-xs text-gray-600 mb-1">
                        {endpoint.path}
                      </div>
                      <div className="text-gray-700">{endpoint.summary}</div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );

  const renderEndpointDetails = () => {
    if (!selectedEndpoint) {
      return (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <BookOpen className="h-16 w-16 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-500">Select an endpoint to view documentation</p>
          </div>
        </div>
      );
    }

    return (
      <div className="flex-1 overflow-y-auto">
        <div className="p-6 space-y-6">
          {/* Endpoint Header */}
          <div className="space-y-4">
            <div className="flex items-center space-x-3">
              <Badge className={getMethodColor(selectedEndpoint.method)}>
                {selectedEndpoint.method}
              </Badge>
              <code className="text-lg font-mono">{selectedEndpoint.path}</code>
              {selectedEndpoint.deprecated && (
                <Badge variant="outline" className="text-orange-600 border-orange-200">
                  <AlertTriangle className="h-3 w-3 mr-1" />
                  Deprecated
                </Badge>
              )}
            </div>

            <h1 className="text-2xl font-bold">{selectedEndpoint.summary}</h1>
            <p className="text-gray-600">{selectedEndpoint.description}</p>

            {/* Tags */}
            {selectedEndpoint.tags.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {selectedEndpoint.tags.map(tag => (
                  <Badge key={tag} variant="outline" className="text-xs">
                    <Hash className="h-3 w-3 mr-1" />
                    {tag}
                  </Badge>
                ))}
              </div>
            )}

            {/* Authentication */}
            {selectedEndpoint.authentication.required && (
              <Alert>
                <Lock className="h-4 w-4" />
                <AlertDescription>
                  This endpoint requires authentication with {selectedEndpoint.authentication.type}.
                  {selectedEndpoint.authentication.scopes && (
                    <span> Required scopes: {selectedEndpoint.authentication.scopes.join(', ')}</span>
                  )}
                </AlertDescription>
              </Alert>
            )}

            {/* Rate Limiting */}
            {selectedEndpoint.rateLimit && (
              <Alert>
                <Clock className="h-4 w-4" />
                <AlertDescription>
                  Rate limit: {selectedEndpoint.rateLimit.requests} requests per {selectedEndpoint.rateLimit.window}
                </AlertDescription>
              </Alert>
            )}
          </div>

          <Tabs defaultValue="documentation" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="documentation">Documentation</TabsTrigger>
              <TabsTrigger value="testing">Try It Out</TabsTrigger>
              <TabsTrigger value="examples">Code Examples</TabsTrigger>
            </TabsList>

            <TabsContent value="documentation" className="space-y-6">
              {/* Parameters */}
              {selectedEndpoint.parameters.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Parameters</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {['path', 'query', 'header'].map(paramType => {
                        const params = selectedEndpoint.parameters.filter(p => p.in === paramType);
                        if (params.length === 0) return null;

                        return (
                          <div key={paramType}>
                            <h4 className="font-medium capitalize mb-2">{paramType} Parameters</h4>
                            <div className="space-y-3">
                              {params.map(param => (
                                <div key={param.name} className="border rounded p-3">
                                  <div className="flex items-center space-x-2 mb-2">
                                    <code className="font-medium">{param.name}</code>
                                    <Badge variant="outline" className="text-xs">
                                      {param.type}
                                    </Badge>
                                    {param.required && (
                                      <Badge className="bg-red-100 text-red-800 text-xs">
                                        Required
                                      </Badge>
                                    )}
                                  </div>
                                  <p className="text-sm text-gray-600 mb-2">{param.description}</p>
                                  {param.example !== undefined && (
                                    <div className="text-xs">
                                      <span className="text-gray-500">Example: </span>
                                      <code className="bg-gray-100 px-1 rounded">{JSON.stringify(param.example)}</code>
                                    </div>
                                  )}
                                  {param.enum && (
                                    <div className="text-xs">
                                      <span className="text-gray-500">Allowed values: </span>
                                      <code className="bg-gray-100 px-1 rounded">{param.enum.join(', ')}</code>
                                    </div>
                                  )}
                                </div>
                              ))}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Request Body */}
              {selectedEndpoint.requestBody && (
                <Card>
                  <CardHeader>
                    <CardTitle>Request Body</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="flex items-center space-x-2">
                        <Badge variant="outline">{selectedEndpoint.requestBody.contentType}</Badge>
                        {selectedEndpoint.requestBody.required && (
                          <Badge className="bg-red-100 text-red-800">Required</Badge>
                        )}
                      </div>

                      {selectedEndpoint.requestBody.examples && (
                        <div>
                          <h4 className="font-medium mb-2">Example</h4>
                          <pre className="bg-gray-100 p-3 rounded text-sm overflow-x-auto">
                            <code>{JSON.stringify(selectedEndpoint.requestBody.examples.default, null, 2)}</code>
                          </pre>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Responses */}
              <Card>
                <CardHeader>
                  <CardTitle>Responses</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {Object.entries(selectedEndpoint.responses).map(([statusCode, response]) => (
                      <div key={statusCode} className="border rounded p-3">
                        <div className="flex items-center space-x-2 mb-2">
                          <Badge className={
                            statusCode.startsWith('2') ? 'bg-green-100 text-green-800' :
                            statusCode.startsWith('4') ? 'bg-orange-100 text-orange-800' :
                            statusCode.startsWith('5') ? 'bg-red-100 text-red-800' :
                            'bg-gray-100 text-gray-800'
                          }>
                            {statusCode}
                          </Badge>
                        </div>
                        <p className="text-sm text-gray-600 mb-2">{response.description}</p>
                        {response.examples && (
                          <div>
                            <h5 className="font-medium text-sm mb-2">Example Response</h5>
                            <pre className="bg-gray-50 p-2 rounded text-xs overflow-x-auto">
                              <code>{JSON.stringify(response.examples.default, null, 2)}</code>
                            </pre>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="testing" className="space-y-6">
              {/* Test Interface */}
              <Card>
                <CardHeader>
                  <CardTitle>Test This Endpoint</CardTitle>
                  <CardDescription>
                    Fill in the parameters and execute a live request
                  </CardDescription>
                </CardHeader>

                <CardContent className="space-y-4">
                  {/* Request URL */}
                  <div>
                    <Label>Request URL</Label>
                    <div className="flex items-center space-x-2 mt-1">
                      <Badge className={getMethodColor(selectedEndpoint.method)}>
                        {selectedEndpoint.method}
                      </Badge>
                      <code className="flex-1 p-2 bg-gray-100 rounded text-sm">
                        {testRequest.endpoint}
                      </code>
                    </div>
                  </div>

                  {/* Parameters */}
                  {selectedEndpoint.parameters.length > 0 && (
                    <div>
                      <Label>Parameters</Label>
                      <div className="space-y-2 mt-2">
                        {selectedEndpoint.parameters.map(param => (
                          <div key={param.name} className="grid grid-cols-3 gap-2 items-center">
                            <Label className="text-sm">{param.name}</Label>
                            <Input
                              value={testRequest.parameters[param.name] || ''}
                              onChange={(e) => setTestRequest(prev => ({
                                ...prev,
                                parameters: { ...prev.parameters, [param.name]: e.target.value }
                              }))}
                              placeholder={param.example?.toString() || param.description}
                              className="col-span-2"
                            />
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Request Body */}
                  {selectedEndpoint.requestBody && ['POST', 'PUT', 'PATCH'].includes(selectedEndpoint.method) && (
                    <div>
                      <Label>Request Body</Label>
                      <textarea
                        value={testRequest.body || ''}
                        onChange={(e) => setTestRequest(prev => ({ ...prev, body: e.target.value }))}
                        className="w-full mt-1 p-2 border rounded font-mono text-sm"
                        rows={8}
                        placeholder="Enter JSON request body..."
                      />
                    </div>
                  )}

                  {/* Headers */}
                  <div>
                    <Label>Headers</Label>
                    <div className="space-y-2 mt-2">
                      {Object.entries(testRequest.headers).map(([key, value]) => (
                        <div key={key} className="grid grid-cols-3 gap-2 items-center">
                          <Input value={key} readOnly className="bg-gray-50" />
                          <Input
                            value={value}
                            onChange={(e) => setTestRequest(prev => ({
                              ...prev,
                              headers: { ...prev.headers, [key]: e.target.value }
                            }))}
                            className="col-span-2"
                          />
                        </div>
                      ))}
                    </div>
                  </div>

                  <Button onClick={executeTest} disabled={isTesting} className="w-full">
                    {isTesting ? (
                      <>
                        <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                        Executing...
                      </>
                    ) : (
                      <>
                        <Play className="h-4 w-4 mr-2" />
                        Execute Request
                      </>
                    )}
                  </Button>
                </CardContent>
              </Card>

              {/* Test Response */}
              {testResponse && (
                <Card>
                  <CardHeader>
                    <CardTitle>Response</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          <Badge className={
                            testResponse.status >= 200 && testResponse.status < 300 ? 'bg-green-100 text-green-800' :
                            testResponse.status >= 400 ? 'bg-red-100 text-red-800' :
                            'bg-gray-100 text-gray-800'
                          }>
                            {testResponse.status} {testResponse.statusText}
                          </Badge>
                          <span className="text-sm text-gray-500">
                            {testResponse.duration}ms • {formatBytes(testResponse.size)}
                          </span>
                        </div>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => copyToClipboard(JSON.stringify(testResponse.data, null, 2))}
                        >
                          <Copy className="h-4 w-4 mr-2" />
                          Copy
                        </Button>
                      </div>

                      <div>
                        <Label>Response Body</Label>
                        <pre className="mt-1 p-3 bg-gray-100 rounded text-sm overflow-x-auto max-h-96">
                          <code>{JSON.stringify(testResponse.data, null, 2)}</code>
                        </pre>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </TabsContent>

            <TabsContent value="examples" className="space-y-6">
              {/* Code Examples */}
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle>Code Examples</CardTitle>
                    <div className="flex items-center space-x-2">
                      <Switch
                        checked={showApiKey}
                        onCheckedChange={setShowApiKey}
                        id="show-api-key"
                      />
                      <Label htmlFor="show-api-key" className="text-sm">
                        Show API key
                      </Label>
                    </div>
                  </div>
                </CardHeader>

                <CardContent>
                  <Tabs value={selectedLanguage} onValueChange={(value) => setSelectedLanguage(value as any)}>
                    <TabsList className="grid w-full grid-cols-5">
                      <TabsTrigger value="curl">cURL</TabsTrigger>
                      <TabsTrigger value="javascript">JavaScript</TabsTrigger>
                      <TabsTrigger value="python">Python</TabsTrigger>
                      <TabsTrigger value="php">PHP</TabsTrigger>
                      <TabsTrigger value="java">Java</TabsTrigger>
                    </TabsList>

                    {(['curl', 'javascript', 'python', 'php', 'java'] as const).map(lang => (
                      <TabsContent key={lang} value={lang}>
                        <div className="relative">
                          <pre className="bg-gray-900 text-gray-100 p-4 rounded text-sm overflow-x-auto">
                            <code>{generateCodeExample(selectedEndpoint, lang)}</code>
                          </pre>
                          <Button
                            variant="outline"
                            size="sm"
                            className="absolute top-2 right-2"
                            onClick={() => copyToClipboard(generateCodeExample(selectedEndpoint, lang))}
                          >
                            <Copy className="h-4 w-4" />
                          </Button>
                        </div>
                      </TabsContent>
                    ))}
                  </Tabs>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    );
  };

  if (!user) {
    return (
      <Alert>
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          Please log in to view API documentation.
        </AlertDescription>
      </Alert>
    );
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="h-8 w-8 animate-spin text-gray-400" />
      </div>
    );
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <AlertTriangle className="h-4 w-4" />
        <AlertDescription>{error}</AlertDescription>
      </Alert>
    );
  }

  return (
    <div className={`flex h-screen bg-white ${className}`}>
      {renderSidebar()}
      {renderEndpointDetails()}
    </div>
  );
}