import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { 
  Loader2, 
  Shield, 
  User, 
  Key, 
  Settings, 
  CheckCircle, 
  AlertCircle, 
  RefreshCw,
  Zap,
  Info
} from 'lucide-react';
import { useOAuth } from '../hooks/useOAuth';
import { apiService } from '../services/api';
import OAuthLogin from './OAuthLogin';

export function OAuthDemo() {
  const {
    isAuthenticated,
    isLoading,
    user,
    token,
    error,
    scopes,
    connectWithOAuth,
    refreshToken,
    logout,
    isTokenExpired,
    hasValidSession,
    getAvailableScopes
  } = useOAuth();

  const [connectionStatus, setConnectionStatus] = useState<string>('disconnected');
  const [connectionLoading, setConnectionLoading] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [oauthConfig, setOauthConfig] = useState<any>(null);
  const [availableScopes, setAvailableScopes] = useState<string[]>([]);

  // Load OAuth configuration
  useEffect(() => {
    const loadConfig = async () => {
      try {
        const config = await apiService.getOAuthConfig();
        setOauthConfig(config);
      } catch (error) {
        console.error('Failed to load OAuth config:', error);
      }
    };

    const loadScopes = async () => {
      try {
        const scopes = await getAvailableScopes();
        setAvailableScopes(scopes);
      } catch (error) {
        console.error('Failed to load available scopes:', error);
      }
    };

    loadConfig();
    loadScopes();
  }, [getAvailableScopes]);

  const handleConnectWithOAuth = async () => {
    setConnectionLoading(true);
    setConnectionError(null);

    try {
      const success = await connectWithOAuth();
      if (success) {
        setConnectionStatus('connected');
        
        // Check bot status after connection
        setTimeout(async () => {
          try {
            const status = await apiService.getBotStatus();
            setConnectionStatus(status.connection_status);
          } catch (error) {
            console.error('Failed to get bot status:', error);
          }
        }, 2000);
      } else {
        setConnectionError('Connection failed');
      }
    } catch (error: any) {
      setConnectionError(error.message || 'Connection failed');
    } finally {
      setConnectionLoading(false);
    }
  };

  const handleRefreshToken = async () => {
    try {
      await refreshToken();
      setConnectionError(null);
    } catch (error: any) {
      setConnectionError(error.message || 'Token refresh failed');
    }
  };

  const formatTokenExpiry = () => {
    if (!token?.info) return 'Unknown';
    
    const createdAt = new Date(token.info.created_at);
    const expiresAt = new Date(createdAt.getTime() + (token.info.expires_in * 1000));
    const now = new Date();
    
    const diffMs = expiresAt.getTime() - now.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    
    if (diffMins < 0) return 'Expired';
    if (diffMins < 60) return `${diffMins} minutes`;
    
    const diffHours = Math.floor(diffMins / 60);
    return `${diffHours} hours`;
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-6 w-6" />
            OAuth 2.0 Demo & Testing
          </CardTitle>
          <CardDescription>
            Test OAuth 2.0 authentication flow with Deriv API
          </CardDescription>
        </CardHeader>
      </Card>

      <Tabs defaultValue="login" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="login">Login</TabsTrigger>
          <TabsTrigger value="status">Status</TabsTrigger>
          <TabsTrigger value="connection">Connection</TabsTrigger>
          <TabsTrigger value="config">Config</TabsTrigger>
        </TabsList>

        <TabsContent value="login" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            <OAuthLogin
              onSuccess={(user, token) => {
                console.log('OAuth Success:', { user, token });
              }}
              onError={(error) => {
                console.error('OAuth Error:', error);
              }}
            />

            {isAuthenticated && (
              <Card>
                <CardHeader>
                  <CardTitle>Quick Actions</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <Button 
                    onClick={handleRefreshToken}
                    variant="outline"
                    disabled={isLoading}
                    className="w-full"
                  >
                    <RefreshCw className="mr-2 h-4 w-4" />
                    Refresh Token
                  </Button>

                  <Button 
                    onClick={handleConnectWithOAuth}
                    disabled={connectionLoading || !hasValidSession()}
                    className="w-full"
                  >
                    {connectionLoading ? (
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    ) : (
                      <Zap className="mr-2 h-4 w-4" />
                    )}
                    Connect to WebSocket
                  </Button>

                  {connectionError && (
                    <Alert variant="destructive">
                      <AlertCircle className="h-4 w-4" />
                      <AlertDescription>{connectionError}</AlertDescription>
                    </Alert>
                  )}
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>

        <TabsContent value="status" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Authentication Status</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span>Status:</span>
                    <Badge variant={isAuthenticated ? "default" : "secondary"}>
                      {isAuthenticated ? "Authenticated" : "Not Authenticated"}
                    </Badge>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span>Token Valid:</span>
                    <Badge variant={hasValidSession() ? "default" : "destructive"}>
                      {hasValidSession() ? "Valid" : "Invalid/Expired"}
                    </Badge>
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span>Loading:</span>
                    <Badge variant={isLoading ? "default" : "secondary"}>
                      {isLoading ? "Yes" : "No"}
                    </Badge>
                  </div>

                  <div className="flex items-center justify-between">
                    <span>Connection:</span>
                    <Badge variant={
                      connectionStatus === 'connected' ? "default" : 
                      connectionStatus === 'connecting' ? "secondary" : 
                      "outline"
                    }>
                      {connectionStatus}
                    </Badge>
                  </div>
                </div>

                {token && (
                  <div className="pt-4 border-t">
                    <h4 className="font-medium mb-2">Token Info</h4>
                    <div className="space-y-1 text-sm">
                      <div>Type: {token.info.token_type}</div>
                      <div>Expires in: {formatTokenExpiry()}</div>
                      <div>Created: {new Date(token.info.created_at).toLocaleString()}</div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>User Information</CardTitle>
              </CardHeader>
              <CardContent>
                {user ? (
                  <div className="space-y-2">
                    {Object.entries(user).map(([key, value]) => (
                      <div key={key} className="flex justify-between">
                        <span className="font-medium">{key}:</span>
                        <span className="text-sm">{String(value)}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-muted-foreground">No user information available</p>
                )}
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Scopes & Permissions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium mb-2">Active Scopes:</h4>
                  <div className="flex flex-wrap gap-1">
                    {scopes.length > 0 ? (
                      scopes.map(scope => (
                        <Badge key={scope} variant="default">
                          {scope}
                        </Badge>
                      ))
                    ) : (
                      <span className="text-muted-foreground">No active scopes</span>
                    )}
                  </div>
                </div>

                <div>
                  <h4 className="font-medium mb-2">Available Scopes:</h4>
                  <div className="flex flex-wrap gap-1">
                    {availableScopes.map(scope => (
                      <Badge key={scope} variant="outline">
                        {scope}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="connection" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>WebSocket Connection</CardTitle>
              <CardDescription>
                Test connection to Deriv WebSocket using OAuth token
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between p-4 border rounded-lg">
                <div>
                  <div className="font-medium">Connection Status</div>
                  <div className="text-sm text-muted-foreground">
                    Current WebSocket connection state
                  </div>
                </div>
                <Badge variant={
                  connectionStatus === 'connected' ? "default" : 
                  connectionStatus === 'connecting' ? "secondary" : 
                  "outline"
                }>
                  {connectionStatus}
                </Badge>
              </div>

              {connectionError && (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{connectionError}</AlertDescription>
                </Alert>
              )}

              <div className="flex gap-2">
                <Button 
                  onClick={handleConnectWithOAuth}
                  disabled={connectionLoading || !hasValidSession()}
                  variant="default"
                >
                  {connectionLoading ? (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  ) : (
                    <Zap className="mr-2 h-4 w-4" />
                  )}
                  Connect
                </Button>

                <Button 
                  onClick={() => {
                    setConnectionStatus('disconnected');
                    setConnectionError(null);
                  }}
                  variant="outline"
                >
                  Reset
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="config" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>OAuth Configuration</CardTitle>
              <CardDescription>
                Current OAuth 2.0 configuration and endpoints
              </CardDescription>
            </CardHeader>
            <CardContent>
              {oauthConfig ? (
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium mb-2">Endpoints</h4>
                    <div className="space-y-1 text-sm font-mono bg-muted p-3 rounded">
                      <div>Authorize: {oauthConfig.oauth_endpoints.authorize_url}</div>
                      <div>Token: {oauthConfig.oauth_endpoints.token_url}</div>
                      <div>UserInfo: {oauthConfig.oauth_endpoints.user_info_url}</div>
                    </div>
                  </div>

                  <div>
                    <h4 className="font-medium mb-2">Client Configuration</h4>
                    <div className="space-y-1 text-sm">
                      <div>Client ID: {oauthConfig.client_configuration.client_id}</div>
                      <div>Has Secret: {oauthConfig.client_configuration.has_client_secret ? 'Yes' : 'No'}</div>
                      <div>Redirect URI: {oauthConfig.client_configuration.default_redirect_uri}</div>
                    </div>
                  </div>

                  <div>
                    <h4 className="font-medium mb-2">Security Features</h4>
                    <div className="flex gap-2">
                      {Object.entries(oauthConfig.security_features).map(([key, value]) => (
                        <Badge key={key} variant={value ? "default" : "outline"}>
                          {key}: {value ? 'Enabled' : 'Disabled'}
                        </Badge>
                      ))}
                    </div>
                  </div>

                  <div>
                    <h4 className="font-medium mb-2">Runtime Info</h4>
                    <div className="text-sm">
                      Active OAuth States: {oauthConfig.active_states_count}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex items-center justify-center p-8">
                  <Loader2 className="h-6 w-6 animate-spin mr-2" />
                  Loading configuration...
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default OAuthDemo;