import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { Loader2, Shield, User, Key, ExternalLink, CheckCircle, AlertCircle } from 'lucide-react';
import { useOAuth } from '../hooks/useOAuth';

export interface OAuthLoginProps {
  onSuccess?: (user: any, token: any) => void;
  onError?: (error: string) => void;
  className?: string;
}

export function OAuthLogin({ onSuccess, onError, className }: OAuthLoginProps) {
  const {
    isAuthenticated,
    isLoading,
    user,
    token,
    error,
    scopes,
    startOAuthFlow,
    handleCallback,
    logout,
    clearError,
    hasValidSession
  } = useOAuth();

  const [availableScopes, setAvailableScopes] = useState<string[]>([]);
  const [selectedScopes, setSelectedScopes] = useState<string[]>(['read', 'trade']);
  const [scopeDescriptions, setScopeDescriptions] = useState<Record<string, string>>({});

  // Handle OAuth callback from URL parameters
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const code = urlParams.get('code');
    const state = urlParams.get('state');
    const oauthError = urlParams.get('error');

    if (oauthError) {
      onError?.(oauthError);
      return;
    }

    if (code && state) {
      handleCallback(code, state)
        .then((response) => {
          onSuccess?.(response.user_info, response.token_info);
          // Clean up URL
          window.history.replaceState({}, document.title, window.location.pathname);
        })
        .catch((error) => {
          onError?.(error.message);
        });
    }
  }, [handleCallback, onSuccess, onError]);

  // Load available scopes
  useEffect(() => {
    const loadScopes = async () => {
      try {
        const scopesResponse = await fetch('/api/oauth/scopes');
        if (scopesResponse.ok) {
          const data = await scopesResponse.json();
          setAvailableScopes(data.available_scopes || []);
          setScopeDescriptions(data.scope_descriptions || {});
        }
      } catch (error) {
        console.error('Failed to load scopes:', error);
      }
    };

    loadScopes();
  }, []);

  // Notify parent on authentication change
  useEffect(() => {
    if (isAuthenticated && user && token) {
      onSuccess?.(user, token);
    }
  }, [isAuthenticated, user, token, onSuccess]);

  // Handle errors
  useEffect(() => {
    if (error) {
      onError?.(error);
    }
  }, [error, onError]);

  const handleStartOAuth = async () => {
    try {
      await startOAuthFlow(selectedScopes);
    } catch (error: any) {
      onError?.(error.message);
    }
  };

  const handleLogout = async () => {
    try {
      await logout();
    } catch (error: any) {
      onError?.(error.message);
    }
  };

  const handleScopeToggle = (scope: string) => {
    setSelectedScopes(prev => 
      prev.includes(scope)
        ? prev.filter(s => s !== scope)
        : [...prev, scope]
    );
  };

  // If already authenticated, show user info
  if (isAuthenticated && hasValidSession()) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <CheckCircle className="h-5 w-5 text-green-500" />
            Authenticated
          </CardTitle>
          <CardDescription>
            Connected to Deriv via OAuth 2.0
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {user && (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <User className="h-4 w-4" />
                <span className="font-medium">User:</span>
                <span>{user.email || user.user_id || 'Unknown'}</span>
              </div>
              {user.account_id && (
                <div className="flex items-center gap-2">
                  <Key className="h-4 w-4" />
                  <span className="font-medium">Account:</span>
                  <span className="font-mono text-sm">{user.account_id}</span>
                </div>
              )}
            </div>
          )}

          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Shield className="h-4 w-4" />
              <span className="font-medium">Scopes:</span>
            </div>
            <div className="flex flex-wrap gap-1">
              {scopes.map(scope => (
                <Badge key={scope} variant="secondary">
                  {scope}
                </Badge>
              ))}
            </div>
          </div>

          {token?.info && (
            <div className="text-sm text-muted-foreground">
              <div>Token expires: {new Date(
                new Date(token.info.created_at).getTime() + 
                (token.info.expires_in * 1000)
              ).toLocaleString()}</div>
            </div>
          )}

          <div className="pt-4 border-t">
            <Button 
              onClick={handleLogout} 
              variant="outline"
              disabled={isLoading}
              className="w-full"
            >
              {isLoading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Logout
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Login form
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Shield className="h-5 w-5" />
          OAuth 2.0 Login
        </CardTitle>
        <CardDescription>
          Connect to Deriv using secure OAuth 2.0 authentication
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              {error}
              <Button 
                variant="ghost" 
                size="sm" 
                onClick={clearError}
                className="ml-2"
              >
                Dismiss
              </Button>
            </AlertDescription>
          </Alert>
        )}

        <div className="space-y-3">
          <div>
            <label className="text-sm font-medium">Permissions (Scopes)</label>
            <p className="text-xs text-muted-foreground mb-2">
              Select the permissions you want to grant to the application
            </p>
          </div>
          
          <div className="space-y-2">
            {availableScopes.map(scope => (
              <div key={scope} className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id={scope}
                  checked={selectedScopes.includes(scope)}
                  onChange={() => handleScopeToggle(scope)}
                  className="rounded border-gray-300"
                />
                <label htmlFor={scope} className="flex-1">
                  <div className="flex items-center gap-2">
                    <Badge variant={selectedScopes.includes(scope) ? "default" : "outline"}>
                      {scope}
                    </Badge>
                    <span className="text-sm">
                      {scopeDescriptions[scope] || `${scope} access`}
                    </span>
                  </div>
                </label>
              </div>
            ))}
          </div>
        </div>

        <div className="pt-4 border-t">
          <Button 
            onClick={handleStartOAuth}
            disabled={isLoading || selectedScopes.length === 0}
            className="w-full"
          >
            {isLoading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Connecting...
              </>
            ) : (
              <>
                <ExternalLink className="mr-2 h-4 w-4" />
                Connect with Deriv OAuth
              </>
            )}
          </Button>
        </div>

        <div className="text-xs text-muted-foreground">
          <div className="flex items-center gap-1">
            <Shield className="h-3 w-3" />
            <span>Secure OAuth 2.0 with PKCE</span>
          </div>
          <div className="mt-1">
            You'll be redirected to Deriv's secure authentication page
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default OAuthLogin;