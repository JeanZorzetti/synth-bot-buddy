/**
 * 🔒 SESSION MANAGER COMPONENT
 * Advanced session management with device tracking and security controls
 */

import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { authService } from '../../services/authService';
import { Button } from '../ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card';
import { Alert, AlertDescription } from '../ui/alert';
import { Badge } from '../ui/badge';
import {
  Monitor,
  Smartphone,
  Tablet,
  Globe,
  MapPin,
  Clock,
  Shield,
  Trash2,
  AlertTriangle,
  CheckCircle,
  RefreshCw,
  Eye,
  LogOut,
  Ban,
  Activity
} from 'lucide-react';

interface Session {
  id: string;
  userId: string;
  deviceType: 'desktop' | 'mobile' | 'tablet' | 'unknown';
  deviceName: string;
  browser: string;
  operatingSystem: string;
  ipAddress: string;
  location: {
    country: string;
    city: string;
    region: string;
  };
  isCurrentSession: boolean;
  lastActivity: string;
  createdAt: string;
  expiresAt: string;
  userAgent: string;
  isActive: boolean;
  isSuspicious: boolean;
  loginCount: number;
}

interface SessionManagerProps {
  className?: string;
}

export function SessionManager({ className = "" }: SessionManagerProps) {
  const { user, logout } = useAuth();
  const [sessions, setSessions] = useState<Session[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isRevoking, setIsRevoking] = useState<string | null>(null);

  useEffect(() => {
    loadSessions();
  }, []);

  const loadSessions = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const userSessions = await authService.getUserSessions();
      setSessions(userSessions);
    } catch (error: any) {
      setError(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  const revokeSession = async (sessionId: string, isCurrentSession: boolean) => {
    try {
      setIsRevoking(sessionId);

      if (isCurrentSession) {
        // If revoking current session, logout
        await logout();
        return;
      }

      await authService.revokeSession(sessionId);
      await loadSessions(); // Refresh the session list
    } catch (error: any) {
      setError(error.message);
    } finally {
      setIsRevoking(null);
    }
  };

  const getDeviceIcon = (deviceType: string) => {
    switch (deviceType) {
      case 'mobile':
        return <Smartphone className="h-5 w-5" />;
      case 'tablet':
        return <Tablet className="h-5 w-5" />;
      case 'desktop':
        return <Monitor className="h-5 w-5" />;
      default:
        return <Globe className="h-5 w-5" />;
    }
  };

  const formatLastActivity = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins} minutes ago`;
    if (diffHours < 24) return `${diffHours} hours ago`;
    return `${diffDays} days ago`;
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const getSessionStatus = (session: Session) => {
    if (!session.isActive) return 'expired';
    if (session.isSuspicious) return 'suspicious';
    if (session.isCurrentSession) return 'current';
    return 'active';
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'current':
        return <Badge className="bg-green-100 text-green-800">Current Session</Badge>;
      case 'active':
        return <Badge variant="outline">Active</Badge>;
      case 'suspicious':
        return <Badge className="bg-red-100 text-red-800">Suspicious</Badge>;
      case 'expired':
        return <Badge variant="secondary">Expired</Badge>;
      default:
        return <Badge variant="secondary">Unknown</Badge>;
    }
  };

  const activeSessions = sessions.filter(s => s.isActive);
  const expiredSessions = sessions.filter(s => !s.isActive);
  const suspiciousSessions = sessions.filter(s => s.isSuspicious);

  if (!user) {
    return (
      <Alert>
        <Shield className="h-4 w-4" />
        <AlertDescription>
          Please log in to view session management.
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Session Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-green-100 rounded-lg">
                <CheckCircle className="h-5 w-5 text-green-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">{activeSessions.length}</p>
                <p className="text-sm text-gray-500">Active Sessions</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-red-100 rounded-lg">
                <AlertTriangle className="h-5 w-5 text-red-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">{suspiciousSessions.length}</p>
                <p className="text-sm text-gray-500">Suspicious</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-gray-100 rounded-lg">
                <Activity className="h-5 w-5 text-gray-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">{sessions.length}</p>
                <p className="text-sm text-gray-500">Total Sessions</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Suspicious Sessions Alert */}
      {suspiciousSessions.length > 0 && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            {suspiciousSessions.length} suspicious session{suspiciousSessions.length > 1 ? 's' : ''} detected.
            Please review and revoke any unauthorized access.
          </AlertDescription>
        </Alert>
      )}

      {/* Active Sessions */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Active Sessions</CardTitle>
              <CardDescription>
                Devices currently logged into your account
              </CardDescription>
            </div>
            <Button
              variant="outline"
              onClick={loadSessions}
              disabled={isLoading}
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
        </CardHeader>

        <CardContent>
          {isLoading ? (
            <div className="space-y-4">
              {[1, 2, 3].map(i => (
                <div key={i} className="animate-pulse">
                  <div className="h-20 bg-gray-200 rounded-lg"></div>
                </div>
              ))}
            </div>
          ) : activeSessions.length === 0 ? (
            <div className="text-center py-8">
              <Shield className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">No active sessions found</p>
            </div>
          ) : (
            <div className="space-y-4">
              {activeSessions.map((session) => (
                <div
                  key={session.id}
                  className={`p-4 border rounded-lg ${
                    session.isSuspicious ? 'border-red-200 bg-red-50' :
                    session.isCurrentSession ? 'border-green-200 bg-green-50' :
                    'border-gray-200'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex items-start space-x-4">
                      <div className="p-2 bg-white rounded-lg shadow-sm">
                        {getDeviceIcon(session.deviceType)}
                      </div>

                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-2 mb-2">
                          <p className="font-medium truncate">
                            {session.deviceName || `${session.browser} on ${session.operatingSystem}`}
                          </p>
                          {getStatusBadge(getSessionStatus(session))}
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm text-gray-600">
                          <div className="flex items-center space-x-2">
                            <Globe className="h-4 w-4" />
                            <span>{session.ipAddress}</span>
                          </div>

                          <div className="flex items-center space-x-2">
                            <MapPin className="h-4 w-4" />
                            <span>
                              {session.location.city}, {session.location.country}
                            </span>
                          </div>

                          <div className="flex items-center space-x-2">
                            <Clock className="h-4 w-4" />
                            <span>Last active: {formatLastActivity(session.lastActivity)}</span>
                          </div>

                          <div className="flex items-center space-x-2">
                            <Activity className="h-4 w-4" />
                            <span>{session.loginCount} logins</span>
                          </div>
                        </div>

                        {session.isSuspicious && (
                          <div className="mt-2 p-2 bg-red-100 rounded text-sm text-red-800">
                            ⚠️ This session has been flagged as suspicious due to unusual activity patterns.
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="flex space-x-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          // Show session details modal
                          console.log('Show session details:', session);
                        }}
                      >
                        <Eye className="h-4 w-4" />
                      </Button>

                      <Button
                        variant={session.isCurrentSession ? "destructive" : "outline"}
                        size="sm"
                        onClick={() => revokeSession(session.id, session.isCurrentSession)}
                        disabled={isRevoking === session.id}
                      >
                        {isRevoking === session.id ? (
                          <RefreshCw className="h-4 w-4 animate-spin" />
                        ) : session.isCurrentSession ? (
                          <LogOut className="h-4 w-4" />
                        ) : (
                          <Ban className="h-4 w-4" />
                        )}
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Security Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Security Actions</CardTitle>
          <CardDescription>
            Additional security measures for your account
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Button
              variant="outline"
              onClick={async () => {
                if (confirm('Are you sure you want to revoke all other sessions?')) {
                  try {
                    const otherSessions = activeSessions.filter(s => !s.isCurrentSession);
                    for (const session of otherSessions) {
                      await authService.revokeSession(session.id);
                    }
                    await loadSessions();
                  } catch (error: any) {
                    setError(error.message);
                  }
                }
              }}
              className="justify-start"
            >
              <Ban className="h-4 w-4 mr-2" />
              Revoke All Other Sessions
            </Button>

            <Button
              variant="outline"
              onClick={() => {
                // Navigate to change password
                console.log('Navigate to change password');
              }}
              className="justify-start"
            >
              <Shield className="h-4 w-4 mr-2" />
              Change Password
            </Button>

            <Button
              variant="outline"
              onClick={() => {
                // Navigate to 2FA setup
                console.log('Navigate to 2FA setup');
              }}
              className="justify-start"
            >
              <CheckCircle className="h-4 w-4 mr-2" />
              Enable Two-Factor Auth
            </Button>

            <Button
              variant="destructive"
              onClick={async () => {
                if (confirm('Are you sure you want to log out of all devices?')) {
                  await logout();
                }
              }}
              className="justify-start"
            >
              <LogOut className="h-4 w-4 mr-2" />
              Log Out Everywhere
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Recent Session History */}
      {expiredSessions.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Recent Session History</CardTitle>
            <CardDescription>
              Previously logged in devices and sessions
            </CardDescription>
          </CardHeader>

          <CardContent>
            <div className="space-y-3">
              {expiredSessions.slice(0, 5).map((session) => (
                <div
                  key={session.id}
                  className="flex items-center justify-between p-3 border rounded-lg bg-gray-50"
                >
                  <div className="flex items-center space-x-3">
                    <div className="p-1 bg-gray-200 rounded">
                      {getDeviceIcon(session.deviceType)}
                    </div>
                    <div>
                      <p className="font-medium text-sm">
                        {session.deviceName || `${session.browser} on ${session.operatingSystem}`}
                      </p>
                      <p className="text-xs text-gray-500">
                        {session.location.city}, {session.location.country} •
                        {formatDate(session.lastActivity)}
                      </p>
                    </div>
                  </div>
                  <Badge variant="secondary">Expired</Badge>
                </div>
              ))}

              {expiredSessions.length > 5 && (
                <Button variant="outline" size="sm" className="w-full">
                  View All History ({expiredSessions.length - 5} more)
                </Button>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}