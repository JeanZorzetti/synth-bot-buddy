'use client';

import React, { useState, useEffect } from 'react';
import { AlertTriangle, Bell, CheckCircle, XCircle, AlertCircle, TrendingUp, TrendingDown, Activity, Zap, Settings, Eye, EyeOff, Volume2, VolumeX } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';

interface Alert {
  id: string;
  title: string;
  message: string;
  type: 'success' | 'warning' | 'error' | 'info';
  severity: 'low' | 'medium' | 'high' | 'critical';
  category: string;
  source: string;
  timestamp: string;
  isRead: boolean;
  isAcknowledged: boolean;
  autoResolve: boolean;
  resolvedAt?: string;
  actions: Array<{
    id: string;
    label: string;
    type: 'primary' | 'secondary';
    action: string;
  }>;
  data: {
    currentValue: number;
    threshold: number;
    unit: string;
    trend: 'up' | 'down' | 'stable';
  };
  visual: {
    showBanner: boolean;
    showNotification: boolean;
    showModal: boolean;
    playSound: boolean;
    blinkDuration: number;
    color: string;
  };
}

interface AlertRule {
  id: string;
  name: string;
  description: string;
  enabled: boolean;
  conditions: {
    metric: string;
    operator: '>' | '<' | '=' | '>=' | '<=';
    threshold: number;
    duration: number; // seconds
  };
  triggers: {
    onThreshold: boolean;
    onTrendChange: boolean;
    onAnomaly: boolean;
  };
  notifications: {
    visual: boolean;
    sound: boolean;
    email: boolean;
    webhook: boolean;
  };
  visualConfig: {
    color: string;
    intensity: 'low' | 'medium' | 'high';
    position: 'top' | 'bottom' | 'center';
    animation: 'fade' | 'slide' | 'bounce' | 'pulse';
  };
  createdAt: string;
  triggeredCount: number;
  lastTriggered?: string;
}

interface AlertStats {
  total: number;
  active: number;
  resolved: number;
  acknowledged: number;
  byType: { [key: string]: number };
  bySeverity: { [key: string]: number };
  byCategory: { [key: string]: number };
  averageResolutionTime: number;
}

const VisualAlerts: React.FC = () => {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [alertRules, setAlertRules] = useState<AlertRule[]>([]);
  const [stats, setStats] = useState<AlertStats | null>(null);
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);
  const [showCreateRule, setShowCreateRule] = useState(false);
  const [globalSettings, setGlobalSettings] = useState({
    enableSounds: true,
    enableVisuals: true,
    enableNotifications: true,
    soundVolume: 70,
    visualIntensity: 'medium' as 'low' | 'medium' | 'high',
    autoAcknowledge: false,
    groupSimilar: true
  });
  const [filterType, setFilterType] = useState('all');
  const [filterSeverity, setFilterSeverity] = useState('all');

  useEffect(() => {
    // Mock data initialization
    const mockAlerts: Alert[] = [
      {
        id: '1',
        title: 'High Trading Volume Alert',
        message: 'Trading volume has exceeded the normal range by 150% in the last 15 minutes.',
        type: 'warning',
        severity: 'medium',
        category: 'Trading',
        source: 'Volume Monitor',
        timestamp: new Date().toISOString(),
        isRead: false,
        isAcknowledged: false,
        autoResolve: true,
        actions: [
          { id: 'scale', label: 'Scale Resources', type: 'primary', action: 'scale_resources' },
          { id: 'investigate', label: 'Investigate', type: 'secondary', action: 'open_investigation' }
        ],
        data: {
          currentValue: 2500000,
          threshold: 1000000,
          unit: 'USD',
          trend: 'up'
        },
        visual: {
          showBanner: true,
          showNotification: true,
          showModal: false,
          playSound: true,
          blinkDuration: 3000,
          color: '#f59e0b'
        }
      },
      {
        id: '2',
        title: 'Critical System Error',
        message: 'Trading engine has encountered a critical error and requires immediate attention.',
        type: 'error',
        severity: 'critical',
        category: 'System',
        source: 'Trading Engine',
        timestamp: new Date(Date.now() - 300000).toISOString(),
        isRead: false,
        isAcknowledged: false,
        autoResolve: false,
        actions: [
          { id: 'restart', label: 'Restart Engine', type: 'primary', action: 'restart_engine' },
          { id: 'escalate', label: 'Escalate', type: 'secondary', action: 'escalate_to_admin' }
        ],
        data: {
          currentValue: 0,
          threshold: 1,
          unit: 'status',
          trend: 'down'
        },
        visual: {
          showBanner: true,
          showNotification: true,
          showModal: true,
          playSound: true,
          blinkDuration: 10000,
          color: '#ef4444'
        }
      },
      {
        id: '3',
        title: 'Performance Milestone Achieved',
        message: 'AI trading bots have achieved a new daily profit record of $125,000.',
        type: 'success',
        severity: 'low',
        category: 'Performance',
        source: 'Performance Monitor',
        timestamp: new Date(Date.now() - 1800000).toISOString(),
        isRead: true,
        isAcknowledged: true,
        autoResolve: true,
        actions: [
          { id: 'share', label: 'Share News', type: 'primary', action: 'share_achievement' }
        ],
        data: {
          currentValue: 125000,
          threshold: 100000,
          unit: 'USD',
          trend: 'up'
        },
        visual: {
          showBanner: false,
          showNotification: true,
          showModal: false,
          playSound: false,
          blinkDuration: 2000,
          color: '#10b981'
        }
      }
    ];

    const mockRules: AlertRule[] = [
      {
        id: '1',
        name: 'High Volume Alert',
        description: 'Trigger when trading volume exceeds threshold',
        enabled: true,
        conditions: {
          metric: 'trading_volume',
          operator: '>',
          threshold: 1000000,
          duration: 300
        },
        triggers: {
          onThreshold: true,
          onTrendChange: false,
          onAnomaly: true
        },
        notifications: {
          visual: true,
          sound: true,
          email: true,
          webhook: false
        },
        visualConfig: {
          color: '#f59e0b',
          intensity: 'medium',
          position: 'top',
          animation: 'pulse'
        },
        createdAt: '2024-01-01T00:00:00Z',
        triggeredCount: 23,
        lastTriggered: '2024-01-15T14:30:00Z'
      },
      {
        id: '2',
        name: 'System Health Check',
        description: 'Monitor system components health',
        enabled: true,
        conditions: {
          metric: 'system_health',
          operator: '<',
          threshold: 95,
          duration: 60
        },
        triggers: {
          onThreshold: true,
          onTrendChange: true,
          onAnomaly: false
        },
        notifications: {
          visual: true,
          sound: true,
          email: true,
          webhook: true
        },
        visualConfig: {
          color: '#ef4444',
          intensity: 'high',
          position: 'center',
          animation: 'bounce'
        },
        createdAt: '2024-01-02T00:00:00Z',
        triggeredCount: 7,
        lastTriggered: '2024-01-14T09:15:00Z'
      }
    ];

    const mockStats: AlertStats = {
      total: 156,
      active: 12,
      resolved: 138,
      acknowledged: 6,
      byType: {
        success: 45,
        warning: 67,
        error: 32,
        info: 12
      },
      bySeverity: {
        low: 78,
        medium: 45,
        high: 23,
        critical: 10
      },
      byCategory: {
        Trading: 89,
        System: 34,
        Performance: 23,
        Security: 10
      },
      averageResolutionTime: 25 // minutes
    };

    setAlerts(mockAlerts);
    setAlertRules(mockRules);
    setStats(mockStats);

    // Simulate real-time alerts
    const interval = setInterval(() => {
      if (Math.random() < 0.1) { // 10% chance every 5 seconds
        generateNewAlert();
      }
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const generateNewAlert = () => {
    const types: Alert['type'][] = ['success', 'warning', 'error', 'info'];
    const severities: Alert['severity'][] = ['low', 'medium', 'high', 'critical'];
    const categories = ['Trading', 'System', 'Performance', 'Security'];

    const newAlert: Alert = {
      id: `alert-${Date.now()}`,
      title: 'New Alert Generated',
      message: 'This is a simulated real-time alert for demonstration.',
      type: types[Math.floor(Math.random() * types.length)],
      severity: severities[Math.floor(Math.random() * severities.length)],
      category: categories[Math.floor(Math.random() * categories.length)],
      source: 'Alert Generator',
      timestamp: new Date().toISOString(),
      isRead: false,
      isAcknowledged: false,
      autoResolve: Math.random() > 0.5,
      actions: [
        { id: 'acknowledge', label: 'Acknowledge', type: 'primary', action: 'acknowledge' }
      ],
      data: {
        currentValue: Math.floor(Math.random() * 100),
        threshold: 50,
        unit: '%',
        trend: Math.random() > 0.5 ? 'up' : 'down'
      },
      visual: {
        showBanner: true,
        showNotification: true,
        showModal: false,
        playSound: globalSettings.enableSounds,
        blinkDuration: 3000,
        color: getAlertColor(types[Math.floor(Math.random() * types.length)])
      }
    };

    setAlerts(prev => [newAlert, ...prev]);

    // Show visual alert if enabled
    if (globalSettings.enableVisuals) {
      showVisualAlert(newAlert);
    }

    // Play sound if enabled
    if (globalSettings.enableSounds && newAlert.visual.playSound) {
      playAlertSound(newAlert.type);
    }
  };

  const getAlertColor = (type: Alert['type']) => {
    switch (type) {
      case 'success': return '#10b981';
      case 'warning': return '#f59e0b';
      case 'error': return '#ef4444';
      case 'info': return '#3b82f6';
      default: return '#6b7280';
    }
  };

  const showVisualAlert = (alert: Alert) => {
    // Implementation for visual alert display
    console.log('Showing visual alert:', alert.title);

    // Create a temporary visual indicator
    if (alert.visual.showBanner) {
      // Implementation for banner alert
    }

    if (alert.visual.showNotification) {
      // Implementation for notification alert
      if (Notification.permission === 'granted') {
        new Notification(alert.title, {
          body: alert.message,
          icon: '/alert-icon.png'
        });
      }
    }
  };

  const playAlertSound = (type: Alert['type']) => {
    // Implementation for audio alert
    const audio = new Audio();
    switch (type) {
      case 'success':
        audio.src = '/sounds/success.mp3';
        break;
      case 'warning':
        audio.src = '/sounds/warning.mp3';
        break;
      case 'error':
        audio.src = '/sounds/error.mp3';
        break;
      default:
        audio.src = '/sounds/info.mp3';
    }
    audio.volume = globalSettings.soundVolume / 100;
    audio.play().catch(console.error);
  };

  const acknowledgeAlert = (alertId: string) => {
    setAlerts(prev => prev.map(alert =>
      alert.id === alertId
        ? { ...alert, isAcknowledged: true, isRead: true }
        : alert
    ));
  };

  const dismissAlert = (alertId: string) => {
    setAlerts(prev => prev.filter(alert => alert.id !== alertId));
  };

  const executeAction = (alert: Alert, actionId: string) => {
    const action = alert.actions.find(a => a.id === actionId);
    if (!action) return;

    console.log(`Executing action: ${action.action} for alert: ${alert.id}`);

    // Implementation for various actions
    switch (action.action) {
      case 'acknowledge':
        acknowledgeAlert(alert.id);
        break;
      case 'scale_resources':
        // Implementation for resource scaling
        break;
      case 'restart_engine':
        // Implementation for engine restart
        break;
      default:
        console.log('Unknown action:', action.action);
    }
  };

  const getAlertIcon = (type: Alert['type']) => {
    switch (type) {
      case 'success': return <CheckCircle className="h-5 w-5" />;
      case 'warning': return <AlertTriangle className="h-5 w-5" />;
      case 'error': return <XCircle className="h-5 w-5" />;
      case 'info': return <AlertCircle className="h-5 w-5" />;
      default: return <Bell className="h-5 w-5" />;
    }
  };

  const getSeverityColor = (severity: Alert['severity']) => {
    switch (severity) {
      case 'low': return 'bg-blue-100 text-blue-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'high': return 'bg-orange-100 text-orange-800';
      case 'critical': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const filteredAlerts = alerts.filter(alert => {
    const matchesType = filterType === 'all' || alert.type === filterType;
    const matchesSeverity = filterSeverity === 'all' || alert.severity === filterSeverity;
    return matchesType && matchesSeverity;
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Visual Alerts System</h1>
          <p className="text-muted-foreground">
            Real-time visual alerts with customizable notifications
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" onClick={() => setShowCreateRule(true)}>
            <Zap className="h-4 w-4 mr-2" />
            Create Rule
          </Button>
          <Button variant="outline">
            <Settings className="h-4 w-4 mr-2" />
            Settings
          </Button>
        </div>
      </div>

      {/* Alert Statistics */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Total Alerts</p>
                  <p className="text-2xl font-bold">{stats.total}</p>
                </div>
                <Bell className="h-8 w-8 text-muted-foreground" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Active Alerts</p>
                  <p className="text-2xl font-bold text-orange-600">{stats.active}</p>
                </div>
                <Activity className="h-8 w-8 text-orange-600" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Resolved</p>
                  <p className="text-2xl font-bold text-green-600">{stats.resolved}</p>
                </div>
                <CheckCircle className="h-8 w-8 text-green-600" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Avg Resolution</p>
                  <p className="text-2xl font-bold">{stats.averageResolutionTime}m</p>
                </div>
                <TrendingUp className="h-8 w-8 text-muted-foreground" />
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Global Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Alert Settings</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <Label>Enable Visual Alerts</Label>
                <Switch
                  checked={globalSettings.enableVisuals}
                  onCheckedChange={(checked) =>
                    setGlobalSettings(prev => ({ ...prev, enableVisuals: checked }))
                  }
                />
              </div>

              <div className="flex items-center justify-between">
                <Label>Enable Sound Alerts</Label>
                <Switch
                  checked={globalSettings.enableSounds}
                  onCheckedChange={(checked) =>
                    setGlobalSettings(prev => ({ ...prev, enableSounds: checked }))
                  }
                />
              </div>

              <div className="flex items-center justify-between">
                <Label>Auto Acknowledge</Label>
                <Switch
                  checked={globalSettings.autoAcknowledge}
                  onCheckedChange={(checked) =>
                    setGlobalSettings(prev => ({ ...prev, autoAcknowledge: checked }))
                  }
                />
              </div>
            </div>

            <div className="space-y-4">
              <div className="space-y-2">
                <Label>Sound Volume: {globalSettings.soundVolume}%</Label>
                <Slider
                  value={[globalSettings.soundVolume]}
                  onValueChange={([value]) =>
                    setGlobalSettings(prev => ({ ...prev, soundVolume: value }))
                  }
                  max={100}
                  step={10}
                />
              </div>

              <div className="space-y-2">
                <Label>Visual Intensity</Label>
                <Select
                  value={globalSettings.visualIntensity}
                  onValueChange={(value: 'low' | 'medium' | 'high') =>
                    setGlobalSettings(prev => ({ ...prev, visualIntensity: value }))
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="low">Low</SelectItem>
                    <SelectItem value="medium">Medium</SelectItem>
                    <SelectItem value="high">High</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-4">
              <Button
                variant="outline"
                onClick={() => generateNewAlert()}
                className="w-full"
              >
                Test Alert
              </Button>

              <Button
                variant="outline"
                onClick={() => {
                  if (Notification.permission !== 'granted') {
                    Notification.requestPermission();
                  }
                }}
                className="w-full"
              >
                Enable Notifications
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Filters */}
      <div className="flex items-center space-x-4">
        <Select value={filterType} onValueChange={setFilterType}>
          <SelectTrigger className="w-48">
            <SelectValue placeholder="Filter by type" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Types</SelectItem>
            <SelectItem value="success">Success</SelectItem>
            <SelectItem value="warning">Warning</SelectItem>
            <SelectItem value="error">Error</SelectItem>
            <SelectItem value="info">Info</SelectItem>
          </SelectContent>
        </Select>

        <Select value={filterSeverity} onValueChange={setFilterSeverity}>
          <SelectTrigger className="w-48">
            <SelectValue placeholder="Filter by severity" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Severities</SelectItem>
            <SelectItem value="low">Low</SelectItem>
            <SelectItem value="medium">Medium</SelectItem>
            <SelectItem value="high">High</SelectItem>
            <SelectItem value="critical">Critical</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Alerts List */}
      <div className="space-y-4">
        {filteredAlerts.map(alert => (
          <Card
            key={alert.id}
            className={`transition-all duration-300 ${
              !alert.isRead ? 'border-l-4 border-l-primary' : ''
            } ${alert.visual.showBanner ? 'shadow-lg' : ''}`}
            style={{
              borderLeftColor: !alert.isRead ? alert.visual.color : undefined
            }}
          >
            <CardContent className="p-6">
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-3 flex-1">
                  <div className={`text-${alert.type === 'success' ? 'green' : alert.type === 'warning' ? 'yellow' : alert.type === 'error' ? 'red' : 'blue'}-600`}>
                    {getAlertIcon(alert.type)}
                  </div>

                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                      <h3 className="font-semibold">{alert.title}</h3>
                      <Badge className={getSeverityColor(alert.severity)}>
                        {alert.severity}
                      </Badge>
                      <Badge variant="outline">{alert.category}</Badge>
                      {alert.isAcknowledged && (
                        <Badge variant="secondary">Acknowledged</Badge>
                      )}
                    </div>

                    <p className="text-sm text-muted-foreground mb-3">
                      {alert.message}
                    </p>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                      <div>
                        <Label className="font-medium">Current Value</Label>
                        <p className="text-muted-foreground">
                          {alert.data.currentValue.toLocaleString()} {alert.data.unit}
                        </p>
                      </div>

                      <div>
                        <Label className="font-medium">Threshold</Label>
                        <p className="text-muted-foreground">
                          {alert.data.threshold.toLocaleString()} {alert.data.unit}
                        </p>
                      </div>

                      <div>
                        <Label className="font-medium">Trend</Label>
                        <div className="flex items-center space-x-1">
                          {alert.data.trend === 'up' ? (
                            <TrendingUp className="h-4 w-4 text-green-600" />
                          ) : alert.data.trend === 'down' ? (
                            <TrendingDown className="h-4 w-4 text-red-600" />
                          ) : (
                            <Activity className="h-4 w-4 text-gray-600" />
                          )}
                          <span className="text-muted-foreground capitalize">
                            {alert.data.trend}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center justify-between mt-4">
                      <div className="text-xs text-muted-foreground">
                        <p>Source: {alert.source}</p>
                        <p>Time: {new Date(alert.timestamp).toLocaleString()}</p>
                      </div>

                      <div className="flex space-x-2">
                        {alert.actions.map(action => (
                          <Button
                            key={action.id}
                            variant={action.type === 'primary' ? 'default' : 'outline'}
                            size="sm"
                            onClick={() => executeAction(alert, action.id)}
                          >
                            {action.label}
                          </Button>
                        ))}
                        {!alert.isAcknowledged && (
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => acknowledgeAlert(alert.id)}
                          >
                            <Eye className="h-4 w-4 mr-1" />
                            Acknowledge
                          </Button>
                        )}
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => dismissAlert(alert.id)}
                        >
                          Dismiss
                        </Button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}

        {filteredAlerts.length === 0 && (
          <Card>
            <CardContent className="p-12 text-center">
              <CheckCircle className="h-12 w-12 mx-auto mb-4 text-green-600" />
              <h3 className="text-lg font-medium mb-2">No Active Alerts</h3>
              <p className="text-muted-foreground">
                All systems are running normally. You'll see alerts here when they occur.
              </p>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
};

export default VisualAlerts;