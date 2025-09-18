'use client';

import React, { useState, useEffect } from 'react';
import { Plus, Settings, Move, Trash2, Edit3, BarChart3, TrendingUp, Target, DollarSign, Users, Activity, Eye, Save, RotateCcw, Layout, Grid3X3 } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Slider } from '@/components/ui/slider';

interface KPIWidget {
  id: string;
  title: string;
  description: string;
  type: 'metric' | 'chart' | 'gauge' | 'list' | 'table';
  position: { x: number; y: number; width: number; height: number };
  config: {
    dataSource: string;
    metric: string;
    visualization: string;
    refreshInterval: number;
    showTrend: boolean;
    showTarget: boolean;
    colorScheme: string;
    thresholds: {
      warning: number;
      critical: number;
    };
  };
  data: {
    current: number;
    previous: number;
    target?: number;
    unit: string;
    trend: 'up' | 'down' | 'stable';
    status: 'good' | 'warning' | 'critical';
  };
  permissions: {
    canEdit: boolean;
    canDelete: boolean;
    canMove: boolean;
  };
}

interface DashboardLayout {
  id: string;
  name: string;
  description: string;
  isDefault: boolean;
  widgets: KPIWidget[];
  settings: {
    autoRefresh: boolean;
    refreshInterval: number;
    showGrid: boolean;
    gridSize: number;
    theme: 'light' | 'dark' | 'auto';
  };
  sharing: {
    isPublic: boolean;
    allowedUsers: string[];
    permalink: string;
  };
  createdAt: string;
  updatedAt: string;
}

interface WidgetTemplate {
  id: string;
  name: string;
  description: string;
  type: KPIWidget['type'];
  defaultConfig: KPIWidget['config'];
  category: string;
  icon: React.ReactNode;
}

const CustomKPIDashboard: React.FC = () => {
  const [layouts, setLayouts] = useState<DashboardLayout[]>([]);
  const [currentLayout, setCurrentLayout] = useState<DashboardLayout | null>(null);
  const [editMode, setEditMode] = useState(false);
  const [selectedWidget, setSelectedWidget] = useState<KPIWidget | null>(null);
  const [showWidgetLibrary, setShowWidgetLibrary] = useState(false);
  const [showLayoutSettings, setShowLayoutSettings] = useState(false);
  const [widgetTemplates, setWidgetTemplates] = useState<WidgetTemplate[]>([]);
  const [draggedWidget, setDraggedWidget] = useState<string | null>(null);

  useEffect(() => {
    // Mock widget templates
    const mockTemplates: WidgetTemplate[] = [
      {
        id: 'revenue-metric',
        name: 'Revenue Metric',
        description: 'Display total revenue with trend',
        type: 'metric',
        defaultConfig: {
          dataSource: 'trading_api',
          metric: 'total_revenue',
          visualization: 'number',
          refreshInterval: 30,
          showTrend: true,
          showTarget: true,
          colorScheme: 'green',
          thresholds: { warning: 80, critical: 90 }
        },
        category: 'Financial',
        icon: <DollarSign className="h-5 w-5" />
      },
      {
        id: 'user-count',
        name: 'Active Users',
        description: 'Current active user count',
        type: 'metric',
        defaultConfig: {
          dataSource: 'user_analytics',
          metric: 'active_users',
          visualization: 'number',
          refreshInterval: 60,
          showTrend: true,
          showTarget: false,
          colorScheme: 'blue',
          thresholds: { warning: 1000, critical: 500 }
        },
        category: 'Users',
        icon: <Users className="h-5 w-5" />
      },
      {
        id: 'performance-gauge',
        name: 'Performance Gauge',
        description: 'System performance indicator',
        type: 'gauge',
        defaultConfig: {
          dataSource: 'system_metrics',
          metric: 'performance_score',
          visualization: 'gauge',
          refreshInterval: 15,
          showTrend: false,
          showTarget: true,
          colorScheme: 'rainbow',
          thresholds: { warning: 70, critical: 50 }
        },
        category: 'System',
        icon: <Target className="h-5 w-5" />
      },
      {
        id: 'trend-chart',
        name: 'Trend Chart',
        description: 'Time series trend visualization',
        type: 'chart',
        defaultConfig: {
          dataSource: 'trading_metrics',
          metric: 'profit_trend',
          visualization: 'line',
          refreshInterval: 30,
          showTrend: true,
          showTarget: false,
          colorScheme: 'blue',
          thresholds: { warning: 0, critical: -10 }
        },
        category: 'Trading',
        icon: <TrendingUp className="h-5 w-5" />
      }
    ];

    // Mock dashboard layouts
    const mockLayouts: DashboardLayout[] = [
      {
        id: 'default',
        name: 'Default Dashboard',
        description: 'Standard trading analytics dashboard',
        isDefault: true,
        widgets: generateDefaultWidgets(),
        settings: {
          autoRefresh: true,
          refreshInterval: 30,
          showGrid: true,
          gridSize: 12,
          theme: 'light'
        },
        sharing: {
          isPublic: false,
          allowedUsers: [],
          permalink: ''
        },
        createdAt: '2024-01-01T00:00:00Z',
        updatedAt: new Date().toISOString()
      }
    ];

    setWidgetTemplates(mockTemplates);
    setLayouts(mockLayouts);
    setCurrentLayout(mockLayouts[0]);
  }, []);

  const generateDefaultWidgets = (): KPIWidget[] => {
    return [
      {
        id: 'widget-1',
        title: 'Total Revenue',
        description: 'Daily trading revenue',
        type: 'metric',
        position: { x: 0, y: 0, width: 3, height: 2 },
        config: {
          dataSource: 'trading_api',
          metric: 'total_revenue',
          visualization: 'number',
          refreshInterval: 30,
          showTrend: true,
          showTarget: true,
          colorScheme: 'green',
          thresholds: { warning: 80000, critical: 50000 }
        },
        data: {
          current: 125840.50,
          previous: 118920.30,
          target: 150000,
          unit: 'USD',
          trend: 'up',
          status: 'good'
        },
        permissions: {
          canEdit: true,
          canDelete: true,
          canMove: true
        }
      },
      {
        id: 'widget-2',
        title: 'Active Trading Bots',
        description: 'Currently running bots',
        type: 'metric',
        position: { x: 3, y: 0, width: 3, height: 2 },
        config: {
          dataSource: 'bot_manager',
          metric: 'active_bots',
          visualization: 'number',
          refreshInterval: 30,
          showTrend: true,
          showTarget: true,
          colorScheme: 'blue',
          thresholds: { warning: 1000, critical: 500 }
        },
        data: {
          current: 1247,
          previous: 1189,
          target: 1500,
          unit: 'bots',
          trend: 'up',
          status: 'good'
        },
        permissions: {
          canEdit: true,
          canDelete: true,
          canMove: true
        }
      },
      {
        id: 'widget-3',
        title: 'Win Rate',
        description: 'Overall trading success rate',
        type: 'gauge',
        position: { x: 6, y: 0, width: 3, height: 2 },
        config: {
          dataSource: 'performance_metrics',
          metric: 'win_rate',
          visualization: 'gauge',
          refreshInterval: 60,
          showTrend: false,
          showTarget: true,
          colorScheme: 'rainbow',
          thresholds: { warning: 70, critical: 60 }
        },
        data: {
          current: 78.6,
          previous: 76.2,
          target: 80,
          unit: '%',
          trend: 'up',
          status: 'good'
        },
        permissions: {
          canEdit: true,
          canDelete: true,
          canMove: true
        }
      },
      {
        id: 'widget-4',
        title: 'Performance Trend',
        description: '7-day performance history',
        type: 'chart',
        position: { x: 0, y: 2, width: 6, height: 3 },
        config: {
          dataSource: 'analytics_db',
          metric: 'performance_trend',
          visualization: 'line',
          refreshInterval: 300,
          showTrend: true,
          showTarget: false,
          colorScheme: 'blue',
          thresholds: { warning: 70, critical: 50 }
        },
        data: {
          current: 82.4,
          previous: 79.1,
          unit: '%',
          trend: 'up',
          status: 'good'
        },
        permissions: {
          canEdit: true,
          canDelete: true,
          canMove: true
        }
      }
    ];
  };

  const addWidget = (template: WidgetTemplate) => {
    if (!currentLayout) return;

    const newWidget: KPIWidget = {
      id: `widget-${Date.now()}`,
      title: template.name,
      description: template.description,
      type: template.type,
      position: findEmptyPosition(),
      config: template.defaultConfig,
      data: generateMockData(),
      permissions: {
        canEdit: true,
        canDelete: true,
        canMove: true
      }
    };

    const updatedLayout = {
      ...currentLayout,
      widgets: [...currentLayout.widgets, newWidget],
      updatedAt: new Date().toISOString()
    };

    setCurrentLayout(updatedLayout);
    setLayouts(prev => prev.map(layout =>
      layout.id === currentLayout.id ? updatedLayout : layout
    ));
    setShowWidgetLibrary(false);
  };

  const findEmptyPosition = () => {
    // Simple algorithm to find empty space
    return { x: 9, y: 0, width: 3, height: 2 };
  };

  const generateMockData = () => {
    return {
      current: Math.floor(Math.random() * 1000),
      previous: Math.floor(Math.random() * 1000),
      target: Math.floor(Math.random() * 1500),
      unit: 'units',
      trend: Math.random() > 0.5 ? 'up' : 'down' as 'up' | 'down',
      status: 'good' as 'good' | 'warning' | 'critical'
    };
  };

  const removeWidget = (widgetId: string) => {
    if (!currentLayout) return;

    const updatedLayout = {
      ...currentLayout,
      widgets: currentLayout.widgets.filter(w => w.id !== widgetId),
      updatedAt: new Date().toISOString()
    };

    setCurrentLayout(updatedLayout);
    setLayouts(prev => prev.map(layout =>
      layout.id === currentLayout.id ? updatedLayout : layout
    ));
  };

  const updateWidget = (widgetId: string, updates: Partial<KPIWidget>) => {
    if (!currentLayout) return;

    const updatedLayout = {
      ...currentLayout,
      widgets: currentLayout.widgets.map(w =>
        w.id === widgetId ? { ...w, ...updates } : w
      ),
      updatedAt: new Date().toISOString()
    };

    setCurrentLayout(updatedLayout);
    setLayouts(prev => prev.map(layout =>
      layout.id === currentLayout.id ? updatedLayout : layout
    ));
  };

  const saveLayout = () => {
    if (!currentLayout) return;

    const updatedLayout = {
      ...currentLayout,
      updatedAt: new Date().toISOString()
    };

    setLayouts(prev => prev.map(layout =>
      layout.id === currentLayout.id ? updatedLayout : layout
    ));

    setEditMode(false);
    console.log('Layout saved successfully');
  };

  const resetLayout = () => {
    if (!currentLayout) return;

    const defaultLayout = layouts.find(l => l.isDefault);
    if (defaultLayout) {
      setCurrentLayout({ ...defaultLayout });
    }
  };

  const getWidgetIcon = (type: KPIWidget['type']) => {
    switch (type) {
      case 'metric': return <BarChart3 className="h-4 w-4" />;
      case 'chart': return <TrendingUp className="h-4 w-4" />;
      case 'gauge': return <Target className="h-4 w-4" />;
      case 'list': return <Activity className="h-4 w-4" />;
      case 'table': return <Grid3X3 className="h-4 w-4" />;
      default: return <BarChart3 className="h-4 w-4" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'good': return 'text-green-600';
      case 'warning': return 'text-yellow-600';
      case 'critical': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const renderWidget = (widget: KPIWidget) => {
    const isSelected = selectedWidget?.id === widget.id;

    return (
      <Card
        key={widget.id}
        className={`relative transition-all duration-200 ${
          isSelected ? 'ring-2 ring-primary' : ''
        } ${editMode ? 'cursor-move' : ''}`}
        style={{
          gridColumn: `span ${widget.position.width}`,
          gridRow: `span ${widget.position.height}`
        }}
        onClick={() => setSelectedWidget(widget)}
        draggable={editMode}
        onDragStart={() => setDraggedWidget(widget.id)}
        onDragEnd={() => setDraggedWidget(null)}
      >
        {editMode && (
          <div className="absolute top-2 right-2 z-10 flex space-x-1">
            <Button
              variant="ghost"
              size="sm"
              className="h-6 w-6 p-0"
              onClick={(e) => {
                e.stopPropagation();
                setSelectedWidget(widget);
              }}
            >
              <Edit3 className="h-3 w-3" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 w-6 p-0"
              onClick={(e) => {
                e.stopPropagation();
                removeWidget(widget.id);
              }}
            >
              <Trash2 className="h-3 w-3" />
            </Button>
          </div>
        )}

        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              {getWidgetIcon(widget.type)}
              <CardTitle className="text-sm font-medium">{widget.title}</CardTitle>
            </div>
            <Badge variant="outline" className="text-xs">
              {widget.type}
            </Badge>
          </div>
          <CardDescription className="text-xs">{widget.description}</CardDescription>
        </CardHeader>

        <CardContent>
          {widget.type === 'metric' && (
            <div className="space-y-2">
              <div className="flex items-baseline space-x-2">
                <span className="text-2xl font-bold">
                  {widget.data.unit === 'USD' ? '$' : ''}
                  {widget.data.current.toLocaleString()}
                  {widget.data.unit !== 'USD' ? widget.data.unit : ''}
                </span>
                <div className={`flex items-center space-x-1 ${
                  widget.data.trend === 'up' ? 'text-green-600' : 'text-red-600'
                }`}>
                  {widget.data.trend === 'up' ? <TrendingUp className="h-4 w-4" /> : <TrendingUp className="h-4 w-4 rotate-180" />}
                  <span className="text-sm">
                    {Math.abs(((widget.data.current - widget.data.previous) / widget.data.previous) * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              {widget.config.showTarget && widget.data.target && (
                <div className="space-y-1">
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Progress to target</span>
                    <span>{Math.round((widget.data.current / widget.data.target) * 100)}%</span>
                  </div>
                  <Progress value={(widget.data.current / widget.data.target) * 100} className="h-2" />
                </div>
              )}

              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">
                  Previous: {widget.data.previous.toLocaleString()}
                </span>
                <span className={getStatusColor(widget.data.status)}>
                  ● {widget.data.status.toUpperCase()}
                </span>
              </div>
            </div>
          )}

          {widget.type === 'gauge' && (
            <div className="space-y-4">
              <div className="relative h-32 flex items-center justify-center">
                <div className="w-24 h-24 rounded-full border-8 border-muted relative">
                  <div
                    className="absolute inset-0 rounded-full border-8 border-primary"
                    style={{
                      clipPath: `polygon(50% 50%, 50% 0%, ${50 + (widget.data.current / 100) * 50}% 0%, 100% 100%, 0% 100%)`
                    }}
                  />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-lg font-bold">{widget.data.current}%</span>
                  </div>
                </div>
              </div>

              {widget.config.showTarget && widget.data.target && (
                <div className="text-center text-xs text-muted-foreground">
                  Target: {widget.data.target}%
                </div>
              )}
            </div>
          )}

          {widget.type === 'chart' && (
            <div className="h-32 flex items-center justify-center bg-muted/30 rounded-lg">
              <div className="text-center">
                <TrendingUp className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
                <p className="text-sm text-muted-foreground">Chart visualization</p>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Custom KPI Dashboard</h1>
          <p className="text-muted-foreground">
            Personalized dashboard with drag-and-drop widgets
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant={editMode ? "default" : "outline"}
            onClick={() => setEditMode(!editMode)}
          >
            <Edit3 className="h-4 w-4 mr-2" />
            {editMode ? 'Exit Edit' : 'Edit Mode'}
          </Button>
          {editMode && (
            <>
              <Button variant="outline" onClick={() => setShowWidgetLibrary(true)}>
                <Plus className="h-4 w-4 mr-2" />
                Add Widget
              </Button>
              <Button variant="outline" onClick={saveLayout}>
                <Save className="h-4 w-4 mr-2" />
                Save
              </Button>
              <Button variant="outline" onClick={resetLayout}>
                <RotateCcw className="h-4 w-4 mr-2" />
                Reset
              </Button>
            </>
          )}
          <Button variant="outline" onClick={() => setShowLayoutSettings(true)}>
            <Settings className="h-4 w-4 mr-2" />
            Settings
          </Button>
        </div>
      </div>

      {/* Dashboard Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <Select
            value={currentLayout?.id || ''}
            onValueChange={(layoutId) => {
              const layout = layouts.find(l => l.id === layoutId);
              if (layout) setCurrentLayout(layout);
            }}
          >
            <SelectTrigger className="w-64">
              <Layout className="h-4 w-4 mr-2" />
              <SelectValue placeholder="Select layout" />
            </SelectTrigger>
            <SelectContent>
              {layouts.map(layout => (
                <SelectItem key={layout.id} value={layout.id}>
                  {layout.name}
                  {layout.isDefault && <Badge variant="secondary" className="ml-2">Default</Badge>}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {currentLayout?.settings.autoRefresh && (
            <Badge variant="outline" className="flex items-center space-x-1">
              <Activity className="h-3 w-3" />
              <span>Auto-refresh: {currentLayout.settings.refreshInterval}s</span>
            </Badge>
          )}
        </div>

        <div className="text-sm text-muted-foreground">
          Last updated: {currentLayout?.updatedAt ? new Date(currentLayout.updatedAt).toLocaleTimeString() : 'Never'}
        </div>
      </div>

      {/* Dashboard Grid */}
      {currentLayout && (
        <div
          className={`grid gap-4 auto-rows-max ${
            currentLayout.settings.showGrid ? 'border border-dashed border-muted p-4 rounded-lg' : ''
          }`}
          style={{
            gridTemplateColumns: `repeat(${currentLayout.settings.gridSize}, 1fr)`
          }}
        >
          {currentLayout.widgets.map(renderWidget)}
        </div>
      )}

      {/* Widget Library Dialog */}
      <Dialog open={showWidgetLibrary} onOpenChange={setShowWidgetLibrary}>
        <DialogContent className="max-w-4xl">
          <DialogHeader>
            <DialogTitle>Widget Library</DialogTitle>
            <DialogDescription>
              Choose from pre-built widgets to add to your dashboard
            </DialogDescription>
          </DialogHeader>

          <Tabs defaultValue="all" className="space-y-4">
            <TabsList>
              <TabsTrigger value="all">All Widgets</TabsTrigger>
              <TabsTrigger value="financial">Financial</TabsTrigger>
              <TabsTrigger value="trading">Trading</TabsTrigger>
              <TabsTrigger value="system">System</TabsTrigger>
              <TabsTrigger value="users">Users</TabsTrigger>
            </TabsList>

            <TabsContent value="all" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {widgetTemplates.map(template => (
                  <Card key={template.id} className="cursor-pointer hover:shadow-md transition-shadow">
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-2">
                          {template.icon}
                          <CardTitle className="text-sm">{template.name}</CardTitle>
                        </div>
                        <Badge variant="outline" className="text-xs">{template.category}</Badge>
                      </div>
                      <CardDescription className="text-xs">{template.description}</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-xs">
                          <span>Type:</span>
                          <Badge variant="secondary">{template.type}</Badge>
                        </div>
                        <div className="flex items-center justify-between text-xs">
                          <span>Refresh:</span>
                          <span>{template.defaultConfig.refreshInterval}s</span>
                        </div>
                        <Button
                          size="sm"
                          className="w-full"
                          onClick={() => addWidget(template)}
                        >
                          <Plus className="h-3 w-3 mr-1" />
                          Add Widget
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>

            {/* Other category tabs would filter widgets by category */}
          </Tabs>
        </DialogContent>
      </Dialog>

      {/* Layout Settings Dialog */}
      <Dialog open={showLayoutSettings} onOpenChange={setShowLayoutSettings}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Dashboard Settings</DialogTitle>
            <DialogDescription>
              Configure your dashboard layout and behavior
            </DialogDescription>
          </DialogHeader>

          {currentLayout && (
            <div className="space-y-6">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <Label>Auto Refresh</Label>
                  <Switch
                    checked={currentLayout.settings.autoRefresh}
                    onCheckedChange={(checked) => {
                      const updatedLayout = {
                        ...currentLayout,
                        settings: { ...currentLayout.settings, autoRefresh: checked }
                      };
                      setCurrentLayout(updatedLayout);
                    }}
                  />
                </div>

                <div className="space-y-2">
                  <Label>Refresh Interval: {currentLayout.settings.refreshInterval}s</Label>
                  <Slider
                    value={[currentLayout.settings.refreshInterval]}
                    onValueChange={([value]) => {
                      const updatedLayout = {
                        ...currentLayout,
                        settings: { ...currentLayout.settings, refreshInterval: value }
                      };
                      setCurrentLayout(updatedLayout);
                    }}
                    min={5}
                    max={300}
                    step={5}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <Label>Show Grid</Label>
                  <Switch
                    checked={currentLayout.settings.showGrid}
                    onCheckedChange={(checked) => {
                      const updatedLayout = {
                        ...currentLayout,
                        settings: { ...currentLayout.settings, showGrid: checked }
                      };
                      setCurrentLayout(updatedLayout);
                    }}
                  />
                </div>

                <div className="space-y-2">
                  <Label>Grid Columns: {currentLayout.settings.gridSize}</Label>
                  <Slider
                    value={[currentLayout.settings.gridSize]}
                    onValueChange={([value]) => {
                      const updatedLayout = {
                        ...currentLayout,
                        settings: { ...currentLayout.settings, gridSize: value }
                      };
                      setCurrentLayout(updatedLayout);
                    }}
                    min={6}
                    max={24}
                    step={2}
                  />
                </div>

                <div className="space-y-2">
                  <Label>Theme</Label>
                  <Select
                    value={currentLayout.settings.theme}
                    onValueChange={(theme: 'light' | 'dark' | 'auto') => {
                      const updatedLayout = {
                        ...currentLayout,
                        settings: { ...currentLayout.settings, theme }
                      };
                      setCurrentLayout(updatedLayout);
                    }}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="light">Light</SelectItem>
                      <SelectItem value="dark">Dark</SelectItem>
                      <SelectItem value="auto">Auto</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <Separator />

              <div className="space-y-4">
                <h4 className="font-medium">Sharing & Collaboration</h4>

                <div className="flex items-center justify-between">
                  <Label>Public Dashboard</Label>
                  <Switch
                    checked={currentLayout.sharing.isPublic}
                    onCheckedChange={(checked) => {
                      const updatedLayout = {
                        ...currentLayout,
                        sharing: { ...currentLayout.sharing, isPublic: checked }
                      };
                      setCurrentLayout(updatedLayout);
                    }}
                  />
                </div>

                {currentLayout.sharing.isPublic && (
                  <div className="space-y-2">
                    <Label>Public URL</Label>
                    <Input
                      value={`${window.location.origin}/dashboard/public/${currentLayout.id}`}
                      readOnly
                    />
                  </div>
                )}
              </div>

              <div className="flex justify-end space-x-2">
                <Button variant="outline" onClick={() => setShowLayoutSettings(false)}>
                  Cancel
                </Button>
                <Button onClick={() => {
                  saveLayout();
                  setShowLayoutSettings(false);
                }}>
                  Save Settings
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default CustomKPIDashboard;