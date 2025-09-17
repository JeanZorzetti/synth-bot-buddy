'use client';

import React, { useState, useEffect } from 'react';
import { AlertTriangle, Clock, ArrowUp, Users, Mail, Phone, MessageSquare, Calendar, Zap } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Separator } from '@/components/ui/separator';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';

interface EscalationRule {
  id: string;
  name: string;
  description: string;
  conditions: {
    priority: string[];
    category: string[];
    responseTime: number; // minutes
    customerTier: string[];
    unresponded: boolean;
    reopened: boolean;
    negativeRating: boolean;
  };
  actions: {
    assignToTeam: string;
    notifyManager: boolean;
    escalatePriority: boolean;
    sendEmail: boolean;
    scheduleCall: boolean;
    createAlert: boolean;
  };
  isActive: boolean;
  createdAt: string;
  triggeredCount: number;
  successRate: number;
}

interface EscalationEvent {
  id: string;
  ticketId: string;
  ticketTitle: string;
  customerId: string;
  customerName: string;
  ruleId: string;
  ruleName: string;
  triggerReason: string;
  escalatedFrom: string;
  escalatedTo: string;
  priority: 'low' | 'medium' | 'high' | 'urgent' | 'critical';
  status: 'pending' | 'in_progress' | 'resolved' | 'failed';
  createdAt: string;
  resolvedAt?: string;
  actions: Array<{
    type: string;
    description: string;
    timestamp: string;
    success: boolean;
  }>;
  assignedAgent?: {
    id: string;
    name: string;
    email: string;
    avatar?: string;
  };
}

interface EscalationStats {
  totalEscalations: number;
  activeEscalations: number;
  averageResolutionTime: number;
  successRate: number;
  escalationsByPriority: { [key: string]: number };
  escalationsByReason: { [key: string]: number };
  trendData: Array<{
    date: string;
    escalations: number;
    resolved: number;
  }>;
}

const TicketEscalation: React.FC = () => {
  const [escalationRules, setEscalationRules] = useState<EscalationRule[]>([]);
  const [escalationEvents, setEscalationEvents] = useState<EscalationEvent[]>([]);
  const [stats, setStats] = useState<EscalationStats | null>(null);
  const [selectedEvent, setSelectedEvent] = useState<EscalationEvent | null>(null);
  const [showCreateRule, setShowCreateRule] = useState(false);
  const [filterStatus, setFilterStatus] = useState('all');
  const [filterPriority, setFilterPriority] = useState('all');

  useEffect(() => {
    // Mock data - replace with real API calls
    const mockRules: EscalationRule[] = [
      {
        id: '1',
        name: 'Critical Priority Auto-Escalation',
        description: 'Automatically escalate critical priority tickets to senior support team',
        conditions: {
          priority: ['critical'],
          category: ['Trading Issues', 'Account Security'],
          responseTime: 15,
          customerTier: ['Premium', 'Enterprise'],
          unresponded: true,
          reopened: false,
          negativeRating: false
        },
        actions: {
          assignToTeam: 'Senior Support',
          notifyManager: true,
          escalatePriority: false,
          sendEmail: true,
          scheduleCall: true,
          createAlert: true
        },
        isActive: true,
        createdAt: '2024-01-01T00:00:00Z',
        triggeredCount: 45,
        successRate: 0.92
      },
      {
        id: '2',
        name: 'Unresponded Ticket Escalation',
        description: 'Escalate tickets that haven\'t received a response within SLA',
        conditions: {
          priority: ['high', 'urgent'],
          category: [],
          responseTime: 60,
          customerTier: [],
          unresponded: true,
          reopened: false,
          negativeRating: false
        },
        actions: {
          assignToTeam: 'Team Lead',
          notifyManager: true,
          escalatePriority: true,
          sendEmail: true,
          scheduleCall: false,
          createAlert: true
        },
        isActive: true,
        createdAt: '2024-01-02T00:00:00Z',
        triggeredCount: 78,
        successRate: 0.85
      },
      {
        id: '3',
        name: 'Negative Rating Follow-up',
        description: 'Escalate tickets with negative customer ratings for quality review',
        conditions: {
          priority: [],
          category: [],
          responseTime: 0,
          customerTier: [],
          unresponded: false,
          reopened: false,
          negativeRating: true
        },
        actions: {
          assignToTeam: 'Quality Assurance',
          notifyManager: true,
          escalatePriority: false,
          sendEmail: true,
          scheduleCall: true,
          createAlert: false
        },
        isActive: true,
        createdAt: '2024-01-03T00:00:00Z',
        triggeredCount: 23,
        successRate: 0.96
      }
    ];

    const mockEvents: EscalationEvent[] = [
      {
        id: '1',
        ticketId: 'T-001',
        ticketTitle: 'Unable to access trading account',
        customerId: 'C-001',
        customerName: 'John Doe',
        ruleId: '1',
        ruleName: 'Critical Priority Auto-Escalation',
        triggerReason: 'Critical priority ticket unresponded for 15 minutes',
        escalatedFrom: 'Level 1 Support',
        escalatedTo: 'Senior Support',
        priority: 'critical',
        status: 'in_progress',
        createdAt: '2024-01-15T14:30:00Z',
        actions: [
          {
            type: 'team_assignment',
            description: 'Assigned to Senior Support team',
            timestamp: '2024-01-15T14:30:00Z',
            success: true
          },
          {
            type: 'manager_notification',
            description: 'Notified support manager via email',
            timestamp: '2024-01-15T14:31:00Z',
            success: true
          },
          {
            type: 'customer_email',
            description: 'Sent escalation notification to customer',
            timestamp: '2024-01-15T14:32:00Z',
            success: true
          }
        ],
        assignedAgent: {
          id: 'agent-1',
          name: 'Sarah Johnson',
          email: 'sarah@company.com',
          avatar: '/avatars/sarah.jpg'
        }
      },
      {
        id: '2',
        ticketId: 'T-002',
        ticketTitle: 'API integration not working',
        customerId: 'C-002',
        customerName: 'TechCorp Ltd',
        ruleId: '2',
        ruleName: 'Unresponded Ticket Escalation',
        triggerReason: 'High priority ticket unresponded for 60 minutes',
        escalatedFrom: 'Level 1 Support',
        escalatedTo: 'Team Lead',
        priority: 'high',
        status: 'resolved',
        createdAt: '2024-01-15T13:00:00Z',
        resolvedAt: '2024-01-15T15:30:00Z',
        actions: [
          {
            type: 'team_assignment',
            description: 'Assigned to Team Lead',
            timestamp: '2024-01-15T13:00:00Z',
            success: true
          },
          {
            type: 'priority_escalation',
            description: 'Priority escalated to urgent',
            timestamp: '2024-01-15T13:01:00Z',
            success: true
          },
          {
            type: 'resolution',
            description: 'Issue resolved and customer notified',
            timestamp: '2024-01-15T15:30:00Z',
            success: true
          }
        ]
      }
    ];

    const mockStats: EscalationStats = {
      totalEscalations: 146,
      activeEscalations: 12,
      averageResolutionTime: 45, // minutes
      successRate: 0.89,
      escalationsByPriority: {
        'critical': 25,
        'urgent': 34,
        'high': 45,
        'medium': 32,
        'low': 10
      },
      escalationsByReason: {
        'Unresponded': 65,
        'Critical Priority': 35,
        'Negative Rating': 23,
        'Reopened': 15,
        'Customer Request': 8
      },
      trendData: [
        { date: '2024-01-10', escalations: 8, resolved: 7 },
        { date: '2024-01-11', escalations: 12, resolved: 10 },
        { date: '2024-01-12', escalations: 6, resolved: 8 },
        { date: '2024-01-13', escalations: 15, resolved: 12 },
        { date: '2024-01-14', escalations: 9, resolved: 11 },
        { date: '2024-01-15', escalations: 11, resolved: 8 }
      ]
    };

    setEscalationRules(mockRules);
    setEscalationEvents(mockEvents);
    setStats(mockStats);
  }, []);

  const handleResolveEscalation = (eventId: string) => {
    setEscalationEvents(prev =>
      prev.map(event =>
        event.id === eventId
          ? {
              ...event,
              status: 'resolved',
              resolvedAt: new Date().toISOString(),
              actions: [
                ...event.actions,
                {
                  type: 'manual_resolution',
                  description: 'Manually marked as resolved',
                  timestamp: new Date().toISOString(),
                  success: true
                }
              ]
            }
          : event
      )
    );
  };

  const handleToggleRule = (ruleId: string) => {
    setEscalationRules(prev =>
      prev.map(rule =>
        rule.id === ruleId ? { ...rule, isActive: !rule.isActive } : rule
      )
    );
  };

  const filteredEvents = escalationEvents.filter(event => {
    const matchesStatus = filterStatus === 'all' || event.status === filterStatus;
    const matchesPriority = filterPriority === 'all' || event.priority === filterPriority;
    return matchesStatus && matchesPriority;
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending': return 'bg-yellow-100 text-yellow-800';
      case 'in_progress': return 'bg-blue-100 text-blue-800';
      case 'resolved': return 'bg-green-100 text-green-800';
      case 'failed': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'critical': return 'bg-red-100 text-red-800';
      case 'urgent': return 'bg-orange-100 text-orange-800';
      case 'high': return 'bg-yellow-100 text-yellow-800';
      case 'medium': return 'bg-blue-100 text-blue-800';
      case 'low': return 'bg-gray-100 text-gray-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Ticket Escalation Management</h1>
          <p className="text-muted-foreground">
            Automated escalation rules and monitoring for critical support issues
          </p>
        </div>
        <Button onClick={() => setShowCreateRule(true)}>
          <Zap className="h-4 w-4 mr-2" />
          Create Rule
        </Button>
      </div>

      {/* Statistics */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Total Escalations</p>
                  <p className="text-2xl font-bold">{stats.totalEscalations}</p>
                </div>
                <ArrowUp className="h-8 w-8 text-muted-foreground" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Active Escalations</p>
                  <p className="text-2xl font-bold text-orange-600">{stats.activeEscalations}</p>
                </div>
                <AlertTriangle className="h-8 w-8 text-orange-600" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Avg Resolution Time</p>
                  <p className="text-2xl font-bold">{stats.averageResolutionTime}m</p>
                </div>
                <Clock className="h-8 w-8 text-muted-foreground" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Success Rate</p>
                  <p className="text-2xl font-bold text-green-600">
                    {Math.round(stats.successRate * 100)}%
                  </p>
                </div>
                <Users className="h-8 w-8 text-green-600" />
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Escalation Rules */}
      <Card>
        <CardHeader>
          <CardTitle>Escalation Rules</CardTitle>
          <CardDescription>
            Configure automatic escalation conditions and actions
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {escalationRules.map(rule => (
              <div key={rule.id} className="border rounded-lg p-4">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                      <h3 className="font-medium">{rule.name}</h3>
                      <Badge variant={rule.isActive ? 'default' : 'secondary'}>
                        {rule.isActive ? 'Active' : 'Inactive'}
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground mb-3">{rule.description}</p>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="font-medium mb-1">Conditions:</p>
                        <ul className="text-muted-foreground space-y-1">
                          {rule.conditions.priority.length > 0 && (
                            <li>• Priority: {rule.conditions.priority.join(', ')}</li>
                          )}
                          {rule.conditions.responseTime > 0 && (
                            <li>• Response time: {rule.conditions.responseTime}min</li>
                          )}
                          {rule.conditions.unresponded && <li>• Unresponded tickets</li>}
                          {rule.conditions.reopened && <li>• Reopened tickets</li>}
                          {rule.conditions.negativeRating && <li>• Negative ratings</li>}
                        </ul>
                      </div>

                      <div>
                        <p className="font-medium mb-1">Actions:</p>
                        <ul className="text-muted-foreground space-y-1">
                          {rule.actions.assignToTeam && (
                            <li>• Assign to: {rule.actions.assignToTeam}</li>
                          )}
                          {rule.actions.notifyManager && <li>• Notify manager</li>}
                          {rule.actions.escalatePriority && <li>• Escalate priority</li>}
                          {rule.actions.sendEmail && <li>• Send email notification</li>}
                          {rule.actions.scheduleCall && <li>• Schedule call</li>}
                        </ul>
                      </div>
                    </div>

                    <div className="flex items-center justify-between mt-3 text-xs text-muted-foreground">
                      <span>Triggered {rule.triggeredCount} times</span>
                      <span>Success rate: {Math.round(rule.successRate * 100)}%</span>
                    </div>
                  </div>

                  <div className="flex space-x-2 ml-4">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleToggleRule(rule.id)}
                    >
                      {rule.isActive ? 'Disable' : 'Enable'}
                    </Button>
                    <Button variant="outline" size="sm">
                      Edit
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Filters */}
      <div className="flex items-center space-x-4">
        <Select value={filterStatus} onValueChange={setFilterStatus}>
          <SelectTrigger className="w-48">
            <SelectValue placeholder="Filter by status" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Status</SelectItem>
            <SelectItem value="pending">Pending</SelectItem>
            <SelectItem value="in_progress">In Progress</SelectItem>
            <SelectItem value="resolved">Resolved</SelectItem>
            <SelectItem value="failed">Failed</SelectItem>
          </SelectContent>
        </Select>

        <Select value={filterPriority} onValueChange={setFilterPriority}>
          <SelectTrigger className="w-48">
            <SelectValue placeholder="Filter by priority" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Priorities</SelectItem>
            <SelectItem value="critical">Critical</SelectItem>
            <SelectItem value="urgent">Urgent</SelectItem>
            <SelectItem value="high">High</SelectItem>
            <SelectItem value="medium">Medium</SelectItem>
            <SelectItem value="low">Low</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Escalation Events */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Escalations</CardTitle>
          <CardDescription>
            Monitor and manage escalated tickets
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {filteredEvents.map(event => (
              <div key={event.id} className="border rounded-lg p-4">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                      <h3 className="font-medium">{event.ticketTitle}</h3>
                      <Badge className={getStatusColor(event.status)}>
                        {event.status.replace('_', ' ')}
                      </Badge>
                      <Badge className={getPriorityColor(event.priority)}>
                        {event.priority}
                      </Badge>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-muted-foreground mb-3">
                      <div>
                        <p><strong>Ticket:</strong> {event.ticketId}</p>
                        <p><strong>Customer:</strong> {event.customerName}</p>
                      </div>
                      <div>
                        <p><strong>Escalated from:</strong> {event.escalatedFrom}</p>
                        <p><strong>Escalated to:</strong> {event.escalatedTo}</p>
                      </div>
                      <div>
                        <p><strong>Rule:</strong> {event.ruleName}</p>
                        <p><strong>Created:</strong> {new Date(event.createdAt).toLocaleString()}</p>
                      </div>
                    </div>

                    <div className="bg-muted/50 rounded p-3 mb-3">
                      <p className="text-sm"><strong>Trigger Reason:</strong> {event.triggerReason}</p>
                    </div>

                    {event.assignedAgent && (
                      <div className="flex items-center space-x-2 mb-3">
                        <Avatar className="h-6 w-6">
                          <AvatarImage src={event.assignedAgent.avatar} />
                          <AvatarFallback>{event.assignedAgent.name[0]}</AvatarFallback>
                        </Avatar>
                        <span className="text-sm">
                          Assigned to {event.assignedAgent.name}
                        </span>
                      </div>
                    )}

                    {/* Actions Timeline */}
                    <div className="space-y-2">
                      <p className="text-sm font-medium">Actions Taken:</p>
                      {event.actions.map((action, index) => (
                        <div key={index} className="flex items-center space-x-2 text-xs">
                          <div className={`w-2 h-2 rounded-full ${action.success ? 'bg-green-500' : 'bg-red-500'}`} />
                          <span className="text-muted-foreground">
                            {new Date(action.timestamp).toLocaleTimeString()}
                          </span>
                          <span>{action.description}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="flex flex-col space-y-2 ml-4">
                    {event.status === 'in_progress' && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleResolveEscalation(event.id)}
                      >
                        Mark Resolved
                      </Button>
                    )}
                    <Button variant="outline" size="sm">
                      View Ticket
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default TicketEscalation;