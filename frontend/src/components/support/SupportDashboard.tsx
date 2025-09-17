/**
 * 🎧 SUPPORT DASHBOARD COMPONENT
 * Complete technical support system with tickets, chat, and knowledge base
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
import { Progress } from '../ui/progress';
import {
  Headphones,
  MessageSquare,
  BookOpen,
  Plus,
  Search,
  Filter,
  Clock,
  AlertTriangle,
  CheckCircle,
  User,
  Mail,
  Phone,
  MessageCircle,
  FileText,
  Star,
  ThumbsUp,
  ThumbsDown,
  Send,
  Upload,
  Download,
  Eye,
  Edit,
  Trash2,
  RefreshCw,
  Settings,
  Bell,
  Users,
  BarChart3,
  TrendingUp,
  Calendar,
  Globe,
  Shield,
  Zap,
  HelpCircle,
  ExternalLink,
  Copy,
  Archive,
  Flag,
  Hash,
  Tag,
  Paperclip,
  Image,
  Video,
  Mic,
  Camera,
  MapPin,
  Timer,
  Activity,
  AlertCircle
} from 'lucide-react';

interface SupportTicket {
  id: string;
  number: string;
  subject: string;
  description: string;
  status: 'open' | 'in_progress' | 'waiting_customer' | 'resolved' | 'closed';
  priority: 'low' | 'medium' | 'high' | 'urgent';
  category: string;
  tags: string[];
  customer: {
    id: string;
    name: string;
    email: string;
    plan: string;
    avatar?: string;
  };
  assignedTo?: {
    id: string;
    name: string;
    avatar?: string;
    department: string;
  };
  createdAt: string;
  updatedAt: string;
  lastResponseAt?: string;
  responseTime?: number;
  resolutionTime?: number;
  satisfaction?: {
    rating: number;
    feedback?: string;
  };
  attachments: TicketAttachment[];
  messages: TicketMessage[];
  escalated: boolean;
  internalNotes: string[];
}

interface TicketMessage {
  id: string;
  content: string;
  author: {
    id: string;
    name: string;
    type: 'customer' | 'agent' | 'system';
    avatar?: string;
  };
  timestamp: string;
  attachments?: TicketAttachment[];
  isInternal: boolean;
}

interface TicketAttachment {
  id: string;
  filename: string;
  size: number;
  type: string;
  url: string;
  uploadedAt: string;
  uploadedBy: string;
}

interface SupportStats {
  totalTickets: number;
  openTickets: number;
  averageResponseTime: number;
  customerSatisfaction: number;
  resolutionRate: number;
  ticketsByStatus: {
    status: string;
    count: number;
    percentage: number;
  }[];
  ticketsByPriority: {
    priority: string;
    count: number;
    percentage: number;
  }[];
  recentActivity: {
    id: string;
    type: 'created' | 'updated' | 'resolved' | 'escalated';
    ticket: string;
    timestamp: string;
    agent?: string;
  }[];
}

interface KnowledgeArticle {
  id: string;
  title: string;
  content: string;
  category: string;
  tags: string[];
  views: number;
  helpful: number;
  notHelpful: number;
  lastUpdated: string;
  author: string;
  featured: boolean;
}

interface SupportDashboardProps {
  className?: string;
  userRole?: 'customer' | 'agent' | 'admin';
}

export function SupportDashboard({ className = "", userRole = 'customer' }: SupportDashboardProps) {
  const { user } = useAuth();
  const { permissions } = usePermissions();

  const [activeTab, setActiveTab] = useState('tickets');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [tickets, setTickets] = useState<SupportTicket[]>([]);
  const [selectedTicket, setSelectedTicket] = useState<SupportTicket | null>(null);
  const [supportStats, setSupportStats] = useState<SupportStats | null>(null);
  const [knowledgeArticles, setKnowledgeArticles] = useState<KnowledgeArticle[]>([]);

  // Create ticket form
  const [isCreatingTicket, setIsCreatingTicket] = useState(false);
  const [ticketForm, setTicketForm] = useState({
    subject: '',
    description: '',
    category: '',
    priority: 'medium' as const,
    tags: [] as string[],
  });

  // Filters and search
  const [filters, setFilters] = useState({
    search: '',
    status: 'all',
    priority: 'all',
    category: 'all',
    assignedTo: 'all',
  });

  // Chat state
  const [newMessage, setNewMessage] = useState('');
  const [isSendingMessage, setIsSendingMessage] = useState(false);

  useEffect(() => {
    loadSupportData();
  }, [filters]);

  const loadSupportData = async () => {
    try {
      setIsLoading(true);
      setError(null);

      const [ticketsData, statsData, articlesData] = await Promise.all([
        apiClient.get<SupportTicket[]>('/support/tickets', { params: filters }),
        userRole !== 'customer' ? apiClient.get<SupportStats>('/support/stats') : Promise.resolve(null),
        apiClient.get<KnowledgeArticle[]>('/support/knowledge-base')
      ]);

      setTickets(ticketsData);
      if (statsData) setSupportStats(statsData);
      setKnowledgeArticles(articlesData);
    } catch (error: any) {
      setError(error.message || 'Failed to load support data');
    } finally {
      setIsLoading(false);
    }
  };

  const createTicket = async () => {
    try {
      setIsLoading(true);
      const newTicket = await apiClient.post<SupportTicket>('/support/tickets', ticketForm);
      setTickets(prev => [newTicket, ...prev]);
      setIsCreatingTicket(false);
      resetTicketForm();
    } catch (error: any) {
      setError(error.message || 'Failed to create ticket');
    } finally {
      setIsLoading(false);
    }
  };

  const sendMessage = async (ticketId: string) => {
    if (!newMessage.trim()) return;

    try {
      setIsSendingMessage(true);
      const message = await apiClient.post<TicketMessage>(`/support/tickets/${ticketId}/messages`, {
        content: newMessage,
        isInternal: false,
      });

      setTickets(prev => prev.map(ticket =>
        ticket.id === ticketId
          ? { ...ticket, messages: [...ticket.messages, message] }
          : ticket
      ));

      if (selectedTicket?.id === ticketId) {
        setSelectedTicket(prev => prev ? {
          ...prev,
          messages: [...prev.messages, message]
        } : null);
      }

      setNewMessage('');
    } catch (error: any) {
      setError(error.message || 'Failed to send message');
    } finally {
      setIsSendingMessage(false);
    }
  };

  const updateTicketStatus = async (ticketId: string, status: SupportTicket['status']) => {
    try {
      const updatedTicket = await apiClient.put<SupportTicket>(`/support/tickets/${ticketId}`, { status });
      setTickets(prev => prev.map(ticket => ticket.id === ticketId ? updatedTicket : ticket));
      if (selectedTicket?.id === ticketId) {
        setSelectedTicket(updatedTicket);
      }
    } catch (error: any) {
      setError(error.message || 'Failed to update ticket');
    }
  };

  const resetTicketForm = () => {
    setTicketForm({
      subject: '',
      description: '',
      category: '',
      priority: 'medium',
      tags: [],
    });
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getStatusBadge = (status: string) => {
    const variants = {
      open: { className: 'bg-blue-100 text-blue-800', icon: MessageCircle },
      in_progress: { className: 'bg-yellow-100 text-yellow-800', icon: Clock },
      waiting_customer: { className: 'bg-orange-100 text-orange-800', icon: User },
      resolved: { className: 'bg-green-100 text-green-800', icon: CheckCircle },
      closed: { className: 'bg-gray-100 text-gray-800', icon: Archive },
    };

    const variant = variants[status as keyof typeof variants] || variants.open;
    const Icon = variant.icon;

    return (
      <Badge className={variant.className}>
        <Icon className="h-3 w-3 mr-1" />
        {status.replace('_', ' ').toUpperCase()}
      </Badge>
    );
  };

  const getPriorityBadge = (priority: string) => {
    const variants = {
      low: 'bg-gray-100 text-gray-800',
      medium: 'bg-blue-100 text-blue-800',
      high: 'bg-orange-100 text-orange-800',
      urgent: 'bg-red-100 text-red-800',
    };

    return (
      <Badge className={variants[priority as keyof typeof variants] || variants.medium}>
        {priority.toUpperCase()}
      </Badge>
    );
  };

  const renderStatsOverview = () => {
    if (!supportStats || userRole === 'customer') return null;

    return (
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-6">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-blue-100 rounded-lg">
                <FileText className="h-5 w-5 text-blue-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">{supportStats.totalTickets}</p>
                <p className="text-sm text-gray-500">Total Tickets</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-orange-100 rounded-lg">
                <Clock className="h-5 w-5 text-orange-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">{supportStats.openTickets}</p>
                <p className="text-sm text-gray-500">Open Tickets</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-green-100 rounded-lg">
                <Timer className="h-5 w-5 text-green-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">{supportStats.averageResponseTime}h</p>
                <p className="text-sm text-gray-500">Avg Response</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-purple-100 rounded-lg">
                <Star className="h-5 w-5 text-purple-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">{(supportStats.customerSatisfaction * 100).toFixed(1)}%</p>
                <p className="text-sm text-gray-500">Satisfaction</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-indigo-100 rounded-lg">
                <TrendingUp className="h-5 w-5 text-indigo-600" />
              </div>
              <div>
                <p className="text-2xl font-bold">{(supportStats.resolutionRate * 100).toFixed(1)}%</p>
                <p className="text-sm text-gray-500">Resolution Rate</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  };

  const renderTicketsList = () => (
    <div className="space-y-4">
      {/* Filters */}
      <Card>
        <CardContent className="p-4">
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            <div className="relative">
              <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
              <Input
                placeholder="Search tickets..."
                value={filters.search}
                onChange={(e) => setFilters(prev => ({ ...prev, search: e.target.value }))}
                className="pl-10"
              />
            </div>

            <select
              value={filters.status}
              onChange={(e) => setFilters(prev => ({ ...prev, status: e.target.value }))}
              className="px-3 py-2 border rounded-md bg-white"
            >
              <option value="all">All Status</option>
              <option value="open">Open</option>
              <option value="in_progress">In Progress</option>
              <option value="waiting_customer">Waiting Customer</option>
              <option value="resolved">Resolved</option>
              <option value="closed">Closed</option>
            </select>

            <select
              value={filters.priority}
              onChange={(e) => setFilters(prev => ({ ...prev, priority: e.target.value }))}
              className="px-3 py-2 border rounded-md bg-white"
            >
              <option value="all">All Priority</option>
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
              <option value="urgent">Urgent</option>
            </select>

            <select
              value={filters.category}
              onChange={(e) => setFilters(prev => ({ ...prev, category: e.target.value }))}
              className="px-3 py-2 border rounded-md bg-white"
            >
              <option value="all">All Categories</option>
              <option value="technical">Technical</option>
              <option value="billing">Billing</option>
              <option value="account">Account</option>
              <option value="feature_request">Feature Request</option>
              <option value="bug_report">Bug Report</option>
            </select>

            <Button onClick={() => setIsCreatingTicket(true)}>
              <Plus className="h-4 w-4 mr-2" />
              New Ticket
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Tickets */}
      <div className="space-y-3">
        {tickets.map((ticket) => (
          <Card key={ticket.id} className="hover:shadow-md transition-shadow cursor-pointer">
            <CardContent className="p-4" onClick={() => setSelectedTicket(ticket)}>
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-3 mb-2">
                    <span className="font-mono text-sm text-gray-500">#{ticket.number}</span>
                    {getStatusBadge(ticket.status)}
                    {getPriorityBadge(ticket.priority)}
                    {ticket.escalated && (
                      <Badge className="bg-red-100 text-red-800">
                        <Flag className="h-3 w-3 mr-1" />
                        Escalated
                      </Badge>
                    )}
                  </div>

                  <h3 className="font-semibold text-lg mb-2">{ticket.subject}</h3>
                  <p className="text-gray-600 text-sm mb-3 line-clamp-2">{ticket.description}</p>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div className="flex items-center space-x-2">
                      <User className="h-4 w-4 text-gray-400" />
                      <span>{ticket.customer.name}</span>
                    </div>

                    <div className="flex items-center space-x-2">
                      <Clock className="h-4 w-4 text-gray-400" />
                      <span>{formatDate(ticket.createdAt)}</span>
                    </div>

                    <div className="flex items-center space-x-2">
                      <MessageCircle className="h-4 w-4 text-gray-400" />
                      <span>{ticket.messages.length} messages</span>
                    </div>

                    {ticket.assignedTo && (
                      <div className="flex items-center space-x-2">
                        <Headphones className="h-4 w-4 text-gray-400" />
                        <span>{ticket.assignedTo.name}</span>
                      </div>
                    )}
                  </div>

                  {ticket.tags.length > 0 && (
                    <div className="flex flex-wrap gap-2 mt-3">
                      {ticket.tags.map((tag, index) => (
                        <Badge key={index} variant="outline" className="text-xs">
                          <Tag className="h-3 w-3 mr-1" />
                          {tag}
                        </Badge>
                      ))}
                    </div>
                  )}
                </div>

                <div className="flex flex-col items-end space-y-2">
                  {ticket.satisfaction && (
                    <div className="flex items-center space-x-1">
                      {[1, 2, 3, 4, 5].map((star) => (
                        <Star
                          key={star}
                          className={`h-4 w-4 ${
                            star <= ticket.satisfaction!.rating
                              ? 'text-yellow-400 fill-current'
                              : 'text-gray-300'
                          }`}
                        />
                      ))}
                    </div>
                  )}

                  <Button variant="outline" size="sm">
                    <Eye className="h-4 w-4 mr-2" />
                    View
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}

        {tickets.length === 0 && (
          <Card>
            <CardContent className="p-8 text-center">
              <Headphones className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">No support tickets found</p>
              <Button onClick={() => setIsCreatingTicket(true)} className="mt-4">
                Create Your First Ticket
              </Button>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );

  const renderTicketDetail = () => {
    if (!selectedTicket) return null;

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <Card className="max-w-4xl w-full mx-4 max-h-[90vh] overflow-y-auto">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <span className="font-mono text-sm text-gray-500">#{selectedTicket.number}</span>
                {getStatusBadge(selectedTicket.status)}
                {getPriorityBadge(selectedTicket.priority)}
              </div>
              <Button variant="ghost" onClick={() => setSelectedTicket(null)}>
                <AlertCircle className="h-4 w-4" />
              </Button>
            </div>
            <CardTitle>{selectedTicket.subject}</CardTitle>
          </CardHeader>

          <CardContent className="space-y-6">
            {/* Ticket Info */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p className="text-sm text-gray-500">Customer</p>
                <p className="font-medium">{selectedTicket.customer.name}</p>
                <p className="text-sm text-gray-400">{selectedTicket.customer.email}</p>
              </div>

              <div>
                <p className="text-sm text-gray-500">Created</p>
                <p className="font-medium">{formatDate(selectedTicket.createdAt)}</p>
              </div>

              <div>
                <p className="text-sm text-gray-500">Category</p>
                <p className="font-medium capitalize">{selectedTicket.category}</p>
              </div>

              {selectedTicket.assignedTo && (
                <div>
                  <p className="text-sm text-gray-500">Assigned To</p>
                  <p className="font-medium">{selectedTicket.assignedTo.name}</p>
                  <p className="text-sm text-gray-400">{selectedTicket.assignedTo.department}</p>
                </div>
              )}
            </div>

            {/* Status Controls */}
            {userRole !== 'customer' && (
              <div className="flex space-x-2">
                <Button
                  variant="outline"
                  onClick={() => updateTicketStatus(selectedTicket.id, 'in_progress')}
                  disabled={selectedTicket.status === 'in_progress'}
                >
                  Start Progress
                </Button>
                <Button
                  variant="outline"
                  onClick={() => updateTicketStatus(selectedTicket.id, 'waiting_customer')}
                  disabled={selectedTicket.status === 'waiting_customer'}
                >
                  Wait Customer
                </Button>
                <Button
                  variant="outline"
                  onClick={() => updateTicketStatus(selectedTicket.id, 'resolved')}
                  disabled={selectedTicket.status === 'resolved'}
                >
                  Mark Resolved
                </Button>
              </div>
            )}

            {/* Messages */}
            <div className="space-y-4">
              <h4 className="font-medium">Conversation</h4>
              <div className="space-y-4 max-h-96 overflow-y-auto">
                {selectedTicket.messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex ${message.author.type === 'customer' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                        message.author.type === 'customer'
                          ? 'bg-blue-500 text-white'
                          : message.author.type === 'system'
                          ? 'bg-gray-100 text-gray-800'
                          : 'bg-gray-200 text-gray-800'
                      }`}
                    >
                      <div className="flex items-center space-x-2 mb-1">
                        <span className="font-medium text-sm">{message.author.name}</span>
                        <span className="text-xs opacity-75">{formatDate(message.timestamp)}</span>
                      </div>
                      <p className="text-sm">{message.content}</p>
                    </div>
                  </div>
                ))}
              </div>

              {/* New Message */}
              <div className="flex space-x-2">
                <Input
                  value={newMessage}
                  onChange={(e) => setNewMessage(e.target.value)}
                  placeholder="Type your message..."
                  className="flex-1"
                  onKeyPress={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      sendMessage(selectedTicket.id);
                    }
                  }}
                />
                <Button
                  onClick={() => sendMessage(selectedTicket.id)}
                  disabled={isSendingMessage || !newMessage.trim()}
                >
                  {isSendingMessage ? (
                    <RefreshCw className="h-4 w-4 animate-spin" />
                  ) : (
                    <Send className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  };

  const renderCreateTicketForm = () => (
    <Card className="max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle>Create Support Ticket</CardTitle>
        <CardDescription>
          Describe your issue and we'll help you resolve it quickly
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="subject">Subject</Label>
          <Input
            id="subject"
            value={ticketForm.subject}
            onChange={(e) => setTicketForm(prev => ({ ...prev, subject: e.target.value }))}
            placeholder="Brief description of your issue"
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <Label htmlFor="category">Category</Label>
            <select
              id="category"
              value={ticketForm.category}
              onChange={(e) => setTicketForm(prev => ({ ...prev, category: e.target.value }))}
              className="w-full p-2 border rounded-md"
            >
              <option value="">Select category</option>
              <option value="technical">Technical Issue</option>
              <option value="billing">Billing Question</option>
              <option value="account">Account Help</option>
              <option value="feature_request">Feature Request</option>
              <option value="bug_report">Bug Report</option>
            </select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="priority">Priority</Label>
            <select
              id="priority"
              value={ticketForm.priority}
              onChange={(e) => setTicketForm(prev => ({ ...prev, priority: e.target.value as any }))}
              className="w-full p-2 border rounded-md"
            >
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
              <option value="urgent">Urgent</option>
            </select>
          </div>
        </div>

        <div className="space-y-2">
          <Label htmlFor="description">Description</Label>
          <textarea
            id="description"
            value={ticketForm.description}
            onChange={(e) => setTicketForm(prev => ({ ...prev, description: e.target.value }))}
            placeholder="Please provide detailed information about your issue..."
            className="w-full p-3 border rounded-md resize-none"
            rows={6}
          />
        </div>

        <div className="flex space-x-4">
          <Button
            onClick={createTicket}
            disabled={!ticketForm.subject || !ticketForm.description || !ticketForm.category || isLoading}
            className="flex-1"
          >
            {isLoading ? (
              <>
                <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                Creating...
              </>
            ) : (
              <>
                <Plus className="h-4 w-4 mr-2" />
                Create Ticket
              </>
            )}
          </Button>

          <Button
            variant="outline"
            onClick={() => {
              setIsCreatingTicket(false);
              resetTicketForm();
            }}
            className="flex-1"
          >
            Cancel
          </Button>
        </div>
      </CardContent>
    </Card>
  );

  const renderKnowledgeBase = () => (
    <div className="space-y-6">
      {/* Search */}
      <Card>
        <CardContent className="p-4">
          <div className="relative">
            <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
            <Input
              placeholder="Search knowledge base..."
              className="pl-10"
            />
          </div>
        </CardContent>
      </Card>

      {/* Featured Articles */}
      <div>
        <h3 className="text-lg font-semibold mb-4">Featured Articles</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {knowledgeArticles.filter(article => article.featured).map((article) => (
            <Card key={article.id} className="hover:shadow-md transition-shadow cursor-pointer">
              <CardContent className="p-4">
                <h4 className="font-medium mb-2">{article.title}</h4>
                <p className="text-sm text-gray-600 mb-3 line-clamp-3">
                  {article.content.substring(0, 150)}...
                </p>
                <div className="flex items-center justify-between text-xs text-gray-500">
                  <span>{article.views} views</span>
                  <div className="flex items-center space-x-2">
                    <ThumbsUp className="h-3 w-3" />
                    <span>{article.helpful}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* Categories */}
      <div>
        <h3 className="text-lg font-semibold mb-4">Browse by Category</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {['Getting Started', 'Account Management', 'Billing', 'Technical', 'API', 'Trading'].map((category) => (
            <Card key={category} className="hover:shadow-md transition-shadow cursor-pointer">
              <CardContent className="p-4 text-center">
                <BookOpen className="h-8 w-8 text-blue-500 mx-auto mb-2" />
                <h4 className="font-medium">{category}</h4>
                <p className="text-sm text-gray-500">
                  {Math.floor(Math.random() * 20) + 5} articles
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );

  if (!user) {
    return (
      <Alert>
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          Please log in to access support.
        </AlertDescription>
      </Alert>
    );
  }

  if (isCreatingTicket) {
    return <div className={className}>{renderCreateTicketForm()}</div>;
  }

  return (
    <div className={`max-w-7xl mx-auto space-y-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Support Center</h1>
          <p className="text-gray-600">Get help when you need it</p>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Stats Overview for Agents/Admins */}
      {renderStatsOverview()}

      {/* Main Content */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3 max-w-md">
          <TabsTrigger value="tickets" className="flex items-center space-x-2">
            <FileText className="h-4 w-4" />
            <span>Tickets</span>
          </TabsTrigger>
          <TabsTrigger value="knowledge" className="flex items-center space-x-2">
            <BookOpen className="h-4 w-4" />
            <span>Knowledge Base</span>
          </TabsTrigger>
          <TabsTrigger value="chat" className="flex items-center space-x-2">
            <MessageSquare className="h-4 w-4" />
            <span>Live Chat</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="tickets">
          {renderTicketsList()}
        </TabsContent>

        <TabsContent value="knowledge">
          {renderKnowledgeBase()}
        </TabsContent>

        <TabsContent value="chat">
          <Card>
            <CardContent className="p-8 text-center">
              <MessageSquare className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <p className="text-gray-500">Live chat coming soon...</p>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Ticket Detail Modal */}
      {renderTicketDetail()}
    </div>
  );
}