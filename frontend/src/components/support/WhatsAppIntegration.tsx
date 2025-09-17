'use client';

import React, { useState, useEffect } from 'react';
import { MessageCircle, Phone, Send, Image, File, Clock, Check, CheckCheck, Settings } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

interface WhatsAppMessage {
  id: string;
  conversationId: string;
  senderId: string;
  senderName: string;
  senderPhone: string;
  content: string;
  type: 'text' | 'image' | 'document' | 'audio' | 'video';
  timestamp: string;
  status: 'sent' | 'delivered' | 'read' | 'failed';
  isFromCustomer: boolean;
  mediaUrl?: string;
  mediaName?: string;
  mediaSize?: number;
  replyToId?: string;
}

interface WhatsAppConversation {
  id: string;
  customerId: string;
  customerName: string;
  customerPhone: string;
  customerAvatar?: string;
  lastMessage: string;
  lastMessageTime: string;
  unreadCount: number;
  status: 'active' | 'waiting' | 'closed';
  assignedAgent?: {
    id: string;
    name: string;
    avatar?: string;
  };
  tags: string[];
  priority: 'low' | 'medium' | 'high' | 'urgent';
}

interface WhatsAppTemplate {
  id: string;
  name: string;
  category: string;
  content: string;
  variables: string[];
  language: string;
  status: 'approved' | 'pending' | 'rejected';
  usageCount: number;
}

interface WhatsAppSettings {
  businessProfile: {
    name: string;
    description: string;
    phone: string;
    email: string;
    website: string;
    address: string;
  };
  automation: {
    welcomeMessage: string;
    awayMessage: string;
    workingHours: {
      enabled: boolean;
      timezone: string;
      schedule: { [key: string]: { start: string; end: string; enabled: boolean } };
    };
    autoAssignment: boolean;
    chatbotEnabled: boolean;
  };
  notifications: {
    newMessage: boolean;
    mentionKeywords: string[];
    emailNotifications: boolean;
    desktopNotifications: boolean;
  };
}

const WhatsAppIntegration: React.FC = () => {
  const [conversations, setConversations] = useState<WhatsAppConversation[]>([]);
  const [messages, setMessages] = useState<WhatsAppMessage[]>([]);
  const [templates, setTemplates] = useState<WhatsAppTemplate[]>([]);
  const [settings, setSettings] = useState<WhatsAppSettings | null>(null);
  const [selectedConversation, setSelectedConversation] = useState<WhatsAppConversation | null>(null);
  const [newMessage, setNewMessage] = useState('');
  const [selectedTemplate, setSelectedTemplate] = useState('');
  const [isConnected, setIsConnected] = useState(true);
  const [activeTab, setActiveTab] = useState('conversations');

  useEffect(() => {
    // Mock data - replace with real WhatsApp Business API integration
    const mockConversations: WhatsAppConversation[] = [
      {
        id: '1',
        customerId: 'C001',
        customerName: 'John Smith',
        customerPhone: '+1234567890',
        customerAvatar: '/avatars/john.jpg',
        lastMessage: 'Hi, I need help with my trading account',
        lastMessageTime: '2024-01-15T14:30:00Z',
        unreadCount: 2,
        status: 'active',
        assignedAgent: {
          id: 'agent1',
          name: 'Sarah Johnson',
          avatar: '/avatars/sarah.jpg'
        },
        tags: ['trading', 'account-issue'],
        priority: 'high'
      },
      {
        id: '2',
        customerId: 'C002',
        customerName: 'Maria Garcia',
        customerPhone: '+1987654321',
        lastMessage: 'Thank you for the quick support!',
        lastMessageTime: '2024-01-15T13:45:00Z',
        unreadCount: 0,
        status: 'closed',
        tags: ['resolved', 'api'],
        priority: 'medium'
      },
      {
        id: '3',
        customerId: 'C003',
        customerName: 'Tech Solutions Inc',
        customerPhone: '+1555123456',
        lastMessage: 'When will the new API features be available?',
        lastMessageTime: '2024-01-15T12:20:00Z',
        unreadCount: 1,
        status: 'waiting',
        tags: ['api', 'feature-request'],
        priority: 'medium'
      }
    ];

    const mockMessages: WhatsAppMessage[] = [
      {
        id: '1',
        conversationId: '1',
        senderId: 'C001',
        senderName: 'John Smith',
        senderPhone: '+1234567890',
        content: 'Hi, I need help with my trading account',
        type: 'text',
        timestamp: '2024-01-15T14:30:00Z',
        status: 'read',
        isFromCustomer: true
      },
      {
        id: '2',
        conversationId: '1',
        senderId: 'agent1',
        senderName: 'Sarah Johnson',
        senderPhone: 'support',
        content: 'Hello John! I\'d be happy to help you with your trading account. Can you please describe the specific issue you\'re experiencing?',
        type: 'text',
        timestamp: '2024-01-15T14:32:00Z',
        status: 'read',
        isFromCustomer: false
      },
      {
        id: '3',
        conversationId: '1',
        senderId: 'C001',
        senderName: 'John Smith',
        senderPhone: '+1234567890',
        content: 'I can\'t login to my account. It says my credentials are invalid',
        type: 'text',
        timestamp: '2024-01-15T14:35:00Z',
        status: 'read',
        isFromCustomer: true
      }
    ];

    const mockTemplates: WhatsAppTemplate[] = [
      {
        id: '1',
        name: 'Welcome Message',
        category: 'greeting',
        content: 'Hello {{name}}! Welcome to our AI Trading Platform. How can we assist you today?',
        variables: ['name'],
        language: 'en',
        status: 'approved',
        usageCount: 156
      },
      {
        id: '2',
        name: 'Account Verification',
        category: 'verification',
        content: 'Hi {{name}}, please verify your account by clicking this link: {{verification_link}}',
        variables: ['name', 'verification_link'],
        language: 'en',
        status: 'approved',
        usageCount: 89
      },
      {
        id: '3',
        name: 'Trading Alert',
        category: 'notification',
        content: 'Trading Alert: {{symbol}} has reached your target price of {{price}}. Take action now!',
        variables: ['symbol', 'price'],
        language: 'en',
        status: 'approved',
        usageCount: 234
      }
    ];

    const mockSettings: WhatsAppSettings = {
      businessProfile: {
        name: 'AI Trading Support',
        description: 'Professional AI-powered trading platform support',
        phone: '+1-800-TRADING',
        email: 'support@aitrading.com',
        website: 'https://aitrading.com',
        address: '123 Financial District, NY'
      },
      automation: {
        welcomeMessage: 'Welcome to AI Trading Support! How can we help you today?',
        awayMessage: 'We\'re currently away but will respond as soon as possible.',
        workingHours: {
          enabled: true,
          timezone: 'America/New_York',
          schedule: {
            monday: { start: '09:00', end: '18:00', enabled: true },
            tuesday: { start: '09:00', end: '18:00', enabled: true },
            wednesday: { start: '09:00', end: '18:00', enabled: true },
            thursday: { start: '09:00', end: '18:00', enabled: true },
            friday: { start: '09:00', end: '18:00', enabled: true },
            saturday: { start: '10:00', end: '16:00', enabled: false },
            sunday: { start: '10:00', end: '16:00', enabled: false }
          }
        },
        autoAssignment: true,
        chatbotEnabled: true
      },
      notifications: {
        newMessage: true,
        mentionKeywords: ['urgent', 'emergency', 'help', 'issue'],
        emailNotifications: true,
        desktopNotifications: true
      }
    };

    setConversations(mockConversations);
    setMessages(mockMessages);
    setTemplates(mockTemplates);
    setSettings(mockSettings);
    setSelectedConversation(mockConversations[0]);
  }, []);

  const sendMessage = () => {
    if (!newMessage.trim() || !selectedConversation) return;

    const message: WhatsAppMessage = {
      id: `msg-${Date.now()}`,
      conversationId: selectedConversation.id,
      senderId: 'current-agent',
      senderName: 'Support Agent',
      senderPhone: 'support',
      content: newMessage,
      type: 'text',
      timestamp: new Date().toISOString(),
      status: 'sent',
      isFromCustomer: false
    };

    setMessages(prev => [...prev, message]);
    setNewMessage('');

    // Update conversation last message
    setConversations(prev =>
      prev.map(conv =>
        conv.id === selectedConversation.id
          ? {
              ...conv,
              lastMessage: newMessage,
              lastMessageTime: new Date().toISOString()
            }
          : conv
      )
    );
  };

  const useTemplate = (template: WhatsAppTemplate) => {
    setNewMessage(template.content);
    setSelectedTemplate('');
  };

  const getMessageIcon = (status: string) => {
    switch (status) {
      case 'sent': return <Check className="h-3 w-3 text-gray-400" />;
      case 'delivered': return <CheckCheck className="h-3 w-3 text-gray-400" />;
      case 'read': return <CheckCheck className="h-3 w-3 text-blue-500" />;
      case 'failed': return <span className="text-red-500">!</span>;
      default: return null;
    }
  };

  const conversationMessages = messages.filter(msg => msg.conversationId === selectedConversation?.id);

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center space-x-2">
          <MessageCircle className="h-6 w-6 text-green-600" />
          <h1 className="text-2xl font-bold">WhatsApp Business</h1>
          <Badge variant={isConnected ? 'default' : 'destructive'}>
            {isConnected ? 'Connected' : 'Disconnected'}
          </Badge>
        </div>
        <Button variant="outline" onClick={() => setActiveTab('settings')}>
          <Settings className="h-4 w-4 mr-2" />
          Settings
        </Button>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="conversations">Conversations</TabsTrigger>
          <TabsTrigger value="templates">Templates</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
          <TabsTrigger value="settings">Settings</TabsTrigger>
        </TabsList>

        <TabsContent value="conversations" className="flex-1 flex">
          <div className="flex h-full w-full">
            {/* Conversations List */}
            <div className="w-1/3 border-r flex flex-col">
              <div className="p-4 border-b">
                <Input placeholder="Search conversations..." />
              </div>
              <ScrollArea className="flex-1">
                <div className="p-2 space-y-2">
                  {conversations.map(conversation => (
                    <div
                      key={conversation.id}
                      onClick={() => setSelectedConversation(conversation)}
                      className={`p-3 rounded-lg cursor-pointer hover:bg-muted/50 ${
                        selectedConversation?.id === conversation.id ? 'bg-muted' : ''
                      }`}
                    >
                      <div className="flex items-start space-x-3">
                        <Avatar className="h-10 w-10">
                          <AvatarImage src={conversation.customerAvatar} />
                          <AvatarFallback>{conversation.customerName[0]}</AvatarFallback>
                        </Avatar>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center justify-between">
                            <p className="font-medium text-sm truncate">
                              {conversation.customerName}
                            </p>
                            <div className="flex items-center space-x-1">
                              {conversation.unreadCount > 0 && (
                                <Badge variant="destructive" className="text-xs">
                                  {conversation.unreadCount}
                                </Badge>
                              )}
                              <span className="text-xs text-muted-foreground">
                                {new Date(conversation.lastMessageTime).toLocaleTimeString([], {
                                  hour: '2-digit',
                                  minute: '2-digit'
                                })}
                              </span>
                            </div>
                          </div>
                          <p className="text-xs text-muted-foreground truncate">
                            {conversation.lastMessage}
                          </p>
                          <div className="flex items-center justify-between mt-1">
                            <div className="flex space-x-1">
                              {conversation.tags.slice(0, 2).map(tag => (
                                <Badge key={tag} variant="secondary" className="text-xs">
                                  {tag}
                                </Badge>
                              ))}
                            </div>
                            <Badge variant={
                              conversation.status === 'active' ? 'default' :
                              conversation.status === 'waiting' ? 'secondary' : 'outline'
                            } className="text-xs">
                              {conversation.status}
                            </Badge>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            </div>

            {/* Chat Area */}
            <div className="flex-1 flex flex-col">
              {selectedConversation ? (
                <>
                  {/* Chat Header */}
                  <div className="p-4 border-b">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <Avatar>
                          <AvatarImage src={selectedConversation.customerAvatar} />
                          <AvatarFallback>{selectedConversation.customerName[0]}</AvatarFallback>
                        </Avatar>
                        <div>
                          <p className="font-medium">{selectedConversation.customerName}</p>
                          <p className="text-sm text-muted-foreground">
                            {selectedConversation.customerPhone}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Button variant="outline" size="sm">
                          <Phone className="h-4 w-4" />
                        </Button>
                        <Select>
                          <SelectTrigger className="w-32">
                            <SelectValue placeholder="Actions" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="close">Close Chat</SelectItem>
                            <SelectItem value="transfer">Transfer</SelectItem>
                            <SelectItem value="block">Block User</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                  </div>

                  {/* Messages */}
                  <ScrollArea className="flex-1 p-4">
                    <div className="space-y-4">
                      {conversationMessages.map(message => (
                        <div
                          key={message.id}
                          className={`flex ${message.isFromCustomer ? 'justify-start' : 'justify-end'}`}
                        >
                          <div
                            className={`max-w-[70%] rounded-lg p-3 ${
                              message.isFromCustomer
                                ? 'bg-muted text-foreground'
                                : 'bg-primary text-primary-foreground'
                            }`}
                          >
                            <p className="text-sm">{message.content}</p>
                            <div className={`flex items-center justify-between mt-1 ${
                              message.isFromCustomer ? 'text-muted-foreground' : 'text-primary-foreground/70'
                            }`}>
                              <span className="text-xs">
                                {new Date(message.timestamp).toLocaleTimeString([], {
                                  hour: '2-digit',
                                  minute: '2-digit'
                                })}
                              </span>
                              {!message.isFromCustomer && (
                                <div className="ml-2">
                                  {getMessageIcon(message.status)}
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>

                  {/* Message Input */}
                  <div className="p-4 border-t">
                    <div className="space-y-3">
                      <div className="flex items-center space-x-2">
                        <Select value={selectedTemplate} onValueChange={setSelectedTemplate}>
                          <SelectTrigger className="w-48">
                            <SelectValue placeholder="Quick templates" />
                          </SelectTrigger>
                          <SelectContent>
                            {templates.map(template => (
                              <SelectItem
                                key={template.id}
                                value={template.id}
                                onClick={() => useTemplate(template)}
                              >
                                {template.name}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="flex space-x-2">
                        <Textarea
                          value={newMessage}
                          onChange={(e) => setNewMessage(e.target.value)}
                          placeholder="Type your message..."
                          onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), sendMessage())}
                          className="flex-1 min-h-[40px] max-h-[120px] resize-none"
                          rows={1}
                        />
                        <div className="flex flex-col space-y-1">
                          <Button variant="outline" size="sm">
                            <Image className="h-4 w-4" />
                          </Button>
                          <Button variant="outline" size="sm">
                            <File className="h-4 w-4" />
                          </Button>
                        </div>
                        <Button onClick={sendMessage} disabled={!newMessage.trim()}>
                          <Send className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </div>
                </>
              ) : (
                <div className="flex-1 flex items-center justify-center">
                  <div className="text-center">
                    <MessageCircle className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                    <p className="text-lg font-medium">Select a conversation</p>
                    <p className="text-muted-foreground">Choose a conversation to start messaging</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="templates" className="flex-1">
          <div className="p-6 space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold">Message Templates</h2>
              <Button>Create Template</Button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {templates.map(template => (
                <Card key={template.id}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg">{template.name}</CardTitle>
                      <Badge variant={
                        template.status === 'approved' ? 'default' :
                        template.status === 'pending' ? 'secondary' : 'destructive'
                      }>
                        {template.status}
                      </Badge>
                    </div>
                    <CardDescription>{template.category}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div className="bg-muted/50 rounded p-3">
                        <p className="text-sm">{template.content}</p>
                      </div>

                      {template.variables.length > 0 && (
                        <div>
                          <p className="text-sm font-medium mb-1">Variables:</p>
                          <div className="flex flex-wrap gap-1">
                            {template.variables.map(variable => (
                              <Badge key={variable} variant="outline" className="text-xs">
                                {variable}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}

                      <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <span>Used {template.usageCount} times</span>
                        <span>{template.language.toUpperCase()}</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="analytics" className="flex-1">
          <div className="p-6">
            <h2 className="text-xl font-semibold mb-6">WhatsApp Analytics</h2>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">Total Conversations</p>
                      <p className="text-2xl font-bold">1,234</p>
                    </div>
                    <MessageCircle className="h-8 w-8 text-muted-foreground" />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">Messages Sent</p>
                      <p className="text-2xl font-bold">5,678</p>
                    </div>
                    <Send className="h-8 w-8 text-muted-foreground" />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">Response Rate</p>
                      <p className="text-2xl font-bold">94%</p>
                    </div>
                    <CheckCheck className="h-8 w-8 text-green-600" />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">Avg Response Time</p>
                      <p className="text-2xl font-bold">2.5m</p>
                    </div>
                    <Clock className="h-8 w-8 text-muted-foreground" />
                  </div>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Message Volume Trends</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64 flex items-center justify-center text-muted-foreground">
                  Chart placeholder - Integrate with your preferred charting library
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="settings" className="flex-1">
          {settings && (
            <div className="p-6 space-y-6">
              <h2 className="text-xl font-semibold">WhatsApp Business Settings</h2>

              <Card>
                <CardHeader>
                  <CardTitle>Business Profile</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Business Name</Label>
                      <Input value={settings.businessProfile.name} />
                    </div>
                    <div className="space-y-2">
                      <Label>Phone Number</Label>
                      <Input value={settings.businessProfile.phone} />
                    </div>
                    <div className="space-y-2">
                      <Label>Email</Label>
                      <Input value={settings.businessProfile.email} />
                    </div>
                    <div className="space-y-2">
                      <Label>Website</Label>
                      <Input value={settings.businessProfile.website} />
                    </div>
                  </div>
                  <div className="space-y-2">
                    <Label>Description</Label>
                    <Textarea value={settings.businessProfile.description} />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Automation Settings</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <Label>Enable Chatbot</Label>
                      <p className="text-sm text-muted-foreground">
                        Automatically respond to common queries
                      </p>
                    </div>
                    <Switch checked={settings.automation.chatbotEnabled} />
                  </div>

                  <div className="flex items-center justify-between">
                    <div>
                      <Label>Auto Assignment</Label>
                      <p className="text-sm text-muted-foreground">
                        Automatically assign conversations to available agents
                      </p>
                    </div>
                    <Switch checked={settings.automation.autoAssignment} />
                  </div>

                  <div className="space-y-2">
                    <Label>Welcome Message</Label>
                    <Textarea value={settings.automation.welcomeMessage} />
                  </div>

                  <div className="space-y-2">
                    <Label>Away Message</Label>
                    <Textarea value={settings.automation.awayMessage} />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Notification Settings</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <Label>New Message Notifications</Label>
                    <Switch checked={settings.notifications.newMessage} />
                  </div>

                  <div className="flex items-center justify-between">
                    <Label>Email Notifications</Label>
                    <Switch checked={settings.notifications.emailNotifications} />
                  </div>

                  <div className="flex items-center justify-between">
                    <Label>Desktop Notifications</Label>
                    <Switch checked={settings.notifications.desktopNotifications} />
                  </div>

                  <div className="space-y-2">
                    <Label>Mention Keywords</Label>
                    <Input
                      value={settings.notifications.mentionKeywords.join(', ')}
                      placeholder="urgent, emergency, help"
                    />
                    <p className="text-xs text-muted-foreground">
                      Comma-separated keywords that trigger priority notifications
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default WhatsAppIntegration;