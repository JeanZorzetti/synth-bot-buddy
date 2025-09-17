'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Send, Paperclip, Smile, Phone, Video, MoreVertical, X, Minimize2, Maximize2 } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Textarea } from '@/components/ui/textarea';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';

interface ChatMessage {
  id: string;
  senderId: string;
  senderName: string;
  senderRole: 'customer' | 'agent' | 'bot';
  content: string;
  timestamp: string;
  type: 'text' | 'file' | 'system';
  fileName?: string;
  fileSize?: number;
  isRead: boolean;
}

interface ChatSession {
  id: string;
  participantId: string;
  participantName: string;
  participantEmail: string;
  agentId?: string;
  agentName?: string;
  status: 'waiting' | 'active' | 'closed';
  startTime: string;
  lastActivity: string;
  subject: string;
  priority: 'low' | 'medium' | 'high' | 'urgent';
  queue: string;
  tags: string[];
}

interface Agent {
  id: string;
  name: string;
  email: string;
  avatar?: string;
  status: 'online' | 'away' | 'busy' | 'offline';
  expertise: string[];
  activeChats: number;
  maxChats: number;
}

const LiveChat: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [currentSession, setCurrentSession] = useState<ChatSession | null>(null);
  const [agent, setAgent] = useState<Agent | null>(null);
  const [isTyping, setIsTyping] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected');
  const [queuePosition, setQueuePosition] = useState<number | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Initialize WebSocket connection
  useEffect(() => {
    if (isOpen && !wsRef.current) {
      connectWebSocket();
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [isOpen]);

  const connectWebSocket = () => {
    setConnectionStatus('connecting');

    // Replace with your actual WebSocket URL
    const wsUrl = process.env.NODE_ENV === 'production'
      ? 'wss://botderivapi.roilabs.com.br/ws/chat'
      : 'ws://localhost:8000/ws/chat';

    wsRef.current = new WebSocket(wsUrl);

    wsRef.current.onopen = () => {
      setConnectionStatus('connected');
      // Send authentication and initialize chat session
      const initMessage = {
        type: 'init_chat',
        data: {
          userId: 'user_123', // Get from auth context
          subject: 'General Support',
          priority: 'medium'
        }
      };
      wsRef.current?.send(JSON.stringify(initMessage));
    };

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleWebSocketMessage(data);
    };

    wsRef.current.onclose = () => {
      setConnectionStatus('disconnected');
      // Attempt to reconnect after 3 seconds
      setTimeout(() => {
        if (isOpen) connectWebSocket();
      }, 3000);
    };

    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnectionStatus('disconnected');
    };
  };

  const handleWebSocketMessage = (data: any) => {
    switch (data.type) {
      case 'session_created':
        setCurrentSession(data.session);
        setQueuePosition(data.queuePosition);
        break;

      case 'agent_assigned':
        setAgent(data.agent);
        setQueuePosition(null);
        break;

      case 'message':
        const newMessage: ChatMessage = {
          id: data.id,
          senderId: data.senderId,
          senderName: data.senderName,
          senderRole: data.senderRole,
          content: data.content,
          timestamp: data.timestamp,
          type: data.messageType || 'text',
          fileName: data.fileName,
          fileSize: data.fileSize,
          isRead: data.senderRole !== 'customer'
        };
        setMessages(prev => [...prev, newMessage]);
        break;

      case 'agent_typing':
        setIsTyping(data.isTyping);
        break;

      case 'queue_update':
        setQueuePosition(data.position);
        break;

      case 'session_closed':
        setCurrentSession(null);
        setAgent(null);
        setQueuePosition(null);
        break;
    }
  };

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = () => {
    if (!message.trim() || !wsRef.current || connectionStatus !== 'connected') return;

    const messageData = {
      type: 'send_message',
      data: {
        sessionId: currentSession?.id,
        content: message.trim(),
        messageType: 'text'
      }
    };

    wsRef.current.send(JSON.stringify(messageData));
    setMessage('');
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const startChat = () => {
    setIsOpen(true);
  };

  const closeChat = () => {
    setIsOpen(false);
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setMessages([]);
    setCurrentSession(null);
    setAgent(null);
    setQueuePosition(null);
  };

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  // Chat Widget (Minimized)
  if (!isOpen) {
    return (
      <div className="fixed bottom-4 right-4 z-50">
        <Button
          onClick={startChat}
          className="rounded-full h-14 w-14 shadow-lg hover:shadow-xl transition-all"
          size="lg"
        >
          <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
        </Button>
      </div>
    );
  }

  // Chat Window
  return (
    <div className={`fixed bottom-4 right-4 z-50 ${isMinimized ? 'w-80' : 'w-96'} ${isMinimized ? 'h-14' : 'h-[600px]'} transition-all duration-300`}>
      <Card className="h-full shadow-xl">
        {/* Header */}
        <CardHeader className="p-4 border-b">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              {agent ? (
                <>
                  <Avatar className="h-8 w-8">
                    <AvatarImage src={agent.avatar} />
                    <AvatarFallback>{agent.name[0]}</AvatarFallback>
                  </Avatar>
                  <div>
                    <p className="font-medium text-sm">{agent.name}</p>
                    <p className="text-xs text-muted-foreground">Support Agent</p>
                  </div>
                </>
              ) : (
                <div>
                  <p className="font-medium text-sm">Live Support</p>
                  <p className="text-xs text-muted-foreground">
                    {queuePosition ? `Position in queue: ${queuePosition}` : 'Connecting...'}
                  </p>
                </div>
              )}
            </div>

            <div className="flex items-center space-x-2">
              {connectionStatus === 'connected' && (
                <Badge variant="secondary" className="text-xs">
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-1"></div>
                  Online
                </Badge>
              )}

              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="sm">
                    <MoreVertical className="h-4 w-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent>
                  <DropdownMenuItem>
                    <Phone className="h-4 w-4 mr-2" />
                    Request Call
                  </DropdownMenuItem>
                  <DropdownMenuItem>
                    <Video className="h-4 w-4 mr-2" />
                    Video Call
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => setIsMinimized(!isMinimized)}>
                    {isMinimized ? <Maximize2 className="h-4 w-4 mr-2" /> : <Minimize2 className="h-4 w-4 mr-2" />}
                    {isMinimized ? 'Maximize' : 'Minimize'}
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={closeChat}>
                    <X className="h-4 w-4 mr-2" />
                    End Chat
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>
        </CardHeader>

        {!isMinimized && (
          <>
            {/* Messages */}
            <CardContent className="p-0 flex-1 flex flex-col h-[400px]">
              <ScrollArea className="flex-1 p-4">
                <div className="space-y-4">
                  {/* Welcome Message */}
                  {messages.length === 0 && (
                    <div className="text-center text-muted-foreground text-sm py-8">
                      <div className="mb-4">
                        <svg className="h-12 w-12 mx-auto mb-3 text-muted-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                        </svg>
                      </div>
                      <p className="font-medium mb-2">Welcome to Live Support!</p>
                      <p>We're here to help. Send us a message and an agent will be with you shortly.</p>
                    </div>
                  )}

                  {messages.map((msg) => (
                    <div
                      key={msg.id}
                      className={`flex ${msg.senderRole === 'customer' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div className={`max-w-[80%] ${msg.senderRole === 'customer' ? 'order-2' : 'order-1'}`}>
                        {msg.senderRole !== 'customer' && (
                          <div className="flex items-center space-x-2 mb-1">
                            <Avatar className="h-6 w-6">
                              <AvatarFallback className="text-xs">{msg.senderName[0]}</AvatarFallback>
                            </Avatar>
                            <span className="text-xs text-muted-foreground">{msg.senderName}</span>
                          </div>
                        )}

                        <div
                          className={`rounded-lg p-3 ${
                            msg.senderRole === 'customer'
                              ? 'bg-primary text-primary-foreground'
                              : msg.senderRole === 'bot'
                              ? 'bg-muted'
                              : 'bg-secondary'
                          }`}
                        >
                          <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                          {msg.type === 'file' && msg.fileName && (
                            <div className="mt-2 p-2 rounded border bg-background/10">
                              <div className="flex items-center space-x-2">
                                <Paperclip className="h-4 w-4" />
                                <span className="text-xs">{msg.fileName}</span>
                                {msg.fileSize && (
                                  <span className="text-xs text-muted-foreground">
                                    ({(msg.fileSize / 1024).toFixed(1)} KB)
                                  </span>
                                )}
                              </div>
                            </div>
                          )}
                        </div>

                        <div className="flex items-center justify-between mt-1">
                          <span className="text-xs text-muted-foreground">
                            {formatTime(msg.timestamp)}
                          </span>
                          {msg.senderRole === 'customer' && (
                            <span className="text-xs text-muted-foreground">
                              {msg.isRead ? '✓✓' : '✓'}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}

                  {/* Typing Indicator */}
                  {isTyping && (
                    <div className="flex justify-start">
                      <div className="bg-secondary rounded-lg p-3">
                        <div className="flex space-x-1">
                          <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce"></div>
                          <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                          <div className="w-2 h-2 bg-muted-foreground rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
                <div ref={messagesEndRef} />
              </ScrollArea>

              {/* Input Area */}
              <div className="p-4 border-t">
                <div className="flex items-end space-x-2">
                  <div className="flex-1">
                    <Textarea
                      value={message}
                      onChange={(e) => setMessage(e.target.value)}
                      onKeyPress={handleKeyPress}
                      placeholder={
                        connectionStatus === 'connected'
                          ? 'Type your message...'
                          : connectionStatus === 'connecting'
                          ? 'Connecting...'
                          : 'Disconnected'
                      }
                      disabled={connectionStatus !== 'connected'}
                      className="min-h-[40px] max-h-[100px] resize-none"
                      rows={1}
                    />
                  </div>

                  <div className="flex space-x-1">
                    <Button variant="ghost" size="sm" disabled={connectionStatus !== 'connected'}>
                      <Paperclip className="h-4 w-4" />
                    </Button>
                    <Button variant="ghost" size="sm" disabled={connectionStatus !== 'connected'}>
                      <Smile className="h-4 w-4" />
                    </Button>
                    <Button
                      onClick={sendMessage}
                      disabled={!message.trim() || connectionStatus !== 'connected'}
                      size="sm"
                    >
                      <Send className="h-4 w-4" />
                    </Button>
                  </div>
                </div>

                {queuePosition && (
                  <p className="text-xs text-muted-foreground mt-2 text-center">
                    You are #{queuePosition} in queue. Estimated wait time: {queuePosition * 2} minutes
                  </p>
                )}
              </div>
            </CardContent>
          </>
        )}
      </Card>
    </div>
  );
};

export default LiveChat;