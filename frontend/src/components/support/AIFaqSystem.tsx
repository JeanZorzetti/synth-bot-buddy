'use client';

import React, { useState, useEffect } from 'react';
import { Search, Bot, ThumbsUp, ThumbsDown, MessageCircle, Lightbulb, Clock, Zap } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Textarea } from '@/components/ui/textarea';

interface FAQItem {
  id: string;
  question: string;
  answer: string;
  category: string;
  tags: string[];
  helpful: number;
  notHelpful: number;
  views: number;
  lastUpdated: string;
  confidence: number;
  relatedQuestions: string[];
}

interface AIResponse {
  id: string;
  query: string;
  answer: string;
  confidence: number;
  sources: Array<{
    id: string;
    title: string;
    type: 'faq' | 'article' | 'manual';
    relevance: number;
  }>;
  timestamp: string;
  wasHelpful?: boolean;
  followUpQuestions: string[];
}

interface QuickAnswer {
  id: string;
  trigger: string;
  response: string;
  category: string;
}

const AIFaqSystem: React.FC = () => {
  const [query, setQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [aiResponse, setAiResponse] = useState<AIResponse | null>(null);
  const [faqs, setFaqs] = useState<FAQItem[]>([]);
  const [quickAnswers, setQuickAnswers] = useState<QuickAnswer[]>([]);
  const [chatHistory, setChatHistory] = useState<AIResponse[]>([]);
  const [showQuickAnswers, setShowQuickAnswers] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState('all');

  useEffect(() => {
    // Mock data - replace with real API calls
    const mockFaqs: FAQItem[] = [
      {
        id: '1',
        question: 'How do I set up my first AI trading bot?',
        answer: 'To set up your first AI trading bot, follow these steps: 1) Create an account and verify your email, 2) Connect your trading account via API, 3) Configure your risk parameters, 4) Select a trading strategy, 5) Start with paper trading to test...',
        category: 'Getting Started',
        tags: ['setup', 'bot', 'beginner'],
        helpful: 89,
        notHelpful: 5,
        views: 1234,
        lastUpdated: '2024-01-15',
        confidence: 0.95,
        relatedQuestions: ['How to configure risk settings?', 'What is paper trading?']
      },
      {
        id: '2',
        question: 'What are the minimum deposit requirements?',
        answer: 'Minimum deposits vary by account type: Trial - $0 (demo only), Basic - $250, Premium - $1,000, Enterprise - $10,000. You can start with our free trial to test the platform...',
        category: 'Billing',
        tags: ['deposit', 'minimum', 'account'],
        helpful: 67,
        notHelpful: 2,
        views: 892,
        lastUpdated: '2024-01-12',
        confidence: 0.98,
        relatedQuestions: ['How to upgrade my account?', 'What payment methods are accepted?']
      },
      {
        id: '3',
        question: 'How accurate are the AI predictions?',
        answer: 'Our AI models achieve 75-85% accuracy in favorable market conditions. Accuracy varies based on market volatility, asset type, and timeframe. Historical backtesting shows consistent performance across different market cycles...',
        category: 'AI & Trading',
        tags: ['accuracy', 'ai', 'predictions'],
        helpful: 123,
        notHelpful: 8,
        views: 2156,
        lastUpdated: '2024-01-14',
        confidence: 0.92,
        relatedQuestions: ['How is accuracy measured?', 'What affects prediction accuracy?']
      }
    ];

    const mockQuickAnswers: QuickAnswer[] = [
      {
        id: '1',
        trigger: 'How to get started',
        response: 'Start by creating an account, then follow our quick setup guide to configure your first trading bot.',
        category: 'Getting Started'
      },
      {
        id: '2',
        trigger: 'Forgot password',
        response: 'Click "Forgot Password" on the login page and check your email for reset instructions.',
        category: 'Account'
      },
      {
        id: '3',
        trigger: 'API key setup',
        response: 'Go to Settings > API Keys, click "Create New Key", set permissions, and copy your key securely.',
        category: 'API'
      },
      {
        id: '4',
        trigger: 'Contact support',
        response: 'Use our live chat, create a support ticket, or email support@yourplatform.com for help.',
        category: 'Support'
      }
    ];

    setFaqs(mockFaqs);
    setQuickAnswers(mockQuickAnswers);
  }, []);

  const handleAIQuery = async () => {
    if (!query.trim()) return;

    setIsLoading(true);
    setShowQuickAnswers(false);

    // Simulate AI processing delay
    await new Promise(resolve => setTimeout(resolve, 2000));

    // Mock AI response - replace with real AI service call
    const mockResponse: AIResponse = {
      id: `ai-${Date.now()}`,
      query: query,
      answer: `Based on your question "${query}", here's what I found: ${generateMockAnswer(query)}`,
      confidence: 0.87,
      sources: [
        {
          id: '1',
          title: 'AI Trading Bot Setup Guide',
          type: 'article',
          relevance: 0.92
        },
        {
          id: '2',
          title: 'Getting Started FAQ',
          type: 'faq',
          relevance: 0.85
        }
      ],
      timestamp: new Date().toISOString(),
      followUpQuestions: [
        'How do I configure risk parameters?',
        'What trading strategies are available?',
        'Can I backtest my strategy?'
      ]
    };

    setAiResponse(mockResponse);
    setChatHistory(prev => [...prev, mockResponse]);
    setIsLoading(false);
  };

  const generateMockAnswer = (question: string): string => {
    const lowerQuestion = question.toLowerCase();

    if (lowerQuestion.includes('setup') || lowerQuestion.includes('start')) {
      return 'To get started with AI trading, first create your account and verify your email. Then connect your trading account through our secure API integration. Configure your risk tolerance and select from our pre-built strategies or create your own. I recommend starting with paper trading to familiarize yourself with the platform.';
    }

    if (lowerQuestion.includes('deposit') || lowerQuestion.includes('money')) {
      return 'Our platform offers flexible account tiers. You can start with a free trial account (no deposit required) to test all features with simulated data. For live trading, minimum deposits are: Basic ($250), Premium ($1,000), and Enterprise ($10,000). All deposits are secured and insured.';
    }

    if (lowerQuestion.includes('accuracy') || lowerQuestion.includes('performance')) {
      return 'Our AI models demonstrate 75-85% accuracy under normal market conditions. Performance varies by asset class - forex typically shows higher accuracy than crypto due to lower volatility. We provide detailed backtesting reports and real-time performance metrics for transparency.';
    }

    return 'I understand your question and I\'m here to help. Based on our knowledge base, I can provide detailed guidance on trading strategies, account setup, risk management, and platform features. Would you like me to elaborate on any specific aspect?';
  };

  const handleFeedback = (responseId: string, isHelpful: boolean) => {
    setChatHistory(prev =>
      prev.map(response =>
        response.id === responseId
          ? { ...response, wasHelpful: isHelpful }
          : response
      )
    );

    if (aiResponse?.id === responseId) {
      setAiResponse(prev =>
        prev ? { ...prev, wasHelpful: isHelpful } : null
      );
    }
  };

  const handleQuickAnswer = (quickAnswer: QuickAnswer) => {
    setQuery(quickAnswer.trigger);
    setAiResponse({
      id: `quick-${Date.now()}`,
      query: quickAnswer.trigger,
      answer: quickAnswer.response,
      confidence: 1.0,
      sources: [],
      timestamp: new Date().toISOString(),
      followUpQuestions: []
    });
    setShowQuickAnswers(false);
  };

  const categories = ['all', 'Getting Started', 'Billing', 'AI & Trading', 'Account', 'API', 'Support'];

  const filteredFaqs = selectedCategory === 'all'
    ? faqs
    : faqs.filter(faq => faq.category === selectedCategory);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center space-y-4">
        <div className="flex items-center justify-center space-x-2">
          <Bot className="h-8 w-8 text-primary" />
          <h1 className="text-3xl font-bold">AI-Powered FAQ</h1>
        </div>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Ask me anything! I'll provide instant, intelligent answers based on our comprehensive knowledge base.
        </p>
      </div>

      {/* AI Query Input */}
      <Card>
        <CardContent className="p-6">
          <div className="space-y-4">
            <div className="relative">
              <div className="flex space-x-2">
                <div className="flex-1 relative">
                  <Bot className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
                  <Input
                    type="text"
                    placeholder="Ask me anything about AI trading, setup, features..."
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleAIQuery()}
                    className="pl-10"
                  />
                </div>
                <Button
                  onClick={handleAIQuery}
                  disabled={isLoading || !query.trim()}
                  className="px-6"
                >
                  {isLoading ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-2 border-background border-t-foreground mr-2"></div>
                      Thinking...
                    </>
                  ) : (
                    <>
                      <Zap className="h-4 w-4 mr-2" />
                      Ask AI
                    </>
                  )}
                </Button>
              </div>
            </div>

            {/* Quick Answers */}
            {showQuickAnswers && (
              <div className="space-y-3">
                <p className="text-sm text-muted-foreground flex items-center">
                  <Lightbulb className="h-4 w-4 mr-2" />
                  Quick answers to common questions:
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                  {quickAnswers.map(answer => (
                    <Button
                      key={answer.id}
                      variant="outline"
                      onClick={() => handleQuickAnswer(answer)}
                      className="justify-start h-auto p-3 text-left"
                    >
                      <div>
                        <p className="font-medium text-sm">{answer.trigger}</p>
                        <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                          {answer.response}
                        </p>
                      </div>
                    </Button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* AI Response */}
      {aiResponse && (
        <Card>
          <CardHeader>
            <div className="flex items-start justify-between">
              <div className="flex items-center space-x-2">
                <Bot className="h-5 w-5 text-primary" />
                <CardTitle className="text-lg">AI Assistant</CardTitle>
                <Badge variant="secondary" className="text-xs">
                  {Math.round(aiResponse.confidence * 100)}% confidence
                </Badge>
              </div>
              <div className="flex items-center space-x-1">
                <Clock className="h-4 w-4 text-muted-foreground" />
                <span className="text-xs text-muted-foreground">
                  {new Date(aiResponse.timestamp).toLocaleTimeString()}
                </span>
              </div>
            </div>
          </CardHeader>

          <CardContent>
            <div className="space-y-4">
              <div className="bg-muted/50 rounded-lg p-4">
                <p className="font-medium text-sm mb-2">Your question:</p>
                <p className="text-sm">{aiResponse.query}</p>
              </div>

              <div>
                <p className="text-sm leading-relaxed">{aiResponse.answer}</p>
              </div>

              {aiResponse.sources.length > 0 && (
                <div className="space-y-2">
                  <p className="text-sm font-medium">Sources:</p>
                  <div className="space-y-1">
                    {aiResponse.sources.map(source => (
                      <div key={source.id} className="flex items-center justify-between text-xs text-muted-foreground bg-muted/30 rounded p-2">
                        <span>{source.title}</span>
                        <div className="flex items-center space-x-2">
                          <Badge variant="outline" className="text-xs">
                            {source.type}
                          </Badge>
                          <span>{Math.round(source.relevance * 100)}% match</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {aiResponse.followUpQuestions.length > 0 && (
                <div className="space-y-2">
                  <p className="text-sm font-medium">Related questions:</p>
                  <div className="flex flex-wrap gap-2">
                    {aiResponse.followUpQuestions.map((question, index) => (
                      <Button
                        key={index}
                        variant="outline"
                        size="sm"
                        onClick={() => setQuery(question)}
                        className="text-xs"
                      >
                        {question}
                      </Button>
                    ))}
                  </div>
                </div>
              )}

              <Separator />

              <div className="flex items-center justify-between">
                <p className="text-sm text-muted-foreground">Was this answer helpful?</p>
                <div className="flex space-x-2">
                  <Button
                    variant={aiResponse.wasHelpful === true ? "default" : "outline"}
                    size="sm"
                    onClick={() => handleFeedback(aiResponse.id, true)}
                  >
                    <ThumbsUp className="h-4 w-4 mr-1" />
                    Yes
                  </Button>
                  <Button
                    variant={aiResponse.wasHelpful === false ? "default" : "outline"}
                    size="sm"
                    onClick={() => handleFeedback(aiResponse.id, false)}
                  >
                    <ThumbsDown className="h-4 w-4 mr-1" />
                    No
                  </Button>
                  <Button variant="outline" size="sm">
                    <MessageCircle className="h-4 w-4 mr-1" />
                    Chat with Human
                  </Button>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Browse FAQs */}
      <Card>
        <CardHeader>
          <CardTitle>Browse All FAQs</CardTitle>
          <CardDescription>
            Or browse through our comprehensive FAQ collection organized by category
          </CardDescription>
        </CardHeader>

        <CardContent>
          <div className="space-y-4">
            {/* Category Filter */}
            <div className="flex flex-wrap gap-2">
              {categories.map(category => (
                <Button
                  key={category}
                  variant={selectedCategory === category ? "default" : "outline"}
                  size="sm"
                  onClick={() => setSelectedCategory(category)}
                >
                  {category === 'all' ? 'All Categories' : category}
                </Button>
              ))}
            </div>

            {/* FAQ List */}
            <div className="space-y-3">
              {filteredFaqs.map(faq => (
                <Card key={faq.id} className="cursor-pointer hover:shadow-md transition-shadow">
                  <CardContent className="p-4">
                    <div className="space-y-3">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <h4 className="font-medium text-sm">{faq.question}</h4>
                          <div className="flex items-center space-x-2 mt-1">
                            <Badge variant="outline" className="text-xs">{faq.category}</Badge>
                            <span className="text-xs text-muted-foreground">
                              {faq.views} views
                            </span>
                          </div>
                        </div>
                        <div className="flex items-center space-x-2 text-xs text-muted-foreground">
                          <div className="flex items-center space-x-1">
                            <ThumbsUp className="h-3 w-3" />
                            <span>{faq.helpful}</span>
                          </div>
                          <div className="flex items-center space-x-1">
                            <ThumbsDown className="h-3 w-3" />
                            <span>{faq.notHelpful}</span>
                          </div>
                        </div>
                      </div>

                      <p className="text-sm text-muted-foreground line-clamp-2">
                        {faq.answer}
                      </p>

                      <div className="flex flex-wrap gap-1">
                        {faq.tags.map(tag => (
                          <Badge key={tag} variant="secondary" className="text-xs">
                            {tag}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default AIFaqSystem;