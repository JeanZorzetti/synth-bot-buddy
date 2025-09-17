'use client';

import React, { useState, useEffect } from 'react';
import { Star, MessageSquare, ThumbsUp, ThumbsDown, Send, Filter, TrendingUp, Award, Calendar } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Checkbox } from '@/components/ui/checkbox';

interface FeedbackItem {
  id: string;
  userId: string;
  userName: string;
  userEmail: string;
  type: 'general' | 'feature_request' | 'bug_report' | 'support_rating' | 'product_review';
  category: string;
  rating: number;
  title: string;
  content: string;
  tags: string[];
  status: 'pending' | 'reviewed' | 'implemented' | 'rejected';
  priority: 'low' | 'medium' | 'high' | 'critical';
  createdAt: string;
  updatedAt: string;
  responses: Array<{
    id: string;
    authorId: string;
    authorName: string;
    content: string;
    timestamp: string;
    isPublic: boolean;
  }>;
  upvotes: number;
  downvotes: number;
  implementationEta?: string;
}

interface FeedbackStats {
  totalFeedback: number;
  averageRating: number;
  ratingDistribution: { [key: number]: number };
  categoryBreakdown: { [key: string]: number };
  statusBreakdown: { [key: string]: number };
  trendData: Array<{
    date: string;
    count: number;
    averageRating: number;
  }>;
}

interface NewFeedback {
  type: string;
  category: string;
  rating: number;
  title: string;
  content: string;
  tags: string[];
  priority: string;
  isAnonymous: boolean;
}

const FeedbackSystem: React.FC = () => {
  const [feedback, setFeedback] = useState<FeedbackItem[]>([]);
  const [stats, setStats] = useState<FeedbackStats | null>(null);
  const [newFeedback, setNewFeedback] = useState<NewFeedback>({
    type: 'general',
    category: '',
    rating: 5,
    title: '',
    content: '',
    tags: [],
    priority: 'medium',
    isAnonymous: false
  });
  const [selectedFilter, setSelectedFilter] = useState('all');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [showSubmissionForm, setShowSubmissionForm] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    // Mock data - replace with real API calls
    const mockFeedback: FeedbackItem[] = [
      {
        id: '1',
        userId: 'user1',
        userName: 'John Doe',
        userEmail: 'john@example.com',
        type: 'feature_request',
        category: 'Trading Interface',
        rating: 4,
        title: 'Add dark mode support',
        content: 'It would be great to have a dark mode option for better visibility during night trading sessions.',
        tags: ['ui', 'accessibility', 'dark-mode'],
        status: 'reviewed',
        priority: 'medium',
        createdAt: '2024-01-15T10:30:00Z',
        updatedAt: '2024-01-16T14:20:00Z',
        responses: [
          {
            id: 'resp1',
            authorId: 'admin1',
            authorName: 'Product Team',
            content: 'Great suggestion! We\'ve added this to our roadmap for Q2 2024.',
            timestamp: '2024-01-16T14:20:00Z',
            isPublic: true
          }
        ],
        upvotes: 23,
        downvotes: 2,
        implementationEta: '2024-06-30'
      },
      {
        id: '2',
        userId: 'user2',
        userName: 'Sarah Smith',
        userEmail: 'sarah@example.com',
        type: 'bug_report',
        category: 'AI Engine',
        rating: 2,
        title: 'AI predictions inconsistent on weekends',
        content: 'I\'ve noticed that the AI predictions seem less accurate during weekend trading hours.',
        tags: ['ai', 'accuracy', 'weekend-trading'],
        status: 'implemented',
        priority: 'high',
        createdAt: '2024-01-12T16:45:00Z',
        updatedAt: '2024-01-18T09:15:00Z',
        responses: [
          {
            id: 'resp2',
            authorId: 'dev1',
            authorName: 'AI Team',
            content: 'We\'ve identified the issue and deployed a fix. Weekend market models have been updated.',
            timestamp: '2024-01-18T09:15:00Z',
            isPublic: true
          }
        ],
        upvotes: 15,
        downvotes: 1
      },
      {
        id: '3',
        userId: 'user3',
        userName: 'Mike Johnson',
        userEmail: 'mike@example.com',
        type: 'product_review',
        category: 'Overall Experience',
        rating: 5,
        title: 'Excellent platform for algorithmic trading',
        content: 'Been using this platform for 6 months now. The AI capabilities are impressive and the interface is intuitive.',
        tags: ['review', 'positive', 'ai', 'interface'],
        status: 'reviewed',
        priority: 'low',
        createdAt: '2024-01-10T14:20:00Z',
        updatedAt: '2024-01-11T08:30:00Z',
        responses: [],
        upvotes: 31,
        downvotes: 0
      }
    ];

    const mockStats: FeedbackStats = {
      totalFeedback: 145,
      averageRating: 4.2,
      ratingDistribution: { 1: 5, 2: 8, 3: 15, 4: 45, 5: 72 },
      categoryBreakdown: {
        'Trading Interface': 35,
        'AI Engine': 28,
        'Overall Experience': 25,
        'API Integration': 20,
        'Billing': 15,
        'Support': 12,
        'Documentation': 10
      },
      statusBreakdown: {
        'pending': 45,
        'reviewed': 68,
        'implemented': 25,
        'rejected': 7
      },
      trendData: [
        { date: '2024-01-01', count: 12, averageRating: 4.1 },
        { date: '2024-01-02', count: 8, averageRating: 4.3 },
        { date: '2024-01-03', count: 15, averageRating: 4.0 },
        { date: '2024-01-04', count: 18, averageRating: 4.4 },
        { date: '2024-01-05', count: 22, averageRating: 4.2 }
      ]
    };

    setFeedback(mockFeedback);
    setStats(mockStats);
  }, []);

  const handleSubmitFeedback = async () => {
    setIsSubmitting(true);

    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1500));

    const feedbackItem: FeedbackItem = {
      id: `feedback-${Date.now()}`,
      userId: 'current-user',
      userName: newFeedback.isAnonymous ? 'Anonymous' : 'Current User',
      userEmail: 'user@example.com',
      type: newFeedback.type as any,
      category: newFeedback.category,
      rating: newFeedback.rating,
      title: newFeedback.title,
      content: newFeedback.content,
      tags: newFeedback.tags,
      status: 'pending',
      priority: newFeedback.priority as any,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      responses: [],
      upvotes: 0,
      downvotes: 0
    };

    setFeedback(prev => [feedbackItem, ...prev]);
    setNewFeedback({
      type: 'general',
      category: '',
      rating: 5,
      title: '',
      content: '',
      tags: [],
      priority: 'medium',
      isAnonymous: false
    });
    setShowSubmissionForm(false);
    setIsSubmitting(false);
  };

  const handleVote = (feedbackId: string, voteType: 'up' | 'down') => {
    setFeedback(prev =>
      prev.map(item =>
        item.id === feedbackId
          ? {
              ...item,
              upvotes: voteType === 'up' ? item.upvotes + 1 : item.upvotes,
              downvotes: voteType === 'down' ? item.downvotes + 1 : item.downvotes
            }
          : item
      )
    );
  };

  const renderStars = (rating: number, size = 'sm') => {
    return (
      <div className="flex">
        {[1, 2, 3, 4, 5].map(star => (
          <Star
            key={star}
            className={`${size === 'sm' ? 'h-4 w-4' : 'h-5 w-5'} ${
              star <= rating ? 'fill-yellow-400 text-yellow-400' : 'text-gray-300'
            }`}
          />
        ))}
      </div>
    );
  };

  const filteredFeedback = feedback.filter(item => {
    const matchesFilter = selectedFilter === 'all' || item.status === selectedFilter;
    const matchesCategory = selectedCategory === 'all' || item.category === selectedCategory;
    return matchesFilter && matchesCategory;
  });

  const categories = ['all', 'Trading Interface', 'AI Engine', 'Overall Experience', 'API Integration', 'Billing', 'Support', 'Documentation'];
  const feedbackTypes = [
    { value: 'general', label: 'General Feedback' },
    { value: 'feature_request', label: 'Feature Request' },
    { value: 'bug_report', label: 'Bug Report' },
    { value: 'support_rating', label: 'Support Rating' },
    { value: 'product_review', label: 'Product Review' }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Feedback & Reviews</h1>
          <p className="text-muted-foreground">
            Share your thoughts and help us improve the platform
          </p>
        </div>
        <Button onClick={() => setShowSubmissionForm(true)}>
          <MessageSquare className="h-4 w-4 mr-2" />
          Submit Feedback
        </Button>
      </div>

      {/* Statistics Dashboard */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Total Feedback</p>
                  <p className="text-2xl font-bold">{stats.totalFeedback}</p>
                </div>
                <MessageSquare className="h-8 w-8 text-muted-foreground" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Average Rating</p>
                  <div className="flex items-center space-x-2">
                    <p className="text-2xl font-bold">{stats.averageRating}</p>
                    {renderStars(Math.round(stats.averageRating))}
                  </div>
                </div>
                <Star className="h-8 w-8 text-muted-foreground" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Implementation Rate</p>
                  <p className="text-2xl font-bold">
                    {Math.round((stats.statusBreakdown.implemented / stats.totalFeedback) * 100)}%
                  </p>
                </div>
                <TrendingUp className="h-8 w-8 text-muted-foreground" />
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">This Month</p>
                  <p className="text-2xl font-bold">28</p>
                  <p className="text-xs text-green-600">+12% from last month</p>
                </div>
                <Calendar className="h-8 w-8 text-muted-foreground" />
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Rating Distribution */}
      {stats && (
        <Card>
          <CardHeader>
            <CardTitle>Rating Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {[5, 4, 3, 2, 1].map(rating => (
                <div key={rating} className="flex items-center space-x-4">
                  <div className="flex items-center space-x-2 w-16">
                    <span className="text-sm">{rating}</span>
                    <Star className="h-4 w-4 fill-yellow-400 text-yellow-400" />
                  </div>
                  <Progress
                    value={(stats.ratingDistribution[rating] / stats.totalFeedback) * 100}
                    className="flex-1"
                  />
                  <span className="text-sm text-muted-foreground w-12">
                    {stats.ratingDistribution[rating]}
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Feedback Submission Form */}
      {showSubmissionForm && (
        <Card>
          <CardHeader>
            <CardTitle>Submit Feedback</CardTitle>
            <CardDescription>
              Help us improve by sharing your thoughts, suggestions, or reporting issues
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="feedback-type">Feedback Type</Label>
                  <Select value={newFeedback.type} onValueChange={(value) => setNewFeedback(prev => ({ ...prev, type: value }))}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {feedbackTypes.map(type => (
                        <SelectItem key={type.value} value={type.value}>
                          {type.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="category">Category</Label>
                  <Select value={newFeedback.category} onValueChange={(value) => setNewFeedback(prev => ({ ...prev, category: value }))}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select category" />
                    </SelectTrigger>
                    <SelectContent>
                      {categories.filter(cat => cat !== 'all').map(category => (
                        <SelectItem key={category} value={category}>
                          {category}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="space-y-2">
                <Label>Overall Rating</Label>
                <RadioGroup
                  value={newFeedback.rating.toString()}
                  onValueChange={(value) => setNewFeedback(prev => ({ ...prev, rating: parseInt(value) }))}
                  className="flex space-x-4"
                >
                  {[1, 2, 3, 4, 5].map(rating => (
                    <div key={rating} className="flex items-center space-x-2">
                      <RadioGroupItem value={rating.toString()} id={`rating-${rating}`} />
                      <Label htmlFor={`rating-${rating}`} className="flex items-center space-x-1">
                        <span>{rating}</span>
                        <Star className="h-4 w-4 fill-yellow-400 text-yellow-400" />
                      </Label>
                    </div>
                  ))}
                </RadioGroup>
              </div>

              <div className="space-y-2">
                <Label htmlFor="title">Title</Label>
                <Input
                  id="title"
                  value={newFeedback.title}
                  onChange={(e) => setNewFeedback(prev => ({ ...prev, title: e.target.value }))}
                  placeholder="Brief summary of your feedback"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="content">Details</Label>
                <Textarea
                  id="content"
                  value={newFeedback.content}
                  onChange={(e) => setNewFeedback(prev => ({ ...prev, content: e.target.value }))}
                  placeholder="Provide detailed feedback, suggestions, or describe the issue..."
                  rows={4}
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Priority</Label>
                  <RadioGroup
                    value={newFeedback.priority}
                    onValueChange={(value) => setNewFeedback(prev => ({ ...prev, priority: value }))}
                  >
                    {['low', 'medium', 'high', 'critical'].map(priority => (
                      <div key={priority} className="flex items-center space-x-2">
                        <RadioGroupItem value={priority} id={`priority-${priority}`} />
                        <Label htmlFor={`priority-${priority}`} className="capitalize">
                          {priority}
                        </Label>
                      </div>
                    ))}
                  </RadioGroup>
                </div>

                <div className="space-y-4">
                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="anonymous"
                      checked={newFeedback.isAnonymous}
                      onCheckedChange={(checked) => setNewFeedback(prev => ({ ...prev, isAnonymous: checked as boolean }))}
                    />
                    <Label htmlFor="anonymous">Submit anonymously</Label>
                  </div>
                </div>
              </div>

              <div className="flex justify-end space-x-2">
                <Button variant="outline" onClick={() => setShowSubmissionForm(false)}>
                  Cancel
                </Button>
                <Button
                  onClick={handleSubmitFeedback}
                  disabled={isSubmitting || !newFeedback.title || !newFeedback.content || !newFeedback.category}
                >
                  {isSubmitting ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-2 border-background border-t-foreground mr-2"></div>
                      Submitting...
                    </>
                  ) : (
                    <>
                      <Send className="h-4 w-4 mr-2" />
                      Submit Feedback
                    </>
                  )}
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Filters */}
      <div className="flex items-center space-x-4">
        <Select value={selectedFilter} onValueChange={setSelectedFilter}>
          <SelectTrigger className="w-48">
            <Filter className="h-4 w-4 mr-2" />
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All Status</SelectItem>
            <SelectItem value="pending">Pending Review</SelectItem>
            <SelectItem value="reviewed">Under Review</SelectItem>
            <SelectItem value="implemented">Implemented</SelectItem>
            <SelectItem value="rejected">Rejected</SelectItem>
          </SelectContent>
        </Select>

        <Select value={selectedCategory} onValueChange={setSelectedCategory}>
          <SelectTrigger className="w-48">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {categories.map(category => (
              <SelectItem key={category} value={category}>
                {category === 'all' ? 'All Categories' : category}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Feedback List */}
      <div className="space-y-4">
        {filteredFeedback.map(item => (
          <Card key={item.id}>
            <CardContent className="p-6">
              <div className="space-y-4">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-2 mb-2">
                      <Badge variant="outline">{item.type.replace('_', ' ')}</Badge>
                      <Badge variant="outline">{item.category}</Badge>
                      <Badge variant={
                        item.status === 'implemented' ? 'default' :
                        item.status === 'reviewed' ? 'secondary' :
                        item.status === 'rejected' ? 'destructive' : 'outline'
                      }>
                        {item.status}
                      </Badge>
                      {item.implementationEta && (
                        <Badge variant="secondary">
                          ETA: {new Date(item.implementationEta).toLocaleDateString()}
                        </Badge>
                      )}
                    </div>
                    <h3 className="font-medium text-lg">{item.title}</h3>
                    <div className="flex items-center space-x-2 mb-2">
                      {renderStars(item.rating)}
                      <span className="text-sm text-muted-foreground">by {item.userName}</span>
                      <span className="text-sm text-muted-foreground">
                        {new Date(item.createdAt).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                </div>

                <p className="text-sm text-muted-foreground">{item.content}</p>

                <div className="flex flex-wrap gap-1">
                  {item.tags.map(tag => (
                    <Badge key={tag} variant="secondary" className="text-xs">
                      {tag}
                    </Badge>
                  ))}
                </div>

                {item.responses.length > 0 && (
                  <div className="border-t pt-4">
                    <h4 className="font-medium mb-2">Team Response:</h4>
                    {item.responses.map(response => (
                      <div key={response.id} className="bg-muted/50 rounded-lg p-3">
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium text-sm">{response.authorName}</span>
                          <span className="text-xs text-muted-foreground">
                            {new Date(response.timestamp).toLocaleDateString()}
                          </span>
                        </div>
                        <p className="text-sm">{response.content}</p>
                      </div>
                    ))}
                  </div>
                )}

                <div className="flex items-center justify-between pt-2">
                  <div className="flex items-center space-x-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleVote(item.id, 'up')}
                    >
                      <ThumbsUp className="h-4 w-4 mr-1" />
                      {item.upvotes}
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleVote(item.id, 'down')}
                    >
                      <ThumbsDown className="h-4 w-4 mr-1" />
                      {item.downvotes}
                    </Button>
                  </div>
                  <Badge variant={
                    item.priority === 'critical' ? 'destructive' :
                    item.priority === 'high' ? 'secondary' :
                    item.priority === 'medium' ? 'outline' : 'default'
                  }>
                    {item.priority} priority
                  </Badge>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
};

export default FeedbackSystem;