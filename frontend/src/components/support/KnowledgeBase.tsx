'use client';

import React, { useState, useEffect } from 'react';
import { Search, Book, Video, Download, Clock, User, Star, ThumbsUp, ThumbsDown, Share2, Bookmark } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Separator } from '@/components/ui/separator';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { ScrollArea } from '@/components/ui/scroll-area';

interface KnowledgeArticle {
  id: string;
  title: string;
  content: string;
  summary: string;
  category: string;
  subcategory: string;
  tags: string[];
  author: {
    id: string;
    name: string;
    avatar?: string;
    role: string;
  };
  createdAt: string;
  updatedAt: string;
  views: number;
  likes: number;
  dislikes: number;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  estimatedReadTime: number;
  relatedArticles: string[];
  attachments: Array<{
    id: string;
    name: string;
    url: string;
    type: 'pdf' | 'image' | 'video' | 'code';
    size: number;
  }>;
}

interface VideoTutorial {
  id: string;
  title: string;
  description: string;
  thumbnailUrl: string;
  videoUrl: string;
  duration: string;
  category: string;
  tags: string[];
  views: number;
  likes: number;
  createdAt: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  series?: {
    id: string;
    name: string;
    position: number;
    totalVideos: number;
  };
}

interface KnowledgeCategory {
  id: string;
  name: string;
  description: string;
  icon: string;
  articleCount: number;
  subcategories: Array<{
    id: string;
    name: string;
    description: string;
    articleCount: number;
  }>;
}

const KnowledgeBase: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedDifficulty, setSelectedDifficulty] = useState('all');
  const [sortBy, setSortBy] = useState('relevance');
  const [articles, setArticles] = useState<KnowledgeArticle[]>([]);
  const [videos, setVideos] = useState<VideoTutorial[]>([]);
  const [categories, setCategories] = useState<KnowledgeCategory[]>([]);
  const [selectedArticle, setSelectedArticle] = useState<KnowledgeArticle | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    // Mock data - replace with real API calls
    const mockCategories: KnowledgeCategory[] = [
      {
        id: '1',
        name: 'Getting Started',
        description: 'Essential guides for new users',
        icon: '🚀',
        articleCount: 12,
        subcategories: [
          { id: '1-1', name: 'Account Setup', description: 'Setting up your account', articleCount: 4 },
          { id: '1-2', name: 'First Steps', description: 'Your first trading session', articleCount: 5 },
          { id: '1-3', name: 'Basic Concepts', description: 'Trading fundamentals', articleCount: 3 }
        ]
      },
      {
        id: '2',
        name: 'Trading Strategies',
        description: 'Advanced trading techniques and strategies',
        icon: '📈',
        articleCount: 25,
        subcategories: [
          { id: '2-1', name: 'Technical Analysis', description: 'Chart patterns and indicators', articleCount: 8 },
          { id: '2-2', name: 'Risk Management', description: 'Protecting your capital', articleCount: 7 },
          { id: '2-3', name: 'AI Strategies', description: 'AI-powered trading approaches', articleCount: 10 }
        ]
      },
      {
        id: '3',
        name: 'API Integration',
        description: 'Developer guides and API documentation',
        icon: '⚙️',
        articleCount: 18,
        subcategories: [
          { id: '3-1', name: 'Authentication', description: 'API keys and security', articleCount: 5 },
          { id: '3-2', name: 'Endpoints', description: 'Available API endpoints', articleCount: 8 },
          { id: '3-3', name: 'Examples', description: 'Code examples and tutorials', articleCount: 5 }
        ]
      }
    ];

    const mockArticles: KnowledgeArticle[] = [
      {
        id: '1',
        title: 'Complete Guide to AI Trading Bot Setup',
        content: 'This comprehensive guide walks you through setting up your first AI trading bot...',
        summary: 'Learn how to configure and deploy your AI trading bot from scratch',
        category: 'Getting Started',
        subcategory: 'First Steps',
        tags: ['setup', 'ai', 'trading', 'beginner'],
        author: {
          id: '1',
          name: 'Sarah Johnson',
          role: 'Trading Expert',
          avatar: '/avatars/sarah.jpg'
        },
        createdAt: '2024-01-15T10:30:00Z',
        updatedAt: '2024-01-16T14:20:00Z',
        views: 1247,
        likes: 89,
        dislikes: 3,
        difficulty: 'beginner',
        estimatedReadTime: 8,
        relatedArticles: ['2', '3'],
        attachments: [
          {
            id: '1',
            name: 'setup-guide.pdf',
            url: '/downloads/setup-guide.pdf',
            type: 'pdf',
            size: 2.1 * 1024 * 1024
          },
          {
            id: '2',
            name: 'configuration-example.json',
            url: '/downloads/config.json',
            type: 'code',
            size: 1024
          }
        ]
      },
      {
        id: '2',
        title: 'Advanced Risk Management Strategies',
        content: 'Risk management is crucial for successful trading. This article covers...',
        summary: 'Master risk management techniques to protect your capital',
        category: 'Trading Strategies',
        subcategory: 'Risk Management',
        tags: ['risk', 'management', 'advanced', 'strategy'],
        author: {
          id: '2',
          name: 'Michael Chen',
          role: 'Risk Analyst',
          avatar: '/avatars/michael.jpg'
        },
        createdAt: '2024-01-12T09:15:00Z',
        updatedAt: '2024-01-14T16:45:00Z',
        views: 892,
        likes: 67,
        dislikes: 2,
        difficulty: 'advanced',
        estimatedReadTime: 12,
        relatedArticles: ['3', '4'],
        attachments: []
      }
    ];

    const mockVideos: VideoTutorial[] = [
      {
        id: '1',
        title: 'AI Trading Bot Tutorial - Part 1: Getting Started',
        description: 'Introduction to AI trading and setting up your first bot',
        thumbnailUrl: '/thumbnails/tutorial-1.jpg',
        videoUrl: '/videos/tutorial-1.mp4',
        duration: '15:30',
        category: 'Getting Started',
        tags: ['tutorial', 'ai', 'basics'],
        views: 3456,
        likes: 234,
        createdAt: '2024-01-10T12:00:00Z',
        difficulty: 'beginner',
        series: {
          id: 'series-1',
          name: 'Complete AI Trading Course',
          position: 1,
          totalVideos: 8
        }
      }
    ];

    setCategories(mockCategories);
    setArticles(mockArticles);
    setVideos(mockVideos);
  }, []);

  const handleSearch = (query: string) => {
    setSearchQuery(query);
    // Implement search logic here
  };

  const filteredArticles = articles.filter(article => {
    const matchesCategory = selectedCategory === 'all' || article.category === selectedCategory;
    const matchesDifficulty = selectedDifficulty === 'all' || article.difficulty === selectedDifficulty;
    const matchesSearch = !searchQuery ||
      article.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      article.summary.toLowerCase().includes(searchQuery.toLowerCase()) ||
      article.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));

    return matchesCategory && matchesDifficulty && matchesSearch;
  });

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString([], {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  if (selectedArticle) {
    return (
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Article Header */}
        <div className="flex items-center justify-between">
          <Button variant="ghost" onClick={() => setSelectedArticle(null)}>
            ← Back to Knowledge Base
          </Button>
          <div className="flex items-center space-x-2">
            <Button variant="outline" size="sm">
              <Bookmark className="h-4 w-4 mr-2" />
              Save
            </Button>
            <Button variant="outline" size="sm">
              <Share2 className="h-4 w-4 mr-2" />
              Share
            </Button>
          </div>
        </div>

        {/* Article Content */}
        <Card>
          <CardHeader>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <Badge variant="outline">{selectedArticle.category}</Badge>
                <Badge variant={
                  selectedArticle.difficulty === 'beginner' ? 'default' :
                  selectedArticle.difficulty === 'intermediate' ? 'secondary' : 'destructive'
                }>
                  {selectedArticle.difficulty}
                </Badge>
              </div>

              <CardTitle className="text-2xl">{selectedArticle.title}</CardTitle>
              <CardDescription className="text-lg">{selectedArticle.summary}</CardDescription>

              <div className="flex items-center justify-between text-sm text-muted-foreground">
                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-2">
                    <Avatar className="h-6 w-6">
                      <AvatarImage src={selectedArticle.author.avatar} />
                      <AvatarFallback>{selectedArticle.author.name[0]}</AvatarFallback>
                    </Avatar>
                    <span>{selectedArticle.author.name}</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <Clock className="h-4 w-4" />
                    <span>{selectedArticle.estimatedReadTime} min read</span>
                  </div>
                  <span>Updated {formatDate(selectedArticle.updatedAt)}</span>
                </div>

                <div className="flex items-center space-x-4">
                  <span>{selectedArticle.views} views</span>
                  <div className="flex items-center space-x-2">
                    <Button variant="ghost" size="sm">
                      <ThumbsUp className="h-4 w-4 mr-1" />
                      {selectedArticle.likes}
                    </Button>
                    <Button variant="ghost" size="sm">
                      <ThumbsDown className="h-4 w-4 mr-1" />
                      {selectedArticle.dislikes}
                    </Button>
                  </div>
                </div>
              </div>

              <div className="flex flex-wrap gap-2">
                {selectedArticle.tags.map(tag => (
                  <Badge key={tag} variant="secondary" className="text-xs">
                    {tag}
                  </Badge>
                ))}
              </div>
            </div>
          </CardHeader>

          <CardContent>
            <div className="prose max-w-none">
              <div className="whitespace-pre-wrap text-sm leading-relaxed">
                {selectedArticle.content}
              </div>
            </div>

            {/* Attachments */}
            {selectedArticle.attachments.length > 0 && (
              <div className="mt-8">
                <h3 className="text-lg font-medium mb-4">Attachments</h3>
                <div className="space-y-2">
                  {selectedArticle.attachments.map(attachment => (
                    <Card key={attachment.id} className="p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <div className="w-10 h-10 rounded bg-muted flex items-center justify-center">
                            {attachment.type === 'pdf' && <Book className="h-5 w-5" />}
                            {attachment.type === 'video' && <Video className="h-5 w-5" />}
                            {attachment.type === 'code' && <span className="text-xs font-mono">{'</>'}</span>}
                            {attachment.type === 'image' && <span className="text-xs">🖼️</span>}
                          </div>
                          <div>
                            <p className="font-medium">{attachment.name}</p>
                            <p className="text-sm text-muted-foreground">
                              {attachment.type.toUpperCase()} • {formatFileSize(attachment.size)}
                            </p>
                          </div>
                        </div>
                        <Button variant="outline" size="sm">
                          <Download className="h-4 w-4 mr-2" />
                          Download
                        </Button>
                      </div>
                    </Card>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center space-y-4">
        <h1 className="text-3xl font-bold">Knowledge Base</h1>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Comprehensive guides, tutorials, and documentation to help you master our AI trading platform.
        </p>
      </div>

      {/* Search and Filters */}
      <Card>
        <CardContent className="p-6">
          <div className="space-y-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
              <Input
                type="text"
                placeholder="Search articles, tutorials, guides..."
                value={searchQuery}
                onChange={(e) => handleSearch(e.target.value)}
                className="pl-10"
              />
            </div>

            <div className="flex flex-wrap gap-4">
              <Select value={selectedCategory} onValueChange={setSelectedCategory}>
                <SelectTrigger className="w-48">
                  <SelectValue placeholder="All Categories" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Categories</SelectItem>
                  {categories.map(category => (
                    <SelectItem key={category.id} value={category.name}>
                      {category.icon} {category.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              <Select value={selectedDifficulty} onValueChange={setSelectedDifficulty}>
                <SelectTrigger className="w-48">
                  <SelectValue placeholder="All Levels" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Levels</SelectItem>
                  <SelectItem value="beginner">Beginner</SelectItem>
                  <SelectItem value="intermediate">Intermediate</SelectItem>
                  <SelectItem value="advanced">Advanced</SelectItem>
                </SelectContent>
              </Select>

              <Select value={sortBy} onValueChange={setSortBy}>
                <SelectTrigger className="w-48">
                  <SelectValue placeholder="Sort by" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="relevance">Relevance</SelectItem>
                  <SelectItem value="newest">Newest First</SelectItem>
                  <SelectItem value="oldest">Oldest First</SelectItem>
                  <SelectItem value="views">Most Viewed</SelectItem>
                  <SelectItem value="likes">Most Liked</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Content Tabs */}
      <Tabs defaultValue="articles" className="space-y-6">
        <TabsList>
          <TabsTrigger value="articles">Articles ({filteredArticles.length})</TabsTrigger>
          <TabsTrigger value="videos">Video Tutorials ({videos.length})</TabsTrigger>
          <TabsTrigger value="categories">Browse by Category</TabsTrigger>
        </TabsList>

        <TabsContent value="articles">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredArticles.map(article => (
              <Card key={article.id} className="cursor-pointer hover:shadow-md transition-shadow" onClick={() => setSelectedArticle(article)}>
                <CardHeader>
                  <div className="flex items-start justify-between mb-2">
                    <Badge variant="outline" className="text-xs">{article.category}</Badge>
                    <Badge variant={
                      article.difficulty === 'beginner' ? 'default' :
                      article.difficulty === 'intermediate' ? 'secondary' : 'destructive'
                    } className="text-xs">
                      {article.difficulty}
                    </Badge>
                  </div>
                  <CardTitle className="text-lg line-clamp-2">{article.title}</CardTitle>
                  <CardDescription className="line-clamp-3">{article.summary}</CardDescription>
                </CardHeader>

                <CardContent>
                  <div className="space-y-4">
                    <div className="flex flex-wrap gap-1">
                      {article.tags.slice(0, 3).map(tag => (
                        <Badge key={tag} variant="secondary" className="text-xs">
                          {tag}
                        </Badge>
                      ))}
                      {article.tags.length > 3 && (
                        <Badge variant="secondary" className="text-xs">
                          +{article.tags.length - 3}
                        </Badge>
                      )}
                    </div>

                    <Separator />

                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <div className="flex items-center space-x-2">
                        <Avatar className="h-5 w-5">
                          <AvatarImage src={article.author.avatar} />
                          <AvatarFallback className="text-xs">{article.author.name[0]}</AvatarFallback>
                        </Avatar>
                        <span>{article.author.name}</span>
                      </div>
                      <div className="flex items-center space-x-1">
                        <Clock className="h-3 w-3" />
                        <span>{article.estimatedReadTime}m</span>
                      </div>
                    </div>

                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span>{article.views} views</span>
                      <div className="flex items-center space-x-2">
                        <div className="flex items-center space-x-1">
                          <ThumbsUp className="h-3 w-3" />
                          <span>{article.likes}</span>
                        </div>
                        <span>•</span>
                        <span>{formatDate(article.updatedAt)}</span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="videos">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {videos.map(video => (
              <Card key={video.id} className="cursor-pointer hover:shadow-md transition-shadow">
                <div className="aspect-video bg-muted rounded-t-lg relative overflow-hidden">
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="w-12 h-12 bg-black/20 rounded-full flex items-center justify-center">
                      <Video className="h-6 w-6 text-white" />
                    </div>
                  </div>
                  <div className="absolute bottom-2 right-2 bg-black/80 text-white text-xs px-2 py-1 rounded">
                    {video.duration}
                  </div>
                </div>

                <CardHeader>
                  <div className="flex items-start justify-between mb-2">
                    <Badge variant="outline" className="text-xs">{video.category}</Badge>
                    <Badge variant={
                      video.difficulty === 'beginner' ? 'default' :
                      video.difficulty === 'intermediate' ? 'secondary' : 'destructive'
                    } className="text-xs">
                      {video.difficulty}
                    </Badge>
                  </div>
                  <CardTitle className="text-lg line-clamp-2">{video.title}</CardTitle>
                  <CardDescription className="line-clamp-2">{video.description}</CardDescription>
                </CardHeader>

                <CardContent>
                  <div className="space-y-3">
                    {video.series && (
                      <div className="text-sm text-muted-foreground">
                        Part {video.series.position} of {video.series.totalVideos} • {video.series.name}
                      </div>
                    )}

                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span>{video.views.toLocaleString()} views</span>
                      <div className="flex items-center space-x-1">
                        <ThumbsUp className="h-3 w-3" />
                        <span>{video.likes}</span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="categories">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {categories.map(category => (
              <Card key={category.id} className="cursor-pointer hover:shadow-md transition-shadow">
                <CardHeader>
                  <div className="flex items-center space-x-3">
                    <div className="text-2xl">{category.icon}</div>
                    <div>
                      <CardTitle className="text-lg">{category.name}</CardTitle>
                      <CardDescription>{category.description}</CardDescription>
                    </div>
                  </div>
                </CardHeader>

                <CardContent>
                  <div className="space-y-3">
                    <div className="text-sm text-muted-foreground">
                      {category.articleCount} articles
                    </div>

                    <div className="space-y-2">
                      {category.subcategories.map(sub => (
                        <div key={sub.id} className="flex items-center justify-between p-2 rounded hover:bg-muted/50 cursor-pointer">
                          <div>
                            <p className="font-medium text-sm">{sub.name}</p>
                            <p className="text-xs text-muted-foreground">{sub.description}</p>
                          </div>
                          <Badge variant="secondary" className="text-xs">
                            {sub.articleCount}
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default KnowledgeBase;