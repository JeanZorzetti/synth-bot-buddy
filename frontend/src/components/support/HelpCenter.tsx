'use client';

import React, { useState, useEffect } from 'react';
import { Search, Book, MessageCircle, FileText, Star, ChevronRight, Filter } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

interface Article {
  id: string;
  title: string;
  content: string;
  category: string;
  tags: string[];
  views: number;
  rating: number;
  lastUpdated: string;
  featured: boolean;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
}

interface FAQ {
  id: string;
  question: string;
  answer: string;
  category: string;
  helpful: number;
  notHelpful: number;
  tags: string[];
}

interface SearchResult {
  type: 'article' | 'faq' | 'video';
  id: string;
  title: string;
  snippet: string;
  relevance: number;
  category: string;
}

const HelpCenter: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [articles, setArticles] = useState<Article[]>([]);
  const [faqs, setFaqs] = useState<FAQ[]>([]);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [isLoading, setIsLoading] = useState(false);

  // Mock data - replace with real API calls
  useEffect(() => {
    const mockArticles: Article[] = [
      {
        id: '1',
        title: 'Getting Started with AI Trading Bot',
        content: 'Complete guide to setting up your first trading bot...',
        category: 'Getting Started',
        tags: ['setup', 'beginner', 'trading'],
        views: 1250,
        rating: 4.8,
        lastUpdated: '2024-01-15',
        featured: true,
        difficulty: 'beginner'
      },
      {
        id: '2',
        title: 'Advanced Risk Management Strategies',
        content: 'Learn how to implement sophisticated risk management...',
        category: 'Risk Management',
        tags: ['risk', 'advanced', 'strategy'],
        views: 890,
        rating: 4.6,
        lastUpdated: '2024-01-12',
        featured: true,
        difficulty: 'advanced'
      },
      {
        id: '3',
        title: 'API Integration Best Practices',
        content: 'Best practices for integrating with Deriv API...',
        category: 'API',
        tags: ['api', 'integration', 'development'],
        views: 750,
        rating: 4.7,
        lastUpdated: '2024-01-10',
        featured: false,
        difficulty: 'intermediate'
      }
    ];

    const mockFaqs: FAQ[] = [
      {
        id: '1',
        question: 'How do I start trading with the AI bot?',
        answer: 'To start trading, first configure your API keys, set your risk parameters, and then activate the trading mode in the dashboard.',
        category: 'Trading',
        helpful: 45,
        notHelpful: 3,
        tags: ['trading', 'setup', 'beginner']
      },
      {
        id: '2',
        question: 'What are the minimum deposit requirements?',
        answer: 'The minimum deposit varies by account type. Trial accounts start from $10, while Premium accounts require $1000 minimum.',
        category: 'Billing',
        helpful: 32,
        notHelpful: 1,
        tags: ['billing', 'deposit', 'account']
      },
      {
        id: '3',
        question: 'How accurate are the AI predictions?',
        answer: 'Our AI models achieve 75-85% accuracy in favorable market conditions. Performance varies based on market volatility and asset type.',
        category: 'AI',
        helpful: 78,
        notHelpful: 5,
        tags: ['ai', 'accuracy', 'performance']
      }
    ];

    setArticles(mockArticles);
    setFaqs(mockFaqs);
  }, []);

  const handleSearch = async (query: string) => {
    setIsLoading(true);
    setSearchQuery(query);

    // Simulate AI-powered search with relevance scoring
    await new Promise(resolve => setTimeout(resolve, 800));

    if (!query.trim()) {
      setSearchResults([]);
      setIsLoading(false);
      return;
    }

    const results: SearchResult[] = [];

    // Search articles
    articles.forEach(article => {
      const relevance = calculateRelevance(query, article.title + ' ' + article.content + ' ' + article.tags.join(' '));
      if (relevance > 0.3) {
        results.push({
          type: 'article',
          id: article.id,
          title: article.title,
          snippet: article.content.substring(0, 150) + '...',
          relevance,
          category: article.category
        });
      }
    });

    // Search FAQs
    faqs.forEach(faq => {
      const relevance = calculateRelevance(query, faq.question + ' ' + faq.answer + ' ' + faq.tags.join(' '));
      if (relevance > 0.3) {
        results.push({
          type: 'faq',
          id: faq.id,
          title: faq.question,
          snippet: faq.answer.substring(0, 150) + '...',
          relevance,
          category: faq.category
        });
      }
    });

    // Sort by relevance
    results.sort((a, b) => b.relevance - a.relevance);

    setSearchResults(results);
    setIsLoading(false);
  };

  const calculateRelevance = (query: string, content: string): number => {
    const queryWords = query.toLowerCase().split(' ');
    const contentWords = content.toLowerCase().split(' ');
    let matches = 0;

    queryWords.forEach(queryWord => {
      contentWords.forEach(contentWord => {
        if (contentWord.includes(queryWord) || queryWord.includes(contentWord)) {
          matches++;
        }
      });
    });

    return matches / Math.max(queryWords.length, contentWords.length);
  };

  const categories = ['all', 'Getting Started', 'Trading', 'API', 'Risk Management', 'Billing', 'AI'];

  const filteredArticles = selectedCategory === 'all'
    ? articles
    : articles.filter(article => article.category === selectedCategory);

  const filteredFaqs = selectedCategory === 'all'
    ? faqs
    : faqs.filter(faq => faq.category === selectedCategory);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center space-y-4">
        <h1 className="text-3xl font-bold">Help Center</h1>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Find answers to your questions, learn about our features, and get the most out of your AI trading experience.
        </p>
      </div>

      {/* Search Bar */}
      <Card>
        <CardContent className="p-6">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground h-4 w-4" />
            <Input
              type="text"
              placeholder="Search for articles, FAQs, tutorials..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSearch(searchQuery)}
              className="pl-10 pr-20"
            />
            <Button
              onClick={() => handleSearch(searchQuery)}
              disabled={isLoading}
              className="absolute right-2 top-1/2 transform -translate-y-1/2"
              size="sm"
            >
              {isLoading ? 'Searching...' : 'Search'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Search Results */}
      {searchResults.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Search Results ({searchResults.length})</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {searchResults.map((result) => (
                <div key={result.id} className="border rounded-lg p-4 hover:bg-muted/50 cursor-pointer">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        {result.type === 'article' ? <Book className="h-4 w-4" /> : <MessageCircle className="h-4 w-4" />}
                        <h3 className="font-medium">{result.title}</h3>
                        <Badge variant="outline" className="text-xs">
                          {result.type === 'article' ? 'Article' : 'FAQ'}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground">{result.snippet}</p>
                      <div className="flex items-center gap-4 mt-2">
                        <span className="text-xs text-muted-foreground">{result.category}</span>
                        <div className="flex items-center gap-1">
                          <Star className="h-3 w-3 fill-yellow-400 text-yellow-400" />
                          <span className="text-xs">Relevance: {Math.round(result.relevance * 100)}%</span>
                        </div>
                      </div>
                    </div>
                    <ChevronRight className="h-4 w-4 text-muted-foreground" />
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="cursor-pointer hover:shadow-md transition-shadow">
          <CardContent className="p-6 text-center">
            <MessageCircle className="h-8 w-8 mx-auto mb-3 text-blue-600" />
            <h3 className="font-medium mb-2">Start Live Chat</h3>
            <p className="text-sm text-muted-foreground">Chat with our support team instantly</p>
          </CardContent>
        </Card>

        <Card className="cursor-pointer hover:shadow-md transition-shadow">
          <CardContent className="p-6 text-center">
            <FileText className="h-8 w-8 mx-auto mb-3 text-green-600" />
            <h3 className="font-medium mb-2">Create Ticket</h3>
            <p className="text-sm text-muted-foreground">Submit a detailed support request</p>
          </CardContent>
        </Card>

        <Card className="cursor-pointer hover:shadow-md transition-shadow">
          <CardContent className="p-6 text-center">
            <Book className="h-8 w-8 mx-auto mb-3 text-purple-600" />
            <h3 className="font-medium mb-2">User Guide</h3>
            <p className="text-sm text-muted-foreground">Complete documentation and tutorials</p>
          </CardContent>
        </Card>
      </div>

      {/* Content Tabs */}
      <Tabs defaultValue="articles" className="space-y-4">
        <div className="flex items-center justify-between">
          <TabsList>
            <TabsTrigger value="articles">Articles</TabsTrigger>
            <TabsTrigger value="faqs">FAQs</TabsTrigger>
            <TabsTrigger value="videos">Video Tutorials</TabsTrigger>
          </TabsList>

          <Select value={selectedCategory} onValueChange={setSelectedCategory}>
            <SelectTrigger className="w-48">
              <Filter className="h-4 w-4 mr-2" />
              <SelectValue placeholder="Filter by category" />
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

        <TabsContent value="articles">
          {/* Featured Articles */}
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-medium mb-4">Featured Articles</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {filteredArticles.filter(article => article.featured).map((article) => (
                  <Card key={article.id} className="cursor-pointer hover:shadow-md transition-shadow">
                    <CardHeader>
                      <div className="flex items-start justify-between">
                        <div>
                          <CardTitle className="text-lg">{article.title}</CardTitle>
                          <CardDescription>{article.category}</CardDescription>
                        </div>
                        <Badge variant={article.difficulty === 'beginner' ? 'default' : article.difficulty === 'intermediate' ? 'secondary' : 'destructive'}>
                          {article.difficulty}
                        </Badge>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <p className="text-sm text-muted-foreground mb-4">
                        {article.content.substring(0, 120)}...
                      </p>
                      <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <div className="flex items-center gap-4">
                          <span>{article.views} views</span>
                          <div className="flex items-center gap-1">
                            <Star className="h-3 w-3 fill-yellow-400 text-yellow-400" />
                            <span>{article.rating}</span>
                          </div>
                        </div>
                        <span>Updated {new Date(article.lastUpdated).toLocaleDateString()}</span>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>

            {/* All Articles */}
            <div>
              <h3 className="text-lg font-medium mb-4">All Articles</h3>
              <div className="space-y-3">
                {filteredArticles.map((article) => (
                  <Card key={article.id} className="cursor-pointer hover:shadow-md transition-shadow">
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <h4 className="font-medium">{article.title}</h4>
                            <Badge variant="outline" className="text-xs">{article.category}</Badge>
                            <Badge variant={article.difficulty === 'beginner' ? 'default' : article.difficulty === 'intermediate' ? 'secondary' : 'destructive'} className="text-xs">
                              {article.difficulty}
                            </Badge>
                          </div>
                          <div className="flex items-center gap-4 text-xs text-muted-foreground">
                            <span>{article.views} views</span>
                            <div className="flex items-center gap-1">
                              <Star className="h-3 w-3 fill-yellow-400 text-yellow-400" />
                              <span>{article.rating}</span>
                            </div>
                            <span>Updated {new Date(article.lastUpdated).toLocaleDateString()}</span>
                          </div>
                        </div>
                        <ChevronRight className="h-4 w-4 text-muted-foreground" />
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="faqs">
          <div className="space-y-4">
            {filteredFaqs.map((faq) => (
              <Card key={faq.id}>
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <CardTitle className="text-lg">{faq.question}</CardTitle>
                    <Badge variant="outline">{faq.category}</Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-muted-foreground mb-4">{faq.answer}</p>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <Button variant="outline" size="sm">
                        👍 Helpful ({faq.helpful})
                      </Button>
                      <Button variant="outline" size="sm">
                        👎 Not Helpful ({faq.notHelpful})
                      </Button>
                    </div>
                    <div className="flex gap-1">
                      {faq.tags.map((tag) => (
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
        </TabsContent>

        <TabsContent value="videos">
          <div className="text-center py-8">
            <div className="max-w-md mx-auto space-y-4">
              <FileText className="h-12 w-12 mx-auto text-muted-foreground" />
              <h3 className="text-lg font-medium">Video Tutorials Coming Soon</h3>
              <p className="text-muted-foreground">
                We're working on creating comprehensive video tutorials to help you get the most out of our platform.
              </p>
              <Button variant="outline">
                Get Notified When Available
              </Button>
            </div>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default HelpCenter;