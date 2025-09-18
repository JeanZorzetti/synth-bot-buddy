'use client';

import React, { useState, useEffect } from 'react';
import { Brain, TrendingUp, TrendingDown, Zap, Target, AlertTriangle, BarChart3, Activity, Lightbulb, Cpu, Eye, Settings } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';

interface AIModel {
  id: string;
  name: string;
  type: 'lstm' | 'transformer' | 'prophet' | 'arima' | 'ensemble';
  status: 'training' | 'ready' | 'predicting' | 'error';
  accuracy: number;
  confidence: number;
  lastTrained: string;
  predictions: Array<{
    timestamp: string;
    value: number;
    confidence: number;
    upperBound: number;
    lowerBound: number;
  }>;
  performance: {
    mse: number;
    mae: number;
    r2: number;
    mape: number;
  };
}

interface TrendInsight {
  id: string;
  type: 'pattern' | 'anomaly' | 'correlation' | 'opportunity' | 'risk';
  title: string;
  description: string;
  confidence: number;
  impact: 'low' | 'medium' | 'high' | 'critical';
  timeframe: string;
  recommendations: string[];
  data: {
    metric: string;
    currentValue: number;
    predictedValue: number;
    changePercentage: number;
  };
  aiGenerated: boolean;
  timestamp: string;
}

interface MarketRegime {
  id: string;
  name: string;
  description: string;
  probability: number;
  characteristics: string[];
  duration: string;
  tradingStrategies: string[];
  riskLevel: 'low' | 'medium' | 'high';
  indicators: Array<{
    name: string;
    value: number;
    signal: 'bullish' | 'bearish' | 'neutral';
  }>;
}

interface AIConfiguration {
  models: {
    enabled: string[];
    weights: { [key: string]: number };
    retrainingFrequency: 'daily' | 'weekly' | 'monthly';
    minConfidence: number;
  };
  insights: {
    enableAutoGeneration: boolean;
    minImpact: 'low' | 'medium' | 'high';
    categories: string[];
    alertThreshold: number;
  };
  forecasting: {
    horizon: number; // days
    updateFrequency: number; // hours
    confidenceIntervals: boolean;
    seasonalityDetection: boolean;
  };
}

const AITrendAnalysis: React.FC = () => {
  const [models, setModels] = useState<AIModel[]>([]);
  const [insights, setInsights] = useState<TrendInsight[]>([]);
  const [marketRegimes, setMarketRegimes] = useState<MarketRegime[]>([]);
  const [configuration, setConfiguration] = useState<AIConfiguration | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('ensemble');
  const [analysisMode, setAnalysisMode] = useState<'auto' | 'manual'>('auto');
  const [showAdvanced, setShowAdvanced] = useState(false);

  useEffect(() => {
    initializeAISystem();
  }, []);

  const initializeAISystem = () => {
    // Mock AI models
    const mockModels: AIModel[] = [
      {
        id: 'lstm-primary',
        name: 'LSTM Neural Network',
        type: 'lstm',
        status: 'ready',
        accuracy: 0.847,
        confidence: 0.92,
        lastTrained: '2024-01-15T10:30:00Z',
        predictions: generatePredictions(7),
        performance: {
          mse: 0.125,
          mae: 0.089,
          r2: 0.847,
          mape: 0.034
        }
      },
      {
        id: 'transformer-advanced',
        name: 'Transformer Model',
        type: 'transformer',
        status: 'ready',
        accuracy: 0.823,
        confidence: 0.88,
        lastTrained: '2024-01-15T08:45:00Z',
        predictions: generatePredictions(7),
        performance: {
          mse: 0.156,
          mae: 0.098,
          r2: 0.823,
          mape: 0.041
        }
      },
      {
        id: 'ensemble-master',
        name: 'Ensemble Model',
        type: 'ensemble',
        status: 'ready',
        accuracy: 0.891,
        confidence: 0.95,
        lastTrained: '2024-01-15T12:00:00Z',
        predictions: generatePredictions(7),
        performance: {
          mse: 0.098,
          mae: 0.067,
          r2: 0.891,
          mape: 0.028
        }
      }
    ];

    // Mock AI insights
    const mockInsights: TrendInsight[] = [
      {
        id: '1',
        type: 'opportunity',
        title: 'Emerging Bullish Pattern Detected',
        description: 'AI models have identified a strong bullish divergence pattern forming in EUR/USD with 89% confidence. Historical analysis suggests 15-20% upward movement typically follows this pattern.',
        confidence: 0.89,
        impact: 'high',
        timeframe: '3-7 days',
        recommendations: [
          'Consider increasing EUR/USD position size',
          'Set stop-loss at 1.0850 level',
          'Take profit targets at 1.1150 and 1.1250'
        ],
        data: {
          metric: 'EUR/USD',
          currentValue: 1.0945,
          predictedValue: 1.1180,
          changePercentage: 2.14
        },
        aiGenerated: true,
        timestamp: new Date().toISOString()
      },
      {
        id: '2',
        type: 'risk',
        title: 'Volatility Spike Warning',
        description: 'Multiple AI models are converging on a high probability volatility event in the next 24-48 hours. Market regime analysis suggests defensive positioning.',
        confidence: 0.76,
        impact: 'medium',
        timeframe: '1-2 days',
        recommendations: [
          'Reduce overall position size by 25%',
          'Increase stop-loss distances',
          'Consider volatility hedging strategies'
        ],
        data: {
          metric: 'Market Volatility',
          currentValue: 12.5,
          predictedValue: 18.2,
          changePercentage: 45.6
        },
        aiGenerated: true,
        timestamp: new Date(Date.now() - 3600000).toISOString()
      },
      {
        id: '3',
        type: 'correlation',
        title: 'Cross-Asset Correlation Breakdown',
        description: 'AI detected unusual correlation breakdown between Gold and USD, presenting arbitrage opportunities in precious metals trading.',
        confidence: 0.82,
        impact: 'medium',
        timeframe: '5-10 days',
        recommendations: [
          'Monitor Gold/USD correlation coefficient',
          'Consider pairs trading strategy',
          'Set up alerts for correlation normalization'
        ],
        data: {
          metric: 'Gold-USD Correlation',
          currentValue: -0.65,
          predictedValue: -0.25,
          changePercentage: 61.5
        },
        aiGenerated: true,
        timestamp: new Date(Date.now() - 7200000).toISOString()
      }
    ];

    // Mock market regimes
    const mockRegimes: MarketRegime[] = [
      {
        id: 'trending-bull',
        name: 'Bullish Trending Market',
        description: 'Strong upward momentum with low volatility and consistent buyer interest',
        probability: 0.67,
        characteristics: [
          'Consistent higher highs and higher lows',
          'Above-average volume on up moves',
          'Low volatility environment',
          'Strong institutional buying'
        ],
        duration: '2-4 weeks',
        tradingStrategies: [
          'Trend following strategies',
          'Momentum-based entries',
          'Buy the dip approach',
          'Breakout trading'
        ],
        riskLevel: 'medium',
        indicators: [
          { name: 'RSI', value: 68, signal: 'bullish' },
          { name: 'MACD', value: 0.045, signal: 'bullish' },
          { name: 'Volume', value: 1.23, signal: 'bullish' }
        ]
      },
      {
        id: 'range-bound',
        name: 'Range-Bound Market',
        description: 'Sideways movement with defined support and resistance levels',
        probability: 0.25,
        characteristics: [
          'Clear support and resistance levels',
          'Mean reversion behavior',
          'Lower volume participation',
          'Consolidation phase'
        ],
        duration: '1-3 weeks',
        tradingStrategies: [
          'Range trading strategies',
          'Mean reversion plays',
          'Support/resistance bounces',
          'Theta decay strategies'
        ],
        riskLevel: 'low',
        indicators: [
          { name: 'RSI', value: 52, signal: 'neutral' },
          { name: 'MACD', value: -0.008, signal: 'neutral' },
          { name: 'Volume', value: 0.89, signal: 'bearish' }
        ]
      }
    ];

    // Mock configuration
    const mockConfig: AIConfiguration = {
      models: {
        enabled: ['lstm-primary', 'transformer-advanced', 'ensemble-master'],
        weights: {
          'lstm-primary': 0.3,
          'transformer-advanced': 0.2,
          'ensemble-master': 0.5
        },
        retrainingFrequency: 'weekly',
        minConfidence: 0.75
      },
      insights: {
        enableAutoGeneration: true,
        minImpact: 'medium',
        categories: ['opportunity', 'risk', 'correlation', 'pattern'],
        alertThreshold: 0.8
      },
      forecasting: {
        horizon: 7,
        updateFrequency: 4,
        confidenceIntervals: true,
        seasonalityDetection: true
      }
    };

    setModels(mockModels);
    setInsights(mockInsights);
    setMarketRegimes(mockRegimes);
    setConfiguration(mockConfig);
  };

  const generatePredictions = (days: number) => {
    const predictions = [];
    let baseValue = 1.0945;

    for (let i = 0; i < days; i++) {
      const trend = 0.002 * i;
      const noise = (Math.random() - 0.5) * 0.01;
      const value = baseValue + trend + noise;
      const confidence = 0.95 - (i * 0.05);
      const spread = value * 0.01 * (1 + i * 0.1);

      predictions.push({
        timestamp: new Date(Date.now() + i * 24 * 60 * 60 * 1000).toISOString(),
        value: parseFloat(value.toFixed(5)),
        confidence,
        upperBound: parseFloat((value + spread).toFixed(5)),
        lowerBound: parseFloat((value - spread).toFixed(5))
      });
    }

    return predictions;
  };

  const getModelStatusIcon = (status: AIModel['status']) => {
    switch (status) {
      case 'ready': return <Zap className="h-4 w-4 text-green-600" />;
      case 'training': return <Cpu className="h-4 w-4 text-blue-600 animate-pulse" />;
      case 'predicting': return <Brain className="h-4 w-4 text-purple-600 animate-pulse" />;
      case 'error': return <AlertTriangle className="h-4 w-4 text-red-600" />;
      default: return <Activity className="h-4 w-4 text-gray-600" />;
    }
  };

  const getInsightIcon = (type: TrendInsight['type']) => {
    switch (type) {
      case 'opportunity': return <Target className="h-5 w-5 text-green-600" />;
      case 'risk': return <AlertTriangle className="h-5 w-5 text-red-600" />;
      case 'pattern': return <BarChart3 className="h-5 w-5 text-blue-600" />;
      case 'correlation': return <Activity className="h-5 w-5 text-purple-600" />;
      case 'anomaly': return <Eye className="h-5 w-5 text-orange-600" />;
      default: return <Lightbulb className="h-5 w-5 text-yellow-600" />;
    }
  };

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'critical': return 'bg-red-100 text-red-800';
      case 'high': return 'bg-orange-100 text-orange-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'low': return 'bg-green-100 text-green-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const retrainModel = (modelId: string) => {
    setModels(prev => prev.map(model =>
      model.id === modelId
        ? { ...model, status: 'training' as const }
        : model
    ));

    // Simulate training completion
    setTimeout(() => {
      setModels(prev => prev.map(model =>
        model.id === modelId
          ? {
              ...model,
              status: 'ready' as const,
              lastTrained: new Date().toISOString(),
              accuracy: Math.min(0.95, model.accuracy + Math.random() * 0.02)
            }
          : model
      ));
    }, 3000);
  };

  const selectedModelData = models.find(m => m.id === selectedModel);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center space-x-2">
            <Brain className="h-8 w-8 text-primary" />
            <span>AI Trend Analysis</span>
          </h1>
          <p className="text-muted-foreground">
            Advanced machine learning models for predictive market analysis
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" onClick={() => setShowAdvanced(!showAdvanced)}>
            <Settings className="h-4 w-4 mr-2" />
            {showAdvanced ? 'Hide' : 'Show'} Advanced
          </Button>
          <Switch
            checked={analysisMode === 'auto'}
            onCheckedChange={(checked) => setAnalysisMode(checked ? 'auto' : 'manual')}
          />
          <Label className="text-sm">Auto Analysis</Label>
        </div>
      </div>

      {/* AI Models Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {models.map(model => (
          <Card key={model.id} className={`cursor-pointer transition-all ${
            selectedModel === model.id ? 'ring-2 ring-primary' : 'hover:shadow-md'
          }`} onClick={() => setSelectedModel(model.id)}>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">{model.name}</CardTitle>
                {getModelStatusIcon(model.status)}
              </div>
              <CardDescription className="capitalize">{model.type} model</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-sm">Accuracy</span>
                  <div className="flex items-center space-x-2">
                    <Progress value={model.accuracy * 100} className="w-16 h-2" />
                    <span className="text-sm font-medium">{(model.accuracy * 100).toFixed(1)}%</span>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm">Confidence</span>
                  <span className="text-sm font-medium">{(model.confidence * 100).toFixed(0)}%</span>
                </div>

                <div className="text-xs text-muted-foreground">
                  Last trained: {new Date(model.lastTrained).toLocaleDateString()}
                </div>

                <Button
                  size="sm"
                  variant="outline"
                  className="w-full"
                  onClick={(e) => {
                    e.stopPropagation();
                    retrainModel(model.id);
                  }}
                  disabled={model.status === 'training'}
                >
                  {model.status === 'training' ? 'Training...' : 'Retrain Model'}
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Main Analysis Tabs */}
      <Tabs defaultValue="insights" className="space-y-4">
        <TabsList>
          <TabsTrigger value="insights">AI Insights</TabsTrigger>
          <TabsTrigger value="predictions">Predictions</TabsTrigger>
          <TabsTrigger value="regimes">Market Regimes</TabsTrigger>
          <TabsTrigger value="performance">Model Performance</TabsTrigger>
        </TabsList>

        <TabsContent value="insights" className="space-y-4">
          <div className="space-y-4">
            {insights.map(insight => (
              <Card key={insight.id}>
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="flex items-center space-x-3">
                      {getInsightIcon(insight.type)}
                      <div>
                        <CardTitle className="text-lg">{insight.title}</CardTitle>
                        <CardDescription className="flex items-center space-x-2 mt-1">
                          <Badge className={getImpactColor(insight.impact)}>
                            {insight.impact} impact
                          </Badge>
                          <Badge variant="outline">{insight.timeframe}</Badge>
                          {insight.aiGenerated && (
                            <Badge variant="secondary">
                              <Brain className="h-3 w-3 mr-1" />
                              AI Generated
                            </Badge>
                          )}
                        </CardDescription>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-medium">
                        {(insight.confidence * 100).toFixed(0)}% confidence
                      </div>
                      <Progress value={insight.confidence * 100} className="w-20 h-2 mt-1" />
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <p className="text-sm text-muted-foreground">{insight.description}</p>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 p-3 bg-muted/30 rounded-lg">
                      <div className="text-center">
                        <div className="text-lg font-bold">{insight.data.metric}</div>
                        <div className="text-xs text-muted-foreground">Asset/Metric</div>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-bold">{insight.data.currentValue}</div>
                        <div className="text-xs text-muted-foreground">Current Value</div>
                      </div>
                      <div className="text-center">
                        <div className={`text-lg font-bold ${
                          insight.data.changePercentage > 0 ? 'text-green-600' : 'text-red-600'
                        }`}>
                          {insight.data.changePercentage > 0 ? '+' : ''}{insight.data.changePercentage.toFixed(1)}%
                        </div>
                        <div className="text-xs text-muted-foreground">Predicted Change</div>
                      </div>
                    </div>

                    <div>
                      <h4 className="font-medium mb-2">AI Recommendations:</h4>
                      <ul className="space-y-1">
                        {insight.recommendations.map((rec, index) => (
                          <li key={index} className="text-sm flex items-start space-x-2">
                            <span className="text-primary mt-1">•</span>
                            <span>{rec}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div className="text-xs text-muted-foreground">
                      Generated: {new Date(insight.timestamp).toLocaleString()}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="predictions">
          {selectedModelData && (
            <Card>
              <CardHeader>
                <CardTitle>{selectedModelData.name} Predictions</CardTitle>
                <CardDescription>
                  7-day forecast with confidence intervals
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="h-64 bg-muted/30 rounded-lg flex items-center justify-center">
                    <div className="text-center">
                      <TrendingUp className="h-12 w-12 mx-auto mb-3 text-muted-foreground" />
                      <p className="text-lg font-medium">Prediction Chart</p>
                      <p className="text-sm text-muted-foreground">
                        Time series prediction with confidence bounds
                      </p>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <h4 className="font-medium mb-3">Next 3 Days</h4>
                      <div className="space-y-2">
                        {selectedModelData.predictions.slice(0, 3).map((pred, index) => (
                          <div key={index} className="flex items-center justify-between p-2 bg-muted/30 rounded">
                            <span className="text-sm">
                              {new Date(pred.timestamp).toLocaleDateString()}
                            </span>
                            <div className="text-right">
                              <div className="font-medium">{pred.value}</div>
                              <div className="text-xs text-muted-foreground">
                                ±{((pred.upperBound - pred.lowerBound) / 2).toFixed(4)}
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div>
                      <h4 className="font-medium mb-3">Model Statistics</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-sm">Mean Squared Error:</span>
                          <span className="text-sm font-medium">{selectedModelData.performance.mse.toFixed(4)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">Mean Absolute Error:</span>
                          <span className="text-sm font-medium">{selectedModelData.performance.mae.toFixed(4)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">R² Score:</span>
                          <span className="text-sm font-medium">{selectedModelData.performance.r2.toFixed(4)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-sm">MAPE:</span>
                          <span className="text-sm font-medium">{(selectedModelData.performance.mape * 100).toFixed(2)}%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="regimes">
          <div className="space-y-4">
            {marketRegimes.map(regime => (
              <Card key={regime.id}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle className="text-lg">{regime.name}</CardTitle>
                      <CardDescription>{regime.description}</CardDescription>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold">{(regime.probability * 100).toFixed(0)}%</div>
                      <div className="text-xs text-muted-foreground">Probability</div>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    <div>
                      <h4 className="font-medium mb-2">Characteristics</h4>
                      <ul className="space-y-1">
                        {regime.characteristics.map((char, index) => (
                          <li key={index} className="text-sm flex items-start space-x-2">
                            <span className="text-primary mt-1">•</span>
                            <span>{char}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div>
                      <h4 className="font-medium mb-2">Trading Strategies</h4>
                      <ul className="space-y-1">
                        {regime.tradingStrategies.map((strategy, index) => (
                          <li key={index} className="text-sm flex items-start space-x-2">
                            <span className="text-green-600 mt-1">✓</span>
                            <span>{strategy}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div>
                      <h4 className="font-medium mb-2">Key Indicators</h4>
                      <div className="space-y-2">
                        {regime.indicators.map((indicator, index) => (
                          <div key={index} className="flex items-center justify-between">
                            <span className="text-sm">{indicator.name}</span>
                            <div className="flex items-center space-x-2">
                              <span className="text-sm font-medium">{indicator.value}</span>
                              <Badge variant={
                                indicator.signal === 'bullish' ? 'default' :
                                indicator.signal === 'bearish' ? 'destructive' : 'secondary'
                              } className="text-xs">
                                {indicator.signal}
                              </Badge>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="mt-4 flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <span className="text-sm">Risk Level:</span>
                      <Badge className={
                        regime.riskLevel === 'low' ? 'bg-green-100 text-green-800' :
                        regime.riskLevel === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-red-100 text-red-800'
                      }>
                        {regime.riskLevel}
                      </Badge>
                    </div>
                    <div className="text-sm text-muted-foreground">
                      Expected duration: {regime.duration}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="performance">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Model Comparison</CardTitle>
                <CardDescription>Performance metrics across all AI models</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {models.map(model => (
                    <div key={model.id} className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="font-medium">{model.name}</span>
                        <span className="text-sm">{(model.accuracy * 100).toFixed(1)}%</span>
                      </div>
                      <Progress value={model.accuracy * 100} className="h-2" />
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Training History</CardTitle>
                <CardDescription>Model accuracy over time</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="h-64 bg-muted/30 rounded-lg flex items-center justify-center">
                  <div className="text-center">
                    <BarChart3 className="h-12 w-12 mx-auto mb-3 text-muted-foreground" />
                    <p className="text-lg font-medium">Training History Chart</p>
                    <p className="text-sm text-muted-foreground">
                      Model accuracy progression over training epochs
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default AITrendAnalysis;