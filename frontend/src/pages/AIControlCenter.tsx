/**
 * AI Control Center - Phase 8 Integration
 * Interface completa para gerenciamento de IA, ensemble e aprendizado em tempo real
 */

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, PieChart, Pie, Cell } from 'recharts';
import { Brain, Target, TrendingUp, AlertTriangle, Settings, Play, Pause, RotateCcw, Cpu, Zap, Activity } from 'lucide-react';
import apiClient, { EnsemblePrediction, ModelPrediction } from '@/services/apiClient';

interface ModelPerformance {
  model_type: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  sharpe_ratio: number;
  total_predictions: number;
  last_updated: string;
}

interface LearningMetrics {
  total_updates: number;
  successful_adaptations: number;
  drift_detections: number;
  average_learning_time: number;
  adaptation_rate: number;
  data_efficiency: number;
}

interface FeatureImportance {
  [feature: string]: number;
}

interface AIStatus {
  ensemble_enabled: boolean;
  learning_enabled: boolean;
  prediction_confidence_threshold: number;
  retraining_in_progress: boolean;
  last_retrain: string;
  models_trained: number;
  total_predictions_today: number;
}

const AIControlCenter: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState<string>('EUR/USD');
  const [currentPrediction, setCurrentPrediction] = useState<EnsemblePrediction | null>(null);
  const [modelPerformance, setModelPerformance] = useState<ModelPerformance[]>([]);
  const [learningMetrics, setLearningMetrics] = useState<LearningMetrics | null>(null);
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance>({});
  const [aiStatus, setAiStatus] = useState<AIStatus>({
    ensemble_enabled: true,
    learning_enabled: true,
    prediction_confidence_threshold: 0.6,
    retraining_in_progress: false,
    last_retrain: '',
    models_trained: 5,
    total_predictions_today: 0
  });
  const [loading, setLoading] = useState<boolean>(true);

  const symbols = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'BTC/USD', 'ETH/USD', 'Gold', 'Oil'];

  useEffect(() => {
    loadAIData();

    // Setup WebSocket for real-time AI updates
    const ws = apiClient.subscribeToAIUpdates((data) => {
      if (data.type === 'prediction_update') {
        setCurrentPrediction(data.prediction);
      } else if (data.type === 'performance_update') {
        setModelPerformance(data.performance);
      } else if (data.type === 'learning_update') {
        setLearningMetrics(data.metrics);
      }
    });

    return () => {
      ws.close();
    };
  }, [selectedSymbol]);

  const loadAIData = async () => {
    setLoading(true);
    try {
      // Load model performance
      const performance = await apiClient.getModelPerformance();
      setModelPerformance(performance.models || []);

      // Load learning insights
      const learning = await apiClient.getLearningInsights();
      setLearningMetrics(learning.adaptation_metrics || null);

      // Load feature importance
      const importance = await apiClient.getFeatureImportance(selectedSymbol);
      setFeatureImportance(importance);

      // Load real AI status
      const realAiStatus = await apiClient.getAIControlStatus();
      setAiStatus(prev => ({
        ...prev,
        last_retrain: realAiStatus.last_retrain || new Date().toISOString(),
        total_predictions_today: realAiStatus.total_predictions_today || 0,
        active_models: realAiStatus.active_models || 0,
        ensemble_accuracy: realAiStatus.ensemble_accuracy || 0,
        drift_detected: realAiStatus.drift_detected || false,
        retraining_in_progress: realAiStatus.retraining_in_progress || false
      }));

    } catch (error) {
      console.error('Error loading AI data:', error);
    } finally {
      setLoading(false);
    }
  };

  const triggerRetrain = async (modelType?: string) => {
    try {
      setAiStatus(prev => ({ ...prev, retraining_in_progress: true }));
      await apiClient.triggerModelRetrain(modelType);

      setTimeout(() => {
        setAiStatus(prev => ({
          ...prev,
          retraining_in_progress: false,
          last_retrain: new Date().toISOString()
        }));
        loadAIData();
      }, 5000);
    } catch (error) {
      console.error('Error triggering retrain:', error);
      setAiStatus(prev => ({ ...prev, retraining_in_progress: false }));
    }
  };

  const updateAISetting = (key: keyof AIStatus, value: any) => {
    setAiStatus(prev => ({ ...prev, [key]: value }));
  };

  const getModelColor = (modelType: string): string => {
    const colors: Record<string, string> = {
      'lstm': '#3B82F6',
      'transformer': '#10B981',
      'cnn': '#F59E0B',
      'random_forest': '#EF4444',
      'gradient_boosting': '#8B5CF6'
    };
    return colors[modelType] || '#6B7280';
  };

  const performanceData = modelPerformance.map(model => ({
    name: model.model_type.toUpperCase(),
    accuracy: model.accuracy * 100,
    precision: model.precision * 100,
    recall: model.recall * 100,
    f1_score: model.f1_score * 100,
    sharpe: model.sharpe_ratio
  }));

  const featureImportanceData = Object.entries(featureImportance)
    .sort(([,a], [,b]) => b - a)
    .slice(0, 15)
    .map(([name, value]) => ({
      name: name.replace(/_/g, ' ').toUpperCase(),
      importance: value * 100
    }));

  const predictionConfidenceData = currentPrediction?.individual_predictions.map(pred => ({
    model: pred.model_type.toUpperCase(),
    confidence: pred.confidence * 100,
    prediction: pred.value
  })) || [];

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center">
            <Brain className="w-8 h-8 mr-3 text-blue-600" />
            AI Control Center
          </h1>
          <p className="text-muted-foreground">Advanced AI ensemble management and learning systems</p>
        </div>
        <div className="flex items-center space-x-2">
          <Badge variant={aiStatus.ensemble_enabled ? "default" : "secondary"}>
            {aiStatus.ensemble_enabled ? "AI Active" : "AI Paused"}
          </Badge>
          <Button onClick={() => loadAIData()} disabled={loading}>
            Refresh Data
          </Button>
        </div>
      </div>

      {/* AI Status Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Ensemble Status</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-2xl font-bold">{aiStatus.models_trained}</p>
                <p className="text-xs text-muted-foreground">Models Active</p>
              </div>
              <Badge variant={aiStatus.ensemble_enabled ? "default" : "secondary"}>
                {aiStatus.ensemble_enabled ? "Running" : "Stopped"}
              </Badge>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Daily Predictions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-2xl font-bold">{aiStatus.total_predictions_today.toLocaleString()}</p>
                <p className="text-xs text-muted-foreground">Today</p>
              </div>
              <Target className="w-8 h-8 text-green-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Learning Status</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-2xl font-bold">
                  {learningMetrics ? (learningMetrics.adaptation_rate * 100).toFixed(1) : '0'}%
                </p>
                <p className="text-xs text-muted-foreground">Adaptation Rate</p>
              </div>
              <Activity className="w-8 h-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Last Retrain</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">
                  {aiStatus.last_retrain ?
                    new Date(aiStatus.last_retrain).toLocaleDateString() :
                    'Never'
                  }
                </p>
                <p className="text-xs text-muted-foreground">Auto Retrain</p>
              </div>
              {aiStatus.retraining_in_progress ? (
                <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
              ) : (
                <RotateCcw className="w-8 h-8 text-purple-500" />
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Symbol Selection */}
      <div className="flex items-center space-x-4">
        <Select value={selectedSymbol} onValueChange={setSelectedSymbol}>
          <SelectTrigger className="w-48">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {symbols.map(symbol => (
              <SelectItem key={symbol} value={symbol}>{symbol}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Main Content */}
      <Tabs defaultValue="ensemble" className="space-y-4">
        <TabsList>
          <TabsTrigger value="ensemble">Ensemble</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="learning">Learning</TabsTrigger>
          <TabsTrigger value="features">Features</TabsTrigger>
          <TabsTrigger value="settings">Settings</TabsTrigger>
        </TabsList>

        {/* Ensemble Tab */}
        <TabsContent value="ensemble" className="space-y-4">
          {currentPrediction && (
            <Card>
              <CardHeader>
                <CardTitle>Current Ensemble Prediction</CardTitle>
                <CardDescription>
                  Latest prediction for {selectedSymbol} with {(currentPrediction.confidence * 100).toFixed(1)}% confidence
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <div className="text-center">
                      <p className="text-3xl font-bold text-blue-600">
                        {(currentPrediction.final_prediction * 100).toFixed(1)}%
                      </p>
                      <p className="text-sm text-muted-foreground">Final Prediction</p>
                    </div>

                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span>Confidence</span>
                        <span>{(currentPrediction.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <Progress value={currentPrediction.confidence * 100} />

                      <div className="flex justify-between">
                        <span>Consensus Level</span>
                        <span>{(currentPrediction.consensus_level * 100).toFixed(1)}%</span>
                      </div>
                      <Progress value={currentPrediction.consensus_level * 100} />
                    </div>
                  </div>

                  <div>
                    <h4 className="font-semibold mb-3">Individual Model Predictions</h4>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={predictionConfidenceData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="model" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="confidence" fill="#3B82F6" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Model Weight Distribution</CardTitle>
                <CardDescription>Current weights in ensemble voting</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={currentPrediction?.weight_distribution ?
                        Object.entries(currentPrediction.weight_distribution).map(([model, weight]) => ({
                          name: model.toUpperCase(),
                          value: weight * 100,
                          fill: getModelColor(model)
                        })) : []
                      }
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      dataKey="value"
                    >
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Ensemble Actions</CardTitle>
                <CardDescription>Control and monitor the AI ensemble</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Button
                    onClick={() => triggerRetrain()}
                    disabled={aiStatus.retraining_in_progress}
                    className="w-full"
                  >
                    {aiStatus.retraining_in_progress ? (
                      <>
                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                        Retraining...
                      </>
                    ) : (
                      <>
                        <RotateCcw className="w-4 h-4 mr-2" />
                        Retrain All Models
                      </>
                    )}
                  </Button>

                  <Button
                    variant="outline"
                    onClick={() => updateAISetting('ensemble_enabled', !aiStatus.ensemble_enabled)}
                    className="w-full"
                  >
                    {aiStatus.ensemble_enabled ? (
                      <>
                        <Pause className="w-4 h-4 mr-2" />
                        Pause Ensemble
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4 mr-2" />
                        Start Ensemble
                      </>
                    )}
                  </Button>
                </div>

                <Alert>
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>
                    Ensemble is running with {aiStatus.models_trained} active models.
                    Average prediction confidence: {currentPrediction ?
                      (currentPrediction.confidence * 100).toFixed(1) : '0'}%
                  </AlertDescription>
                </Alert>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Performance Tab */}
        <TabsContent value="performance" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Model Performance Comparison</CardTitle>
                <CardDescription>Accuracy and precision metrics</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="accuracy" fill="#3B82F6" name="Accuracy %" />
                    <Bar dataKey="precision" fill="#10B981" name="Precision %" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Performance Radar</CardTitle>
                <CardDescription>Multi-dimensional performance view</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <RadarChart data={performanceData}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="name" />
                    <PolarRadiusAxis domain={[0, 100]} />
                    <Radar
                      name="Accuracy"
                      dataKey="accuracy"
                      stroke="#3B82F6"
                      fill="#3B82F6"
                      fillOpacity={0.1}
                    />
                    <Radar
                      name="Precision"
                      dataKey="precision"
                      stroke="#10B981"
                      fill="#10B981"
                      fillOpacity={0.1}
                    />
                    <Tooltip />
                  </RadarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Detailed Performance Metrics</CardTitle>
              <CardDescription>Individual model statistics</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {modelPerformance.map((model, index) => (
                  <div key={model.model_type} className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between mb-3">
                      <h4 className="font-semibold">{model.model_type.toUpperCase()}</h4>
                      <Badge style={{ backgroundColor: getModelColor(model.model_type) }}>
                        {(model.accuracy * 100).toFixed(1)}% Accuracy
                      </Badge>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <p className="text-muted-foreground">Precision</p>
                        <p className="font-medium">{(model.precision * 100).toFixed(1)}%</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Recall</p>
                        <p className="font-medium">{(model.recall * 100).toFixed(1)}%</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">F1 Score</p>
                        <p className="font-medium">{(model.f1_score * 100).toFixed(1)}%</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Sharpe Ratio</p>
                        <p className="font-medium">{model.sharpe_ratio.toFixed(2)}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Learning Tab */}
        <TabsContent value="learning" className="space-y-4">
          {learningMetrics && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Total Updates</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold">{learningMetrics.total_updates.toLocaleString()}</p>
                  <p className="text-xs text-muted-foreground">Learning iterations</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Success Rate</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold">{(learningMetrics.adaptation_rate * 100).toFixed(1)}%</p>
                  <p className="text-xs text-muted-foreground">Successful adaptations</p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-sm">Drift Detections</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold">{learningMetrics.drift_detections}</p>
                  <p className="text-xs text-muted-foreground">Performance drifts</p>
                </CardContent>
              </Card>
            </div>
          )}

          <Card>
            <CardHeader>
              <CardTitle>Learning Progress</CardTitle>
              <CardDescription>Real-time learning system metrics</CardDescription>
            </CardHeader>
            <CardContent>
              {learningMetrics && (
                <div className="space-y-6">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span>Adaptation Rate</span>
                      <span>{(learningMetrics.adaptation_rate * 100).toFixed(1)}%</span>
                    </div>
                    <Progress value={learningMetrics.adaptation_rate * 100} />
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span>Data Efficiency</span>
                      <span>{(learningMetrics.data_efficiency * 100).toFixed(1)}%</span>
                    </div>
                    <Progress value={learningMetrics.data_efficiency * 100} />
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-muted-foreground">Average Learning Time</p>
                      <p className="font-medium">{learningMetrics.average_learning_time.toFixed(2)}ms</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Successful Adaptations</p>
                      <p className="font-medium">{learningMetrics.successful_adaptations.toLocaleString()}</p>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Features Tab */}
        <TabsContent value="features" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Feature Importance for {selectedSymbol}</CardTitle>
              <CardDescription>Top features driving model predictions</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={featureImportanceData} layout="horizontal" margin={{ left: 100 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="name" type="category" width={100} />
                  <Tooltip />
                  <Bar dataKey="importance" fill="#6366F1" />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Settings Tab */}
        <TabsContent value="settings" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>AI Configuration</CardTitle>
              <CardDescription>Adjust AI behavior and parameters</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Enable Ensemble</p>
                  <p className="text-sm text-muted-foreground">Turn on/off the AI ensemble system</p>
                </div>
                <Switch
                  checked={aiStatus.ensemble_enabled}
                  onCheckedChange={(checked) => updateAISetting('ensemble_enabled', checked)}
                />
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium">Enable Online Learning</p>
                  <p className="text-sm text-muted-foreground">Allow models to adapt in real-time</p>
                </div>
                <Switch
                  checked={aiStatus.learning_enabled}
                  onCheckedChange={(checked) => updateAISetting('learning_enabled', checked)}
                />
              </div>

              <div className="space-y-3">
                <div className="flex justify-between">
                  <p className="font-medium">Prediction Confidence Threshold</p>
                  <span>{(aiStatus.prediction_confidence_threshold * 100).toFixed(0)}%</span>
                </div>
                <Slider
                  value={[aiStatus.prediction_confidence_threshold * 100]}
                  onValueChange={([value]) => updateAISetting('prediction_confidence_threshold', value / 100)}
                  max={100}
                  min={10}
                  step={5}
                />
                <p className="text-sm text-muted-foreground">
                  Minimum confidence required for predictions to be used
                </p>
              </div>

              <div className="pt-4 border-t">
                <Button onClick={() => triggerRetrain()} disabled={aiStatus.retraining_in_progress}>
                  <RotateCcw className="w-4 h-4 mr-2" />
                  Trigger Full Retrain
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default AIControlCenter;