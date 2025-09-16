import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Slider } from '@/components/ui/slider';
import {
  GraduationCap,
  Brain,
  Activity,
  Play,
  Pause,
  RotateCcw,
  Settings,
  TrendingUp,
  Target,
  Zap,
  Database,
  LineChart,
  Eye,
  Cpu,
  Clock,
  CheckCircle2,
  AlertCircle,
  Download,
  Upload,
  Save,
  RefreshCw
} from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';

interface TrainingSession {
  id: string;
  model_name: string;
  symbol: string;
  status: 'running' | 'completed' | 'failed' | 'pending';
  progress: number;
  accuracy: number;
  loss: number;
  epochs_completed: number;
  epochs_total: number;
  start_time: string;
  duration: number;
}

interface DatasetInfo {
  symbol: string;
  total_ticks: number;
  date_range: {
    start: string;
    end: string;
  };
  quality_score: number;
  features_count: number;
  size_mb: number;
}

interface ModelConfig {
  lstm_units: number;
  dropout_rate: number;
  learning_rate: number;
  batch_size: number;
  epochs: number;
  sequence_length: number;
  use_attention: boolean;
  use_bidirectional: boolean;
}

export default function Training() {
  const [currentSession, setCurrentSession] = useState<TrainingSession | null>({
    id: 'session_001',
    model_name: 'LSTM-v2.1',
    symbol: 'R_75',
    status: 'running',
    progress: 65,
    accuracy: 0.678,
    loss: 0.234,
    epochs_completed: 65,
    epochs_total: 100,
    start_time: new Date().toISOString(),
    duration: 2340
  });

  const [datasets, setDatasets] = useState<DatasetInfo[]>([
    {
      symbol: 'R_75',
      total_ticks: 284567,
      date_range: { start: '2024-01-01', end: '2024-01-15' },
      quality_score: 94.2,
      features_count: 22,
      size_mb: 45.7
    },
    {
      symbol: 'R_100',
      total_ticks: 198432,
      date_range: { start: '2024-01-01', end: '2024-01-15' },
      quality_score: 91.8,
      features_count: 22,
      size_mb: 32.1
    },
    {
      symbol: 'R_50',
      total_ticks: 356789,
      date_range: { start: '2024-01-01', end: '2024-01-15' },
      quality_score: 96.1,
      features_count: 22,
      size_mb: 58.3
    }
  ]);

  const [modelConfig, setModelConfig] = useState<ModelConfig>({
    lstm_units: 128,
    dropout_rate: 0.3,
    learning_rate: 0.001,
    batch_size: 32,
    epochs: 100,
    sequence_length: 60,
    use_attention: true,
    use_bidirectional: true
  });

  const [trainingHistory, setTrainingHistory] = useState([
    { epoch: 1, accuracy: 0.52, loss: 0.89, val_accuracy: 0.51, val_loss: 0.91 },
    { epoch: 10, accuracy: 0.58, loss: 0.74, val_accuracy: 0.57, val_loss: 0.76 },
    { epoch: 20, accuracy: 0.62, loss: 0.65, val_accuracy: 0.61, val_loss: 0.67 },
    { epoch: 30, accuracy: 0.65, loss: 0.58, val_accuracy: 0.64, val_loss: 0.61 },
    { epoch: 40, accuracy: 0.67, loss: 0.52, val_accuracy: 0.66, val_loss: 0.54 },
    { epoch: 50, accuracy: 0.68, loss: 0.48, val_accuracy: 0.67, val_loss: 0.49 },
    { epoch: 60, accuracy: 0.678, loss: 0.234, val_accuracy: 0.675, val_loss: 0.238 }
  ]);

  const [isCollectingData, setIsCollectingData] = useState(false);
  const [collectionProgress, setCollectionProgress] = useState(0);

  // Simular coleta de dados em tempo real
  useEffect(() => {
    let interval: NodeJS.Timeout;

    if (isCollectingData) {
      interval = setInterval(() => {
        setCollectionProgress(prev => {
          if (prev >= 100) {
            setIsCollectingData(false);
            return 100;
          }
          return prev + Math.random() * 3 + 1;
        });

        // Simular atualizações dos datasets
        setDatasets(prev => prev.map(dataset => ({
          ...dataset,
          total_ticks: dataset.total_ticks + Math.floor(Math.random() * 50 + 10),
          size_mb: dataset.size_mb + Math.random() * 0.1
        })));
      }, 1000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isCollectingData]);

  // Simular progresso do treinamento
  useEffect(() => {
    let interval: NodeJS.Timeout;

    if (currentSession?.status === 'running') {
      interval = setInterval(() => {
        setCurrentSession(prev => {
          if (!prev || prev.progress >= 100) return prev;

          const newProgress = Math.min(prev.progress + Math.random() * 2, 100);
          const newEpoch = Math.floor((newProgress / 100) * prev.epochs_total);

          return {
            ...prev,
            progress: newProgress,
            epochs_completed: newEpoch,
            accuracy: Math.min(0.95, prev.accuracy + Math.random() * 0.001),
            loss: Math.max(0.05, prev.loss - Math.random() * 0.001),
            duration: prev.duration + 3
          };
        });
      }, 3000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [currentSession?.status]);

  const startTraining = () => {
    setCurrentSession({
      id: `session_${Date.now()}`,
      model_name: 'LSTM-v2.2',
      symbol: 'R_75',
      status: 'running',
      progress: 0,
      accuracy: 0.5,
      loss: 1.0,
      epochs_completed: 0,
      epochs_total: modelConfig.epochs,
      start_time: new Date().toISOString(),
      duration: 0
    });
  };

  const pauseTraining = () => {
    setCurrentSession(prev => prev ? { ...prev, status: 'pending' } : null);
  };

  const stopTraining = () => {
    setCurrentSession(prev => prev ? { ...prev, status: 'completed' } : null);
  };

  const startDataCollection = () => {
    setIsCollectingData(true);
    setCollectionProgress(0);
  };

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours}h ${minutes}m ${secs}s`;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'text-green-500';
      case 'completed': return 'text-blue-500';
      case 'failed': return 'text-red-500';
      case 'pending': return 'text-yellow-500';
      default: return 'text-gray-500';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return <Play className="h-4 w-4 text-green-500" />;
      case 'completed': return <CheckCircle2 className="h-4 w-4 text-blue-500" />;
      case 'failed': return <AlertCircle className="h-4 w-4 text-red-500" />;
      case 'pending': return <Pause className="h-4 w-4 text-yellow-500" />;
      default: return <Clock className="h-4 w-4 text-gray-500" />;
    }
  };

  return (
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">🎓 Treinamento</h1>
            <p className="text-muted-foreground">
              Coleta de dados tick-a-tick e treinamento da IA/ML
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant={currentSession?.status === 'running' ? 'default' : 'secondary'} className="gap-1">
              <Brain className="h-3 w-3" />
              {currentSession?.status === 'running' ? 'Treinando' : 'Parado'}
            </Badge>
            <Badge variant={isCollectingData ? 'default' : 'secondary'} className="gap-1">
              <Database className="h-3 w-3" />
              {isCollectingData ? 'Coletando' : 'Dados OK'}
            </Badge>
          </div>
        </div>

        <Tabs defaultValue="training" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="training" className="flex items-center gap-2">
              <GraduationCap className="h-4 w-4" />
              Treinamento
            </TabsTrigger>
            <TabsTrigger value="data" className="flex items-center gap-2">
              <Database className="h-4 w-4" />
              Dados
            </TabsTrigger>
            <TabsTrigger value="config" className="flex items-center gap-2">
              <Settings className="h-4 w-4" />
              Configuração
            </TabsTrigger>
            <TabsTrigger value="analysis" className="flex items-center gap-2">
              <LineChart className="h-4 w-4" />
              Análise
            </TabsTrigger>
          </TabsList>

          {/* Aba de Treinamento */}
          <TabsContent value="training" className="space-y-6">
            {/* Sessão de Treinamento Atual */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <div className="flex items-center">
                    <Brain className="h-5 w-5 mr-2 text-purple-500" />
                    Sessão de Treinamento Atual
                  </div>
                  <div className="flex items-center gap-2">
                    {currentSession && getStatusIcon(currentSession.status)}
                    <span className={`text-sm font-medium ${currentSession ? getStatusColor(currentSession.status) : ''}`}>
                      {currentSession?.status || 'Inativo'}
                    </span>
                  </div>
                </CardTitle>
              </CardHeader>
              <CardContent>
                {currentSession ? (
                  <div className="space-y-6">
                    {/* Progress Bar Principal */}
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span>Progresso do Treinamento</span>
                        <span>{currentSession.progress.toFixed(1)}%</span>
                      </div>
                      <Progress value={currentSession.progress} className="h-3" />
                      <div className="flex justify-between text-xs text-muted-foreground mt-1">
                        <span>Epoch {currentSession.epochs_completed}/{currentSession.epochs_total}</span>
                        <span>Tempo: {formatDuration(currentSession.duration)}</span>
                      </div>
                    </div>

                    {/* Métricas */}
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                      <div className="text-center p-4 bg-muted/50 rounded-lg">
                        <div className="text-2xl font-bold text-green-600">
                          {(currentSession.accuracy * 100).toFixed(1)}%
                        </div>
                        <div className="text-sm text-muted-foreground">Acurácia</div>
                      </div>
                      <div className="text-center p-4 bg-muted/50 rounded-lg">
                        <div className="text-2xl font-bold text-blue-600">
                          {currentSession.loss.toFixed(3)}
                        </div>
                        <div className="text-sm text-muted-foreground">Loss</div>
                      </div>
                      <div className="text-center p-4 bg-muted/50 rounded-lg">
                        <div className="text-2xl font-bold text-purple-600">
                          {currentSession.symbol}
                        </div>
                        <div className="text-sm text-muted-foreground">Símbolo</div>
                      </div>
                      <div className="text-center p-4 bg-muted/50 rounded-lg">
                        <div className="text-2xl font-bold text-orange-600">
                          {currentSession.model_name}
                        </div>
                        <div className="text-sm text-muted-foreground">Modelo</div>
                      </div>
                    </div>

                    {/* Controles */}
                    <div className="flex items-center gap-3">
                      {currentSession.status === 'running' ? (
                        <>
                          <Button onClick={pauseTraining} variant="outline">
                            <Pause className="h-4 w-4 mr-2" />
                            Pausar
                          </Button>
                          <Button onClick={stopTraining} variant="destructive">
                            <AlertCircle className="h-4 w-4 mr-2" />
                            Parar
                          </Button>
                        </>
                      ) : (
                        <Button onClick={startTraining} className="bg-green-600 hover:bg-green-700">
                          <Play className="h-4 w-4 mr-2" />
                          Iniciar Treinamento
                        </Button>
                      )}
                      <Button variant="outline">
                        <Save className="h-4 w-4 mr-2" />
                        Salvar Modelo
                      </Button>
                      <Button variant="outline">
                        <Download className="h-4 w-4 mr-2" />
                        Exportar
                      </Button>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <Brain className="h-12 w-12 mx-auto mb-4 text-muted-foreground opacity-50" />
                    <h3 className="text-lg font-medium mb-2">Nenhum treinamento ativo</h3>
                    <p className="text-muted-foreground mb-4">
                      Configure os parâmetros e inicie um novo treinamento
                    </p>
                    <Button onClick={startTraining} className="bg-green-600 hover:bg-green-700">
                      <Play className="h-4 w-4 mr-2" />
                      Iniciar Treinamento
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Gráfico de Progresso */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <LineChart className="h-5 w-5 mr-2" />
                  Evolução do Treinamento
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <h4 className="text-sm font-medium mb-2">Acurácia</h4>
                      <div className="h-32 bg-muted/30 rounded-lg flex items-center justify-center">
                        <span className="text-muted-foreground text-sm">Gráfico de Acurácia vs Epochs</span>
                      </div>
                    </div>
                    <div>
                      <h4 className="text-sm font-medium mb-2">Loss</h4>
                      <div className="h-32 bg-muted/30 rounded-lg flex items-center justify-center">
                        <span className="text-muted-foreground text-sm">Gráfico de Loss vs Epochs</span>
                      </div>
                    </div>
                  </div>

                  {/* Tabela de Histórico */}
                  <div>
                    <h4 className="text-sm font-medium mb-2">Últimas Epochs</h4>
                    <div className="border rounded-lg">
                      <div className="grid grid-cols-5 gap-4 p-3 border-b bg-muted/30 text-sm font-medium">
                        <span>Epoch</span>
                        <span>Accuracy</span>
                        <span>Loss</span>
                        <span>Val Accuracy</span>
                        <span>Val Loss</span>
                      </div>
                      {trainingHistory.slice(-5).map((entry) => (
                        <div key={entry.epoch} className="grid grid-cols-5 gap-4 p-3 border-b text-sm">
                          <span>{entry.epoch}</span>
                          <span className="text-green-600">{(entry.accuracy * 100).toFixed(1)}%</span>
                          <span className="text-red-600">{entry.loss.toFixed(3)}</span>
                          <span className="text-blue-600">{(entry.val_accuracy * 100).toFixed(1)}%</span>
                          <span className="text-orange-600">{entry.val_loss.toFixed(3)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Aba de Dados */}
          <TabsContent value="data" className="space-y-6">
            {/* Coleta em Tempo Real */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <div className="flex items-center">
                    <Activity className="h-5 w-5 mr-2 text-blue-500" />
                    Coleta de Dados em Tempo Real
                  </div>
                  <Badge variant={isCollectingData ? 'default' : 'secondary'}>
                    {isCollectingData ? 'Coletando' : 'Parado'}
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {isCollectingData && (
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span>Progresso da Coleta</span>
                        <span>{collectionProgress.toFixed(1)}%</span>
                      </div>
                      <Progress value={collectionProgress} className="h-2" />
                    </div>
                  )}

                  <div className="flex items-center gap-3">
                    <Button
                      onClick={startDataCollection}
                      disabled={isCollectingData}
                      className="bg-blue-600 hover:bg-blue-700"
                    >
                      <Database className="h-4 w-4 mr-2" />
                      {isCollectingData ? 'Coletando...' : 'Iniciar Coleta'}
                    </Button>
                    <Button variant="outline">
                      <RefreshCw className="h-4 w-4 mr-2" />
                      Atualizar Datasets
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Datasets Disponíveis */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Database className="h-5 w-5 mr-2" />
                  Datasets Disponíveis
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {datasets.map((dataset) => (
                    <div key={dataset.symbol} className="border rounded-lg p-4">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-2">
                          <Badge variant="outline">{dataset.symbol}</Badge>
                          <span className="font-medium">{dataset.total_ticks.toLocaleString()} ticks</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge variant={dataset.quality_score > 95 ? 'default' : dataset.quality_score > 90 ? 'secondary' : 'destructive'}>
                            {dataset.quality_score.toFixed(1)}% qualidade
                          </Badge>
                          <span className="text-sm text-muted-foreground">{dataset.size_mb.toFixed(1)} MB</span>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <span className="text-muted-foreground">Período:</span>
                          <div className="font-medium">{dataset.date_range.start} - {dataset.date_range.end}</div>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Features:</span>
                          <div className="font-medium">{dataset.features_count} features</div>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Qualidade:</span>
                          <div className="font-medium">{dataset.quality_score.toFixed(1)}%</div>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Tamanho:</span>
                          <div className="font-medium">{dataset.size_mb.toFixed(1)} MB</div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Aba de Configuração */}
          <TabsContent value="config" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Settings className="h-5 w-5 mr-2" />
                  Configuração do Modelo
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Arquitetura */}
                <div>
                  <h3 className="text-lg font-medium mb-4">Arquitetura da Rede Neural</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <div>
                        <Label htmlFor="lstm-units">LSTM Units</Label>
                        <div className="flex items-center space-x-4 mt-2">
                          <Slider
                            value={[modelConfig.lstm_units]}
                            onValueChange={(value) => setModelConfig(prev => ({ ...prev, lstm_units: value[0] }))}
                            max={512}
                            min={32}
                            step={32}
                            className="flex-1"
                          />
                          <span className="w-16 text-sm">{modelConfig.lstm_units}</span>
                        </div>
                      </div>

                      <div>
                        <Label htmlFor="dropout">Dropout Rate</Label>
                        <div className="flex items-center space-x-4 mt-2">
                          <Slider
                            value={[modelConfig.dropout_rate]}
                            onValueChange={(value) => setModelConfig(prev => ({ ...prev, dropout_rate: value[0] }))}
                            max={0.8}
                            min={0.1}
                            step={0.1}
                            className="flex-1"
                          />
                          <span className="w-16 text-sm">{modelConfig.dropout_rate.toFixed(1)}</span>
                        </div>
                      </div>

                      <div>
                        <Label htmlFor="sequence-length">Sequence Length</Label>
                        <div className="flex items-center space-x-4 mt-2">
                          <Slider
                            value={[modelConfig.sequence_length]}
                            onValueChange={(value) => setModelConfig(prev => ({ ...prev, sequence_length: value[0] }))}
                            max={120}
                            min={20}
                            step={10}
                            className="flex-1"
                          />
                          <span className="w-16 text-sm">{modelConfig.sequence_length}</span>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-4">
                      <div className="flex items-center space-x-2">
                        <Switch
                          checked={modelConfig.use_attention}
                          onCheckedChange={(checked) => setModelConfig(prev => ({ ...prev, use_attention: checked }))}
                        />
                        <Label>Usar Attention Mechanism</Label>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Switch
                          checked={modelConfig.use_bidirectional}
                          onCheckedChange={(checked) => setModelConfig(prev => ({ ...prev, use_bidirectional: checked }))}
                        />
                        <Label>LSTM Bidirecional</Label>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Parâmetros de Treinamento */}
                <div>
                  <h3 className="text-lg font-medium mb-4">Parâmetros de Treinamento</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <Label htmlFor="learning-rate">Learning Rate</Label>
                      <Input
                        type="number"
                        value={modelConfig.learning_rate}
                        onChange={(e) => setModelConfig(prev => ({ ...prev, learning_rate: parseFloat(e.target.value) }))}
                        step="0.0001"
                        min="0.0001"
                        max="0.01"
                      />
                    </div>

                    <div>
                      <Label htmlFor="batch-size">Batch Size</Label>
                      <Input
                        type="number"
                        value={modelConfig.batch_size}
                        onChange={(e) => setModelConfig(prev => ({ ...prev, batch_size: parseInt(e.target.value) }))}
                        min="8"
                        max="128"
                        step="8"
                      />
                    </div>

                    <div>
                      <Label htmlFor="epochs">Epochs</Label>
                      <Input
                        type="number"
                        value={modelConfig.epochs}
                        onChange={(e) => setModelConfig(prev => ({ ...prev, epochs: parseInt(e.target.value) }))}
                        min="10"
                        max="500"
                        step="10"
                      />
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  <Button className="bg-green-600 hover:bg-green-700">
                    <Save className="h-4 w-4 mr-2" />
                    Salvar Configuração
                  </Button>
                  <Button variant="outline">
                    <Upload className="h-4 w-4 mr-2" />
                    Carregar Preset
                  </Button>
                  <Button variant="outline">
                    <RotateCcw className="h-4 w-4 mr-2" />
                    Restaurar Padrão
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Aba de Análise */}
          <TabsContent value="analysis" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Eye className="h-5 w-5 mr-2" />
                  Análise de Padrões Descobertos
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-6">
                  <Alert>
                    <Brain className="h-4 w-4" />
                    <AlertDescription>
                      A IA identificou 23 padrões únicos nos dados de tick.
                      Os padrões com maior confiança estão sendo utilizados para trading.
                    </AlertDescription>
                  </Alert>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h4 className="font-medium mb-3">Padrões de Alta</h4>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                          <span className="text-sm">Divergência de Momentum</span>
                          <Badge variant="default">85% confiança</Badge>
                        </div>
                        <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                          <span className="text-sm">Quebra de Resistência</span>
                          <Badge variant="default">78% confiança</Badge>
                        </div>
                        <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                          <span className="text-sm">Padrão Velocity</span>
                          <Badge variant="default">72% confiança</Badge>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h4 className="font-medium mb-3">Padrões de Baixa</h4>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between p-3 bg-red-50 rounded-lg">
                          <span className="text-sm">Reversão de Tendência</span>
                          <Badge variant="destructive">82% confiança</Badge>
                        </div>
                        <div className="flex items-center justify-between p-3 bg-red-50 rounded-lg">
                          <span className="text-sm">Volume Spike Negativo</span>
                          <Badge variant="destructive">75% confiança</Badge>
                        </div>
                        <div className="flex items-center justify-between p-3 bg-red-50 rounded-lg">
                          <span className="text-sm">Suporte Quebrado</span>
                          <Badge variant="destructive">69% confiança</Badge>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h4 className="font-medium mb-3">Feature Importance</h4>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Price Velocity</span>
                        <div className="flex items-center gap-2">
                          <div className="w-32 bg-muted rounded-full h-2">
                            <div className="bg-blue-500 h-2 rounded-full" style={{ width: '92%' }} />
                          </div>
                          <span className="text-sm w-12">92%</span>
                        </div>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Volatility 5min</span>
                        <div className="flex items-center gap-2">
                          <div className="w-32 bg-muted rounded-full h-2">
                            <div className="bg-blue-500 h-2 rounded-full" style={{ width: '87%' }} />
                          </div>
                          <span className="text-sm w-12">87%</span>
                        </div>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm">Momentum</span>
                        <div className="flex items-center gap-2">
                          <div className="w-32 bg-muted rounded-full h-2">
                            <div className="bg-blue-500 h-2 rounded-full" style={{ width: '84%' }} />
                          </div>
                          <span className="text-sm w-12">84%</span>
                        </div>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm">RSI Tick</span>
                        <div className="flex items-center gap-2">
                          <div className="w-32 bg-muted rounded-full h-2">
                            <div className="bg-blue-500 h-2 rounded-full" style={{ width: '76%' }} />
                          </div>
                          <span className="text-sm w-12">76%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
  );
}