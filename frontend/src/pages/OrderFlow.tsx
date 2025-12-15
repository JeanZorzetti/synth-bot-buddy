import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell, LineChart, Line, Area, AreaChart } from 'recharts';
import { Activity, TrendingUp, TrendingDown, Minus, AlertCircle, BarChart3, Layers } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface OrderBookData {
  bid_volume: number;
  ask_volume: number;
  bid_pressure: number;
  ask_pressure: number;
  imbalance: 'bullish' | 'bearish' | 'neutral';
  spread: number;
  spread_pct: number;
  best_bid: number;
  best_ask: number;
  depth_ratio: number;
  bid_walls: Array<{ price: number; volume: number }>;
  ask_walls: Array<{ price: number; volume: number }>;
}

interface AggressiveOrders {
  aggressive_buys: Array<{ price: number; size: number; timestamp: string }>;
  aggressive_sells: Array<{ price: number; size: number; timestamp: string }>;
  aggressive_sentiment: 'bullish' | 'bearish' | 'neutral';
  total_buy_volume: number;
  total_sell_volume: number;
  buy_pressure: number;
  delta: number;
  aggression_intensity: number;
}

interface VolumeProfile {
  poc: number;
  vah: number;
  val: number;
  value_area_volume_pct: number;
  total_volume: number;
  volume_profile: Array<{ price: number; volume: number; percentage: number }>;
}

interface TapeReading {
  buy_pressure: number;
  sell_pressure: number;
  interpretation: string;
  absorption: {
    detected: boolean;
    type: string;
    strength: number;
  };
  momentum: {
    speed: string;
    trades_per_minute: number;
    acceleration: number;
  };
  buy_volume: number;
  sell_volume: number;
  num_trades: number;
}

export default function OrderFlow() {
  const [orderBook, setOrderBook] = useState<OrderBookData | null>(null);
  const [aggressiveOrders, setAggressiveOrders] = useState<AggressiveOrders | null>(null);
  const [volumeProfile, setVolumeProfile] = useState<VolumeProfile | null>(null);
  const [tapeReading, setTapeReading] = useState<TapeReading | null>(null);
  const [loading, setLoading] = useState(false);
  const [symbol, setSymbol] = useState('R_100');

  const loadOrderFlowData = async () => {
    setLoading(true);
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'https://botderivapi.roilabs.com.br';

      // Mock data para demonstração (em produção, virá da API)
      const mockOrderBook: OrderBookData = {
        bid_volume: 15800,
        ask_volume: 9200,
        bid_pressure: 63.2,
        ask_pressure: 36.8,
        imbalance: 'bullish',
        spread: 0.15,
        spread_pct: 0.015,
        best_bid: 1234.85,
        best_ask: 1235.00,
        depth_ratio: 1.72,
        bid_walls: [
          { price: 1234.50, volume: 5000 },
          { price: 1233.00, volume: 3200 }
        ],
        ask_walls: [
          { price: 1235.50, volume: 2800 }
        ]
      };

      const mockAggressiveOrders: AggressiveOrders = {
        aggressive_buys: [
          { price: 1235.00, size: 1500, timestamp: '2024-12-14T10:30:15' },
          { price: 1235.10, size: 2000, timestamp: '2024-12-14T10:30:45' }
        ],
        aggressive_sells: [
          { price: 1234.90, size: 800, timestamp: '2024-12-14T10:30:30' }
        ],
        aggressive_sentiment: 'bullish',
        total_buy_volume: 18500,
        total_sell_volume: 8200,
        buy_pressure: 69.3,
        delta: 10300,
        aggression_intensity: 75.5
      };

      const mockVolumeProfile: VolumeProfile = {
        poc: 1234.75,
        vah: 1235.20,
        val: 1234.30,
        value_area_volume_pct: 70.0,
        total_volume: 125000,
        volume_profile: Array.from({ length: 20 }, (_, i) => ({
          price: 1234.00 + (i * 0.10),
          volume: Math.random() * 10000 + 2000,
          percentage: Math.random() * 100
        })).sort((a, b) => b.price - a.price)
      };

      const mockTapeReading: TapeReading = {
        buy_pressure: 68.5,
        sell_pressure: 31.5,
        interpretation: 'forte pressão compradora',
        absorption: {
          detected: true,
          type: 'absorbing_sells',
          strength: 65
        },
        momentum: {
          speed: 'fast',
          trades_per_minute: 45,
          acceleration: 12.5
        },
        buy_volume: 24500,
        sell_volume: 11200,
        num_trades: 127
      };

      setOrderBook(mockOrderBook);
      setAggressiveOrders(mockAggressiveOrders);
      setVolumeProfile(mockVolumeProfile);
      setTapeReading(mockTapeReading);
    } catch (error) {
      console.error('Error loading order flow data:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadOrderFlowData();
    const interval = setInterval(loadOrderFlowData, 10000); // Atualizar a cada 10s
    return () => clearInterval(interval);
  }, [symbol]);

  const getSentimentIcon = (sentiment: 'bullish' | 'bearish' | 'neutral') => {
    if (sentiment === 'bullish') return <TrendingUp className="h-5 w-5 text-emerald-500" />;
    if (sentiment === 'bearish') return <TrendingDown className="h-5 w-5 text-red-500" />;
    return <Minus className="h-5 w-5 text-gray-500" />;
  };

  const getSentimentColor = (sentiment: 'bullish' | 'bearish' | 'neutral') => {
    if (sentiment === 'bullish') return 'text-emerald-500';
    if (sentiment === 'bearish') return 'text-red-500';
    return 'text-gray-500';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white flex items-center gap-3">
              <Activity className="h-8 w-8 text-blue-500" />
              Order Flow Analysis
            </h1>
            <p className="text-gray-400 mt-1">
              Análise institucional de fluxo de ordens em tempo real
            </p>
          </div>
          <Button
            onClick={loadOrderFlowData}
            disabled={loading}
            className="bg-blue-600 hover:bg-blue-700"
          >
            {loading ? 'Atualizando...' : 'Atualizar Dados'}
          </Button>
        </div>

        {/* Order Book Analysis */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Order Book Depth */}
          <Card className="bg-gray-800/50 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-blue-500" />
                Order Book Depth
              </CardTitle>
              <CardDescription className="text-gray-400">
                Profundidade do livro de ordens (Bid vs Ask)
              </CardDescription>
            </CardHeader>
            <CardContent>
              {orderBook && (
                <div className="space-y-4">
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart
                      data={[
                        { name: 'Bid', volume: orderBook.bid_volume, fill: '#10b981' },
                        { name: 'Ask', volume: orderBook.ask_volume, fill: '#ef4444' }
                      ]}
                      layout="horizontal"
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis type="category" dataKey="name" stroke="#9ca3af" />
                      <YAxis type="number" stroke="#9ca3af" />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                        labelStyle={{ color: '#fff' }}
                      />
                      <Bar dataKey="volume" radius={[8, 8, 0, 0]}>
                        {[
                          { fill: '#10b981' },
                          { fill: '#ef4444' }
                        ].map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.fill} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>

                  {/* Metrics */}
                  <div className="grid grid-cols-3 gap-4">
                    <div className="bg-gray-700/50 p-3 rounded-lg">
                      <p className="text-xs text-gray-400">Bid Pressure</p>
                      <p className="text-xl font-bold text-emerald-500">
                        {orderBook.bid_pressure.toFixed(1)}%
                      </p>
                    </div>
                    <div className="bg-gray-700/50 p-3 rounded-lg">
                      <p className="text-xs text-gray-400">Imbalance</p>
                      <p className={`text-xl font-bold flex items-center gap-1 ${getSentimentColor(orderBook.imbalance)}`}>
                        {getSentimentIcon(orderBook.imbalance)}
                        {orderBook.imbalance}
                      </p>
                    </div>
                    <div className="bg-gray-700/50 p-3 rounded-lg">
                      <p className="text-xs text-gray-400">Spread</p>
                      <p className="text-xl font-bold text-blue-400">
                        {orderBook.spread_pct.toFixed(3)}%
                      </p>
                    </div>
                  </div>

                  {/* Walls Detection */}
                  {(orderBook.bid_walls.length > 0 || orderBook.ask_walls.length > 0) && (
                    <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-3">
                      <div className="flex items-start gap-2">
                        <AlertCircle className="h-5 w-5 text-yellow-500 mt-0.5" />
                        <div>
                          <p className="text-sm font-medium text-yellow-500">Muros Detectados</p>
                          <p className="text-xs text-gray-400 mt-1">
                            {orderBook.bid_walls.length} Bid Walls, {orderBook.ask_walls.length} Ask Walls
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Aggressive Orders */}
          <Card className="bg-gray-800/50 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <TrendingUp className="h-5 w-5 text-orange-500" />
                Aggressive Orders
              </CardTitle>
              <CardDescription className="text-gray-400">
                Ordens agressivas detectadas (&gt;3x média)
              </CardDescription>
            </CardHeader>
            <CardContent>
              {aggressiveOrders && (
                <div className="space-y-4">
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart
                      data={[
                        { name: 'Buy', volume: aggressiveOrders.total_buy_volume, fill: '#10b981' },
                        { name: 'Sell', volume: aggressiveOrders.total_sell_volume, fill: '#ef4444' }
                      ]}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="name" stroke="#9ca3af" />
                      <YAxis stroke="#9ca3af" />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                        labelStyle={{ color: '#fff' }}
                      />
                      <Bar dataKey="volume" radius={[8, 8, 0, 0]}>
                        {[
                          { fill: '#10b981' },
                          { fill: '#ef4444' }
                        ].map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.fill} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>

                  {/* Metrics */}
                  <div className="grid grid-cols-3 gap-4">
                    <div className="bg-gray-700/50 p-3 rounded-lg">
                      <p className="text-xs text-gray-400">Sentiment</p>
                      <p className={`text-lg font-bold flex items-center gap-1 ${getSentimentColor(aggressiveOrders.aggressive_sentiment)}`}>
                        {getSentimentIcon(aggressiveOrders.aggressive_sentiment)}
                        {aggressiveOrders.aggressive_sentiment}
                      </p>
                    </div>
                    <div className="bg-gray-700/50 p-3 rounded-lg">
                      <p className="text-xs text-gray-400">Delta</p>
                      <p className="text-xl font-bold text-blue-400">
                        {aggressiveOrders.delta > 0 ? '+' : ''}{aggressiveOrders.delta.toLocaleString()}
                      </p>
                    </div>
                    <div className="bg-gray-700/50 p-3 rounded-lg">
                      <p className="text-xs text-gray-400">Intensity</p>
                      <p className="text-xl font-bold text-purple-400">
                        {aggressiveOrders.aggression_intensity.toFixed(1)}%
                      </p>
                    </div>
                  </div>

                  {/* Recent Aggressive Orders */}
                  <div className="space-y-2">
                    <p className="text-sm font-medium text-gray-300">Ordens Recentes</p>
                    <div className="space-y-1 max-h-32 overflow-y-auto">
                      {aggressiveOrders.aggressive_buys.slice(0, 3).map((order, i) => (
                        <div key={i} className="bg-emerald-500/10 border border-emerald-500/30 rounded p-2 text-xs">
                          <span className="text-emerald-500 font-medium">BUY</span>
                          <span className="text-gray-400 mx-2">@{order.price}</span>
                          <span className="text-white">{order.size.toLocaleString()}</span>
                        </div>
                      ))}
                      {aggressiveOrders.aggressive_sells.slice(0, 2).map((order, i) => (
                        <div key={i} className="bg-red-500/10 border border-red-500/30 rounded p-2 text-xs">
                          <span className="text-red-500 font-medium">SELL</span>
                          <span className="text-gray-400 mx-2">@{order.price}</span>
                          <span className="text-white">{order.size.toLocaleString()}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Volume Profile & Tape Reading */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Volume Profile */}
          <Card className="bg-gray-800/50 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Layers className="h-5 w-5 text-purple-500" />
                Volume Profile
              </CardTitle>
              <CardDescription className="text-gray-400">
                Distribuição de volume por nível de preço
              </CardDescription>
            </CardHeader>
            <CardContent>
              {volumeProfile && (
                <div className="space-y-4">
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart
                      data={volumeProfile.volume_profile.slice(0, 15)}
                      layout="vertical"
                      margin={{ left: 20, right: 10 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis type="number" stroke="#9ca3af" />
                      <YAxis
                        type="category"
                        dataKey="price"
                        stroke="#9ca3af"
                        tickFormatter={(value) => value.toFixed(2)}
                      />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }}
                        labelStyle={{ color: '#fff' }}
                        formatter={(value: number) => value.toLocaleString()}
                      />
                      <Bar dataKey="volume" radius={[0, 4, 4, 0]}>
                        {volumeProfile.volume_profile.slice(0, 15).map((entry, index) => {
                          let fill = '#6366f1';
                          if (entry.price === volumeProfile.poc) fill = '#f59e0b'; // POC = amber
                          if (entry.price >= volumeProfile.val && entry.price <= volumeProfile.vah) fill = '#8b5cf6'; // Value Area = purple
                          return <Cell key={`cell-${index}`} fill={fill} />;
                        })}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>

                  {/* Key Levels */}
                  <div className="grid grid-cols-3 gap-3">
                    <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-3">
                      <p className="text-xs text-amber-400">POC</p>
                      <p className="text-lg font-bold text-white">
                        {volumeProfile.poc.toFixed(2)}
                      </p>
                    </div>
                    <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-3">
                      <p className="text-xs text-purple-400">VAH</p>
                      <p className="text-lg font-bold text-white">
                        {volumeProfile.vah.toFixed(2)}
                      </p>
                    </div>
                    <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-3">
                      <p className="text-xs text-purple-400">VAL</p>
                      <p className="text-lg font-bold text-white">
                        {volumeProfile.val.toFixed(2)}
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Tape Reading */}
          <Card className="bg-gray-800/50 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Activity className="h-5 w-5 text-cyan-500" />
                Tape Reading
              </CardTitle>
              <CardDescription className="text-gray-400">
                Análise de fluxo de trades em tempo real
              </CardDescription>
            </CardHeader>
            <CardContent>
              {tapeReading && (
                <div className="space-y-4">
                  {/* Pressure Bars */}
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-gray-400">Buy Pressure</span>
                        <span className="text-emerald-500 font-medium">{tapeReading.buy_pressure.toFixed(1)}%</span>
                      </div>
                      <div className="h-4 bg-gray-700 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-500"
                          style={{ width: `${tapeReading.buy_pressure}%` }}
                        />
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <span className="text-gray-400">Sell Pressure</span>
                        <span className="text-red-500 font-medium">{tapeReading.sell_pressure.toFixed(1)}%</span>
                      </div>
                      <div className="h-4 bg-gray-700 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-red-500 to-red-400 transition-all duration-500"
                          style={{ width: `${tapeReading.sell_pressure}%` }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Interpretation */}
                  <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
                    <p className="text-sm font-medium text-blue-400 mb-1">Interpretação</p>
                    <p className="text-lg text-white font-medium">{tapeReading.interpretation}</p>
                  </div>

                  {/* Absorption Detection */}
                  {tapeReading.absorption.detected && (
                    <div className="bg-orange-500/10 border border-orange-500/30 rounded-lg p-4">
                      <div className="flex items-start gap-2">
                        <AlertCircle className="h-5 w-5 text-orange-500 mt-0.5" />
                        <div>
                          <p className="text-sm font-medium text-orange-500">Absorção Detectada</p>
                          <p className="text-xs text-gray-400 mt-1">
                            Tipo: {tapeReading.absorption.type} | Força: {tapeReading.absorption.strength}%
                          </p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Momentum & Stats */}
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-gray-700/50 p-3 rounded-lg">
                      <p className="text-xs text-gray-400">Velocidade</p>
                      <p className="text-lg font-bold text-cyan-400 capitalize">
                        {tapeReading.momentum.speed}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        {tapeReading.momentum.trades_per_minute} trades/min
                      </p>
                    </div>
                    <div className="bg-gray-700/50 p-3 rounded-lg">
                      <p className="text-xs text-gray-400">Aceleração</p>
                      <p className="text-lg font-bold text-purple-400">
                        {tapeReading.momentum.acceleration > 0 ? '+' : ''}{tapeReading.momentum.acceleration.toFixed(1)}%
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        {tapeReading.num_trades} trades
                      </p>
                    </div>
                  </div>

                  {/* Volume Stats */}
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-emerald-500/10 border border-emerald-500/30 rounded p-2">
                      <p className="text-xs text-emerald-400">Buy Volume</p>
                      <p className="text-lg font-bold text-white">
                        {tapeReading.buy_volume.toLocaleString()}
                      </p>
                    </div>
                    <div className="bg-red-500/10 border border-red-500/30 rounded p-2">
                      <p className="text-xs text-red-400">Sell Volume</p>
                      <p className="text-lg font-bold text-white">
                        {tapeReading.sell_volume.toLocaleString()}
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Info Footer */}
        <Card className="bg-gray-800/30 border-gray-700">
          <CardContent className="p-4">
            <div className="flex items-start gap-3 text-sm text-gray-400">
              <AlertCircle className="h-5 w-5 text-blue-500 mt-0.5 flex-shrink-0" />
              <div>
                <p className="font-medium text-gray-300">Sobre Order Flow Analysis</p>
                <p className="mt-1">
                  A análise de Order Flow identifica a intenção de traders institucionais através do livro de ordens,
                  ordens agressivas, volume profile e leitura de tape. Use essas informações para confirmar sinais
                  técnicos e identificar pontos de entrada/saída com maior probabilidade.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
