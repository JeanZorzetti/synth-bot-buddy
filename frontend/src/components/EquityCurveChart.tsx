import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';

interface EquityPoint {
  timestamp: string;
  capital: number;
  profit_loss: number;
  trade_id: string;
}

interface EquityCurveChartProps {
  data: EquityPoint[];
  initialCapital: number;
}

export function EquityCurveChart({ data, initialCapital }: EquityCurveChartProps) {
  if (!data || data.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Equity Curve</CardTitle>
          <CardDescription>Evolução do capital ao longo do tempo</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-64 text-muted-foreground">
            Aguardando trades para exibir equity curve...
          </div>
        </CardContent>
      </Card>
    );
  }

  // Formatar timestamp para exibição
  const formatTime = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit' });
  };

  // Formatar valores monetários
  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('pt-BR', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
    }).format(value);
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const point = payload[0].payload;
      const pnlColor = point.profit_loss >= 0 ? 'text-green-600' : 'text-red-600';

      return (
        <div className="bg-white p-3 border border-gray-200 rounded shadow-lg">
          <p className="text-sm font-semibold mb-1">{formatTime(point.timestamp)}</p>
          <p className="text-sm">
            Capital: <span className="font-bold">{formatCurrency(point.capital)}</span>
          </p>
          <p className={`text-sm ${pnlColor}`}>
            P&L: <span className="font-bold">{formatCurrency(point.profit_loss)}</span>
          </p>
          <p className="text-xs text-muted-foreground mt-1">Trade: {point.trade_id.slice(-8)}</p>
        </div>
      );
    }
    return null;
  };

  // Preparar dados para o gráfico
  const chartData = data.map(point => ({
    ...point,
    time: formatTime(point.timestamp),
  }));

  // Calcular min/max para ajustar escala do Y-axis
  const capitals = data.map(d => d.capital);
  const minCapital = Math.min(...capitals, initialCapital);
  const maxCapital = Math.max(...capitals, initialCapital);
  const padding = (maxCapital - minCapital) * 0.1; // 10% padding

  return (
    <Card>
      <CardHeader>
        <CardTitle>Equity Curve</CardTitle>
        <CardDescription>
          Evolução do capital ao longo de {data.length} trades
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis
              dataKey="time"
              tick={{ fontSize: 12 }}
              tickLine={false}
              axisLine={{ stroke: '#e0e0e0' }}
            />
            <YAxis
              domain={[minCapital - padding, maxCapital + padding]}
              tick={{ fontSize: 12 }}
              tickLine={false}
              axisLine={{ stroke: '#e0e0e0' }}
              tickFormatter={(value) => `$${(value / 1000).toFixed(1)}k`}
            />
            <Tooltip content={<CustomTooltip />} />

            {/* Linha de referência do capital inicial */}
            <ReferenceLine
              y={initialCapital}
              stroke="#94a3b8"
              strokeDasharray="5 5"
              label={{
                value: 'Inicial',
                position: 'insideTopRight',
                fill: '#64748b',
                fontSize: 12,
              }}
            />

            {/* Linha do equity */}
            <Line
              type="monotone"
              dataKey="capital"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={{ r: 3, fill: '#3b82f6' }}
              activeDot={{ r: 5, fill: '#2563eb' }}
            />
          </LineChart>
        </ResponsiveContainer>

        {/* Legenda */}
        <div className="flex items-center justify-center gap-6 mt-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
            <span>Capital Atual</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-0.5 bg-gray-400 border-dashed"></div>
            <span>Capital Inicial</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
