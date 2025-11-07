/**
 * OrderExecutor Component
 * Componente para executar ordens na Deriv API
 * Objetivo 1: Interface para execução de ordens
 */

import { useState } from 'react';
import { executeOrder, OrderResult } from '@/services/orderService';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import {
  TrendingUp,
  TrendingDown,
  Loader2,
  CheckCircle2,
  AlertCircle,
  ExternalLink,
  DollarSign,
  Clock
} from 'lucide-react';

export const OrderExecutor = () => {
  const [token, setToken] = useState('');
  const [contractType, setContractType] = useState<'CALL' | 'PUT'>('CALL');
  const [symbol, setSymbol] = useState('R_75');
  const [amount, setAmount] = useState('1.00');
  const [duration, setDuration] = useState('5');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<OrderResult | null>(null);

  // Carregar token do localStorage
  useState(() => {
    const savedToken = localStorage.getItem('deriv_api_key') ||
                      localStorage.getItem('deriv_primary_token');
    if (savedToken) {
      setToken(savedToken);
    }
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      const orderResult = await executeOrder({
        token,
        contractType,
        symbol,
        amount: parseFloat(amount),
        duration: parseInt(duration),
        durationUnit: 'm'
      });

      setResult(orderResult);

      // Se sucesso, salvar token
      if (orderResult.success) {
        localStorage.setItem('deriv_api_key', token);
      }
    } catch (error) {
      setResult({
        success: false,
        error: error instanceof Error ? error.message : 'Erro desconhecido',
      });
    } finally {
      setLoading(false);
    }
  };

  const isFormValid = () => {
    if (!token || token.length < 10) return false;
    const amountNum = parseFloat(amount);
    if (isNaN(amountNum) || amountNum < 0.35 || amountNum > 100) return false;
    const durationNum = parseInt(duration);
    if (isNaN(durationNum) || durationNum < 1 || durationNum > 60) return false;
    return true;
  };

  const profitPotential = result?.payout && result?.buy_price
    ? (result.payout - result.buy_price).toFixed(2)
    : null;

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="text-center space-y-2">
        <h1 className="text-3xl font-bold">Execução de Ordens</h1>
        <p className="text-muted-foreground">
          Execute ordens na Deriv API - Objetivo 1 ✓
        </p>
      </div>

      {/* Form Card */}
      <Card>
        <CardHeader>
          <CardTitle>Nova Ordem</CardTitle>
          <CardDescription>
            Configure os parâmetros da ordem e execute
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Token */}
            <div className="space-y-2">
              <Label htmlFor="token">Token API da Deriv</Label>
              <Input
                id="token"
                type="password"
                value={token}
                onChange={(e) => setToken(e.target.value)}
                placeholder="Cole seu token da conta Demo aqui"
                required
                minLength={10}
              />
              <p className="text-xs text-muted-foreground">
                Gere um token em{' '}
                <a
                  href="https://app.deriv.com/account/api-token"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:underline"
                >
                  app.deriv.com/account/api-token
                </a>
              </p>
            </div>

            {/* Contract Type and Symbol */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="contractType">Tipo de Contrato</Label>
                <Select value={contractType} onValueChange={(v) => setContractType(v as 'CALL' | 'PUT')}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="CALL">
                      <div className="flex items-center">
                        <TrendingUp className="h-4 w-4 mr-2 text-green-600" />
                        CALL (Rise)
                      </div>
                    </SelectItem>
                    <SelectItem value="PUT">
                      <div className="flex items-center">
                        <TrendingDown className="h-4 w-4 mr-2 text-red-600" />
                        PUT (Fall)
                      </div>
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="symbol">Símbolo</Label>
                <Select value={symbol} onValueChange={setSymbol}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="R_75">Volatility 75 Index</SelectItem>
                    <SelectItem value="R_100">Volatility 100 Index</SelectItem>
                    <SelectItem value="R_50">Volatility 50 Index</SelectItem>
                    <SelectItem value="R_25">Volatility 25 Index</SelectItem>
                    <SelectItem value="R_10">Volatility 10 Index</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Amount and Duration */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="amount">
                  <DollarSign className="h-4 w-4 inline mr-1" />
                  Valor (USD)
                </Label>
                <Input
                  id="amount"
                  type="number"
                  step="0.01"
                  min="0.35"
                  max="100"
                  value={amount}
                  onChange={(e) => setAmount(e.target.value)}
                  required
                />
                <p className="text-xs text-muted-foreground">
                  Mínimo: $0.35 | Máximo: $100.00
                </p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="duration">
                  <Clock className="h-4 w-4 inline mr-1" />
                  Duração (minutos)
                </Label>
                <Input
                  id="duration"
                  type="number"
                  min="1"
                  max="60"
                  value={duration}
                  onChange={(e) => setDuration(e.target.value)}
                  required
                />
                <p className="text-xs text-muted-foreground">
                  Mínimo: 1 min | Máximo: 60 min
                </p>
              </div>
            </div>

            {/* Submit Button */}
            <Button
              type="submit"
              className="w-full"
              size="lg"
              disabled={loading || !isFormValid()}
            >
              {loading ? (
                <>
                  <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                  Executando Ordem...
                </>
              ) : (
                <>
                  <TrendingUp className="h-5 w-5 mr-2" />
                  Executar Ordem
                </>
              )}
            </Button>
          </form>
        </CardContent>
      </Card>

      {/* Result Card */}
      {result && (
        <Card className={result.success ? 'border-green-200 bg-green-50' : 'border-red-200 bg-red-50'}>
          <CardContent className="pt-6">
            {result.success ? (
              <div className="space-y-4">
                <div className="flex items-center space-x-2">
                  <CheckCircle2 className="h-6 w-6 text-green-600" />
                  <h3 className="text-lg font-bold text-green-800">Ordem Executada com Sucesso!</h3>
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground">Contract ID</p>
                    <p className="font-mono font-bold text-lg">{result.contract_id}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Preço Pago</p>
                    <p className="font-bold text-lg">${result.buy_price?.toFixed(2)}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Payout Potencial</p>
                    <p className="font-bold text-lg">${result.payout?.toFixed(2)}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground">Lucro se Ganhar</p>
                    <p className="font-bold text-lg text-green-600">
                      +${profitPotential}
                    </p>
                  </div>
                </div>

                {result.longcode && (
                  <div className="pt-4 border-t">
                    <p className="text-xs text-muted-foreground mb-2">Descrição do Contrato:</p>
                    <p className="text-sm">{result.longcode}</p>
                  </div>
                )}

                <Button
                  variant="outline"
                  className="w-full"
                  onClick={() => window.open(`https://app.deriv.com/contract/${result.contract_id}`, '_blank')}
                >
                  <ExternalLink className="h-4 w-4 mr-2" />
                  Ver Contrato na Plataforma Deriv
                </Button>

                <div className="text-center">
                  <Badge variant="secondary" className="text-xs">
                    Objetivo 1 Concluído: Execução de Ordem ✓
                  </Badge>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex items-center space-x-2">
                  <AlertCircle className="h-6 w-6 text-red-600" />
                  <h3 className="text-lg font-bold text-red-800">Erro ao Executar Ordem</h3>
                </div>
                <Alert variant="destructive">
                  <AlertDescription>
                    {result.error}
                  </AlertDescription>
                </Alert>
                <div className="text-sm text-muted-foreground">
                  <p className="font-semibold mb-2">Possíveis causas:</p>
                  <ul className="list-disc list-inside space-y-1 ml-2">
                    <li>Token inválido ou expirado</li>
                    <li>Conta sem saldo suficiente</li>
                    <li>Backend não está rodando (porta 8000)</li>
                    <li>Símbolo ou parâmetros inválidos</li>
                  </ul>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
};
