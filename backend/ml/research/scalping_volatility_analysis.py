"""
Análise de Volatilidade para Scalping - Fase 0.1

Este script analisa a viabilidade de scalping em diferentes ativos Deriv,
calculando métricas como ATR, tempo para atingir targets, e microestrutura de mercado.

Objetivo: Identificar ativos com volatilidade suficiente para scalping (0.5%-2% targets em 5-15 min)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
import sys
import asyncio

# Adicionar caminho do backend ao sys.path
backend_path = Path(__file__).parent.parent.parent
sys.path.append(str(backend_path))


class ScalpingVolatilityAnalyzer:
    """
    Analisa viabilidade de scalping para diferentes ativos Deriv
    """

    def __init__(self, symbol: str, timeframe: str = '1min'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.df = None
        self.results = {}

    async def collect_historical_data(self, days: int = 180):
        """
        Coleta dados históricos via WebSocket Deriv API

        Args:
            days: Número de dias de histórico (padrão: 180 = 6 meses)
        """
        import websockets
        import json

        print(f"\nColetando dados historicos de {self.symbol} ({days} dias)...")

        # Endpoint WebSocket
        app_id = 1089
        uri = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"

        try:
            async with websockets.connect(uri) as websocket:
                print("   [OK] Conectado ao Deriv WebSocket")

                # Calcular timestamps
                end_time = int(datetime.now().timestamp())
                start_time = int((datetime.now() - timedelta(days=days)).timestamp())

                # Coletar candles (máximo 5000 por request)
                all_candles = []
                chunk_size = 5000
                current_end = end_time

                while True:
                    # Request candles
                    # Granularity: 60 = 1min (M1), 300 = 5min (M5)
                    granularity_map = {'1min': 60, '5min': 300}
                    granularity = granularity_map.get(self.timeframe, 60)

                    request = {
                        "ticks_history": self.symbol,
                        "adjust_start_time": 1,
                        "count": chunk_size,
                        "end": current_end,
                        "granularity": granularity,
                        "style": "candles"
                    }

                    await websocket.send(json.dumps(request))
                    response_str = await websocket.recv()
                    response = json.loads(response_str)

                    if 'error' in response:
                        print(f"   [ERRO] Erro ao coletar candles: {response['error']['message']}")
                        break

                    candles = response.get('candles', [])
                    if not candles:
                        break

                    all_candles.extend(candles)
                    print(f"   [PROGRESS] Coletados {len(all_candles)} candles...")

                    # Verificar se chegou ao início
                    oldest_time = int(candles[0]['epoch'])
                    if oldest_time <= start_time:
                        break

                    # Próximo chunk
                    current_end = oldest_time - 1

                    # Evitar rate limiting
                    await asyncio.sleep(0.5)

                # Converter para DataFrame
                df = pd.DataFrame(all_candles)
                df['timestamp'] = pd.to_datetime(df['epoch'], unit='s')
                df['open'] = df['open'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['close'] = df['close'].astype(float)

                # Filtrar apenas dados dentro do período
                df = df[df['epoch'] >= start_time]
                df = df.sort_values('timestamp').reset_index(drop=True)

                self.df = df
                print(f"   [OK] {len(df)} candles coletados ({df['timestamp'].min()} a {df['timestamp'].max()})")

                # Salvar CSV
                output_dir = Path(__file__).parent / "data"
                output_dir.mkdir(exist_ok=True)
                csv_path = output_dir / f"{self.symbol}_{self.timeframe}_{days}days.csv"
                df.to_csv(csv_path, index=False)
                print(f"   [SAVE] Dados salvos em {csv_path}")

        except Exception as e:
            print(f"   [ERRO] Erro ao coletar dados: {e}")
            import traceback
            traceback.print_exc()
            raise

    def load_data_from_csv(self, csv_path: str):
        """Carrega dados de CSV existente"""
        self.df = pd.read_csv(csv_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        print(f"[OK] Dados carregados: {len(self.df)} candles de {csv_path}")

    def calculate_atr_metrics(self, period: int = 14) -> Dict:
        """
        Calcula ATR e métricas relacionadas

        Returns:
            Dict com métricas de ATR
        """
        print("\n[DATA] Calculando métricas de ATR...")

        # Calcular True Range
        self.df['high_low'] = self.df['high'] - self.df['low']
        self.df['high_close'] = abs(self.df['high'] - self.df['close'].shift(1))
        self.df['low_close'] = abs(self.df['low'] - self.df['close'].shift(1))
        self.df['true_range'] = self.df[['high_low', 'high_close', 'low_close']].max(axis=1)

        # ATR
        self.df['atr'] = self.df['true_range'].rolling(window=period).mean()
        self.df['atr_pct'] = (self.df['atr'] / self.df['close']) * 100

        metrics = {
            'atr_mean': self.df['atr'].mean(),
            'atr_pct_mean': self.df['atr_pct'].mean(),
            'atr_pct_median': self.df['atr_pct'].median(),
            'atr_pct_std': self.df['atr_pct'].std(),
            'atr_pct_min': self.df['atr_pct'].min(),
            'atr_pct_max': self.df['atr_pct'].max(),
            'atr_pct_p25': self.df['atr_pct'].quantile(0.25),
            'atr_pct_p75': self.df['atr_pct'].quantile(0.75),
        }

        print(f"   ATR médio: {metrics['atr_pct_mean']:.4f}%")
        print(f"   ATR mediano: {metrics['atr_pct_median']:.4f}%")

        return metrics

    def calculate_time_to_target(self, target_pct: float, stop_loss_pct: float, max_candles: int = 20) -> Dict:
        """
        Calcula tempo médio para atingir target antes de hit stop loss

        Args:
            target_pct: Target em % (ex: 1.0 para 1%)
            stop_loss_pct: Stop loss em % (ex: 0.5 para 0.5%)
            max_candles: Máximo de candles para simular (padrão: 20 = 20 min)

        Returns:
            Dict com métricas de tempo para target
        """
        results = []

        # Simular em amostra (pular candles para speed)
        sample_indices = range(100, len(self.df) - max_candles, 10)

        for i in sample_indices:
            entry_price = self.df.iloc[i]['close']
            target_price = entry_price * (1 + target_pct / 100)
            stop_price = entry_price * (1 - stop_loss_pct / 100)

            # Simular movimento do preço
            hit_target = False
            hit_stop = False
            time_to_exit = 0
            max_drawdown = 0

            for j in range(i + 1, min(i + max_candles, len(self.df))):
                high = self.df.iloc[j]['high']
                low = self.df.iloc[j]['low']

                # Verificar drawdown
                current_drawdown = ((low - entry_price) / entry_price) * 100
                max_drawdown = min(max_drawdown, current_drawdown)

                # Verificar hit (assumir que high ocorre antes de low se ambos atingidos)
                if high >= target_price:
                    hit_target = True
                    time_to_exit = j - i
                    break
                if low <= stop_price:
                    hit_stop = True
                    time_to_exit = j - i
                    break

            # Se não hit nada, considera timeout
            if not hit_target and not hit_stop:
                time_to_exit = max_candles

            results.append({
                'success': hit_target,
                'hit_stop': hit_stop,
                'timeout': not hit_target and not hit_stop,
                'time_minutes': time_to_exit,
                'drawdown': max_drawdown,
                'hour': self.df.iloc[i]['timestamp'].hour
            })

        df_results = pd.DataFrame(results)

        # Calcular métricas agregadas
        success_trades = df_results[df_results['success']]

        metrics = {
            'total_simulations': len(results),
            'success_rate': (df_results['success'].mean() * 100),
            'stop_rate': (df_results['hit_stop'].mean() * 100),
            'timeout_rate': (df_results['timeout'].mean() * 100),
            'avg_time_minutes_all': df_results['time_minutes'].mean(),
            'avg_time_minutes_success': success_trades['time_minutes'].mean() if len(success_trades) > 0 else 0,
            'median_time_minutes_success': success_trades['time_minutes'].median() if len(success_trades) > 0 else 0,
            'avg_drawdown': df_results['drawdown'].mean(),
            'worst_drawdown': df_results['drawdown'].min(),
        }

        # Melhor horário
        hourly_success = df_results.groupby('hour')['success'].mean()
        if len(hourly_success) > 0:
            best_hour = hourly_success.idxmax()
            metrics['best_hour'] = int(best_hour)
            metrics['best_hour_success_rate'] = float(hourly_success.max() * 100)
            metrics['worst_hour'] = int(hourly_success.idxmin())
            metrics['worst_hour_success_rate'] = float(hourly_success.min() * 100)
        else:
            metrics['best_hour'] = 0
            metrics['best_hour_success_rate'] = 0
            metrics['worst_hour'] = 0
            metrics['worst_hour_success_rate'] = 0

        return metrics

    def calculate_tick_metrics(self) -> Dict:
        """
        Calcula métricas de microestrutura de mercado

        Returns:
            Dict com métricas de tick/spread
        """
        # Volatilidade intrabar (high-low range)
        self.df['intrabar_range'] = self.df['high'] - self.df['low']
        self.df['intrabar_volatility_pct'] = (self.df['intrabar_range'] / self.df['close']) * 100

        # Gaps entre candles
        self.df['gap'] = abs(self.df['open'] - self.df['close'].shift(1))
        self.df['gap_pct'] = (self.df['gap'] / self.df['close'].shift(1)) * 100

        metrics = {
            'avg_intrabar_volatility_pct': self.df['intrabar_volatility_pct'].mean(),
            'median_intrabar_volatility_pct': self.df['intrabar_volatility_pct'].median(),
            'max_intrabar_volatility_pct': self.df['intrabar_volatility_pct'].max(),
            'avg_gap_pct': self.df['gap_pct'].mean(),
            'max_gap_pct': self.df['gap_pct'].max(),
        }

        return metrics

    def analyze_hourly_patterns(self) -> pd.DataFrame:
        """
        Analisa padrões de volatilidade por hora do dia

        Returns:
            DataFrame com métricas por hora
        """
        self.df['hour'] = self.df['timestamp'].dt.hour

        hourly = self.df.groupby('hour').agg({
            'atr_pct': 'mean',
            'intrabar_volatility_pct': 'mean',
            'true_range': 'mean',
            'close': 'count'  # Volume de candles
        }).round(4)

        hourly.columns = ['atr_pct_mean', 'intrabar_vol_mean', 'true_range_mean', 'candle_count']

        return hourly

    def evaluate_scalping_viability(self) -> Dict:
        """
        Avalia se o ativo é viável para scalping baseado em critérios objetivos

        Returns:
            Dict com veredicto e razões
        """
        print("\n[TARGET] Avaliando viabilidade de scalping...")

        # Critérios de aprovação
        criteria = {
            'atr_pct_mean': {'min': 0.05, 'ideal': 0.10, 'value': 0},
            'time_to_1pct_target': {'max': 10, 'ideal': 5, 'value': 0},
            'success_rate_1pct': {'min': 60, 'ideal': 70, 'value': 0},
        }

        # Calcular métricas
        atr_metrics = self.calculate_atr_metrics()
        criteria['atr_pct_mean']['value'] = atr_metrics['atr_pct_mean']

        target_1pct = self.calculate_time_to_target(target_pct=1.0, stop_loss_pct=0.5, max_candles=15)
        criteria['time_to_1pct_target']['value'] = target_1pct['avg_time_minutes_success']
        criteria['success_rate_1pct']['value'] = target_1pct['success_rate']

        # Avaliar cada critério
        approved = True
        reasons = []

        # ATR
        if criteria['atr_pct_mean']['value'] < criteria['atr_pct_mean']['min']:
            approved = False
            reasons.append(
                f"[ERRO] ATR muito baixo ({criteria['atr_pct_mean']['value']:.4f}% < {criteria['atr_pct_mean']['min']:.2f}%)"
            )
        elif criteria['atr_pct_mean']['value'] >= criteria['atr_pct_mean']['ideal']:
            reasons.append(
                f"[OK] ATR excelente ({criteria['atr_pct_mean']['value']:.4f}% >= {criteria['atr_pct_mean']['ideal']:.2f}%)"
            )
        else:
            reasons.append(
                f"[AVISO] ATR aceitável ({criteria['atr_pct_mean']['value']:.4f}%)"
            )

        # Tempo para target
        if criteria['time_to_1pct_target']['value'] > criteria['time_to_1pct_target']['max']:
            approved = False
            reasons.append(
                f"[ERRO] Tempo para 1% TP muito longo ({criteria['time_to_1pct_target']['value']:.1f} min > {criteria['time_to_1pct_target']['max']} min)"
            )
        elif criteria['time_to_1pct_target']['value'] <= criteria['time_to_1pct_target']['ideal']:
            reasons.append(
                f"[OK] Tempo para TP excelente ({criteria['time_to_1pct_target']['value']:.1f} min <= {criteria['time_to_1pct_target']['ideal']} min)"
            )
        else:
            reasons.append(
                f"[AVISO] Tempo para TP aceitável ({criteria['time_to_1pct_target']['value']:.1f} min)"
            )

        # Success rate
        if criteria['success_rate_1pct']['value'] < criteria['success_rate_1pct']['min']:
            approved = False
            reasons.append(
                f"[ERRO] Taxa de sucesso baixa ({criteria['success_rate_1pct']['value']:.1f}% < {criteria['success_rate_1pct']['min']}%)"
            )
        elif criteria['success_rate_1pct']['value'] >= criteria['success_rate_1pct']['ideal']:
            reasons.append(
                f"[OK] Taxa de sucesso excelente ({criteria['success_rate_1pct']['value']:.1f}% >= {criteria['success_rate_1pct']['ideal']}%)"
            )
        else:
            reasons.append(
                f"[AVISO] Taxa de sucesso aceitável ({criteria['success_rate_1pct']['value']:.1f}%)"
            )

        return {
            'approved': approved,
            'criteria': criteria,
            'reasons': reasons
        }

    def generate_report(self, output_path: str):
        """
        Gera relatório completo de viabilidade de scalping

        Args:
            output_path: Caminho para salvar o relatório .md
        """
        print(f"\n{'='*70}")
        print(f"ANÁLISE DE VIABILIDADE DE SCALPING: {self.symbol}")
        print(f"{'='*70}\n")

        # 1. ATR Metrics
        print("## 1. MÉTRICAS DE VOLATILIDADE (ATR)")
        atr_metrics = self.calculate_atr_metrics()
        for key, value in atr_metrics.items():
            print(f"   - {key}: {value:.4f}")

        # 2. Time to Target Analysis
        print("\n## 2. ANÁLISE DE TEMPO PARA TARGETS")
        targets = [
            {'target': 0.5, 'sl': 0.25, 'name': 'Micro'},
            {'target': 1.0, 'sl': 0.5, 'name': 'Padrão'},
            {'target': 1.5, 'sl': 0.75, 'name': 'Agressivo'},
            {'target': 2.0, 'sl': 1.0, 'name': 'Swing-Scalp'},
        ]

        target_results = []
        for t in targets:
            result = self.calculate_time_to_target(t['target'], t['sl'], max_candles=20)
            target_results.append({**t, **result})

            print(f"\n   {t['name']}: Target +{t['target']}% | SL -{t['sl']}%")
            print(f"   - Taxa de sucesso: {result['success_rate']:.1f}%")
            print(f"   - Taxa de stop: {result['stop_rate']:.1f}%")
            print(f"   - Taxa de timeout: {result['timeout_rate']:.1f}%")
            print(f"   - Tempo médio (sucesso): {result['avg_time_minutes_success']:.1f} min")
            print(f"   - Drawdown médio: {result['avg_drawdown']:.2f}%")
            print(f"   - Melhor horário: {result['best_hour']}h ({result['best_hour_success_rate']:.1f}% win rate)")

        # 3. Tick Metrics
        print("\n## 3. MICROESTRUTURA DE MERCADO")
        tick_metrics = self.calculate_tick_metrics()
        for key, value in tick_metrics.items():
            print(f"   - {key}: {value:.4f}%")

        # 4. Hourly Patterns
        print("\n## 4. PADRÕES POR HORA DO DIA")
        hourly = self.analyze_hourly_patterns()
        best_hour = hourly['atr_pct_mean'].idxmax()
        worst_hour = hourly['atr_pct_mean'].idxmin()
        print(f"   Hora com maior volatilidade: {best_hour}h (ATR: {hourly.loc[best_hour, 'atr_pct_mean']:.4f}%)")
        print(f"   Hora com menor volatilidade: {worst_hour}h (ATR: {hourly.loc[worst_hour, 'atr_pct_mean']:.4f}%)")

        # 5. Veredicto Final
        print("\n## 5. VEREDICTO FINAL")
        viability = self.evaluate_scalping_viability()

        for reason in viability['reasons']:
            print(f"   {reason}")

        if viability['approved']:
            print(f"\n   [SUCCESS] {self.symbol} é VIÁVEL para scalping!")
            print(f"   [INFO] Recomendação: SL 0.5%, TP 1.0%, Timeout 15 min, Melhor horário: {target_results[1]['best_hour']}h")
        else:
            print(f"\n   [ERRO] {self.symbol} NÃO é viável para scalping")
            print(f"   [DICA] Sugestão: Considerar swing trading ou buscar outro ativo")

        # Salvar relatório em Markdown
        self._save_markdown_report(output_path, atr_metrics, target_results, tick_metrics, hourly, viability)

        return viability['approved']

    def _save_markdown_report(self, output_path: str, atr_metrics: Dict, target_results: List[Dict],
                               tick_metrics: Dict, hourly: pd.DataFrame, viability: Dict):
        """Salva relatório em formato Markdown"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# ANÁLISE DE VIABILIDADE DE SCALPING: {self.symbol}\n\n")
            f.write(f"**Data da análise**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Período analisado**: {self.df['timestamp'].min()} a {self.df['timestamp'].max()}\n\n")
            f.write(f"**Total de candles**: {len(self.df):,}\n\n")

            # Veredicto no topo
            if viability['approved']:
                f.write("## [OK] VEREDICTO: VIÁVEL PARA SCALPING\n\n")
            else:
                f.write("## [ERRO] VEREDICTO: NÃO VIÁVEL PARA SCALPING\n\n")

            f.write("### Critérios de Avaliação\n\n")
            for reason in viability['reasons']:
                f.write(f"- {reason}\n")
            f.write("\n---\n\n")

            # ATR
            f.write("## 1. Métricas de Volatilidade (ATR)\n\n")
            f.write("| Métrica | Valor |\n")
            f.write("|---------|-------|\n")
            for key, value in atr_metrics.items():
                f.write(f"| {key} | {value:.4f}% |\n")
            f.write("\n")

            # Targets
            f.write("## 2. Análise de Tempo para Targets\n\n")
            f.write("| Cenário | Target | SL | Success Rate | Tempo Médio (Success) | Drawdown Médio | Melhor Hora |\n")
            f.write("|---------|--------|----|--------------|-----------------------|----------------|-------------|\n")
            for t in target_results:
                f.write(f"| {t['name']} | +{t['target']}% | -{t['sl']}% | {t['success_rate']:.1f}% | "
                       f"{t['avg_time_minutes_success']:.1f} min | {t['avg_drawdown']:.2f}% | "
                       f"{t['best_hour']}h ({t['best_hour_success_rate']:.1f}%) |\n")
            f.write("\n")

            # Microestrutura
            f.write("## 3. Microestrutura de Mercado\n\n")
            f.write("| Métrica | Valor |\n")
            f.write("|---------|-------|\n")
            for key, value in tick_metrics.items():
                f.write(f"| {key} | {value:.4f}% |\n")
            f.write("\n")

            # Padrões horários
            f.write("## 4. Padrões por Hora do Dia\n\n")
            f.write("| Hora | ATR Médio (%) | Volatilidade Intrabar (%) | Candles |\n")
            f.write("|------|---------------|---------------------------|----------|\n")
            for hour, row in hourly.iterrows():
                f.write(f"| {hour:02d}h | {row['atr_pct_mean']:.4f} | {row['intrabar_vol_mean']:.4f} | {int(row['candle_count'])} |\n")
            f.write("\n")

            # Recomendação
            f.write("## 5. Recomendação Final\n\n")
            if viability['approved']:
                best_target = target_results[1]  # Padrão: 1% TP / 0.5% SL
                f.write(f"### [OK] {self.symbol} é VIÁVEL para scalping\n\n")
                f.write("**Configuração Recomendada:**\n\n")
                f.write(f"- **Stop Loss**: {best_target['sl']}%\n")
                f.write(f"- **Take Profit**: {best_target['target']}%\n")
                f.write(f"- **Timeout**: 15 minutos\n")
                f.write(f"- **Melhor horário**: {best_target['best_hour']}h - {(best_target['best_hour']+4)%24}h\n")
                f.write(f"- **Win rate esperado**: {best_target['success_rate']:.1f}%\n")
                f.write(f"- **Tempo médio por trade**: {best_target['avg_time_minutes_success']:.1f} min\n")
            else:
                f.write(f"### [ERRO] {self.symbol} NÃO é viável para scalping\n\n")
                f.write("**Razões:**\n\n")
                for reason in viability['reasons']:
                    if '[ERRO]' in reason:
                        f.write(f"- {reason}\n")
                f.write("\n**Sugestão**: Considerar swing trading ou buscar outro ativo.\n")

        print(f"\n[SAVE] Relatório salvo em {output_path}")


async def analyze_multiple_symbols(timeframe='5min'):
    """
    Analisa múltiplos símbolos e gera relatório comparativo

    Args:
        timeframe: '1min' ou '5min' (default: '5min' para Fase 0.2)
    """
    symbols = [
        '1HZ75V',   # Volatility 75
        '1HZ100V',  # Volatility 100
    ]

    results = []
    output_dir = Path(__file__).parent / "reports"
    output_dir.mkdir(exist_ok=True)

    for symbol in symbols:
        print(f"\n{'='*70}")
        print(f"ANALISANDO: {symbol}")
        print(f"{'='*70}")

        analyzer = ScalpingVolatilityAnalyzer(symbol=symbol, timeframe=timeframe)

        # Verificar se já existe CSV
        data_dir = Path(__file__).parent / "data"
        csv_path = data_dir / f"{symbol}_{timeframe}_180days.csv"

        if csv_path.exists():
            print(f"[OK] Dados encontrados em {csv_path}")
            analyzer.load_data_from_csv(str(csv_path))
        else:
            print(f"[DOWNLOAD] Coletando novos dados...")
            try:
                await analyzer.collect_historical_data(days=180)
            except Exception as e:
                print(f"[ERRO] Erro ao coletar dados de {symbol}: {e}")
                continue

        # Gerar relatório individual
        report_path = output_dir / f"scalping_viability_{symbol}.md"
        is_viable = analyzer.generate_report(str(report_path))

        # Armazenar resultados
        results.append({
            'symbol': symbol,
            'viable': is_viable,
            'atr_pct_mean': analyzer.results.get('atr_pct_mean', 0) if hasattr(analyzer, 'results') else analyzer.calculate_atr_metrics()['atr_pct_mean']
        })

    # Gerar relatório comparativo
    _generate_comparative_report(results, str(output_dir / "scalping_assets_comparison.md"))


def _generate_comparative_report(results: List[Dict], output_path: str):
    """Gera relatório comparativo entre ativos"""

    print(f"\n{'='*70}")
    print("RELATÓRIO COMPARATIVO - VIABILIDADE DE SCALPING")
    print(f"{'='*70}\n")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# RELATÓRIO COMPARATIVO: VIABILIDADE DE SCALPING NOS ATIVOS DERIV\n\n")
        f.write(f"**Data**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Resumo de Viabilidade\n\n")
        f.write("| Ativo | Status | ATR Médio (%) |\n")
        f.write("|-------|--------|---------------|\n")

        viable_count = 0
        for r in results:
            status = "[OK] VIÁVEL" if r['viable'] else "[ERRO] NÃO VIÁVEL"
            f.write(f"| {r['symbol']} | {status} | {r['atr_pct_mean']:.4f} |\n")
            if r['viable']:
                viable_count += 1

        f.write(f"\n**Total de ativos viáveis**: {viable_count}/{len(results)}\n\n")

        if viable_count >= 2:
            f.write("## [OK] CONCLUSÃO: SCALPING É VIÁVEL\n\n")
            f.write(f"Foram identificados **{viable_count} ativos viáveis** para scalping ML.\n\n")
            f.write("**Próximo passo**: Avançar para Fase 0.2 (Análise de Features para Scalping)\n")
        elif viable_count == 1:
            f.write("## [AVISO] CONCLUSÃO: SCALPING É POSSÍVEL MAS LIMITADO\n\n")
            f.write("Apenas **1 ativo viável** foi identificado. Scalping é possível mas com opções limitadas.\n\n")
            f.write("**Recomendação**: Prosseguir com cautela para Fase 0.2 ou considerar swing trading.\n")
        else:
            f.write("## [ERRO] CONCLUSÃO: SCALPING NÃO É VIÁVEL\n\n")
            f.write("**NENHUM ativo** atingiu os critérios mínimos para scalping.\n\n")
            f.write("**Recomendação**: DESISTIR de scalping e FOCAR em swing trading (R_100 já validado).\n")

    print(f"[SAVE] Relatório comparativo salvo em {output_path}")


if __name__ == "__main__":
    # Executar análise completa
    asyncio.run(analyze_multiple_symbols())
