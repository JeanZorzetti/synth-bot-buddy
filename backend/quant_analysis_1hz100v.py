#!/usr/bin/env python3
"""
ANÁLISE QUANTITATIVA - VOLATILITY 100 (1s) INDEX

Objetivo: Otimizar estratégia "Color Streak Martingale" para 1HZ100V
Autor: Sistema Abutre - Quant Research
"""
import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from collections import Counter
import websockets
from dotenv import load_dotenv

load_dotenv()

DERIV_APP_ID = os.getenv("DERIV_APP_ID", "99188")
DERIV_API_TOKEN = os.getenv("DERIV_API_TOKEN")
DERIV_WS_URL = f"wss://ws.derivws.com/websockets/v3?app_id={DERIV_APP_ID}"


class StreakAnalyzer:
    """Analisa distribuição de streaks para 1HZ100V"""

    def __init__(self, candles: pd.DataFrame):
        self.candles = candles
        self.streaks_red = []
        self.streaks_green = []

    def calculate_streaks(self) -> Dict:
        """
        Calcula todas as sequências consecutivas de velas da mesma cor

        Returns:
            Dict com distribuição de streaks
        """
        current_color = None
        current_streak = 0

        for _, candle in self.candles.iterrows():
            # Determinar cor (Red = close < open, Green = close >= open)
            color = 'red' if candle['close'] < candle['open'] else 'green'

            if color == current_color:
                current_streak += 1
            else:
                # Guardar streak anterior
                if current_color is not None:
                    if current_color == 'red':
                        self.streaks_red.append(current_streak)
                    else:
                        self.streaks_green.append(current_streak)

                # Iniciar nova streak
                current_color = color
                current_streak = 1

        # Última streak
        if current_color == 'red':
            self.streaks_red.append(current_streak)
        else:
            self.streaks_green.append(current_streak)

        return self._analyze_distribution()

    def _analyze_distribution(self) -> Dict:
        """Análise estatística das streaks"""
        all_streaks = self.streaks_red + self.streaks_green

        return {
            'total_streaks': len(all_streaks),
            'red_streaks': {
                'count': len(self.streaks_red),
                'distribution': Counter(self.streaks_red),
                'max': max(self.streaks_red) if self.streaks_red else 0,
                'mean': np.mean(self.streaks_red) if self.streaks_red else 0,
                'median': np.median(self.streaks_red) if self.streaks_red else 0,
                'p95': np.percentile(self.streaks_red, 95) if self.streaks_red else 0,
                'p99': np.percentile(self.streaks_red, 99) if self.streaks_red else 0
            },
            'green_streaks': {
                'count': len(self.streaks_green),
                'distribution': Counter(self.streaks_green),
                'max': max(self.streaks_green) if self.streaks_green else 0,
                'mean': np.mean(self.streaks_green) if self.streaks_green else 0,
                'median': np.median(self.streaks_green) if self.streaks_green else 0,
                'p95': np.percentile(self.streaks_green, 95) if self.streaks_green else 0,
                'p99': np.percentile(self.streaks_green, 99) if self.streaks_green else 0
            }
        }

    def recommend_delay(self, risk_tolerance: str = 'moderate') -> int:
        """
        Recomenda Delay ideal baseado em análise probabilística

        Args:
            risk_tolerance: 'conservative' (p99), 'moderate' (p95), 'aggressive' (median)

        Returns:
            Delay recomendado
        """
        red_p99 = np.percentile(self.streaks_red, 99) if self.streaks_red else 0
        green_p99 = np.percentile(self.streaks_green, 99) if self.streaks_green else 0

        red_p95 = np.percentile(self.streaks_red, 95) if self.streaks_red else 0
        green_p95 = np.percentile(self.streaks_green, 95) if self.streaks_green else 0

        red_median = np.median(self.streaks_red) if self.streaks_red else 0
        green_median = np.median(self.streaks_green) if self.streaks_green else 0

        if risk_tolerance == 'conservative':
            # Delay = 50% do P99 (máxima segurança)
            return int(min(red_p99, green_p99) * 0.5)
        elif risk_tolerance == 'moderate':
            # Delay = 60% do P95 (equilíbrio)
            return int(min(red_p95, green_p95) * 0.6)
        else:  # aggressive
            # Delay = 80% da mediana (mais sinais)
            return int(min(red_median, green_median) * 0.8)


class MartingaleBacktester:
    """Backtest da estratégia Color Streak Martingale"""

    def __init__(
        self,
        candles: pd.DataFrame,
        initial_balance: float = 10000.0,
        initial_stake: float = 0.35,
        martingale_multiplier: float = 2.0,
        delay: int = 8,
        max_martingale_levels: int = 10,
        stop_loss: float = 100.0
    ):
        self.candles = candles.copy()
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.initial_stake = initial_stake
        self.current_stake = initial_stake
        self.martingale_multiplier = martingale_multiplier
        self.delay = delay
        self.max_levels = max_martingale_levels
        self.stop_loss = stop_loss

        self.trades = []
        self.current_level = 1
        self.consecutive_reds = 0
        self.consecutive_greens = 0

    def run(self) -> Dict:
        """
        Executa backtest completo

        Returns:
            Relatório de performance
        """
        for idx in range(1, len(self.candles)):
            prev_candle = self.candles.iloc[idx - 1]

            # Atualizar contadores
            prev_color = 'red' if prev_candle['close'] < prev_candle['open'] else 'green'

            if prev_color == 'red':
                self.consecutive_reds += 1
                self.consecutive_greens = 0
            else:
                self.consecutive_greens += 1
                self.consecutive_reds = 0

            # Verificar sinal de entrada
            signal = None
            if self.consecutive_reds >= self.delay:
                signal = 'CALL'  # Apostar que próxima será verde
            elif self.consecutive_greens >= self.delay:
                signal = 'PUT'   # Apostar que próxima será vermelha

            if signal:
                # Executar trade
                entry_candle = self.candles.iloc[idx]
                result = self._execute_trade(signal, entry_candle)

                # Stop loss check
                if self.balance <= (self.initial_balance - self.stop_loss):
                    print(f"⚠️ STOP LOSS atingido! Balance: ${self.balance:.2f}")
                    break

        return self._generate_report()

    def _execute_trade(self, signal: str, entry_candle: pd.Series) -> str:
        """Executa um trade e retorna resultado"""
        entry_price = entry_candle['close']
        exit_price = entry_candle['close']  # Simplified: assume exit at same candle

        # Determinar resultado (simplified win rate ~50%)
        actual_color = 'green' if exit_price >= entry_price else 'red'

        if signal == 'CALL':
            is_win = (actual_color == 'green')
        else:  # PUT
            is_win = (actual_color == 'red')

        # Payout 95% (typical for Deriv)
        payout_ratio = 0.95

        if is_win:
            profit = self.current_stake * payout_ratio
            self.balance += profit

            # Reset
            self.current_stake = self.initial_stake
            self.current_level = 1
            self.consecutive_reds = 0
            self.consecutive_greens = 0

            result = 'WIN'
        else:
            profit = -self.current_stake
            self.balance += profit

            # Martingale
            self.current_level += 1
            if self.current_level <= self.max_levels:
                self.current_stake *= self.martingale_multiplier
            else:
                # Max level reached - reset
                print(f"⚠️ Nível Martingale MÁXIMO atingido ({self.max_levels})!")
                self.current_stake = self.initial_stake
                self.current_level = 1

            result = 'LOSS'

        # Registrar trade
        self.trades.append({
            'timestamp': entry_candle['epoch'],
            'signal': signal,
            'stake': self.current_stake / (self.martingale_multiplier if result == 'WIN' else 1),
            'level': self.current_level - 1 if result == 'WIN' else self.current_level,
            'result': result,
            'profit': profit,
            'balance': self.balance
        })

        return result

    def _generate_report(self) -> Dict:
        """Gera relatório de performance"""
        df_trades = pd.DataFrame(self.trades)

        if len(df_trades) == 0:
            return {'error': 'Nenhum trade executado'}

        wins = df_trades[df_trades['result'] == 'WIN']
        losses = df_trades[df_trades['result'] == 'LOSS']

        total_profit = df_trades['profit'].sum()
        win_rate = len(wins) / len(df_trades) * 100

        # Drawdown
        df_trades['cumulative'] = df_trades['profit'].cumsum()
        df_trades['peak'] = df_trades['cumulative'].cummax()
        df_trades['drawdown'] = df_trades['peak'] - df_trades['cumulative']
        max_drawdown = df_trades['drawdown'].max()

        # Profit factor
        gross_profit = wins['profit'].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses['profit'].sum()) if len(losses) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        return {
            'total_trades': len(df_trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate_pct': win_rate,
            'total_profit': total_profit,
            'roi_pct': (total_profit / self.initial_balance) * 100,
            'final_balance': self.balance,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'max_level_used': df_trades['level'].max(),
            'avg_win': wins['profit'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['profit'].mean() if len(losses) > 0 else 0
        }


async def fetch_candles_1hz100v(limit: int = 5000) -> pd.DataFrame:
    """
    Busca candles históricos do Volatility 100 (1s) Index

    Args:
        limit: Número de candles (max: 5000)

    Returns:
        DataFrame com OHLC
    """
    print(f"📊 Buscando {limit} candles do 1HZ100V...")

    async with websockets.connect(DERIV_WS_URL) as ws:
        # Authorize
        if DERIV_API_TOKEN:
            await ws.send(json.dumps({"authorize": DERIV_API_TOKEN}))
            auth_response = json.loads(await ws.recv())
            if "error" in auth_response:
                raise Exception(f"Auth error: {auth_response['error']}")

        # Request candles
        end_time = int(datetime.now().timestamp())

        # 1HZ100V usa granularidade de 60s (1 minuto)
        await ws.send(json.dumps({
            "ticks_history": "1HZ100V",
            "adjust_start_time": 1,
            "count": limit,
            "end": end_time,
            "granularity": 60,  # 1 minute candles
            "style": "candles"
        }))

        response = json.loads(await ws.recv())

        if "error" in response:
            raise Exception(f"API error: {response['error']}")

        candles = response.get("candles", [])

        # Converter para DataFrame
        df = pd.DataFrame(candles)
        df['epoch'] = pd.to_datetime(df['epoch'], unit='s')
        df = df.rename(columns={'epoch': 'timestamp'})

        print(f"✅ {len(df)} candles baixados!")
        return df


async def main():
    """Execução principal da análise"""
    print("=" * 60)
    print("ANÁLISE QUANTITATIVA - VOLATILITY 100 (1s) INDEX")
    print("Estratégia: Color Streak Martingale")
    print("=" * 60)
    print()

    # 1. BUSCAR DADOS
    candles = await fetch_candles_1hz100v(limit=5000)
    print(f"📅 Período: {candles['timestamp'].min()} até {candles['timestamp'].max()}")
    print()

    # 2. ANÁLISE DE STREAKS
    print("🔍 ANÁLISE DE DISTRIBUIÇÃO DE STREAKS")
    print("-" * 60)
    analyzer = StreakAnalyzer(candles)
    streak_stats = analyzer.calculate_streaks()

    print(f"Total de sequências: {streak_stats['total_streaks']}")
    print()
    print("VERMELHAS:")
    print(f"  - Máximo: {streak_stats['red_streaks']['max']}")
    print(f"  - Média: {streak_stats['red_streaks']['mean']:.2f}")
    print(f"  - Mediana: {streak_stats['red_streaks']['median']:.2f}")
    print(f"  - P95: {streak_stats['red_streaks']['p95']:.2f}")
    print(f"  - P99: {streak_stats['red_streaks']['p99']:.2f}")
    print()
    print("VERDES:")
    print(f"  - Máximo: {streak_stats['green_streaks']['max']}")
    print(f"  - Média: {streak_stats['green_streaks']['mean']:.2f}")
    print(f"  - Mediana: {streak_stats['green_streaks']['median']:.2f}")
    print(f"  - P95: {streak_stats['green_streaks']['p95']:.2f}")
    print(f"  - P99: {streak_stats['green_streaks']['p99']:.2f}")
    print()

    # 3. RECOMENDAÇÕES DE DELAY
    print("💡 RECOMENDAÇÕES DE DELAY")
    print("-" * 60)
    delay_conservative = analyzer.recommend_delay('conservative')
    delay_moderate = analyzer.recommend_delay('moderate')
    delay_aggressive = analyzer.recommend_delay('aggressive')

    print(f"Conservador (P99 * 0.5): {delay_conservative} velas")
    print(f"Moderado (P95 * 0.6):    {delay_moderate} velas")
    print(f"Agressivo (Median * 0.8): {delay_aggressive} velas")
    print()

    # 4. BACKTEST COM DIFERENTES CONFIGURAÇÕES
    print("📈 BACKTESTS - OTIMIZAÇÃO DE PARÂMETROS")
    print("=" * 60)

    configs = [
        {'delay': delay_conservative, 'mult': 2.0, 'name': 'Conservador (Delay alto, Mart 2x)'},
        {'delay': delay_moderate, 'mult': 2.0, 'name': 'Moderado (Delay médio, Mart 2x)'},
        {'delay': delay_aggressive, 'mult': 2.0, 'name': 'Agressivo (Delay baixo, Mart 2x)'},
        {'delay': delay_moderate, 'mult': 1.5, 'name': 'Oscar\'s Grind (Delay médio, Mart 1.5x)'},
    ]

    results = []

    for config in configs:
        print(f"\n🧪 Testando: {config['name']}")
        print(f"   Delay: {config['delay']} | Multiplicador: {config['mult']}")

        backtester = MartingaleBacktester(
            candles=candles,
            initial_balance=10000.0,
            initial_stake=0.35,
            martingale_multiplier=config['mult'],
            delay=config['delay'],
            max_martingale_levels=10,
            stop_loss=100.0
        )

        report = backtester.run()

        if 'error' not in report:
            print(f"   ✅ Trades: {report['total_trades']} | Win Rate: {report['win_rate_pct']:.2f}%")
            print(f"   💰 Profit: ${report['total_profit']:.2f} ({report['roi_pct']:.2f}%)")
            print(f"   📉 Max Drawdown: ${report['max_drawdown']:.2f}")
            print(f"   ⚡ Profit Factor: {report['profit_factor']:.2f}")
            print(f"   🎯 Max Level: {report['max_level_used']}")

            results.append({**config, **report})

    # 5. RELATÓRIO FINAL
    print("\n" + "=" * 60)
    print("🏆 CONFIGURAÇÃO RECOMENDADA")
    print("=" * 60)

    # Ordenar por ROI
    best_config = max(results, key=lambda x: x['roi_pct'])

    print(f"Estratégia: {best_config['name']}")
    print(f"Delay: {best_config['delay']} velas")
    print(f"Multiplicador Martingale: {best_config['mult']}x")
    print(f"Win Rate: {best_config['win_rate_pct']:.2f}%")
    print(f"ROI: {best_config['roi_pct']:.2f}%")
    print(f"Profit Factor: {best_config['profit_factor']:.2f}")
    print(f"Max Drawdown: ${best_config['max_drawdown']:.2f}")
    print()
    print("💡 NOTA: No V100 (1s), a volatilidade é MAIOR que no V100 padrão.")
    print("   Considere usar Delay mais alto ou multiplicador mais baixo (1.5x).")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
