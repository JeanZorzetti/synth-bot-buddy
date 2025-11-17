"""
Análise Técnica Principal
Integra todos os indicadores e gera sinais de trading
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

from .indicators import TrendIndicators, MomentumIndicators, VolatilityIndicators
from .patterns import CandlestickPatterns, SupportResistanceDetector, ChartFormationDetector


class TradingSignal:
    """
    Representa um sinal de trading
    """
    def __init__(self, symbol: str, signal_type: str, strength: float,
                 confidence: float, indicators: Dict, reason: str,
                 entry_price: float, stop_loss: float, take_profit: float):
        self.timestamp = datetime.now()
        self.symbol = symbol
        self.signal_type = signal_type  # BUY, SELL, NEUTRAL
        self.strength = strength  # 0-100
        self.confidence = confidence  # 0-100
        self.indicators = indicators
        self.reason = reason
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.risk_reward_ratio = abs((take_profit - entry_price) / (entry_price - stop_loss)) if stop_loss != entry_price else 0

    def to_dict(self) -> Dict:
        """Converte sinal para dicionário"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'strength': round(self.strength, 2),
            'confidence': round(self.confidence, 2),
            'indicators': self.indicators,
            'reason': self.reason,
            'entry_price': round(self.entry_price, 5),
            'stop_loss': round(self.stop_loss, 5),
            'take_profit': round(self.take_profit, 5),
            'risk_reward_ratio': round(self.risk_reward_ratio, 2)
        }


class TechnicalAnalysis:
    """
    Classe principal de análise técnica
    Calcula todos os indicadores e gera sinais de trading
    """

    def __init__(self):
        self.trend = TrendIndicators()
        self.momentum = MomentumIndicators()
        self.volatility = VolatilityIndicators()

        # Detectores de padrões (Fase 2)
        self.candlestick_detector = CandlestickPatterns()
        self.sr_detector = SupportResistanceDetector(window=10, min_touches=1, zone_width_pct=0.3)
        self.chart_detector = ChartFormationDetector(tolerance_pct=1.5, min_bars=15)

    def calculate_all_indicators(self, df: pd.DataFrame) -> Dict:
        """
        Calcula todos os indicadores para um dataframe de preços

        Args:
            df: DataFrame com colunas: open, high, low, close, volume

        Returns:
            Dicionário com todos os indicadores calculados
        """
        if len(df) < 200:
            raise ValueError("Necessário pelo menos 200 barras de dados para análise completa")

        indicators = {}

        # Preços
        close = df['close']
        high = df['high']
        low = df['low']

        # === INDICADORES DE TENDÊNCIA ===
        try:
            # Médias móveis
            mas = self.trend.calculate_all_moving_averages(close)
            indicators.update(mas)

            # Força da tendência
            indicators['trend_strength'] = self.trend.calculate_trend_strength(close)

            # Identificar tendência atual
            current_price = close.iloc[-1]
            if 'sma_20' in indicators and 'sma_50' in indicators and 'sma_200' in indicators:
                trend_info = self.trend.identify_trend(
                    current_price,
                    indicators['sma_20'].iloc[-1],
                    indicators['sma_50'].iloc[-1],
                    indicators['sma_200'].iloc[-1]
                )
                indicators['trend'] = trend_info

        except Exception as e:
            print(f"Erro ao calcular indicadores de tendência: {e}")

        # === INDICADORES DE MOMENTUM ===
        try:
            # RSI
            indicators['rsi'] = self.momentum.rsi(close)

            # MACD
            macd_data = self.momentum.macd(close)
            indicators.update(macd_data)

            # Stochastic
            stoch_data = self.momentum.stochastic(high, low, close)
            indicators.update(stoch_data)

        except Exception as e:
            print(f"Erro ao calcular indicadores de momentum: {e}")

        # === INDICADORES DE VOLATILIDADE ===
        try:
            # Bollinger Bands
            bb_data = self.volatility.bollinger_bands(close)
            indicators['bb_upper'] = bb_data['upper']
            indicators['bb_middle'] = bb_data['middle']
            indicators['bb_lower'] = bb_data['lower']
            indicators['bb_width'] = bb_data['width']

            # ATR
            indicators['atr'] = self.volatility.atr(high, low, close)

        except Exception as e:
            print(f"Erro ao calcular indicadores de volatilidade: {e}")

        return indicators

    def generate_signal(self, df: pd.DataFrame, symbol: str) -> TradingSignal:
        """
        Gera sinal de trading baseado em todos os indicadores

        Args:
            df: DataFrame com dados de preço
            symbol: Símbolo do ativo

        Returns:
            TradingSignal com recomendação
        """
        import logging
        logger = logging.getLogger(__name__)

        # Calcular todos os indicadores
        indicators = self.calculate_all_indicators(df)

        # Valores atuais
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]

        # Inicializar contadores de sinais
        buy_signals = []
        sell_signals = []
        buy_strength = 0
        sell_strength = 0

        logger.info(f"\n{'='*60}")
        logger.info(f"ANÁLISE DE SINAIS PARA {symbol}")
        logger.info(f"Preço atual: {current_price:.5f}")
        logger.info(f"{'='*60}")

        # === ANÁLISE DE TENDÊNCIA ===
        if 'ema_9' in indicators and 'ema_21' in indicators:
            crossover = self.trend.detect_crossover(
                indicators['ema_9'],
                indicators['ema_21']
            )
            logger.info(f"\n[TENDÊNCIA - EMA Crossover]")
            logger.info(f"  EMA 9: {indicators['ema_9'].iloc[-1]:.5f}")
            logger.info(f"  EMA 21: {indicators['ema_21'].iloc[-1]:.5f}")
            logger.info(f"  Sinal: {crossover['signal']}")
            logger.info(f"  Strength: {crossover['strength']:.2f}")

            if crossover['signal'] == 'BUY':
                buy_signals.append(crossover['description'])
                buy_strength += crossover['strength']
                logger.info(f"  ✓ VOTO BUY adicionado")
            elif crossover['signal'] == 'SELL':
                sell_signals.append(crossover['description'])
                sell_strength += crossover['strength']
                logger.info(f"  ✓ VOTO SELL adicionado")
            else:
                logger.info(f"  - NEUTRAL")

        # === ANÁLISE DE RSI ===
        if 'rsi' in indicators:
            rsi_current = indicators['rsi'].iloc[-1]
            rsi_interp = self.momentum.interpret_rsi(rsi_current)

            logger.info(f"\n[RSI]")
            logger.info(f"  Valor: {rsi_current:.2f}")
            logger.info(f"  Sinal: {rsi_interp['signal']}")
            logger.info(f"  Condição: {rsi_interp['condition']}")
            logger.info(f"  Strength: {rsi_interp['strength']:.2f}")

            if rsi_interp['signal'] == 'BUY':
                buy_signals.append(rsi_interp['description'])
                buy_strength += rsi_interp['strength']
                logger.info(f"  ✓ VOTO BUY adicionado")
            elif rsi_interp['signal'] == 'SELL':
                sell_signals.append(rsi_interp['description'])
                sell_strength += rsi_interp['strength']
                logger.info(f"  ✓ VOTO SELL adicionado")
            else:
                logger.info(f"  - NEUTRAL")

        # === ANÁLISE DE MACD ===
        if 'histogram' in indicators:
            macd_line = indicators['macd_line'].iloc[-1]
            signal_line = indicators['signal_line'].iloc[-1]
            histogram = indicators['histogram'].iloc[-1]
            prev_histogram = indicators['histogram'].iloc[-2]

            macd_interp = self.momentum.interpret_macd(
                macd_line, signal_line, histogram, prev_histogram
            )

            logger.info(f"\n[MACD]")
            logger.info(f"  MACD Line: {macd_line:.5f}")
            logger.info(f"  Signal Line: {signal_line:.5f}")
            logger.info(f"  Histogram: {histogram:.5f} (anterior: {prev_histogram:.5f})")
            logger.info(f"  Sinal: {macd_interp['signal']}")
            logger.info(f"  Strength: {macd_interp['strength']:.2f}")

            if macd_interp['signal'] == 'BUY':
                buy_signals.append(macd_interp['description'])
                buy_strength += macd_interp['strength']
                logger.info(f"  ✓ VOTO BUY adicionado")
            elif macd_interp['signal'] == 'SELL':
                sell_signals.append(macd_interp['description'])
                sell_strength += macd_interp['strength']
                logger.info(f"  ✓ VOTO SELL adicionado")
            else:
                logger.info(f"  - NEUTRAL")

        # === ANÁLISE DE BOLLINGER BANDS ===
        if 'bb_upper' in indicators:
            bb_upper = indicators['bb_upper'].iloc[-1]
            bb_middle = indicators['bb_middle'].iloc[-1]
            bb_lower = indicators['bb_lower'].iloc[-1]
            bb_width = indicators['bb_width'].iloc[-1]

            bb_interp = self.volatility.interpret_bollinger(
                current_price, bb_upper, bb_middle, bb_lower, bb_width
            )

            logger.info(f"\n[BOLLINGER BANDS]")
            logger.info(f"  Preço: {current_price:.5f}")
            logger.info(f"  Upper: {bb_upper:.5f}")
            logger.info(f"  Middle: {bb_middle:.5f}")
            logger.info(f"  Lower: {bb_lower:.5f}")
            logger.info(f"  Width: {bb_width:.5f}")
            logger.info(f"  Sinal: {bb_interp['signal']}")
            logger.info(f"  Condição: {bb_interp['condition']}")
            logger.info(f"  Strength: {bb_interp['strength']:.2f}")

            if bb_interp['signal'] == 'BUY':
                buy_signals.append(bb_interp['description'])
                buy_strength += bb_interp['strength']
                logger.info(f"  ✓ VOTO BUY adicionado")
            elif bb_interp['signal'] == 'SELL':
                sell_signals.append(bb_interp['description'])
                sell_strength += bb_interp['strength']
                logger.info(f"  ✓ VOTO SELL adicionado")
            else:
                logger.info(f"  - NEUTRAL")

        # === ANÁLISE DE STOCHASTIC ===
        if 'k_percent' in indicators:
            k_current = indicators['k_percent'].iloc[-1]
            d_current = indicators['d_percent'].iloc[-1]
            k_prev = indicators['k_percent'].iloc[-2]
            d_prev = indicators['d_percent'].iloc[-2]

            stoch_interp = self.momentum.interpret_stochastic(
                k_current, d_current, k_prev, d_prev
            )

            logger.info(f"\n[STOCHASTIC]")
            logger.info(f"  %K: {k_current:.2f} (anterior: {k_prev:.2f})")
            logger.info(f"  %D: {d_current:.2f} (anterior: {d_prev:.2f})")
            logger.info(f"  Sinal: {stoch_interp['signal']}")
            logger.info(f"  Strength: {stoch_interp['strength']:.2f}")

            if stoch_interp['signal'] == 'BUY':
                buy_signals.append(stoch_interp['description'])
                buy_strength += stoch_interp['strength']
                logger.info(f"  ✓ VOTO BUY adicionado")
            elif stoch_interp['signal'] == 'SELL':
                sell_signals.append(stoch_interp['description'])
                sell_strength += stoch_interp['strength']
                logger.info(f"  ✓ VOTO SELL adicionado")
            else:
                logger.info(f"  - NEUTRAL")

        # === ANÁLISE DE PADRÕES (FASE 2) ===
        logger.info(f"\n{'='*60}")
        logger.info(f"ANÁLISE DE PADRÕES")
        logger.info(f"{'='*60}")

        # Detectar padrões de candlestick
        try:
            candlestick_patterns = self.candlestick_detector.detect_all_patterns(df, lookback=50)

            logger.info(f"\n[PADRÕES DE CANDLESTICK]")
            logger.info(f"  Total detectado: {len(candlestick_patterns)}")

            # Contar votos por tipo
            pattern_buy = [p for p in candlestick_patterns if p.signal == 'BUY']
            pattern_sell = [p for p in candlestick_patterns if p.signal == 'SELL']

            logger.info(f"  BUY patterns: {len(pattern_buy)}")
            logger.info(f"  SELL patterns: {len(pattern_sell)}")

            # Top 3 padrões mais confiantes
            top_patterns = sorted(candlestick_patterns, key=lambda p: p.confidence, reverse=True)[:3]
            for p in top_patterns:
                logger.info(f"    • {p.name} ({p.signal}) - Confiança: {p.confidence:.0f}%")

            # Adicionar votos se houver confluência (2+ padrões na mesma direção)
            if len(pattern_buy) >= 2:
                avg_confidence = sum(p.confidence for p in pattern_buy[:3]) / min(len(pattern_buy), 3)
                pattern_desc = f"{len(pattern_buy)} padrões de candlestick bullish detectados"
                buy_signals.append(pattern_desc)
                buy_strength += avg_confidence * 0.8  # 80% do peso de um indicador técnico
                logger.info(f"  ✓ VOTO BUY adicionado (candlestick patterns)")

            if len(pattern_sell) >= 2:
                avg_confidence = sum(p.confidence for p in pattern_sell[:3]) / min(len(pattern_sell), 3)
                pattern_desc = f"{len(pattern_sell)} padrões de candlestick bearish detectados"
                sell_signals.append(pattern_desc)
                sell_strength += avg_confidence * 0.8
                logger.info(f"  ✓ VOTO SELL adicionado (candlestick patterns)")

        except Exception as e:
            logger.warning(f"  Erro ao detectar padrões de candlestick: {e}")

        # Detectar níveis de suporte/resistência e breakouts
        try:
            sr_analysis = self.sr_detector.get_analysis_summary(df)

            logger.info(f"\n[SUPORTE E RESISTÊNCIA]")
            logger.info(f"  Níveis detectados: {sr_analysis['total_levels']}")
            logger.info(f"  Suportes: {sr_analysis['support_levels']}")
            logger.info(f"  Resistências: {sr_analysis['resistance_levels']}")

            # Breakout detectado?
            if sr_analysis['breakout_detected']:
                breakout = sr_analysis['breakout_detected']
                logger.info(f"\n  🚀 BREAKOUT DETECTADO:")
                logger.info(f"    Tipo: {breakout['type']}")
                logger.info(f"    Nível: {breakout['level_price']:.5f}")
                logger.info(f"    Força do nível: {breakout['level_strength']:.0f}")

                # Breakout bullish = voto BUY com alto peso
                if breakout['type'] == 'bullish_breakout':
                    breakout_strength = breakout['level_strength']
                    buy_signals.append(f"Breakout bullish em {breakout['level_price']:.5f}")
                    buy_strength += breakout_strength
                    logger.info(f"  ✓✓ VOTO BUY FORTE adicionado (breakout)")

                # Breakdown bearish = voto SELL com alto peso
                elif breakout['type'] == 'bearish_breakdown':
                    breakout_strength = breakout['level_strength']
                    sell_signals.append(f"Breakdown bearish em {breakout['level_price']:.5f}")
                    sell_strength += breakout_strength
                    logger.info(f"  ✓✓ VOTO SELL FORTE adicionado (breakdown)")

            # Bounce detectado?
            if sr_analysis['bounce_detected']:
                bounce = sr_analysis['bounce_detected']
                logger.info(f"\n  ↩️  BOUNCE DETECTADO:")
                logger.info(f"    Tipo: {bounce['type']}")
                logger.info(f"    Nível: {bounce['level_price']:.5f}")
                logger.info(f"    Força do nível: {bounce['level_strength']:.0f}")

                # Bounce bullish em suporte = voto BUY
                if bounce['type'] == 'bullish_bounce':
                    bounce_strength = bounce['level_strength'] * 0.7  # 70% do peso de breakout
                    buy_signals.append(f"Rejeição de suporte em {bounce['level_price']:.5f}")
                    buy_strength += bounce_strength
                    logger.info(f"  ✓ VOTO BUY adicionado (bounce)")

                # Bounce bearish em resistência = voto SELL
                elif bounce['type'] == 'bearish_bounce':
                    bounce_strength = bounce['level_strength'] * 0.7
                    sell_signals.append(f"Rejeição de resistência em {bounce['level_price']:.5f}")
                    sell_strength += bounce_strength
                    logger.info(f"  ✓ VOTO SELL adicionado (bounce)")

            # Informar níveis próximos (contexto)
            if sr_analysis['nearest_support']:
                ns = sr_analysis['nearest_support']
                logger.info(f"  Suporte mais próximo: {ns['price']:.5f} ({ns['distance_pct']:.2f}%)")

            if sr_analysis['nearest_resistance']:
                nr = sr_analysis['nearest_resistance']
                logger.info(f"  Resistência mais próxima: {nr['price']:.5f} ({nr['distance_pct']:.2f}%)")

        except Exception as e:
            logger.warning(f"  Erro ao detectar S/R: {e}")

        # Detectar formações gráficas
        try:
            chart_formations = self.chart_detector.detect_all_formations(df, lookback=100)

            logger.info(f"\n[FORMAÇÕES GRÁFICAS]")
            logger.info(f"  Total detectado: {len(chart_formations)}")

            # Filtrar apenas formações confirmadas ou completed
            relevant_formations = [f for f in chart_formations if f.status in ['confirmed', 'completed']]
            logger.info(f"  Confirmadas/Completed: {len(relevant_formations)}")

            # Contar por tipo
            formation_buy = [f for f in relevant_formations if f.signal == 'BUY']
            formation_sell = [f for f in relevant_formations if f.signal == 'SELL']

            logger.info(f"  BUY formations: {len(formation_buy)}")
            logger.info(f"  SELL formations: {len(formation_sell)}")

            # Top 2 formações mais confiantes
            top_formations = sorted(relevant_formations, key=lambda f: f.confidence, reverse=True)[:2]
            for f in top_formations:
                logger.info(f"    • {f.name} ({f.status}) - {f.signal} - Confiança: {f.confidence:.0f}%")

            # Adicionar votos para formações confirmadas de alta confiança
            if formation_buy:
                # Pegar as top 2 formações BUY
                top_buy_formations = sorted(formation_buy, key=lambda f: f.confidence, reverse=True)[:2]
                for f in top_buy_formations:
                    if f.confidence >= 60:  # Apenas formações com >60% confiança
                        formation_desc = f"Formação {f.name} ({f.status})"
                        buy_signals.append(formation_desc)
                        buy_strength += f.confidence * 0.9  # 90% do peso de um indicador
                        logger.info(f"  ✓ VOTO BUY adicionado ({f.name})")

            if formation_sell:
                top_sell_formations = sorted(formation_sell, key=lambda f: f.confidence, reverse=True)[:2]
                for f in top_sell_formations:
                    if f.confidence >= 60:
                        formation_desc = f"Formação {f.name} ({f.status})"
                        sell_signals.append(formation_desc)
                        sell_strength += f.confidence * 0.9
                        logger.info(f"  ✓ VOTO SELL adicionado ({f.name})")

        except Exception as e:
            logger.warning(f"  Erro ao detectar formações gráficas: {e}")

        logger.info(f"{'='*60}")

        # === DECISÃO FINAL ===
        total_signals = len(buy_signals) + len(sell_signals)

        logger.info(f"\n{'='*60}")
        logger.info(f"RESUMO DOS VOTOS:")
        logger.info(f"  BUY signals: {len(buy_signals)} (strength total: {buy_strength:.2f})")
        for signal in buy_signals:
            logger.info(f"    • {signal}")
        logger.info(f"  SELL signals: {len(sell_signals)} (strength total: {sell_strength:.2f})")
        for signal in sell_signals:
            logger.info(f"    • {signal}")
        logger.info(f"  Total signals: {total_signals}")
        logger.info(f"{'='*60}")

        # Calcular ATR para stop loss
        atr_value = indicators['atr'].iloc[-1] if 'atr' in indicators else current_price * 0.015

        # Determinar sinal e calcular níveis
        logger.info(f"\n{'='*60}")
        logger.info(f"DECISÃO FINAL:")
        logger.info(f"  Requisitos para BUY: >= 3 sinais BUY E buy_strength > sell_strength")
        logger.info(f"  Requisitos para SELL: >= 3 sinais SELL E sell_strength > buy_strength")
        logger.info(f"  BUY signals: {len(buy_signals)} | BUY strength: {buy_strength:.2f}")
        logger.info(f"  SELL signals: {len(sell_signals)} | SELL strength: {sell_strength:.2f}")

        if len(buy_signals) >= 3 and buy_strength > sell_strength:
            # SINAL DE COMPRA
            signal_type = 'BUY'
            strength = min(buy_strength / len(buy_signals), 100)
            confidence = min((len(buy_signals) / total_signals * 100), 100) if total_signals > 0 else 0
            reason = "Confluência de indicadores: " + "; ".join(buy_signals)

            logger.info(f"  ✓✓✓ SINAL: BUY")
            logger.info(f"  Strength: {strength:.2f}")
            logger.info(f"  Confidence: {confidence:.2f}%")

            # Níveis de preço
            entry_price = current_price
            stop_loss = self.volatility.calculate_atr_stop_loss(
                entry_price, atr_value, is_long=True, multiplier=2.0
            )
            take_profit = entry_price + (abs(entry_price - stop_loss) * 2)  # R:R 1:2

        elif len(sell_signals) >= 3 and sell_strength > buy_strength:
            # SINAL DE VENDA
            signal_type = 'SELL'
            strength = min(sell_strength / len(sell_signals), 100)
            confidence = min((len(sell_signals) / total_signals * 100), 100) if total_signals > 0 else 0
            reason = "Confluência de indicadores: " + "; ".join(sell_signals)

            logger.info(f"  ✓✓✓ SINAL: SELL")
            logger.info(f"  Strength: {strength:.2f}")
            logger.info(f"  Confidence: {confidence:.2f}%")

            # Níveis de preço
            entry_price = current_price
            stop_loss = self.volatility.calculate_atr_stop_loss(
                entry_price, atr_value, is_long=False, multiplier=2.0
            )
            take_profit = entry_price - (abs(stop_loss - entry_price) * 2)  # R:R 1:2

        else:
            # SEM SINAL CLARO
            signal_type = 'NEUTRAL'
            strength = 0
            confidence = 0
            reason = "Sem confluência de indicadores suficiente"
            entry_price = current_price
            stop_loss = current_price
            take_profit = current_price

            logger.info(f"  ⊘ SINAL: NEUTRAL")
            if len(buy_signals) < 3 and len(sell_signals) < 3:
                logger.info(f"  Motivo: Menos de 3 sinais em ambas direções")
            elif len(buy_signals) >= 3 and buy_strength <= sell_strength:
                logger.info(f"  Motivo: {len(buy_signals)} sinais BUY, mas sell_strength >= buy_strength")
            elif len(sell_signals) >= 3 and sell_strength <= buy_strength:
                logger.info(f"  Motivo: {len(sell_signals)} sinais SELL, mas buy_strength >= sell_strength")

        logger.info(f"{'='*60}\n")

        # Preparar resumo dos indicadores
        indicator_summary = {
            'rsi': round(indicators['rsi'].iloc[-1], 2) if 'rsi' in indicators else None,
            'macd_histogram': round(indicators['histogram'].iloc[-1], 4) if 'histogram' in indicators else None,
            'bb_position': 'upper' if current_price > bb_upper else 'lower' if current_price < bb_lower else 'middle' if 'bb_upper' in indicators else None,
            'stochastic_k': round(k_current, 2) if 'k_percent' in indicators else None,
            'atr': round(atr_value, 5),
            'trend': indicators.get('trend', {}).get('direction', 'UNKNOWN')
        }

        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            indicators=indicator_summary,
            reason=reason,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

    def get_current_indicators(self, df: pd.DataFrame) -> Dict:
        """
        Retorna valores atuais de todos os indicadores

        Args:
            df: DataFrame com dados de preço

        Returns:
            Dicionário com valores atuais dos indicadores
        """
        indicators = self.calculate_all_indicators(df)

        # Extrair apenas valores atuais (última linha)
        current = {}
        for key, value in indicators.items():
            if isinstance(value, pd.Series):
                current[key] = round(value.iloc[-1], 5) if not pd.isna(value.iloc[-1]) else None
            elif isinstance(value, dict):
                current[key] = value
            else:
                current[key] = value

        return current
