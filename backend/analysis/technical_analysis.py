"""
Análise Técnica Principal
Integra todos os indicadores e gera sinais de trading
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

from .indicators import TrendIndicators, MomentumIndicators, VolatilityIndicators


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
