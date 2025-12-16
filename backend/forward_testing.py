"""
Forward Testing System - Integração ML Predictor + Paper Trading

Este módulo implementa um sistema automatizado de forward testing que:
1. Coleta dados reais do mercado via Deriv API
2. Gera previsões usando o modelo ML treinado
3. Executa trades no Paper Trading Engine
4. Registra todas as métricas e bugs encontrados
5. Gera relatórios automáticos de validação
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import json
import os
from pathlib import Path

import pandas as pd
import numpy as np

from ml_predictor import MLPredictor
from paper_trading_engine import PaperTradingEngine, PositionType
from deriv_api_legacy import DerivAPI

logger = logging.getLogger(__name__)


class ForwardTestingEngine:
    """
    Motor de Forward Testing automatizado

    Roda paper trading com previsões ML em tempo real,
    registrando métricas e bugs para validação.
    """

    def __init__(
        self,
        symbol: str = "R_100",
        initial_capital: float = 10000.0,
        confidence_threshold: float = 0.60,
        max_position_size_pct: float = 2.0,  # 2% do capital por trade
        stop_loss_pct: float = 2.0,  # 2% stop loss
        take_profit_pct: float = 4.0,  # 4% take profit (risk:reward 1:2)
        log_dir: str = "forward_testing_logs"
    ):
        """
        Inicializa o sistema de forward testing

        Args:
            symbol: Símbolo a tradear
            initial_capital: Capital inicial para paper trading
            confidence_threshold: Threshold mínimo de confidence para executar trade
            max_position_size_pct: Tamanho máximo da posição (% do capital)
            stop_loss_pct: Stop loss percentual
            take_profit_pct: Take profit percentual
            log_dir: Diretório para logs e relatórios
        """
        self.symbol = symbol
        self.confidence_threshold = confidence_threshold
        self.max_position_size_pct = max_position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # Componentes
        self.ml_predictor = MLPredictor()
        self.paper_trading = PaperTradingEngine(initial_capital=initial_capital)

        # Deriv API para dados reais
        self.deriv_api = DerivAPI()
        self.deriv_connected = False
        self.deriv_api_token = os.getenv("DERIV_API_TOKEN", "")
        self.tick_subscription_active = False  # Track subscription status

        # Estado
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.last_prediction_time: Optional[datetime] = None

        # Logs e métricas
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.bug_log: List[Dict] = []
        self.prediction_log: List[Dict] = []
        self.trade_log: List[Dict] = []

        # Controle de frequência (evitar overtrading)
        self.min_time_between_predictions = 60  # 1 minuto entre previsões

        logger.info(f"ForwardTestingEngine inicializado para {symbol}")
        logger.info(f"Parâmetros: confidence_threshold={confidence_threshold}, "
                   f"position_size={max_position_size_pct}%, "
                   f"stop_loss={stop_loss_pct}%, take_profit={take_profit_pct}%")

    async def start(self):
        """Inicia sessão de forward testing"""
        if self.is_running:
            logger.warning("Forward testing já está rodando")
            return

        self.is_running = True
        self.start_time = datetime.now()
        self.paper_trading.start()

        logger.info("="*60)
        logger.info("FORWARD TESTING INICIADO")
        logger.info(f"Início: {self.start_time.isoformat()}")
        logger.info(f"Símbolo: {self.symbol}")
        logger.info(f"Capital Inicial: ${self.paper_trading.initial_capital:,.2f}")
        logger.info("="*60)

        # Iniciar loop de trading
        await self._trading_loop()

    async def stop(self):
        """Para sessão de forward testing"""
        self.is_running = False
        self.paper_trading.stop()

        # Desconectar da Deriv API
        if self.deriv_connected:
            # Cancelar todas as subscrições antes de desconectar
            if self.tick_subscription_active:
                try:
                    await self.deriv_api._send_request({"forget_all": "ticks"})
                    self.tick_subscription_active = False
                    logger.info("📴 Subscrições de ticks canceladas")
                except Exception as e:
                    logger.warning(f"Erro ao cancelar subscrições: {e}")

            await self.deriv_api.disconnect()
            self.deriv_connected = False
            logger.info("🔌 Desconectado da Deriv API")

        logger.info("="*60)
        logger.info("FORWARD TESTING PARADO")
        logger.info(f"Duração: {datetime.now() - self.start_time if self.start_time else 'N/A'}")
        logger.info("="*60)

        # Gerar relatório final
        self.generate_validation_report()

    async def _trading_loop(self):
        """
        Loop principal de trading

        1. Coleta dados do mercado
        2. Gera previsão ML
        3. Executa trade se confidence > threshold
        4. Atualiza posições existentes
        5. Registra métricas
        """
        while self.is_running:
            try:
                # Verificar se tempo mínimo passou desde última previsão
                if self.last_prediction_time:
                    elapsed = (datetime.now() - self.last_prediction_time).total_seconds()
                    if elapsed < self.min_time_between_predictions:
                        await asyncio.sleep(1)
                        continue

                # 1. Coletar dados do mercado
                market_data = await self._fetch_market_data()
                if not market_data:
                    logger.warning("Falha ao coletar dados do mercado")
                    await asyncio.sleep(5)
                    continue

                current_price = market_data['close']

                # 2. Atualizar posições existentes (stop loss / take profit)
                self.paper_trading.update_positions({self.symbol: current_price})

                # 3. Verificar se pode abrir nova posição
                if len(self.paper_trading.positions) >= self.paper_trading.max_positions:
                    logger.debug(f"Limite de posições atingido ({self.paper_trading.max_positions})")
                    await asyncio.sleep(10)
                    continue

                # 4. Gerar previsão ML
                prediction = await self._generate_prediction(market_data)
                if not prediction:
                    logger.warning("Falha ao gerar previsão ML")
                    await asyncio.sleep(5)
                    continue

                self.last_prediction_time = datetime.now()

                # Registrar previsão
                self.prediction_log.append({
                    'timestamp': self.last_prediction_time.isoformat(),
                    'prediction': prediction['prediction'],
                    'confidence': prediction['confidence'],
                    'price': current_price
                })

                # 5. Decidir se executa trade
                if prediction['confidence'] >= self.confidence_threshold:
                    await self._execute_trade(prediction, current_price)
                else:
                    logger.debug(f"Confidence {prediction['confidence']:.2%} < threshold {self.confidence_threshold:.2%}, não executando trade")

                # Aguardar antes da próxima iteração
                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Erro no trading loop: {e}", exc_info=True)
                self._log_bug("trading_loop_error", str(e))
                await asyncio.sleep(5)

    async def _fetch_market_data(self) -> Optional[Dict]:
        """
        Coleta dados REAIS do mercado via Deriv API

        Returns:
            Dict com OHLC + indicadores técnicos ou None se falhar
        """
        try:
            # Conectar à Deriv API se ainda não conectado
            if not self.deriv_connected:
                if not self.deriv_api_token:
                    raise ValueError("DERIV_API_TOKEN não configurado nas variáveis de ambiente")

                await self.deriv_api.connect()
                await self.deriv_api.authorize(self.deriv_api_token)
                self.deriv_connected = True
                logger.info("✅ Conectado e autenticado na Deriv API para dados reais")

            # Obter tick atual (preço real)
            # Se já temos subscrição ativa, reutilizar. Caso contrário, fazer uma nova subscrição
            if not self.tick_subscription_active:
                tick_response = await self.deriv_api.ticks(self.symbol, subscribe=True)
                self.tick_subscription_active = True
            else:
                # Reutilizar subscrição existente - apenas aguardar próximo tick
                # Para isso, vamos fazer um forget_all e resubscrever
                try:
                    await self.deriv_api._send_request({"forget_all": "ticks"})
                    self.tick_subscription_active = False
                except:
                    pass  # Ignorar erro de forget

                tick_response = await self.deriv_api.ticks(self.symbol, subscribe=True)
                self.tick_subscription_active = True

            if 'tick' not in tick_response:
                logger.warning(f"Resposta sem tick: {tick_response}")
                self._log_bug("tick_response_invalid", "Tick não encontrado na resposta")
                return None

            tick = tick_response['tick']
            current_price = float(tick['quote'])

            # Para OHLC, precisamos manter histórico de ticks
            # Por simplicidade, usamos o preço atual como aproximação
            # Em produção real, usar candles history API

            return {
                'timestamp': datetime.fromtimestamp(tick['epoch']).isoformat(),
                'open': current_price,  # Simplificação
                'high': current_price,
                'low': current_price,
                'close': current_price,
                'volume': 1000,  # Volume não disponível em ticks
                'symbol': tick['symbol']
            }

        except Exception as e:
            logger.error(f"Erro ao coletar dados REAIS do mercado: {e}", exc_info=True)
            self._log_bug("market_data_fetch_error", str(e), severity="CRITICAL")

            # Fallback para mock apenas em caso de erro crítico
            logger.warning("⚠️ Usando dados mock como fallback temporário")
            base_price = 100.0
            volatility = np.random.normal(0, 0.5)
            close_price = base_price * (1 + volatility / 100)

            return {
                'timestamp': datetime.now().isoformat(),
                'open': close_price * 0.999,
                'high': close_price * 1.001,
                'low': close_price * 0.998,
                'close': close_price,
                'volume': 1000,
                'symbol': self.symbol
            }

    async def _generate_prediction(self, market_data: Dict) -> Optional[Dict]:
        """
        Gera previsão usando modelo ML

        Args:
            market_data: Dados do mercado

        Returns:
            Dict com prediction e confidence ou None se falhar
        """
        try:
            # Criar DataFrame com histórico de candles
            # Acumular ticks em buffer para formar candles
            if not hasattr(self, 'price_buffer'):
                self.price_buffer = []

            self.price_buffer.append({
                'timestamp': market_data['timestamp'],
                'close': market_data['close'],
                'high': market_data['high'],
                'low': market_data['low'],
                'volume': market_data['volume']
            })

            # Manter buffer de 250 pontos (ML precisa de 200+ para features)
            if len(self.price_buffer) > 250:
                self.price_buffer = self.price_buffer[-250:]

            # Precisa de pelo menos 200 pontos para features
            if len(self.price_buffer) < 200:
                logger.debug(f"Buffer insuficiente: {len(self.price_buffer)}/200 pontos")
                return {
                    'prediction': 'NO_MOVE',
                    'confidence': 0.0,
                    'signal_strength': 'NONE',
                    'reason': f'Aguardando histórico ({len(self.price_buffer)}/200)'
                }

            # Converter buffer para DataFrame
            df = pd.DataFrame(self.price_buffer)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

            # Adicionar coluna 'open' se não existir (usar close anterior)
            if 'open' not in df.columns:
                df['open'] = df['close'].shift(1).fillna(df['close'])

            # Gerar previsão usando ML Predictor
            prediction = self.ml_predictor.predict(df, return_confidence=True)

            logger.info(f"✅ Previsão ML: {prediction['prediction']} (confidence: {prediction['confidence']:.2%})")

            return prediction

        except Exception as e:
            logger.error(f"Erro ao gerar previsão: {e}", exc_info=True)
            self._log_bug("prediction_generation_error", str(e), severity="ERROR")
            return None

    async def _execute_trade(self, prediction: Dict, current_price: float):
        """
        Executa trade baseado na previsão

        Args:
            prediction: Dict com prediction e confidence
            current_price: Preço atual do mercado
        """
        try:
            # Determinar direção
            if prediction['prediction'] == 'UP':
                position_type = PositionType.LONG
            elif prediction['prediction'] == 'DOWN':
                position_type = PositionType.SHORT
            else:
                logger.warning(f"Previsão inválida: {prediction['prediction']}")
                return

            # Calcular tamanho da posição
            position_size = self.paper_trading.capital * (self.max_position_size_pct / 100)

            # Calcular stop loss e take profit
            if position_type == PositionType.LONG:
                stop_loss = current_price * (1 - self.stop_loss_pct / 100)
                take_profit = current_price * (1 + self.take_profit_pct / 100)
            else:  # SHORT
                stop_loss = current_price * (1 + self.stop_loss_pct / 100)
                take_profit = current_price * (1 - self.take_profit_pct / 100)

            # Executar ordem
            position = await self.paper_trading.execute_order(
                symbol=self.symbol,
                position_type=position_type,
                size=position_size,
                current_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

            if position:
                logger.info(f"✅ TRADE EXECUTADO: {position_type.value} {self.symbol} @ ${current_price:.4f}")
                logger.info(f"   Size: ${position_size:.2f} | Confidence: {prediction['confidence']:.2%}")
                logger.info(f"   SL: ${stop_loss:.4f} | TP: ${take_profit:.4f}")

                # Registrar trade
                self.trade_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'position_id': position.id,
                    'type': position_type.value,
                    'entry_price': current_price,
                    'size': position_size,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': prediction['confidence'],
                    'prediction': prediction['prediction']
                })
            else:
                logger.warning("Falha ao executar ordem")
                self._log_bug("order_execution_failed", "Paper trading retornou None")

        except Exception as e:
            logger.error(f"Erro ao executar trade: {e}", exc_info=True)
            self._log_bug("trade_execution_error", str(e))

    def _log_bug(self, bug_type: str, description: str, severity: str = "ERROR"):
        """
        Registra bug encontrado durante forward testing

        Args:
            bug_type: Tipo do bug
            description: Descrição detalhada
            severity: Severidade (ERROR, WARNING, CRITICAL)
        """
        bug = {
            'timestamp': datetime.now().isoformat(),
            'type': bug_type,
            'description': description,
            'severity': severity
        }

        self.bug_log.append(bug)

        # Salvar em arquivo
        bug_file = self.log_dir / "bugs.jsonl"
        with open(bug_file, 'a') as f:
            f.write(json.dumps(bug) + '\n')

        logger.error(f"🐛 BUG REGISTRADO: [{bug_type}] {description}")

    def get_status(self) -> Dict:
        """
        Retorna status atual do forward testing

        Returns:
            Dict com métricas atuais
        """
        metrics = self.paper_trading.get_metrics()

        # Adicionar métricas de forward testing
        duration_seconds = 0
        if self.start_time:
            duration_seconds = (datetime.now() - self.start_time).total_seconds()

        return {
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'duration_seconds': duration_seconds,
            'duration_hours': duration_seconds / 3600,
            'duration_days': duration_seconds / 86400,
            'symbol': self.symbol,
            'total_predictions': len(self.prediction_log),
            'total_trades': len(self.trade_log),
            'total_bugs': len(self.bug_log),
            'last_prediction_time': self.last_prediction_time.isoformat() if self.last_prediction_time else None,
            'paper_trading_metrics': metrics
        }

    def generate_validation_report(self) -> str:
        """
        Gera relatório completo de validação

        Returns:
            Path do relatório gerado
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.log_dir / f"validation_report_{timestamp}.md"

        metrics = self.paper_trading.get_metrics()
        status = self.get_status()

        # Calcular métricas adicionais
        predictions_df = pd.DataFrame(self.prediction_log)
        trades_df = pd.DataFrame(self.trade_log)
        bugs_df = pd.DataFrame(self.bug_log)

        # Estatísticas de previsões
        if len(predictions_df) > 0:
            avg_confidence = predictions_df['confidence'].mean()
            high_conf_pct = (predictions_df['confidence'] >= self.confidence_threshold).mean() * 100
        else:
            avg_confidence = 0
            high_conf_pct = 0

        # Gerar relatório
        report = f"""# Relatório de Validação - Forward Testing

## Informações Gerais

- **Símbolo**: {self.symbol}
- **Início**: {status['start_time']}
- **Duração**: {status['duration_days']:.1f} dias ({status['duration_hours']:.1f} horas)
- **Status**: {'🟢 Ativo' if self.is_running else '🔴 Parado'}

## Parâmetros de Trading

- **Capital Inicial**: ${self.paper_trading.initial_capital:,.2f}
- **Confidence Threshold**: {self.confidence_threshold:.1%}
- **Tamanho Máximo de Posição**: {self.max_position_size_pct}% do capital
- **Stop Loss**: {self.stop_loss_pct}%
- **Take Profit**: {self.take_profit_pct}%
- **Risk:Reward Ratio**: 1:{self.take_profit_pct/self.stop_loss_pct:.1f}

## Performance de Trading

### Métricas Gerais
- **Capital Atual**: ${metrics['current_capital']:,.2f}
- **P&L Total**: ${metrics['total_pnl']:,.2f} ({metrics['total_pnl_pct']:+.2f}%)
- **Capital Máximo**: ${metrics['peak_capital']:,.2f}
- **Max Drawdown**: {metrics['max_drawdown_pct']:.2f}%

### Trades
- **Total de Trades**: {metrics['total_trades']}
- **Trades Vencedores**: {metrics['winning_trades']}
- **Trades Perdedores**: {metrics['losing_trades']}
- **Win Rate**: {metrics['win_rate_pct']:.2f}%
- **Profit Factor**: {metrics['profit_factor']:.2f}
- **Sharpe Ratio**: {metrics['sharpe_ratio']:.2f}
- **Lucro Médio por Trade**: ${metrics['avg_profit_per_trade']:,.2f}

## Previsões ML

- **Total de Previsões**: {status['total_predictions']}
- **Confidence Média**: {avg_confidence:.2%}
- **Previsões com Alta Confidence (>{self.confidence_threshold:.0%})**: {high_conf_pct:.1f}%
- **Execução Rate**: {(len(trades_df) / len(predictions_df) * 100) if len(predictions_df) > 0 else 0:.1f}% (trades executados / previsões)

## Bugs e Problemas

- **Total de Bugs Registrados**: {len(bugs_df)}

"""

        if len(bugs_df) > 0:
            report += "### Bugs por Tipo\n\n"
            bug_counts = bugs_df['type'].value_counts()
            for bug_type, count in bug_counts.items():
                report += f"- **{bug_type}**: {count}\n"

            report += "\n### Bugs Críticos\n\n"
            critical_bugs = bugs_df[bugs_df['severity'] == 'CRITICAL']
            if len(critical_bugs) > 0:
                for _, bug in critical_bugs.iterrows():
                    report += f"- [{bug['timestamp']}] {bug['description']}\n"
            else:
                report += "Nenhum bug crítico encontrado ✅\n"
        else:
            report += "**Nenhum bug registrado! ✅**\n"

        report += f"""
## Validação de Objetivos

### Critérios de Aprovação (FASE 8)

| Métrica | Objetivo | Atual | Status |
|---------|----------|-------|--------|
| Win Rate | > 60% | {metrics['win_rate_pct']:.1f}% | {'✅ PASS' if metrics['win_rate_pct'] >= 60 else '❌ FAIL'} |
| Sharpe Ratio | > 1.5 | {metrics['sharpe_ratio']:.2f} | {'✅ PASS' if metrics['sharpe_ratio'] >= 1.5 else '❌ FAIL'} |
| Max Drawdown | < 15% | {metrics['max_drawdown_pct']:.1f}% | {'✅ PASS' if metrics['max_drawdown_pct'] < 15 else '❌ FAIL'} |
| Profit Factor | > 1.5 | {metrics['profit_factor']:.2f} | {'✅ PASS' if metrics['profit_factor'] >= 1.5 else '❌ FAIL'} |

### Status Geral

"""

        # Verificar aprovação
        passed_criteria = sum([
            metrics['win_rate_pct'] >= 60,
            metrics['sharpe_ratio'] >= 1.5,
            metrics['max_drawdown_pct'] < 15,
            metrics['profit_factor'] >= 1.5
        ])

        if passed_criteria == 4:
            report += "**🎉 SISTEMA APROVADO PARA PRODUÇÃO!**\n\n"
            report += "Todos os critérios de validação foram atendidos. O sistema está pronto para trading real.\n"
        elif passed_criteria >= 2:
            report += "**⚠️ APROVAÇÃO PARCIAL**\n\n"
            report += f"Sistema atendeu {passed_criteria}/4 critérios. Revisar e otimizar antes de produção.\n"
        else:
            report += "**❌ REPROVADO**\n\n"
            report += f"Sistema atendeu apenas {passed_criteria}/4 critérios. Necessário ajustes significativos.\n"

        report += """
## Próximos Passos

1. Analisar trades perdedores para identificar padrões
2. Ajustar thresholds de confidence se necessário
3. Considerar otimização de stop loss / take profit
4. Avaliar adicionar filtros de contexto de mercado
5. Testar em outros símbolos para validar robustez

---

*Relatório gerado automaticamente pelo Forward Testing Engine*
*Data: """ + datetime.now().isoformat() + "*\n"

        # Salvar relatório
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"📊 Relatório de validação gerado: {report_file}")

        return str(report_file)


# Singleton global para gerenciar instância de forward testing
_forward_testing_instance: Optional[ForwardTestingEngine] = None


def get_forward_testing_engine() -> ForwardTestingEngine:
    """Retorna instância singleton do forward testing engine"""
    global _forward_testing_instance
    if _forward_testing_instance is None:
        _forward_testing_instance = ForwardTestingEngine()
    return _forward_testing_instance
