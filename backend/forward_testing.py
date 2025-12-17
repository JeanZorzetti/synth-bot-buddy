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
from alert_system import AlertSystem
from auto_restart_system import AutoRestartSystem, SystemCheckpoint

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
        symbols: Optional[List[str]] = None,  # Lista de símbolos para multi-symbol trading
        initial_capital: float = 10000.0,
        confidence_threshold: float = 0.40,
        max_position_size_pct: float = 2.0,  # 2% do capital por trade
        max_positions_per_symbol: int = 1,  # Máximo de posições por símbolo
        stop_loss_pct: float = 2.0,  # 2% stop loss
        take_profit_pct: float = 4.0,  # 4% take profit (risk:reward 1:2)
        position_timeout_minutes: int = 30,  # Timeout para fechar posição automaticamente
        trailing_stop_enabled: bool = False,  # Trailing stop loss
        trailing_stop_distance_pct: float = 0.5,  # Distância do trailing (0.5%)
        log_dir: str = "forward_testing_logs"
    ):
        """
        Inicializa o sistema de forward testing

        Args:
            symbol: Símbolo primário (usado se symbols não for fornecido)
            symbols: Lista de símbolos para multi-symbol trading (None = single symbol)
            initial_capital: Capital inicial para paper trading
            confidence_threshold: Threshold mínimo de confidence para executar trade
            max_position_size_pct: Tamanho máximo da posição (% do capital)
            max_positions_per_symbol: Máximo de posições abertas por símbolo
            stop_loss_pct: Stop loss percentual
            take_profit_pct: Take profit percentual
            position_timeout_minutes: Timeout para fechar posição automaticamente
            trailing_stop_enabled: Se trailing stop loss está ativado
            trailing_stop_distance_pct: Distância do trailing em %
            log_dir: Diretório para logs e relatórios
        """
        # Configurar símbolos
        if symbols:
            self.symbols = symbols
            self.multi_symbol_mode = True
        else:
            self.symbols = [symbol]
            self.multi_symbol_mode = False

        self.symbol = symbol  # Manter compatibilidade com código existente
        self.confidence_threshold = confidence_threshold
        self.max_position_size_pct = max_position_size_pct
        self.max_positions_per_symbol = max_positions_per_symbol
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.position_timeout_minutes = position_timeout_minutes
        self.trailing_stop_enabled = trailing_stop_enabled
        self.trailing_stop_distance_pct = trailing_stop_distance_pct

        # Rastreamento de posições por símbolo
        self.positions_by_symbol: Dict[str, int] = {sym: 0 for sym in self.symbols}

        # Logs e métricas (definir ANTES de usar em outros componentes)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Componentes
        self.ml_predictor = MLPredictor()
        self.paper_trading = PaperTradingEngine(initial_capital=initial_capital)
        self.alert_system = AlertSystem()
        self.auto_restart = AutoRestartSystem(
            check_interval=30,  # Verificar a cada 30 segundos
            max_failures=3,  # 3 falhas antes de restart
            checkpoint_dir=str(self.log_dir / "checkpoints")
        )

        # Deriv API para dados reais
        self.deriv_api = DerivAPI()
        self.deriv_connected = False
        self.deriv_api_token = os.getenv("DERIV_API_TOKEN", "")

        # Estado
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.last_prediction_time: Optional[datetime] = None

        self.bug_log: List[Dict] = []
        self.prediction_log: List[Dict] = []
        self.trade_log: List[Dict] = []

        # Controle de frequência (evitar overtrading)
        self.min_time_between_predictions = 60  # 1 minuto entre previsões

        logger.info(f"ForwardTestingEngine inicializado para {symbol}")
        logger.info(f"Parâmetros: confidence_threshold={confidence_threshold}, "
                   f"position_size={max_position_size_pct}%, "
                   f"stop_loss={stop_loss_pct}%, take_profit={take_profit_pct}%")

        # Configurar callbacks do auto-restart
        self.auto_restart.set_health_check_callback(self._health_check)
        self.auto_restart.set_restart_callback(self._perform_restart)

    async def _health_check(self) -> bool:
        """
        Verifica saúde do sistema

        Returns:
            True se sistema está saudável, False caso contrário
        """
        try:
            # 1. Verificar se está rodando
            if not self.is_running:
                logger.warning("Health check: Sistema não está rodando")
                return False

            # 2. Verificar se teve predições recentes (últimos 5 minutos)
            if self.last_prediction_time:
                elapsed = (datetime.now() - self.last_prediction_time).total_seconds()
                if elapsed > 300:  # 5 minutos
                    logger.warning(f"Health check: Última previsão há {elapsed:.0f}s (> 300s)")
                    return False

            # 3. Verificar se API está conectada
            if not self.deriv_connected:
                logger.warning("Health check: Deriv API não conectada")
                return False

            # 4. Verificar se capital não zerou
            if self.paper_trading.capital <= 0:
                logger.error("Health check: Capital zerado!")
                return False

            # Sistema saudável
            return True

        except Exception as e:
            logger.error(f"Erro durante health check: {e}", exc_info=True)
            return False

    def _save_checkpoint(self):
        """Salva checkpoint do estado atual"""
        try:
            # Coletar posições abertas
            open_positions = []
            for pos_id, position in self.paper_trading.open_positions.items():
                open_positions.append({
                    'id': pos_id,
                    'symbol': position.symbol,
                    'type': position.position_type.value,
                    'entry_price': position.entry_price,
                    'size': position.size,
                    'stop_loss': position.stop_loss,
                    'take_profit': position.take_profit,
                    'entry_time': position.entry_time.isoformat()
                })

            # Criar checkpoint
            metrics = self.paper_trading.get_metrics()
            checkpoint = SystemCheckpoint(
                timestamp=datetime.now().isoformat(),
                symbol=self.symbol,
                capital=self.paper_trading.capital,
                total_trades=metrics['total_trades'],
                win_rate=metrics['win_rate_pct'],
                is_running=self.is_running,
                last_prediction_time=self.last_prediction_time.isoformat() if self.last_prediction_time else None,
                open_positions=open_positions
            )

            self.auto_restart.save_checkpoint(checkpoint)

        except Exception as e:
            logger.error(f"Erro ao salvar checkpoint: {e}", exc_info=True)

    def _load_checkpoint(self) -> bool:
        """
        Carrega último checkpoint e restaura estado

        Returns:
            True se checkpoint foi carregado com sucesso
        """
        try:
            checkpoint = self.auto_restart.load_checkpoint()

            if not checkpoint:
                logger.info("Nenhum checkpoint para restaurar")
                return False

            # Restaurar capital
            self.paper_trading.capital = checkpoint.capital
            logger.info(f"✅ Capital restaurado: ${checkpoint.capital:,.2f}")

            # Restaurar posições abertas (se houver)
            if checkpoint.open_positions:
                logger.info(f"⚠️ Checkpoint tinha {len(checkpoint.open_positions)} posições abertas")
                logger.info("   (Posições não serão restauradas - sistema iniciará limpo)")

            return True

        except Exception as e:
            logger.error(f"Erro ao carregar checkpoint: {e}", exc_info=True)
            return False

    async def _perform_restart(self) -> bool:
        """
        Executa restart do sistema

        Returns:
            True se restart foi bem-sucedido
        """
        try:
            logger.info("🔄 Executando restart do sistema...")

            # 1. Salvar checkpoint antes de parar
            self._save_checkpoint()

            # 2. Parar sistema atual
            await self.stop()

            # Aguardar cleanup
            await asyncio.sleep(5)

            # 3. Carregar checkpoint
            self._load_checkpoint()

            # 4. Reiniciar sistema
            await self.start()

            logger.info("✅ Restart completado com sucesso")
            return True

        except Exception as e:
            logger.error(f"Erro durante restart: {e}", exc_info=True)
            return False

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
        if self.multi_symbol_mode:
            logger.info(f"🔄 MULTI-SYMBOL MODE: {len(self.symbols)} símbolos")
            logger.info(f"Símbolos: {', '.join(self.symbols)}")
            logger.info(f"Max posições por símbolo: {self.max_positions_per_symbol}")
        else:
            logger.info(f"Símbolo: {self.symbol}")
        logger.info(f"Capital Inicial: ${self.paper_trading.initial_capital:,.2f}")
        logger.info(f"Token Deriv configurado: {'SIM' if self.deriv_api_token else 'NÃO ❌'}")
        logger.info(f"Modelo ML carregado: {self.ml_predictor.model_path.name}")
        logger.info("="*60)

        # Iniciar watchdog de auto-restart em background
        asyncio.create_task(self.auto_restart.start_monitoring())
        logger.info("🔍 Auto-restart watchdog iniciado")

        # Iniciar loop de trading
        try:
            await self._trading_loop()
        except Exception as e:
            logger.error(f"❌ ERRO CRÍTICO no trading loop: {e}", exc_info=True)
            self._save_checkpoint()  # Salvar checkpoint antes de crash
            self.is_running = False
            raise

    async def stop(self):
        """Para sessão de forward testing"""
        self.is_running = False
        self.paper_trading.stop()

        # Parar watchdog
        self.auto_restart.stop_monitoring()
        logger.info("🛑 Auto-restart watchdog parado")

        # Desconectar da Deriv API
        if self.deriv_connected:
            await self.deriv_api.disconnect()
            self.deriv_connected = False
            logger.info("🔌 Desconectado da Deriv API")

        logger.info("="*60)
        logger.info("FORWARD TESTING PARADO")
        logger.info(f"Duração: {datetime.now() - self.start_time if self.start_time else 'N/A'}")
        logger.info("="*60)

        # Gerar relatório final
        self.generate_validation_report()

    async def _process_symbol(self, symbol: str):
        """
        Processa um único símbolo: coleta dados, gera previsão, executa trades

        Args:
            symbol: Símbolo a processar
        """
        try:
            # 1. Coletar dados do mercado para este símbolo
            market_data = await self._fetch_market_data_for_symbol(symbol)
            if not market_data:
                logger.warning(f"❌ Falha ao coletar dados do mercado para {symbol}")
                return

            logger.info(f"✅ [{symbol}] Market data coletado: preço={market_data['close']:.5f}")
            current_price = market_data['close']

            # 2. Atualizar posições existentes deste símbolo
            self.paper_trading.update_positions({symbol: current_price})

            # 2.5. Fechar posições com timeout
            self._check_position_timeouts_for_symbol(symbol, current_price)

            # 3. Verificar se pode abrir nova posição neste símbolo
            current_positions = sum(1 for pos in self.paper_trading.positions.values() if pos.symbol == symbol)
            if current_positions >= self.max_positions_per_symbol:
                logger.info(f"⏸️ [{symbol}] Limite de posições atingido ({current_positions}/{self.max_positions_per_symbol})")
                return

            # 4. Gerar previsão ML
            logger.info(f"🧠 [{symbol}] Gerando previsão ML...")
            prediction = await self._generate_prediction(market_data)
            if not prediction:
                return

            logger.info(f"📊 [{symbol}] Previsão: {prediction['prediction']} (confidence: {prediction.get('confidence', 0):.2%})")

            # 5. Executar trade se confidence > threshold
            confidence = prediction.get('confidence', 0)
            if confidence >= self.confidence_threshold and prediction['prediction'] in ['UP', 'DOWN', 'PRICE_UP', 'PRICE_DOWN']:
                logger.info(f"🎯 [{symbol}] Confidence {confidence:.2%} > {self.confidence_threshold:.2%} - Executando trade!")
                await self._execute_trade_for_symbol(prediction, current_price, symbol)
            else:
                logger.info(f"⏭️ [{symbol}] Confidence insuficiente ou sem sinal: {confidence:.2%}")

        except Exception as e:
            logger.error(f"Erro ao processar {symbol}: {e}", exc_info=True)
            self._log_bug(f"symbol_processing_error_{symbol}", str(e), severity="ERROR")

    async def _trading_loop(self):
        """
        Loop principal de trading

        1. Itera sobre todos os símbolos ativos
        2. Para cada símbolo: coleta dados, gera previsão, executa trades
        3. Atualiza posições existentes
        4. Registra métricas
        """
        while self.is_running:
            try:
                # Verificar se tempo mínimo passou desde última previsão
                if self.last_prediction_time:
                    elapsed = (datetime.now() - self.last_prediction_time).total_seconds()
                    if elapsed < self.min_time_between_predictions:
                        await asyncio.sleep(1)
                        continue

                # Processar cada símbolo
                for symbol in self.symbols:
                    if not self.is_running:
                        break
                    await self._process_symbol(symbol)

                # Verificar condições de alerta (global)
                self._check_alert_conditions()

                # Aguardar antes da próxima iteração
                await asyncio.sleep(10 if self.multi_symbol_mode else 5)

            except Exception as e:
                logger.error(f"Erro no trading loop: {e}", exc_info=True)
                self._log_bug("trading_loop_error", str(e))
                await asyncio.sleep(5)

    async def _fetch_market_data_for_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Coleta dados do mercado para um símbolo específico

        Args:
            symbol: Símbolo a coletar dados

        Returns:
            Dict com dados do mercado ou None se falhar
        """
        # Temporariamente trocar o símbolo ativo
        original_symbol = self.symbol
        self.symbol = symbol

        # Chamar método original
        result = await self._fetch_market_data()

        # Restaurar símbolo original
        self.symbol = original_symbol

        return result

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

                # IMPORTANTE: Cancelar TODAS as subscrições antigas ao conectar
                # Isso previne erro "You are already subscribed to R_100"
                try:
                    await self.deriv_api._send_request({"forget_all": "ticks"})
                    logger.info("🧹 Subscrições antigas canceladas")
                except Exception as e:
                    logger.debug(f"Nenhuma subscrição antiga para cancelar: {e}")

                self.deriv_connected = True
                logger.info("✅ Conectado e autenticado na Deriv API para dados reais")

            # Obter último tick via ticks_history (NUNCA cria subscrição)
            # Mais seguro que ticks() - evita "already subscribed" definitivamente
            logger.info(f"📊 Solicitando último tick para {self.symbol}")
            response = await self.deriv_api.get_latest_tick(self.symbol)
            logger.info(f"✅ Resposta recebida da Deriv API")

            if 'history' not in response or not response['history'].get('prices'):
                logger.warning(f"Resposta sem histórico: {response}")
                self._log_bug("tick_response_invalid", f"Resposta sem histórico: {response}")
                return None

            # Extrair último tick do histórico
            history = response['history']
            prices = history['prices']
            times = history['times']

            if not prices or not times:
                logger.warning("Histórico vazio")
                return None

            current_price = float(prices[-1])
            tick_time = int(times[-1])

            logger.debug(f"✅ Tick recebido: {current_price} @ {tick_time}")

            return {
                'timestamp': datetime.fromtimestamp(tick_time).isoformat(),
                'open': current_price,
                'high': current_price,
                'low': current_price,
                'close': current_price,
                'volume': 1000,
                'symbol': self.symbol
            }

        except Exception as e:
            logger.error(f"❌ CRÍTICO: Falha ao coletar dados REAIS do mercado: {e}", exc_info=True)
            self._log_bug("market_data_fetch_error", str(e), severity="CRITICAL")

            # NÃO usar fallback mock - Forward Testing PRECISA de dados reais
            logger.error("❌ Forward Testing NÃO PODE funcionar sem dados reais da Deriv API!")
            logger.error("   Possíveis causas:")
            logger.error("   1. DERIV_API_TOKEN não configurado ou inválido")
            logger.error("   2. Deriv API está offline ou rejeitando requisições")
            logger.error("   3. Símbolo inválido ou não disponível")
            logger.error("   4. Problema de rede/conectividade")

            # Retornar None para forçar retry no próximo ciclo
            return None

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

    async def _execute_trade_for_symbol(self, prediction: Dict, current_price: float, symbol: str):
        """
        Executa trade para um símbolo específico

        Args:
            prediction: Dict com prediction e confidence
            current_price: Preço atual do mercado
            symbol: Símbolo a tradear
        """
        # Temporariamente trocar o símbolo ativo
        original_symbol = self.symbol
        self.symbol = symbol

        # Chamar método original
        await self._execute_trade(prediction, current_price)

        # Restaurar símbolo original
        self.symbol = original_symbol

    async def _execute_trade(self, prediction: Dict, current_price: float):
        """
        Executa trade baseado na previsão

        Args:
            prediction: Dict com prediction e confidence
            current_price: Preço atual do mercado
        """
        try:
            # Determinar direção
            pred_value = prediction['prediction']
            if pred_value in ['UP', 'PRICE_UP']:
                position_type = PositionType.LONG
            elif pred_value in ['DOWN', 'PRICE_DOWN']:
                position_type = PositionType.SHORT
            else:
                logger.warning(f"Previsão inválida: {pred_value}")
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
                take_profit=take_profit,
                trailing_stop_enabled=self.trailing_stop_enabled,
                trailing_stop_distance_pct=self.trailing_stop_distance_pct
            )

            if position:
                logger.info(f"✅ TRADE EXECUTADO: {position_type.value} {self.symbol} @ ${current_price:.4f}")
                logger.info(f"   Size: ${position_size:.2f} | Confidence: {prediction['confidence']:.2%}")
                logger.info(f"   SL: ${stop_loss:.4f} | TP: ${take_profit:.4f}")
                if self.trailing_stop_enabled:
                    logger.info(f"   🔄 Trailing SL: ATIVADO ({self.trailing_stop_distance_pct}% distance)")

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

    def _check_position_timeouts_for_symbol(self, symbol: str, current_price: float):
        """
        Verifica e fecha posições de um símbolo específico que excederam o tempo limite

        Args:
            symbol: Símbolo a verificar
            current_price: Preço atual do mercado
        """
        now = datetime.now()
        timeout_seconds = self.position_timeout_minutes * 60

        for position_id, position in list(self.paper_trading.positions.items()):
            if position.symbol != symbol:
                continue

            # Calcular tempo desde entrada
            position_age_seconds = (now - position.entry_time).total_seconds()

            if position_age_seconds >= timeout_seconds:
                # Calcular P&L atual
                pnl, pnl_pct = position.calculate_pnl(current_price)

                logger.info(f"⏰ [{symbol}] TIMEOUT: Posição {position_id[-8:]} aberta há {position_age_seconds/60:.1f} min")
                logger.info(f"   Fechando com P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")

                # Fechar posição com razão 'timeout'
                self.paper_trading.close_position(position_id, current_price, exit_reason='timeout')

    def _check_position_timeouts(self, current_price: float):
        """
        Verifica e fecha posições que excederam o tempo limite

        Args:
            current_price: Preço atual do mercado
        """
        now = datetime.now()
        timeout_seconds = self.position_timeout_minutes * 60

        for position_id, position in list(self.paper_trading.positions.items()):
            # Calcular tempo desde entrada
            position_age_seconds = (now - position.entry_time).total_seconds()

            if position_age_seconds >= timeout_seconds:
                # Calcular P&L atual
                pnl, pnl_pct = position.calculate_pnl(current_price)

                logger.info(f"⏰ TIMEOUT: Posição {position_id[-8:]} aberta há {position_age_seconds/60:.1f} min")
                logger.info(f"   Fechando com P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")

                # Fechar posição com razão 'timeout'
                self.paper_trading.close_position(position_id, current_price, exit_reason='timeout')

    def _check_alert_conditions(self):
        """
        Verifica condições de alerta e gera alertas apropriados
        """
        try:
            # Obter métricas atuais
            metrics = self.paper_trading.get_metrics()

            # Pegar último trade se existir
            last_trade = None
            if self.paper_trading.trade_history:
                last_trade_obj = self.paper_trading.trade_history[-1]
                last_trade = {
                    'id': last_trade_obj.id,
                    'profit_loss': last_trade_obj.profit_loss,
                    'exit_reason': getattr(last_trade_obj, 'exit_reason', None)
                }

            # Calcular timeout rate e sl hit rate
            timeout_rate_pct = 0
            sl_hit_rate_pct = 0

            if self.paper_trading.trade_history:
                total = len(self.paper_trading.trade_history)
                timeout_count = sum(1 for t in self.paper_trading.trade_history
                                   if hasattr(t, 'exit_reason') and t.exit_reason == 'timeout')
                sl_count = sum(1 for t in self.paper_trading.trade_history
                              if hasattr(t, 'exit_reason') and t.exit_reason == 'stop_loss')

                timeout_rate_pct = (timeout_count / total) * 100 if total > 0 else 0
                sl_hit_rate_pct = (sl_count / total) * 100 if total > 0 else 0

            # Verificar todas as condições
            new_alerts = self.alert_system.check_all_conditions(
                current_capital=metrics['current_capital'],
                initial_capital=metrics['initial_capital'],
                win_rate_pct=metrics['win_rate_pct'],
                timeout_rate_pct=timeout_rate_pct,
                sl_hit_rate_pct=sl_hit_rate_pct,
                last_trade=last_trade,
                total_trades=metrics['total_trades']
            )

            # Novos alertas são automaticamente adicionados ao alert_system
            # Aqui podemos apenas logar quantos foram gerados
            if new_alerts:
                logger.info(f"🔔 {len(new_alerts)} novo(s) alerta(s) gerado(s)")

        except Exception as e:
            logger.error(f"Erro ao verificar alertas: {e}", exc_info=True)

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

    def get_live_metrics(self) -> Dict:
        """
        Retorna métricas detalhadas em tempo real para dashboard

        Returns:
            Dict com métricas expandidas incluindo equity curve
        """
        metrics = self.paper_trading.get_metrics()

        # Equity Curve (últimos 100 pontos)
        equity_history = self.paper_trading.equity_curve[-100:] if self.paper_trading.equity_curve else []

        # Calcular métricas de execução
        total_positions = len(self.paper_trading.trade_history)
        closed_positions = self.paper_trading.trade_history

        # Taxa de timeout (trades fechados por timeout vs SL/TP)
        timeout_trades = 0
        sl_trades = 0
        tp_trades = 0

        for trade in closed_positions:
            # Determinar razão de fechamento baseado no P&L e tempo
            if hasattr(trade, 'exit_reason'):
                if trade.exit_reason == 'timeout':
                    timeout_trades += 1
                elif trade.exit_reason == 'stop_loss':
                    sl_trades += 1
                elif trade.exit_reason == 'take_profit':
                    tp_trades += 1

        timeout_rate = (timeout_trades / len(closed_positions)) if closed_positions else 0
        sl_hit_rate = (sl_trades / len(closed_positions)) if closed_positions else 0
        tp_hit_rate = (tp_trades / len(closed_positions)) if closed_positions else 0

        # Duração média de trades
        avg_duration = 0
        if closed_positions:
            durations = [(t.exit_time - t.entry_time).total_seconds() / 60 for t in closed_positions if t.exit_time]
            avg_duration = sum(durations) / len(durations) if durations else 0

        # Win rate por período (últimas 10, 20, 50 trades)
        win_rates = {}
        for period in [10, 20, 50]:
            recent_trades = closed_positions[-period:] if len(closed_positions) >= period else closed_positions
            if recent_trades:
                wins = sum(1 for t in recent_trades if t.profit_loss > 0)
                win_rates[f'last_{period}'] = (wins / len(recent_trades)) * 100

        return {
            # Métricas básicas
            'capital': {
                'current': metrics['current_capital'],
                'initial': metrics['initial_capital'],
                'peak': max(equity_history) if equity_history else metrics['initial_capital'],
                'pnl': metrics['total_pnl'],
                'pnl_pct': metrics['total_pnl_pct']
            },

            # Equity curve
            'equity_curve': equity_history,

            # Performance
            'performance': {
                'win_rate': metrics['win_rate_pct'],
                'win_rates_by_period': win_rates,
                'sharpe_ratio': metrics['sharpe_ratio'],
                'profit_factor': metrics['profit_factor'],
                'max_drawdown_pct': metrics['max_drawdown_pct']
            },

            # Execution quality
            'execution': {
                'total_trades': total_positions,
                'open_positions': len(self.paper_trading.positions),
                'closed_positions': len(closed_positions),
                'avg_duration_minutes': round(avg_duration, 2),
                'timeout_rate': round(timeout_rate * 100, 1),
                'sl_hit_rate': round(sl_hit_rate * 100, 1),
                'tp_hit_rate': round(tp_hit_rate * 100, 1)
            },

            # Breakdown
            'trades_breakdown': {
                'winning': metrics['winning_trades'],
                'losing': metrics['losing_trades'],
                'by_exit_reason': {
                    'timeout': timeout_trades,
                    'stop_loss': sl_trades,
                    'take_profit': tp_trades
                }
            },

            # Timestamp
            'updated_at': datetime.now().isoformat()
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
        try:
            logger.info("🚀 Inicializando Forward Testing Engine...")
            _forward_testing_instance = ForwardTestingEngine()
            logger.info("✅ Forward Testing Engine inicializado com sucesso")
        except FileNotFoundError as e:
            logger.error(f"❌ CRÍTICO: Modelo ML não encontrado: {e}")
            logger.error("   Procurar por: backend/ml/models/xgboost_improved_learning_rate_*.pkl")
            logger.error("   O Forward Testing NÃO PODE funcionar sem o modelo ML!")
            raise
        except Exception as e:
            logger.error(f"❌ CRÍTICO: Falha ao inicializar Forward Testing Engine: {e}", exc_info=True)
            raise
    return _forward_testing_instance
