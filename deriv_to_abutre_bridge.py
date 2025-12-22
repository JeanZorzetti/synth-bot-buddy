#!/usr/bin/env python3
"""
DERIV → ABUTRE BRIDGE
Conecta na Deriv API, recebe ticks reais de 1HZ100V e envia para API Abutre
"""
import asyncio
import websockets
import json
import requests
from datetime import datetime
from collections import deque
from typing import Deque, Optional
import logging

# Configuração
DERIV_WS_URL = "wss://ws.derivws.com/websockets/v3?app_id=1089"
ABUTRE_API_URL = "https://botderivapi.roilabs.com.br/api/abutre/events"
SYMBOL = "1HZ100V"
STREAK_THRESHOLD = 8

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DerivAbutreBridge:
    def __init__(self):
        self.candle_history: Deque = deque(maxlen=20)  # Últimas 20 candles
        self.current_streak = 0
        self.last_color = None
        self.last_candle_time = None
        self.connection = None

    async def connect_deriv(self):
        """Conecta no WebSocket da Deriv"""
        logger.info(f"Conectando na Deriv API: {DERIV_WS_URL}")
        self.connection = await websockets.connect(DERIV_WS_URL)
        logger.info("✅ Conectado na Deriv API")

        # Assinar ticks de 1 segundo
        subscribe_msg = {
            "ticks": SYMBOL,
            "subscribe": 1
        }
        await self.connection.send(json.dumps(subscribe_msg))
        logger.info(f"📊 Assinado em {SYMBOL} (ticks de 1s)")

    async def process_tick(self, tick_data: dict):
        """Processa cada tick recebido da Deriv"""
        try:
            quote = tick_data['tick']['quote']
            epoch = tick_data['tick']['epoch']
            timestamp = datetime.utcfromtimestamp(epoch)

            # Simular candle de 1 segundo (OHLC = mesmo valor)
            candle = {
                'timestamp': timestamp.isoformat() + 'Z',
                'symbol': SYMBOL,
                'open': quote,
                'high': quote,
                'low': quote,
                'close': quote,
                'color': self.calculate_color(quote)
            }

            # Adicionar ao histórico
            self.candle_history.append(candle)

            # Atualizar streak
            self.update_streak(candle['color'])

            # Enviar candle para API
            self.send_candle(candle)

            # Verificar trigger de Abutre (8+ streak)
            if self.current_streak >= STREAK_THRESHOLD:
                self.send_trigger(timestamp, self.current_streak, self.last_color)

            logger.info(
                f"📈 Tick: {quote:.2f} | "
                f"Color: {'🟢' if candle['color'] == 1 else '🔴'} | "
                f"Streak: {self.current_streak}"
            )

        except Exception as e:
            logger.error(f"Erro ao processar tick: {e}")

    def calculate_color(self, current_price: float) -> int:
        """Calcula cor do candle (1 = GREEN, -1 = RED)"""
        if len(self.candle_history) == 0:
            return 1  # Primeira candle = verde por padrão

        last_close = self.candle_history[-1]['close']
        return 1 if current_price >= last_close else -1

    def update_streak(self, color: int):
        """Atualiza contagem de streak"""
        if color == self.last_color:
            self.current_streak += 1
        else:
            self.current_streak = 1
            self.last_color = color

    def send_candle(self, candle: dict):
        """Envia candle para API Abutre"""
        try:
            response = requests.post(
                f"{ABUTRE_API_URL}/candle",
                json=candle,
                timeout=5
            )
            if response.status_code != 201:
                logger.warning(f"Erro ao enviar candle: {response.status_code}")
        except Exception as e:
            logger.error(f"Erro ao enviar candle: {e}")

    def send_trigger(self, timestamp: datetime, streak_count: int, direction: str):
        """Envia trigger de Abutre (8+ streak)"""
        try:
            direction_str = "GREEN" if direction == 1 else "RED"

            trigger = {
                'timestamp': timestamp.isoformat() + 'Z',
                'streak_count': streak_count,
                'direction': direction_str
            }

            response = requests.post(
                f"{ABUTRE_API_URL}/trigger",
                json=trigger,
                timeout=5
            )

            if response.status_code == 201:
                logger.warning(
                    f"🚨 TRIGGER ABUTRE! {streak_count} {direction_str} candles consecutivas"
                )
            else:
                logger.error(f"Erro ao enviar trigger: {response.status_code}")

        except Exception as e:
            logger.error(f"Erro ao enviar trigger: {e}")

    async def run(self):
        """Loop principal"""
        try:
            await self.connect_deriv()

            logger.info("🔄 Aguardando ticks da Deriv...")
            logger.info(f"📍 Monitorando {SYMBOL} para streaks de {STREAK_THRESHOLD}+")
            logger.info(f"🌐 Enviando dados para: {ABUTRE_API_URL}")
            logger.info("")

            async for message in self.connection:
                data = json.loads(message)

                # Processar apenas mensagens de tick
                if 'tick' in data:
                    await self.process_tick(data)

                # Verificar erros
                if 'error' in data:
                    logger.error(f"Erro da Deriv: {data['error']}")

        except websockets.exceptions.ConnectionClosed:
            logger.error("⚠️ Conexão com Deriv fechada. Reconectando em 5s...")
            await asyncio.sleep(5)
            await self.run()

        except KeyboardInterrupt:
            logger.info("\n👋 Desconectando...")
            if self.connection:
                await self.connection.close()

        except Exception as e:
            logger.error(f"Erro fatal: {e}")
            raise


async def main():
    """Ponto de entrada"""
    logger.info("=" * 60)
    logger.info("DERIV → ABUTRE BRIDGE")
    logger.info("=" * 60)
    logger.info("")

    bridge = DerivAbutreBridge()
    await bridge.run()


if __name__ == "__main__":
    asyncio.run(main())
