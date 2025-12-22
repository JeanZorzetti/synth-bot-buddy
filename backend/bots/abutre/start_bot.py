"""
ABUTRE BOT - Startup Script

Wrapper para iniciar o bot Abutre corretamente resolvendo imports
"""
import sys
from pathlib import Path

# Adicionar o diretório backend ao path para resolver imports
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

# Agora podemos importar o bot
if __name__ == "__main__":
    import argparse

    # Parse arguments
    parser = argparse.ArgumentParser(description='Abutre Trading Bot')
    parser.add_argument('--demo', action='store_true', help='Use demo account')
    parser.add_argument('--paper-trading', action='store_true', help='Monitor only, no execution')
    parser.add_argument('--port', type=int, default=8000, help='WebSocket server port')
    args = parser.parse_args()

    # Import após adicionar ao path
    from bots.abutre.main import AbutreBot
    import asyncio

    # Criar e iniciar bot
    bot = AbutreBot(
        demo_mode=args.demo,
        paper_trading=args.paper_trading,
        ws_port=args.port
    )

    # Rodar asyncio loop
    asyncio.run(bot.run())
