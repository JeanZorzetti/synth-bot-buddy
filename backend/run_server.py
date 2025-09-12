#!/usr/bin/env python3
"""
Script simples para executar o servidor Deriv Trading Bot
"""
import uvicorn
import os
from pathlib import Path

if __name__ == "__main__":
    # Definir configurações do servidor
    host = "0.0.0.0"
    port = 8000
    
    print("Iniciando Synth Bot Buddy Backend...")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print("="*50)
    
    # Executar o servidor
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,  # Auto-reload durante desenvolvimento
        log_level="info"
    )