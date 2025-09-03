#!/usr/bin/env python3
"""
Startup script para o Synth Bot Buddy Backend
Facilita a inicialização com verificações de ambiente
"""

import sys
import os
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all requirements are installed"""
    try:
        import fastapi
        import uvicorn
        import websockets
        import pydantic
        import dotenv
        print("✅ Todas as dependências estão instaladas")
        return True
    except ImportError as e:
        print(f"❌ Dependência faltando: {e}")
        print("💡 Execute: pip install -r requirements.txt")
        return False

def check_env_file():
    """Check if .env file exists (optional now - tokens are set dynamically)"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists():
        print("ℹ️  Arquivo .env não encontrado (opcional)")
        print("💡 Tokens são configurados dinamicamente via frontend")
        if env_example.exists():
            print("💡 Para configuração manual: cp .env.example .env")
        return True  # Not required anymore
    
    # Check if token is configured (optional)
    with open(env_file, 'r') as f:
        content = f.read()
        if "your_deriv_api_token_here" in content:
            print("ℹ️  Token não configurado no .env (será usado token do frontend)")
        else:
            print("✅ Arquivo .env configurado")
    
    return True

def main():
    print("🚀 Iniciando Synth Bot Buddy Backend...")
    print("=" * 50)
    
    # Already in backend directory, no need to change
    if not Path("main.py").exists():
        print("❌ main.py não encontrado no diretório atual")
        print("💡 Execute o script a partir do diretório backend/")
        return 1
    
    print(f"📁 Diretório atual: {Path.cwd()}")
    
    # Check requirements
    print("\n1. Verificando dependências...")
    if not check_requirements():
        return 1
    
    # Check environment
    print("\n2. Verificando configuração...")
    if not check_env_file():
        return 1
    
    # Start server
    print("\n3. Iniciando servidor...")
    print("🌐 Servidor estará disponível em: http://localhost:8000")
    print("📚 Documentação da API: http://localhost:8000/docs")
    print("🛑 Para parar o servidor: Ctrl+C")
    print("=" * 50)
    
    try:
        # Run with uvicorn
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--reload", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Servidor parado pelo usuário")
    except FileNotFoundError:
        print("❌ uvicorn não encontrado. Instale com: pip install uvicorn[standard]")
        return 1
    except Exception as e:
        print(f"❌ Erro ao iniciar servidor: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())