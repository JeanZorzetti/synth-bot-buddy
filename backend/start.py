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
    """Check if .env file exists and is configured"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists():
        if env_example.exists():
            print("⚠️  Arquivo .env não encontrado")
            print("💡 Copie .env.example para .env e configure seu token:")
            print("   cp .env.example .env")
            return False
        else:
            print("❌ Arquivos de configuração não encontrados")
            return False
    
    # Check if token is configured
    with open(env_file, 'r') as f:
        content = f.read()
        if "your_deriv_api_token_here" in content:
            print("⚠️  Token da API não configurado no arquivo .env")
            print("💡 Configure seu DERIV_API_TOKEN no arquivo .env")
            return False
    
    print("✅ Arquivo .env configurado")
    return True

def main():
    print("🚀 Iniciando Synth Bot Buddy Backend...")
    print("=" * 50)
    
    # Change to backend directory if not already there
    if not Path("main.py").exists():
        backend_path = Path("backend")
        if backend_path.exists():
            os.chdir(backend_path)
            print(f"📁 Mudando para diretório: {backend_path.absolute()}")
    
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