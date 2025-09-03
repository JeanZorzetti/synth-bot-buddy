#!/usr/bin/env python3
"""
Script para instalar dependências do backend
"""

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install Python requirements"""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("❌ requirements.txt não encontrado")
        return False
    
    try:
        print("📦 Instalando dependências...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependências instaladas com sucesso!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao instalar dependências: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return False

def main():
    print("🔧 Instalação de Dependências - Synth Bot Buddy")
    print("=" * 50)
    
    if install_requirements():
        print("\n✅ Instalação concluída!")
        print("💡 Agora você pode executar: python start.py")
    else:
        print("\n❌ Instalação falhou")
        print("💡 Tente instalar manualmente: pip install -r requirements.txt")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())