#!/usr/bin/env python3
"""
Script para instalar dependências do backend
Funciona com python, python3 ou py launcher
"""

import subprocess
import sys
import shutil
from pathlib import Path

def find_python_command():
    """Find available Python command"""
    commands = ["python", "python3", "py"]
    
    for cmd in commands:
        if shutil.which(cmd):
            try:
                # Test if it works
                result = subprocess.run([cmd, "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ Encontrado: {cmd} ({result.stdout.strip()})")
                    return cmd
            except:
                continue
    
    return None

def install_requirements(python_cmd):
    """Install Python requirements"""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("❌ requirements.txt não encontrado")
        return False
    
    try:
        print(f"📦 Instalando dependências com {python_cmd}...")
        subprocess.check_call([
            python_cmd, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependências instaladas com sucesso!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao instalar dependências: {e}")
        print("💡 Tente: pip install fastapi uvicorn[standard] websockets pydantic python-dotenv requests")
        return False
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return False

def main():
    print("🔧 Instalação de Dependências - Synth Bot Buddy")
    print("=" * 50)
    
    # Find Python command
    python_cmd = find_python_command()
    if not python_cmd:
        print("❌ Python não encontrado!")
        print("💡 Instale Python de:")
        print("   - https://python.org")
        print("   - Microsoft Store (search: python)")
        print("   - Anaconda")
        input("Pressione Enter para sair...")
        return 1
    
    if install_requirements(python_cmd):
        print("\n✅ Instalação concluída!")
        print(f"💡 Execute: {python_cmd} run.py")
        print("💡 Ou use: run.bat (Windows)")
    else:
        print("\n❌ Instalação falhou")
        print("💡 Tente manualmente ou use install.bat")
        return 1
    
    input("Pressione Enter para continuar...")
    return 0

if __name__ == "__main__":
    sys.exit(main())