#!/usr/bin/env python3
"""
Script simples para executar o backend
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("🚀 Synth Bot Buddy Backend")
    print("=" * 30)
    
    # Check if main.py exists
    if not Path("main.py").exists():
        print("❌ main.py não encontrado")
        print("💡 Execute este script do diretório backend/")
        return 1
    
    try:
        # Run with uvicorn
        print("🌐 Iniciando servidor em http://localhost:8000")
        print("📚 Documentação: http://localhost:8000/docs")
        print("🛑 Para parar: Ctrl+C")
        print("-" * 30)
        
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--reload", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Servidor parado")
    except FileNotFoundError:
        print("❌ uvicorn não encontrado")
        print("💡 Instale: pip install uvicorn[standard]")
        return 1
    except Exception as e:
        print(f"❌ Erro: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())