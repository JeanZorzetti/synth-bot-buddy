#!/bin/bash
# Script para instalar dependências do backend
# Execute este arquivo antes de rodar o test_simple_order.py

echo ""
echo "========================================"
echo "  Instalando Dependências do Backend"
echo "========================================"
echo ""

# Verificar se Python está instalado
if ! command -v python3 &> /dev/null; then
    echo "❌ ERRO: Python não encontrado!"
    echo "Por favor, instale Python 3.11+ de https://python.org"
    exit 1
fi

echo "[1/3] Verificando versão do Python..."
python3 --version

echo ""
echo "[2/3] Instalando dependências mínimas..."
python3 -m pip install --upgrade pip
python3 -m pip install websockets ujson fastapi uvicorn pydantic python-dotenv

echo ""
echo "[3/3] Verificando instalação..."
python3 -c "import websockets; import ujson; print('✓ websockets OK'); print('✓ ujson OK')"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "  ✓ Instalação Concluída com Sucesso!"
    echo "========================================"
    echo ""
    echo "Agora você pode executar:"
    echo "  python3 test_simple_order.py"
    echo ""
else
    echo ""
    echo "========================================"
    echo "  ✗ Erro na Instalação"
    echo "========================================"
    echo ""
    echo "Tente instalar manualmente:"
    echo "  pip3 install -r requirements.txt"
    echo ""
fi
