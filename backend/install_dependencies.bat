@echo off
REM Script para instalar dependências do backend
REM Execute este arquivo antes de rodar o test_simple_order.py

echo.
echo ========================================
echo   Instalando Dependencias do Backend
echo ========================================
echo.

REM Verificar se Python está instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERRO: Python nao encontrado!
    echo Por favor, instale Python 3.11+ de https://python.org
    pause
    exit /b 1
)

echo [1/3] Verificando versao do Python...
python --version

echo.
echo [2/3] Instalando dependencias minimas...
python -m pip install --upgrade pip
python -m pip install websockets ujson fastapi uvicorn pydantic python-dotenv

echo.
echo [3/3] Verificando instalacao...
python -c "import websockets; import ujson; print('✓ websockets OK'); print('✓ ujson OK')"

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo   ✓ Instalacao Concluida com Sucesso!
    echo ========================================
    echo.
    echo Agora voce pode executar:
    echo   python test_simple_order.py
    echo.
) else (
    echo.
    echo ========================================
    echo   ✗ Erro na Instalacao
    echo ========================================
    echo.
    echo Tente instalar manualmente:
    echo   pip install -r requirements.txt
    echo.
)

pause
