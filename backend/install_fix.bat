@echo off
REM Script de instalação específico para Python 3.13
REM Usa o caminho completo do Python

echo.
echo ========================================
echo   Instalando Dependencias - Python 3.13
echo ========================================
echo.

REM Caminho completo do Python
set PYTHON_PATH=C:\Python313\python.exe

REM Verificar se existe
if not exist "%PYTHON_PATH%" (
    echo ERRO: Python nao encontrado em %PYTHON_PATH%
    echo.
    echo Verifique o caminho da instalacao do Python
    pause
    exit /b 1
)

echo [1/4] Verificando Python...
"%PYTHON_PATH%" --version

echo.
echo [2/4] Atualizando pip...
"%PYTHON_PATH%" -m pip install --upgrade pip

echo.
echo [3/4] Instalando dependencias essenciais...
"%PYTHON_PATH%" -m pip install websockets ujson

echo.
echo [4/4] Verificando instalacao...
"%PYTHON_PATH%" -c "import websockets; import ujson; print('✓ websockets OK'); print('✓ ujson OK')"

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo   ✓ Instalacao Concluida com Sucesso!
    echo ========================================
    echo.
    echo Agora execute:
    echo   C:\Python313\python.exe test_simple_order.py
    echo.
) else (
    echo.
    echo ========================================
    echo   ✗ Erro na Instalacao
    echo ========================================
    echo.
)

pause
