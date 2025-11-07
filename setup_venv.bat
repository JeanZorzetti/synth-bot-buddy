@echo off
REM Script para criar ambiente virtual e instalar dependências
REM Resolve problemas de instalação do Python

echo.
echo ============================================================
echo   SETUP: Criando Ambiente Virtual + Instalando Dependencias
echo ============================================================
echo.

cd /d "%~dp0"

REM Verificar se venv já existe
if exist ".venv" (
    echo [!] Ambiente virtual ja existe em .venv
    echo [!] Removendo ambiente antigo...
    rmdir /s /q .venv
)

echo [1/5] Criando ambiente virtual...
C:\Python313\python.exe -m venv .venv

if %errorlevel% neq 0 (
    echo.
    echo ERRO ao criar ambiente virtual!
    echo.
    echo Alternativa: Reinstalar Python
    echo 1. Desinstale Python atual
    echo 2. Baixe de: https://www.python.org/downloads/
    echo 3. Durante instalacao, marque "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

echo [2/5] Ativando ambiente virtual...
call .venv\Scripts\activate.bat

echo [3/5] Atualizando pip...
python -m pip install --upgrade pip

echo [4/5] Instalando dependencias essenciais...
pip install websockets ujson fastapi uvicorn pydantic python-dotenv aiohttp

echo [5/5] Verificando instalacao...
python -c "import websockets; import ujson; print('✓ websockets OK'); print('✓ ujson OK')"

if %errorlevel% equ 0 (
    echo.
    echo ============================================================
    echo   ✓ Setup Concluido com Sucesso!
    echo ============================================================
    echo.
    echo COMO USAR:
    echo.
    echo 1. Ative o ambiente virtual:
    echo    .venv\Scripts\activate
    echo.
    echo 2. Execute o teste:
    echo    cd backend
    echo    python test_simple_order.py
    echo.
    echo 3. Ou rode o backend:
    echo    cd backend
    echo    python start.py
    echo.
) else (
    echo.
    echo ============================================================
    echo   ✗ Erro na Instalacao
    echo ============================================================
    echo.
)

pause
