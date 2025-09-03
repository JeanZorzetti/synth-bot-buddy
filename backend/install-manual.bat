@echo off
echo 🔧 Instalação Manual - Synth Bot Buddy
echo =======================================

echo 📦 Instalando dependências individualmente...
echo.

REM Find working Python command
set PYTHON_CMD=
py --version >nul 2>&1
if %errorlevel% equ 0 set PYTHON_CMD=py

if "%PYTHON_CMD%"=="" (
    python --version >nul 2>&1
    if %errorlevel% equ 0 set PYTHON_CMD=python
)

if "%PYTHON_CMD%"=="" (
    echo ❌ Python não encontrado!
    pause
    exit /b 1
)

echo ✅ Usando: %PYTHON_CMD%
echo.

echo 📦 Instalando FastAPI...
%PYTHON_CMD% -m pip install fastapi
echo.

echo 📦 Instalando Uvicorn...  
%PYTHON_CMD% -m pip install uvicorn[standard]
echo.

echo 📦 Instalando WebSockets...
%PYTHON_CMD% -m pip install websockets
echo.

echo 📦 Instalando Pydantic...
%PYTHON_CMD% -m pip install pydantic
echo.

echo 📦 Instalando python-dotenv...
%PYTHON_CMD% -m pip install python-dotenv
echo.

echo 📦 Instalando requests...
%PYTHON_CMD% -m pip install requests
echo.

echo ✅ Instalação concluída!
echo 💡 Execute: .\run.bat
pause