@echo off
echo 🔧 Instalação de Dependências - Synth Bot Buddy
echo ===============================================

REM Try different Python commands
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Usando python
    python -m pip install -r requirements.txt
    goto :success
)

python3 --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Usando python3
    python3 -m pip install -r requirements.txt
    goto :success
)

py --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Usando py launcher
    py -m pip install -r requirements.txt
    goto :success
)

echo ❌ Python não encontrado!
echo 💡 Instale Python de:
echo    - https://python.org
echo    - Microsoft Store (python)
echo    - Anaconda
pause
goto :end

:success
echo.
echo ✅ Dependências instaladas!
echo 💡 Execute agora: run.bat
pause

:end