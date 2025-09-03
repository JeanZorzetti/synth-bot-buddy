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
    REM Try different pip methods
    py -m pip install -r requirements.txt >nul 2>&1
    if %errorlevel% neq 0 (
        echo ⚠️ pip via py falhou, tentando ensurepip...
        py -m ensurepip --upgrade >nul 2>&1
        py -m pip install -r requirements.txt >nul 2>&1
        if %errorlevel% neq 0 (
            echo ⚠️ Tentando pip direto...
            pip install -r requirements.txt >nul 2>&1
        )
    )
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