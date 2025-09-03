@echo off
echo 🔧 Reparando pip - Synth Bot Buddy
echo ===================================

REM Try to repair pip installation
py --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Python encontrado
    echo 🔧 Instalando/reparando pip...
    py -m ensurepip --upgrade
    py -m pip install --upgrade pip
    echo.
    echo ✅ pip reparado!
    echo 💡 Execute agora: .\install.bat
    pause
    goto :end
)

python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Python encontrado  
    echo 🔧 Instalando/reparando pip...
    python -m ensurepip --upgrade
    python -m pip install --upgrade pip
    echo.
    echo ✅ pip reparado!
    echo 💡 Execute agora: .\install.bat
    pause
    goto :end
)

echo ❌ Python não encontrado!
echo 💡 Instale Python primeiro
pause

:end