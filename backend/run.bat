@echo off
echo 🚀 Synth Bot Buddy Backend
echo =========================

REM Try different Python commands
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Iniciando com python
    python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
    goto :end
)

python3 --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Iniciando com python3
    python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
    goto :end
)

py --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Iniciando com py launcher
    py -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
    goto :end
)

echo ❌ Python não encontrado!
echo 💡 Execute primeiro: install.bat
pause

:end