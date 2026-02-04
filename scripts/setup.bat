@echo off
setlocal enabledelayedexpansion

echo ========================================
echo        Moltbot Setup Script
echo ========================================
echo.

:: Check Python
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10+ first.
    pause
    exit /b 1
)
python --version

:: Create virtual environment
echo.
echo [2/5] Creating virtual environment...
cd /d "%~dp0\.."
if exist "venv" (
    echo Virtual environment already exists.
) else (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created.
)

:: Activate and install dependencies
echo.
echo [3/5] Installing Python dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)
echo Dependencies installed.

:: Check for llama.cpp
echo.
echo [4/5] Checking llama.cpp installation...
where llama-server >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: llama-server not found in PATH.
    echo.
    echo You need to install llama.cpp with Vulkan support:
    echo   1. Download from: https://github.com/ggerganov/llama.cpp/releases
    echo   2. Get the 'vulkan' build for Windows
    echo   3. Extract and add to PATH, or put llama-server.exe in this folder
    echo.
    echo Or build from source with Vulkan:
    echo   git clone https://github.com/ggerganov/llama.cpp
    echo   cd llama.cpp
    echo   cmake -B build -DGGML_VULKAN=ON
    echo   cmake --build build --config Release
    echo.
) else (
    echo llama-server found in PATH.
)

:: Download models prompt
echo.
echo [5/5] Model download...
echo.
echo Would you like to download the models now?
echo   - Qwen2.5-7B-Instruct Q5 (~5.4 GB) - forecaster
echo   - Qwen3-Coder-30B-A3B Q4 (~18.6 GB) - coder
echo   Total: ~24 GB
echo.
set /p DOWNLOAD="Download models? [Y/n]: "
if /i "!DOWNLOAD!"=="" set DOWNLOAD=Y
if /i "!DOWNLOAD!"=="Y" (
    python scripts\download_models.py all
)

:: Final instructions
echo.
echo ========================================
echo            Setup Complete!
echo ========================================
echo.
echo Next steps:
echo.
echo 1. Configure Telegram bot:
echo    - Talk to @BotFather on Telegram
echo    - Create a new bot with /newbot
echo    - Copy the token to config.yaml
echo    - Get your user ID (talk to @userinfobot)
echo    - Add your ID to allowed_user_ids in config.yaml
echo.
echo 2. Install llama.cpp (if not done):
echo    - Download Vulkan build from GitHub releases
echo    - Add to PATH or copy to this folder
echo.
echo 3. Run Moltbot:
echo    venv\Scripts\activate
echo    python run.py
echo.
echo Or use the start script:
echo    start.bat
echo.
pause
