@echo off
echo ========================================
echo Subtitle Translation Tool - Final Setup
echo ========================================
echo.

set "VENV_NAME=.venv_final"
set "PYTHON_DIR=python311"
set "PYTHON_INSTALLER=python-3.11.7-amd64.exe"
set "PYTHON_URL=https://www.python.org/ftp/python/3.11.7/%PYTHON_INSTALLER%"

echo [INFO] Starting setup...
echo.

:: Step 1: Clean old environment
echo [1/7] Cleaning old environment...
if exist "%VENV_NAME%" (
    echo Found old environment, removing...
    rmdir /s /q "%VENV_NAME%"
    timeout /t 1 /nobreak >nul
)
echo [OK] Cleanup complete
echo.

:: Step 2: Check and install Python
echo [2/7] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found, checking local installation...
    if exist "%PYTHON_DIR%\python.exe" (
        echo [OK] Found local Python at %PYTHON_DIR%
        set "PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%PATH%"
    ) else (
        echo [INFO] Downloading Python 3.11.7...
        echo This may take a few minutes...
        
        set "PYTHON_DOWNLOADED=0"
        
        :: Source 1: Official Python.org
        echo Trying source 1: Python.org...
        powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_INSTALLER%'"
        if exist "%PYTHON_INSTALLER%" (
            set "PYTHON_DOWNLOADED=1"
            echo [OK] Python installer downloaded
        )
        
        :: Source 2: Mirror if needed
        if not "%PYTHON_DOWNLOADED%"=="1" (
            echo Trying source 2: Mirror...
            powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://registry.npmmirror.com/-/binary/python/3.11.7/%PYTHON_INSTALLER%' -OutFile '%PYTHON_INSTALLER%'"
            if exist "%PYTHON_INSTALLER%" (
                set "PYTHON_DOWNLOADED=1"
                echo [OK] Python installer downloaded from mirror
            )
        )
        
        :: If download succeeded, install
        if "%PYTHON_DOWNLOADED%"=="1" (
            echo Installing Python 3.11.7...
            echo Please wait, this may take a few minutes...
            start /wait "" "%PYTHON_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=0 Include_test=0 TargetDir="%~dp0%PYTHON_DIR%"
            if exist "%PYTHON_DIR%\python.exe" (
                echo [OK] Python installed successfully to %PYTHON_DIR%
                set "PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%PATH%"
                del "%PYTHON_INSTALLER%"
            ) else (
                echo [ERROR] Python installation failed
                pause
                exit /b 1
            )
        ) else (
            echo [ERROR] Failed to download Python
            echo Please install Python 3.8+ manually from https://www.python.org/downloads/
            pause
            exit /b 1
        )
    )
)
for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo [OK] %%i
echo.

:: Step 3: Create venv
echo [3/7] Creating virtual environment: %VENV_NAME%
python -m venv "%VENV_NAME%"
if errorlevel 1 (
    echo [ERROR] Failed to create venv
    pause
    exit /b 1
)
echo [OK] Virtual environment created
echo.

:: Step 4: Activate and upgrade pip
echo [4/7] Activating and upgrading pip...
call "%VENV_NAME%\Scripts\activate.bat"
python -m pip install --upgrade pip
echo [OK] pip upgraded
echo.

:: Step 5: Install packages
echo [5/7] Installing packages...
echo This may take a few minutes...
echo.

:: Install key packages
echo Installing main packages...
pip install gradio torch transformers

:: Try to install whisper (may fail due to permissions)
echo.
echo Installing whisper...
pip install openai-whisper 2>nul || echo [WARN] whisper may need manual install

:: Install other packages
echo.
echo Installing other packages...
pip install ffmpy ImageIO imageio-ffmpeg moviepy numpy pandas pydantic requests tqdm rich

echo [OK] Installation complete
echo.

:: Step 6: Verify installation
echo [6/7] Verifying packages...
set OK_COUNT=0
set TOTAL_COUNT=0

echo Checking gradio...
python -c "import gradio; print('  OK: gradio')" 2>nul
if errorlevel 1 (
    echo   FAIL: gradio
) else (
    set /a OK_COUNT+=1
)
set /a TOTAL_COUNT+=1

echo Checking torch...
python -c "import torch; print('  OK: torch')" 2>nul
if errorlevel 1 (
    echo   FAIL: torch
) else (
    set /a OK_COUNT+=1
)
set /a TOTAL_COUNT+=1

echo Checking transformers...
python -c "import transformers; print('  OK: transformers')" 2>nul
if errorlevel 1 (
    echo   FAIL: transformers
) else (
    set /a OK_COUNT+=1
)
set /a TOTAL_COUNT+=1

echo.
echo [RESULT] %OK_COUNT%/%TOTAL_COUNT% packages OK
echo.

:: Step 7: Complete
echo [7/7] Setup complete!
echo.
echo ========================================
if %OK_COUNT% geq 2 (
    echo Setup successful!
) else (
    echo Setup complete, some packages may need manual install
)
echo ========================================
echo.
echo Python directory: %PYTHON_DIR%
echo Virtual environment: %VENV_NAME%
echo.
echo To use:
echo   1. Activate: %VENV_NAME%\Scripts\activate.bat
echo   2. Run: python ui.py
echo.
echo ========================================
echo.
echo Setup complete! You can now activate the environment.
echo.
pause
