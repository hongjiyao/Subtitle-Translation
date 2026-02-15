@echo off
echo ========================================
echo Subtitle Translation Tool - Setup
echo ========================================
echo.

set "PYTHON_DIR=python311"
set "PYTHON_INSTALLER=python-3.11.7-amd64.exe"
set "PYTHON_URL=https://www.python.org/ftp/python/3.11.7/%PYTHON_INSTALLER%"

echo Starting installer...
echo.

:: Check Python
echo Checking Python...
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

:: Run Python setup script
echo [INFO] Running setup_all.py...
echo.
python setup_all.py

:: Check result
if errorlevel 1 (
    echo.
    echo [ERROR] Installation failed
    pause
    exit /b 1
) else (
    echo.
    echo [SUCCESS] Installation complete!
)

echo.
echo ========================================
echo What would you like to do next?
echo ========================================
echo.
echo [1] Download all models (download_all_models.py)
echo [2] Run the UI (ui.py)
echo [3] Do both (download models then run UI)
echo [4] Exit
echo.
set /p CHOICE="Enter your choice (1-4): "

if "%CHOICE%"=="1" goto download_models
if "%CHOICE%"=="2" goto run_ui
if "%CHOICE%"=="3" goto both
if "%CHOICE%"=="4" goto end
echo Invalid choice, exiting...
goto end

:download_models
echo.
echo ========================================
echo Downloading all models...
echo ========================================
echo.
python download_all_models.py
if errorlevel 1 (
    echo.
    echo [ERROR] Model download failed
) else (
    echo.
    echo [SUCCESS] Models downloaded!
)
goto end

:run_ui
echo.
echo ========================================
echo Starting UI...
echo ========================================
echo.
python ui.py
goto end

:both
echo.
echo ========================================
echo Step 1: Downloading all models...
echo ========================================
echo.
python download_all_models.py
if errorlevel 1 (
    echo.
    echo [ERROR] Model download failed
    echo.
    echo Starting UI anyway...
) else (
    echo.
    echo [SUCCESS] Models downloaded!
)
echo.
echo ========================================
echo Step 2: Starting UI...
echo ========================================
echo.
python ui.py
goto end

:end
echo.
echo ========================================
echo.
pause
