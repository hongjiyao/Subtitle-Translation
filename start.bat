@echo off
echo ========================================
echo Subtitle Translation Tool - Setup
echo ========================================
echo.

set "PYTHON_DIR=python311"
set "PYTHON_INSTALLER=python-3.11.7-amd64.exe"
set "PYTHON_URL=https://www.python.org/ftp/python/3.11.7/%PYTHON_INSTALLER%"
set "VC_REDIST_INSTALLER=VC_redist.x64.exe"
set "VC_REDIST_URL=https://download.visualstudio.microsoft.com/download/pr/6f02464a-5e9b-486d-a506-c99a17db9a83/8995548DFFFCDE7C49987029C764355612BA6850EE09A7B6F0FDDC85BDC5C280/VC_redist.x64.exe"

echo Starting installer...
echo.

:: Check and install VC Redist
echo Checking Visual C++ Redistributable...
reg query "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" >nul 2>&1
if errorlevel 1 (
    echo VC Redist not found, checking alternative registry key...
    reg query "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\{e2803110-78b3-4664-a479-3611a381656a}" >nul 2>&1
    if errorlevel 1 (
        echo [INFO] Downloading VC Redist...
        echo This may take a few minutes...
        
        set "VC_DOWNLOADED=0"
        
        :: Download VC Redist
        powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%VC_REDIST_URL%' -OutFile '%VC_REDIST_INSTALLER%'"
        if exist "%VC_REDIST_INSTALLER%" (
            set "VC_DOWNLOADED=1"
            echo [OK] VC Redist installer downloaded
        )
        
        :: If download succeeded, install
        if "%VC_DOWNLOADED%"=="1" (
            echo Installing VC Redist...
            echo Please wait, this may take a few minutes...
            start /wait "" "%VC_REDIST_INSTALLER%" /quiet /norestart
            echo [OK] VC Redist installed successfully
            del "%VC_REDIST_INSTALLER%"
        ) else (
            echo [WARN] Failed to download VC Redist
            echo Some features may not work properly
        )
    ) else (
        echo [OK] VC Redist already installed
    )
) else (
    echo [OK] VC Redist already installed
)
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

:: Set virtual environment Python path
echo [INFO] Setting up virtual environment Python...
set "VENV_PYTHON=.venv_final\Scripts\python.exe"
if exist "%VENV_PYTHON%" (
    echo [OK] Found virtual environment Python at %VENV_PYTHON%
) else (
    echo [WARN] Virtual environment Python not found, using system Python
    set "VENV_PYTHON=python"
)
echo.

:: Run Python setup script
echo [INFO] Running setup_all.py...
echo.
"%VENV_PYTHON%" setup_all.py

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
"%VENV_PYTHON%" download_all_models.py
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
echo UI is starting, please wait...
echo.
start "" /B "%VENV_PYTHON%" ui.py

:: Wait for UI to start (give it some time to initialize)
echo Waiting for UI to initialize...
timeout /t 5 /nobreak >nul

:: Open browser after UI has started
echo Opening browser to http://localhost:7870...
start http://localhost:7870
goto end

:both
echo.
echo ========================================
echo Step 1: Downloading all models...
echo ========================================
echo.
"%VENV_PYTHON%" download_all_models.py
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
echo UI is starting, please wait...
echo.
start "" /B "%VENV_PYTHON%" ui.py

:: Wait for UI to start (give it some time to initialize)
echo Waiting for UI to initialize...
timeout /t 5 /nobreak >nul

:: Open browser after UI has started
echo Opening browser to http://localhost:7870...
start http://localhost:7870
goto end

:end
echo.
echo ========================================
echo.
pause
