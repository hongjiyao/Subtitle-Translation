@echo off
setlocal enabledelayedexpansion

echo ========================================
echo Subtitle Translation Tool - Final Setup
echo ========================================
echo.

set "VENV_NAME=.venv_final"
set "PACKAGES_DIR=packages"
set "PYTHON_DIR=python311"
set "PYTHON_INSTALLER=python-3.11.7-amd64.exe"
set "PYTHON_URL=https://www.python.org/ftp/python/3.11.7/%PYTHON_INSTALLER%"

echo [INFO] Starting setup...
echo.

:: Step 1: Check if packages directory exists and has files
echo [1/7] Checking local packages directory...
set "PACKAGES_EXIST=0"
set "LOCAL_PKG_COUNT=0"
if exist "%PACKAGES_DIR%" (
    for %%f in ("%PACKAGES_DIR%\*.whl") do set /a LOCAL_PKG_COUNT+=1
    if !LOCAL_PKG_COUNT! gtr 0 (
        set "PACKAGES_EXIST=1"
        echo [OK] Found existing packages directory with !LOCAL_PKG_COUNT! files
    ) else (
        echo [INFO] Packages directory is empty, will download packages
    )
) else (
    echo [INFO] Packages directory does not exist, will be created
)
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
        if not "!PYTHON_DOWNLOADED!"=="1" (
            echo Trying source 2: Mirror...
            powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://registry.npmmirror.com/-/binary/python/3.11.7/%PYTHON_INSTALLER%' -OutFile '%PYTHON_INSTALLER%'"
            if exist "%PYTHON_INSTALLER%" (
                set "PYTHON_DOWNLOADED=1"
                echo [OK] Python installer downloaded from mirror
            )
        )

        :: If download succeeded, install
        if "!PYTHON_DOWNLOADED!"=="1" (
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

:: Step 3: Create packages directory
echo [3/7] Creating packages directory...
if not exist "%PACKAGES_DIR%" (
    mkdir "%PACKAGES_DIR%"
    echo [OK] Packages directory created: %PACKAGES_DIR%
) else (
    echo [OK] Packages directory already exists: %PACKAGES_DIR%
)
echo.

:: Step 4: Create virtual environment
echo [4/7] Creating virtual environment...
:: Check if venv exists
if exist "%VENV_NAME%" (
    echo [INFO] Virtual environment already exists, skipping creation...
) else (
    python -m venv "%VENV_NAME%"
    if errorlevel 1 (
        echo [ERROR] Failed to create venv
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)
echo.

:: Step 5: Upgrade pip and download packages to local directory
echo [5/7] Downloading and installing packages...
echo This may take several minutes...
echo.

:: Upgrade pip first
echo Upgrading pip...
"%VENV_NAME%\Scripts\python.exe" -m pip install --upgrade pip
echo [OK] pip upgraded
echo.

:: Check if PyTorch packages exist locally
echo Checking PyTorch packages...
set "TORCH_LOCAL=0"
dir "%PACKAGES_DIR%\torch*.whl" >nul 2>&1
if not errorlevel 1 goto :torch_found

:torch_not_found
:: Download PyTorch to local directory
echo Downloading PyTorch (CUDA 12.6) to %PACKAGES_DIR%...
"%VENV_NAME%\Scripts\pip.exe" download torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 -i "https://download.pytorch.org/whl/cu126" -d "%PACKAGES_DIR%"
if errorlevel 1 (
    echo [WARN] PyTorch download may have failed, continuing...
) else (
    echo [OK] PyTorch downloaded to %PACKAGES_DIR%
)
goto :torch_check_done

:torch_found
echo [INFO] Found local PyTorch packages, skipping download...

:torch_check_done
echo.

:: Install PyTorch from local directory
echo Installing PyTorch from local package...
"%VENV_NAME%\Scripts\pip.exe" install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --no-index --find-links="%PACKAGES_DIR%"
if errorlevel 1 (
    echo [WARN] PyTorch installation may have failed, continuing...
) else (
    echo [OK] PyTorch installed
)
echo.

:: Check if requirements.txt packages exist locally
if exist "requirements.txt" (
    echo Checking local packages for requirements.txt...
    
    :: Download packages from requirements.txt
    echo Downloading packages from requirements.txt to %PACKAGES_DIR%...
    "%VENV_NAME%\Scripts\pip.exe" download -r requirements.txt -d "%PACKAGES_DIR%"
    if errorlevel 1 (
        echo [WARN] Some packages may have failed to download
    ) else (
        echo [OK] Requirements downloaded
    )
    echo.

    :: Install packages from local directory
    echo Installing packages from local directory...
    "%VENV_NAME%\Scripts\pip.exe" install --no-index --find-links="%PACKAGES_DIR%" -r requirements.txt
    if errorlevel 1 (
        echo [WARN] Some packages may have failed to install
    ) else (
        echo [OK] All packages installed successfully!
    )
) else (
    echo [WARN] requirements.txt not found, downloading basic packages
    echo Downloading main packages...
    if not exist "%PACKAGES_DIR%" mkdir "%PACKAGES_DIR%"
    "%VENV_NAME%\Scripts\pip.exe" download gradio transformers openai-whisper sentencepiece ffmpy ImageIO imageio-ffmpeg moviepy numpy pandas pydantic requests tqdm rich soundfile torchcodec accelerate -d "%PACKAGES_DIR%"
    echo [OK] Basic packages downloaded
    echo.

    :: Install basic packages from local directory
    echo Installing basic packages from local directory...
    "%VENV_NAME%\Scripts\pip.exe" install --no-index --find-links="%PACKAGES_DIR%" gradio transformers openai-whisper sentencepiece ffmpy ImageIO imageio-ffmpeg moviepy numpy pandas pydantic requests tqdm rich soundfile torchcodec accelerate
    if errorlevel 1 (
        echo [WARN] Some basic packages may have failed to install
    ) else (
        echo [OK] Basic packages installed
    )
)
echo.

:: Step 6: Verify packages were downloaded
echo [6/7] Verifying downloaded packages...
if exist "%PACKAGES_DIR%" (
    set COUNT=0
    for %%f in ("%PACKAGES_DIR%\*.whl") do set /a COUNT+=1
    echo [OK] Packages directory exists with !COUNT! wheel files
) else (
    echo [WARN] Packages directory not found
)
echo.

:: Step 7: Verify venv packages
echo [7/7] Verifying installed packages in venv...
"%VENV_NAME%\Scripts\pip.exe" list --format=freeze | findstr /C:"gradio" >nul
if errorlevel 1 (
    echo [WARN] Some packages may not be installed correctly
) else (
    echo [OK] Packages verified in virtual environment
)
echo.

:: Step 8: Complete
echo [8/8] Setup complete!
echo.
echo ========================================
echo Setup successful!
echo ========================================
echo.
echo Python directory: %PYTHON_DIR%
echo Virtual environment: %VENV_NAME%
echo Local packages: %PACKAGES_DIR% (%~dp0%PACKAGES_DIR%)
if exist "%PACKAGES_DIR%" (
    set FINAL_COUNT=0
    for %%f in ("%PACKAGES_DIR%\*.whl") do set /a FINAL_COUNT+=1
    echo Package files cached: !FINAL_COUNT!
)
echo.
echo To install packages from local directory in the future:
echo   %VENV_NAME%\Scripts\pip.exe install --no-index --find-links="%PACKAGES_DIR%" -r requirements.txt
echo.
echo Or to install a specific package:
echo   %VENV_NAME%\Scripts\pip.exe install --no-index --find-links="%PACKAGES_DIR%" package_name
echo.
echo ========================================
echo.
pause