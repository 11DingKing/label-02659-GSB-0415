@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

:: SciBERT NER Service 启动脚本 (Windows)

echo.
echo ==========================================
echo   SciBERT NER Service Setup (Windows)
echo ==========================================
echo.

:: 检查 Python
echo [INFO] Checking Python installation...
set PYTHON_CMD=

:: 尝试 python3
where python3 >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=2" %%i in ('python3 --version 2^>^&1') do set PYVER=%%i
    set PYTHON_CMD=python3
    goto :found_python
)

:: 尝试 python
where python >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYVER=%%i
    set PYTHON_CMD=python
    goto :found_python
)

:: 尝试 py
where py >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=2" %%i in ('py --version 2^>^&1') do set PYVER=%%i
    set PYTHON_CMD=py
    goto :found_python
)

echo [ERROR] Python not found!
echo [INFO] Please install Python 3.9+ from https://www.python.org/downloads/
echo [INFO] Make sure to check "Add Python to PATH" during installation.
pause
exit /b 1

:found_python
echo [SUCCESS] Found Python %PYVER% (%PYTHON_CMD%)

:: 检查版本
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set MAJOR=%%a
    set MINOR=%%b
)
if %MAJOR% lss 3 (
    echo [ERROR] Python 3.9+ required, found %PYVER%
    pause
    exit /b 1
)
if %MAJOR% equ 3 if %MINOR% lss 9 (
    echo [ERROR] Python 3.9+ required, found %PYVER%
    pause
    exit /b 1
)

:: 创建虚拟环境
set VENV_DIR=backend\venv
if not exist "%VENV_DIR%" (
    echo [INFO] Creating virtual environment...
    %PYTHON_CMD% -m venv "%VENV_DIR%"
    echo [SUCCESS] Virtual environment created
) else (
    echo [INFO] Virtual environment already exists
)

:: 激活虚拟环境
echo [INFO] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
echo [SUCCESS] Virtual environment activated

:: 安装依赖
echo [INFO] Installing dependencies...
pip install --upgrade pip -q
pip install -r backend\requirements.txt -q
echo [SUCCESS] Dependencies installed

:: 启动服务
echo.
echo ==========================================
echo   SciBERT NER Service
echo   API: http://localhost:8000
echo   Docs: http://localhost:8000/docs
echo ==========================================
echo.

cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000

pause
