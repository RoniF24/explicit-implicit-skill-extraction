@echo off
setlocal enabledelayedexpansion

REM ==========================================================
REM setup_env.bat - Create venv and install requirements
REM ==========================================================

echo.
echo ================================
echo   NLP Project Environment Setup
echo ================================
echo.

REM ---- Move to the script directory (project root) ----
cd /d "%~dp0"

REM ---- Check Python ----
python --version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python is not found in PATH.
  echo         Install Python 3.10+ and check "Add python.exe to PATH".
  exit /b 1
)
echo [OK] Python found.

REM ---- Check requirements file ----
if not exist "requirements.txt" (
  echo [ERROR] requirements.txt not found in:
  echo         %cd%
  exit /b 1
)
echo [OK] requirements.txt found.

REM ---- Create venv if missing ----
if not exist ".venv\Scripts\activate.bat" (
  echo [INFO] Creating virtual environment in .venv ...
  python -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    exit /b 1
  )
  echo [OK] Virtual environment created.
) else (
  echo [OK] Virtual environment already exists.
)

REM ---- Activate venv ----
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo [ERROR] Failed to activate virtual environment.
  exit /b 1
)
echo [OK] Virtual environment activated.

REM ---- Upgrade pip ----
echo [INFO] Upgrading pip ...
python -m pip install --upgrade pip
if errorlevel 1 (
  echo [WARN] pip upgrade failed (continuing anyway) ...
) else (
  echo [OK] pip upgraded.
)

REM ---- Install requirements (log to file) ----
echo [INFO] Installing requirements (this may take a while) ...
echo [INFO] Log will be saved to install_log.txt
python -m pip install -r requirements.txt > install_log.txt 2>&1
set INSTALL_EXIT=%ERRORLEVEL%

if not "%INSTALL_EXIT%"=="0" (
  echo [ERROR] Installation failed. See install_log.txt for details.
  echo.
  echo --- Last 25 lines of log ---
  powershell -NoProfile -Command "Get-Content -Path 'install_log.txt' -Tail 25"
  exit /b 1
)

echo [OK] All requirements installed successfully.

REM ---- Quick sanity check: list installed packages ----
echo.
echo [INFO] Installed packages snapshot:
python -m pip list

echo.
echo ================================
echo   DONE - Environment is ready
echo ================================
echo.
exit /b 0
