@echo off
REM ==========================================================
REM setup_env.bat - minimal venv + requirements installer
REM Run with: call setup_env.bat
REM ==========================================================

cd /d "%~dp0"

if not exist "requirements.txt" (
  echo [ERROR] requirements.txt not found in: %cd%
  exit /b 1
)

REM ---- Create venv if missing ----
if not exist ".venv\Scripts\python.exe" (
  echo [INFO] Creating virtual environment in .venv ...
  python -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    exit /b 1
  )
)

REM ---- Activate venv (stays active because you run with CALL) ----
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo [ERROR] Failed to activate virtual environment.
  exit /b 1
)

REM ---- Upgrade pip (optional but recommended) ----
python -m pip install --upgrade pip

REM ---- Install requirements ----
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Installation failed.
  exit /b 1
)

echo [OK] Environment ready. VENV is active.
