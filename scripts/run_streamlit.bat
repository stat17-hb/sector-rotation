@echo off
setlocal

set "REPO_ROOT=%~dp0.."
cd /d "%REPO_ROOT%" || exit /b 1

set "VENV_PY=%REPO_ROOT%\.venv\Scripts\python.exe"

powershell -NoProfile -ExecutionPolicy Bypass -File "%REPO_ROOT%\scripts\setup_local_env.ps1"
if errorlevel 1 (
    echo Failed to prepare the local virtual environment.
    pause
    exit /b 1
)

"%VENV_PY%" -m streamlit run app.py %*
