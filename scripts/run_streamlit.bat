@echo off
setlocal

set "REPO_ROOT=%~dp0.."
cd /d "%REPO_ROOT%"

call conda activate sector-rotation
if errorlevel 1 (
    echo Failed to activate conda environment "sector-rotation".
    exit /b 1
)

python -m streamlit run app.py %*
