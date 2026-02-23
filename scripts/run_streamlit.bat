@echo off
setlocal

set "REPO_ROOT=%~dp0.."
cd /d "%REPO_ROOT%" || exit /b 1

set "CONDA_BAT="
if defined CONDA_EXE (
    for %%I in ("%CONDA_EXE%") do set "CONDA_BAT=%%~dpI..\condabin\conda.bat"
)
if not defined CONDA_BAT if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"

if defined CONDA_BAT (
    call "%CONDA_BAT%" activate sector-rotation
) else (
    call conda activate sector-rotation
)

if errorlevel 1 (
    echo Failed to activate conda environment "sector-rotation".
    echo Ensure conda is installed and the env "sector-rotation" exists.
    pause
    exit /b 1
)

python -m streamlit run app.py %*
