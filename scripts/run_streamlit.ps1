param(
    [string]$Port = "8501",
    [switch]$Headless
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

# Initialize conda in non-interactive PowerShell sessions.
$condaHook = (conda shell.powershell hook | Out-String)
if ($LASTEXITCODE -ne 0) {
    throw "Failed to initialize conda shell hook."
}
Invoke-Expression $condaHook

conda activate sector-rotation
if ($LASTEXITCODE -ne 0) {
    throw "Failed to activate conda environment 'sector-rotation'."
}

$args = @("-m", "streamlit", "run", "app.py", "--server.port", $Port)
if ($Headless) {
    $args += @("--server.headless", "true")
}

python @args
