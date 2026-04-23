param(
    [string]$WorkerSpec = "1:executor",
    [Parameter(Mandatory = $true)]
    [string]$Task,
    [int]$BootstrapTimeoutSec = 90,
    [int]$BootstrapPollSec = 10
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path ".").Path
$repairScript = Join-Path $repoRoot "scripts\repair_omx_team_worker_bootstrap.ps1"

if (-not (Test-Path $repairScript)) {
    throw "Bootstrap repair script not found: $repairScript"
}

$watcher = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @(
        "-NoLogo",
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $repairScript,
        "-Watch",
        "-TimeoutSec", $BootstrapTimeoutSec,
        "-PollSec", $BootstrapPollSec
    ) `
    -WorkingDirectory $repoRoot `
    -PassThru

try {
    & omx team $WorkerSpec $Task
}
finally {
    if ($watcher -and -not $watcher.HasExited) {
        Stop-Process -Id $watcher.Id -Force
    }
}
