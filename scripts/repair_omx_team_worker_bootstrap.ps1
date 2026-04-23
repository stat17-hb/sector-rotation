param(
    [string]$TeamName = "",
    [switch]$Watch,
    [int]$TimeoutSec = 90,
    [int]$PollSec = 10
)

$ErrorActionPreference = "Stop"

function Get-LatestTeamDir {
    param([string]$StateRoot)

    $teamRoot = Join-Path $StateRoot "team"
    if (-not (Test-Path $teamRoot)) {
        throw "No team state directory found at $teamRoot"
    }

    if ($TeamName) {
        $target = Join-Path $teamRoot $TeamName
        if (-not (Test-Path $target)) {
            throw "Team '$TeamName' not found under $teamRoot"
        }
        return $target
    }

    $latest = Get-ChildItem $teamRoot -Directory |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if (-not $latest) {
        throw "No team directories found under $teamRoot"
    }
    return $latest.FullName
}

function Wait-LatestTeamDir {
    param(
        [string]$StateRoot,
        [int]$TimeoutSec,
        [int]$PollSec
    )

    $deadline = (Get-Date).AddSeconds([Math]::Max(1, $TimeoutSec))
    while ((Get-Date) -lt $deadline) {
        try {
            return Get-LatestTeamDir -StateRoot $StateRoot
        }
        catch {
            Start-Sleep -Seconds ([Math]::Max(1, $PollSec))
        }
    }

    throw "No team state directory became available under $(Join-Path $StateRoot 'team') within ${TimeoutSec}s"
}

function Read-Json {
    param([string]$Path)
    Get-Content $Path -Raw | ConvertFrom-Json
}

function Test-StartupEvidence {
    param([string]$TeamDir)

    $taskPath = Join-Path $TeamDir "tasks\task-1.json"
    if (Test-Path $taskPath) {
        $task = Read-Json -Path $taskPath
        if ($task.status -eq "in_progress" -or $task.status -eq "completed") {
            return $true
        }
    }

    $leaderMailboxPath = Join-Path $TeamDir "mailbox\leader-fixed.json"
    if (Test-Path $leaderMailboxPath) {
        $leaderMailbox = Read-Json -Path $leaderMailboxPath
        if ($leaderMailbox.messages | Where-Object { $_.body -like "ACK:*initialized" }) {
            return $true
        }
    }

    return $false
}

$repoRoot = (Resolve-Path ".").Path
$stateRoot = Join-Path $repoRoot ".omx\state"
$teamDir = if ($Watch) {
    Wait-LatestTeamDir -StateRoot $stateRoot -TimeoutSec $TimeoutSec -PollSec $PollSec
}
else {
    Get-LatestTeamDir -StateRoot $stateRoot
}
$requestsPath = Join-Path $teamDir "dispatch\requests.json"

if (-not (Test-Path $requestsPath)) {
    throw "Dispatch request file not found: $requestsPath"
}

$requests = Read-Json -Path $requestsPath
$startupRequest = $requests |
    Where-Object { $_.kind -eq "inbox" -and $_.trigger_message } |
    Sort-Object created_at -Descending |
    Select-Object -First 1

$paneId = ""
$trigger = ""

if ($startupRequest) {
    $paneId = [string]$startupRequest.pane_id
    $trigger = [string]$startupRequest.trigger_message
}
else {
    $configPath = Join-Path $teamDir "config.json"
    if (-not (Test-Path $configPath)) {
        throw "No startup inbox request found in $requestsPath and no config.json fallback was available"
    }

    $config = Read-Json -Path $configPath
    $worker = $config.workers | Select-Object -First 1
    if (-not $worker) {
        throw "No startup inbox request found and config.json has no workers"
    }

    $paneId = [string]$worker.pane_id
    $workerName = [string]$worker.name
    $teamNameResolved = [string]$config.name
    $inboxPath = Join-Path $teamDir "workers\$workerName\inbox.md"
    $trigger = "Read $($inboxPath -replace '\\','/'), work now, report progress, continue assigned work or next feasible task."
}

if (-not $paneId.StartsWith("%")) {
    throw "Startup request has invalid pane id: $paneId"
}

if ([string]::IsNullOrWhiteSpace($trigger)) {
    throw "Startup request has empty trigger text"
}

tmux send-keys -t $paneId -l -- $trigger
Start-Sleep -Milliseconds 200
tmux send-keys -t $paneId C-m
Start-Sleep -Milliseconds 200
tmux send-keys -t $paneId C-m

Write-Output "Resent startup trigger to $paneId for team $(Split-Path $teamDir -Leaf)"

if ($Watch) {
    $deadline = (Get-Date).AddSeconds([Math]::Max(1, $TimeoutSec))
    while ((Get-Date) -lt $deadline) {
        if (Test-StartupEvidence -TeamDir $teamDir) {
            Write-Output "Startup evidence detected for team $(Split-Path $teamDir -Leaf)"
            exit 0
        }

        Start-Sleep -Seconds ([Math]::Max(1, $PollSec))
        tmux send-keys -t $paneId -l -- $trigger
        Start-Sleep -Milliseconds 200
        tmux send-keys -t $paneId C-m
        Start-Sleep -Milliseconds 200
        tmux send-keys -t $paneId C-m
        Write-Output "Retried startup trigger to $paneId for team $(Split-Path $teamDir -Leaf)"
    }

    throw "Startup evidence was not detected within ${TimeoutSec}s for team $(Split-Path $teamDir -Leaf)"
}
