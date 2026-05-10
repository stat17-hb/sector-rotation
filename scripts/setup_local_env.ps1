param(
    [switch]$Dev
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvDir = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$Requirements = Join-Path $RepoRoot "requirements.txt"
$DevRequirements = Join-Path $RepoRoot "requirements-dev.txt"
$StampFile = Join-Path $VenvDir ".requirements.sha256"

function Get-FileHashText {
    param([string[]]$Paths)

    $hashes = foreach ($Path in $Paths) {
        if (Test-Path $Path) {
            $stream = [System.IO.File]::OpenRead($Path)
            try {
                $sha256 = [System.Security.Cryptography.SHA256]::Create()
                try {
                    $bytes = $sha256.ComputeHash($stream)
                    -join ($bytes | ForEach-Object { $_.ToString("x2") })
                } finally {
                    $sha256.Dispose()
                }
            } finally {
                $stream.Dispose()
            }
        }
    }

    return ($hashes -join "`n")
}

function Test-PythonVersion {
    param([string]$PythonExe)

    if (-not $PythonExe) {
        return $false
    }

    try {
        & $PythonExe -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 13) else 1)" 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Test-PythonCommand {
    param([string[]]$CommandParts)

    if (-not $CommandParts -or $CommandParts.Length -eq 0) {
        return $false
    }

    $exe = $CommandParts[0]
    $args = if ($CommandParts.Length -gt 1) { $CommandParts[1..($CommandParts.Length - 1)] } else { @() }

    try {
        & $exe @args -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 13) else 1)" 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Get-PythonDescription {
    param([string[]]$CommandParts)

    if (-not $CommandParts -or $CommandParts.Length -eq 0) {
        return ""
    }

    $exe = $CommandParts[0]
    $args = if ($CommandParts.Length -gt 1) { $CommandParts[1..($CommandParts.Length - 1)] } else { @() }

    try {
        $description = (& $exe @args -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} at {sys.executable}')" 2>$null)
        if ($LASTEXITCODE -eq 0 -and $description) {
            return ($description | Select-Object -First 1)
        }
    } catch {
    }

    return ""
}

function Resolve-CondaEnvPython {
    param([string]$EnvName)

    $conda = Get-Command conda -ErrorAction SilentlyContinue
    if (-not $conda) {
        return $null
    }

    try {
        $envList = (& conda env list) 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $envList) {
            return $null
        }

        foreach ($line in $envList) {
            $trimmed = $line.Trim()
            if (-not $trimmed -or $trimmed.StartsWith("#")) {
                continue
            }

            $parts = $trimmed -split "\s+"
            $envPath = $parts[$parts.Length - 1]
            if ((Split-Path $envPath -Leaf) -eq $EnvName) {
                $candidate = Join-Path $envPath "python.exe"
                if (Test-PythonVersion $candidate) {
                    return $candidate
                }
            }
        }
    } catch {
    }

    return $null
}

function Resolve-BasePython {
    $rejected = New-Object System.Collections.Generic.List[string]

    if ($env:SECTOR_ROTATION_PYTHON) {
        if (Test-PythonVersion $env:SECTOR_ROTATION_PYTHON) {
            return @($env:SECTOR_ROTATION_PYTHON)
        }
        $rejected.Add("SECTOR_ROTATION_PYTHON=$($env:SECTOR_ROTATION_PYTHON)")
    }

    $condaEnvPython = Resolve-CondaEnvPython "sector-rotation"
    if ($condaEnvPython) {
        return @($condaEnvPython)
    }

    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        if (Test-PythonCommand @("py", "-3.11")) {
            return @("py", "-3.11")
        }
        $rejected.Add("py -3.11")
    } else {
        $rejected.Add("py launcher not found")
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        if (Test-PythonCommand @("python")) {
            return @("python")
        }
        $description = Get-PythonDescription @("python")
        if ($description) {
            $rejected.Add("PATH python is unsupported: $description")
        } else {
            $rejected.Add("PATH python is unusable")
        }
    } else {
        $rejected.Add("PATH python not found")
    }

    $details = ($rejected -join "; ")
    throw "Python 3.11 or 3.12 is required. Install Python 3.11, create conda env 'sector-rotation', set SECTOR_ROTATION_PYTHON, or make `py -3.11` available on PATH. Checked: $details"
}

Set-Location $RepoRoot

$CreatedVenv = $false
if (-not (Test-Path $VenvPython)) {
    $basePython = @(Resolve-BasePython)
    $basePythonExe = $basePython[0]
    $basePythonArgs = if ($basePython.Length -gt 1) { $basePython[1..($basePython.Length - 1)] } else { @() }
    $basePythonDescription = Get-PythonDescription $basePython
    Write-Host "Creating local virtual environment at $VenvDir"
    if ($basePythonDescription) {
        Write-Host "Using base Python: $basePythonDescription"
    }
    & $basePythonExe @basePythonArgs -m venv $VenvDir
    $CreatedVenv = $true
}

if ($CreatedVenv) {
    & $VenvPython -m pip install --upgrade pip setuptools wheel
}

$requirementFiles = @($Requirements)
if ($Dev) {
    $requirementFiles += $DevRequirements
}

$currentHash = Get-FileHashText $requirementFiles
$previousHash = if (Test-Path $StampFile) { Get-Content $StampFile -Raw } else { "" }

$streamlitCheck = & $VenvPython -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('streamlit') else 1)" 2>$null
$needsInstall = ($LASTEXITCODE -ne 0) -or ($currentHash.Trim() -ne $previousHash.Trim())

if ($needsInstall) {
    foreach ($req in $requirementFiles) {
        Write-Host "Installing dependencies from $req"
        & $VenvPython -m pip install -r $req
    }
    $currentHash | Set-Content -Encoding ASCII -Path $StampFile
} else {
    Write-Host "Local virtual environment is up to date."
}

& $VenvPython -c "import sys, streamlit; print(f'Python: {sys.executable}'); print(f'Streamlit: {streamlit.__version__}')"
