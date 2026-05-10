$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$LocalPythonRoot = Join-Path $env:LOCALAPPDATA "Programs\Python"
$ProgramFilesPythonRoot = $env:ProgramFiles

$CandidateCommands = @(
    @{ Label = "py -3.12"; Command = "py"; Args = @("-3.12") },
    @{ Label = "py -3.11"; Command = "py"; Args = @("-3.11") },
    @{ Label = "py -3.10"; Command = "py"; Args = @("-3.10") },
    @{ Label = "Python312 local install"; Command = (Join-Path $LocalPythonRoot "Python312\python.exe"); Args = @() },
    @{ Label = "Python311 local install"; Command = (Join-Path $LocalPythonRoot "Python311\python.exe"); Args = @() },
    @{ Label = "Python310 local install"; Command = (Join-Path $LocalPythonRoot "Python310\python.exe"); Args = @() },
    @{ Label = "Python312 program files"; Command = (Join-Path $ProgramFilesPythonRoot "Python312\python.exe"); Args = @() },
    @{ Label = "Python311 program files"; Command = (Join-Path $ProgramFilesPythonRoot "Python311\python.exe"); Args = @() },
    @{ Label = "Python310 program files"; Command = (Join-Path $ProgramFilesPythonRoot "Python310\python.exe"); Args = @() },
    @{ Label = "python"; Command = "python"; Args = @() }
)

$PythonCommand = $null
$PythonArgs = @()

foreach ($candidate in $CandidateCommands) {
    $isPath = $candidate.Command -match '[\\/]'
    if ($isPath) {
        if (-not (Test-Path $candidate.Command)) {
            continue
        }
    } elseif (-not (Get-Command $candidate.Command -ErrorAction SilentlyContinue)) {
        continue
    }

    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $versionJson = & $candidate.Command @($candidate.Args) -c "import sys, json; print(json.dumps({'major': sys.version_info.major, 'minor': sys.version_info.minor, 'exe': sys.executable}))" 2>$null
    } catch {
        $versionJson = $null
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }

    if (-not $versionJson) {
        continue
    }

    $version = $versionJson | ConvertFrom-Json
    if ($version.major -eq 3 -and $version.minor -ge 10 -and $version.minor -le 12) {
        $PythonCommand = $candidate.Command
        $PythonArgs = $candidate.Args
        Write-Host "Using compatible Python: $($candidate.Label) -> $($version.exe)" -ForegroundColor Green
        break
    }

    Write-Host "Skipping $($candidate.Label): Python $($version.major).$($version.minor) is not supported by this pinned PyTorch stack." -ForegroundColor Yellow
}

if (-not $PythonCommand) {
    throw @"
No compatible Python interpreter found.

This project currently requires Python 3.10, 3.11, or 3.12 because torch==2.5.1/torchvision==0.20.1 do not support your Python 3.14 environment.

Install Python 3.11, then rerun:
  powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
"@
}

if (-not (Test-Path ".venv")) {
    & $PythonCommand @($PythonArgs) -m venv .venv
} else {
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $venvVersionJson = & ".\.venv\Scripts\python.exe" -c "import sys, json; print(json.dumps({'major': sys.version_info.major, 'minor': sys.version_info.minor}))" 2>$null
    } catch {
        $venvVersionJson = $null
    } finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }

    if ($venvVersionJson) {
        $venvVersion = $venvVersionJson | ConvertFrom-Json
    } else {
        $venvVersion = $null
    }

    if (-not $venvVersion -or -not ($venvVersion.major -eq 3 -and $venvVersion.minor -ge 10 -and $venvVersion.minor -le 12)) {
        if ($venvVersion) {
            Write-Host "Existing .venv uses Python $($venvVersion.major).$($venvVersion.minor), recreating it with a compatible interpreter." -ForegroundColor Yellow
        } else {
            Write-Host "Existing .venv is not usable, recreating it with a compatible interpreter." -ForegroundColor Yellow
        }
        Remove-Item ".venv" -Recurse -Force
        & $PythonCommand @($PythonArgs) -m venv .venv
    }
}

if (-not (Test-Path ".env") -and (Test-Path ".env.example")) {
    Copy-Item ".env.example" ".env"
}

$PipArgs = @("--timeout", "120", "--retries", "10")

& ".\.venv\Scripts\python.exe" -m pip install @PipArgs --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install @PipArgs "numpy<2.0.0"
& ".\.venv\Scripts\python.exe" -m pip install @PipArgs torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
& ".\.venv\Scripts\python.exe" -m pip install @PipArgs -r requirements.txt
& ".\.venv\Scripts\python.exe" -m spacy download en_core_web_sm
& ".\.venv\Scripts\playwright.exe" install chromium

Write-Host "Setup complete. Run scripts\run.ps1 or use the VS Code launch configuration."
