$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
            return
        }
        $key, $value = $line.Split("=", 2)
        [Environment]::SetEnvironmentVariable($key.Trim(), $value.Trim().Trim('"'), "Process")
    }
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Local virtual environment not found: $ProjectRoot\.venv" -ForegroundColor Yellow
    Write-Host "Run setup first:" -ForegroundColor Yellow
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1" -ForegroundColor Cyan
    exit 1
}

& ".\.venv\Scripts\python.exe" -c "import torch" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "The local .venv exists, but torch is not installed." -ForegroundColor Yellow
    Write-Host "Run setup again with Python 3.10, 3.11, or 3.12:" -ForegroundColor Yellow
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1" -ForegroundColor Cyan
    exit 1
}

& ".\.venv\Scripts\python.exe" ".\src\banyanTreev3_agentic.py"
