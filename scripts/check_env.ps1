$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host ".venv not found. Run scripts\setup.ps1 first." -ForegroundColor Yellow
    exit 1
}

& ".\.venv\Scripts\python.exe" -c @"
import sys
print("Python:", sys.version)
print("Executable:", sys.executable)
try:
    import torch
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("Torch import failed:", repr(exc))
"@
