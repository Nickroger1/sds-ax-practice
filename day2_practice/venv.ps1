# 1. Set execution policy for the current process
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

# 2. Create the Python virtual environment if it doesn't exist
if (-not (Test-Path -Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv .venv
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Yellow
}

# 3. Activate the virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
..\.venv\Scripts\activate