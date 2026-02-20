# PowerShell script to activate the virtual environment
# Usage: .\activate_venv.ps1

Write-Host "Activating virtual environment..." -ForegroundColor Green
& ".\venv\Scripts\Activate.ps1"

Write-Host "`nVirtual environment activated!" -ForegroundColor Green
Write-Host "Python: $(python --version)" -ForegroundColor Cyan
Write-Host "Pip: $(pip --version)" -ForegroundColor Cyan
Write-Host "`nTo deactivate, run: deactivate" -ForegroundColor Yellow
