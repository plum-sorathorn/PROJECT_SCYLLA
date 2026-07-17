# ============================================================
# PROJECT: SCYLLA // Dev-Mode Launcher (PowerShell)
# Quick start: Python ODP on 6900 + frontend direct file access.
# No C++ build required. Frontend must be opened manually.
# Run: .\scripts\start_dev.ps1
# ============================================================

$ProjectRoot = Split-Path $PSScriptRoot -Parent
$VenvPath = "$ProjectRoot\backend\.venv"

Write-Host "[SCYLLA DEV] Starting Python ODP backend..." -ForegroundColor Yellow

# Ensure venv exists
if (-not (Test-Path "$VenvPath\Scripts\python.exe")) {
    Write-Host "[ERROR] Virtual environment not found. Run deploy.ps1 first." -ForegroundColor Red
    exit 1
}

# Patch app.js to point to port 6900 in dev mode
$AppJsPath = "$ProjectRoot\frontend\app.js"
$AppJs = Get-Content $AppJsPath -Raw
if ($AppJs -match "127\.0\.0\.1:8080") {
    Write-Host "[DEV] Patching app.js API_BASE to port 6900 for dev mode..." -ForegroundColor DarkYellow
    $AppJs = $AppJs -replace "127\.0\.0\.1:8080", "127.0.0.1:6900"
    Set-Content -Path $AppJsPath -Value $AppJs
}

# Start Python ODP
$OdpJob = Start-Process -FilePath "$VenvPath\Scripts\python.exe" `
    -ArgumentList "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "6900", "--reload" `
    -WorkingDirectory "$ProjectRoot\backend" `
    -PassThru -WindowStyle Normal

Write-Host "[SCYLLA DEV] ODP running at http://127.0.0.1:6900" -ForegroundColor Green
Write-Host "[SCYLLA DEV] Open frontend\index.html in your browser." -ForegroundColor Cyan
Write-Host ""
Write-Host "  API Docs: http://127.0.0.1:6900/docs" -ForegroundColor DarkGray
Write-Host "  Health:   http://127.0.0.1:6900/health" -ForegroundColor DarkGray
Write-Host ""
Write-Host "[SCYLLA DEV] Press Ctrl+C to stop." -ForegroundColor DarkGray

Wait-Process -Id $OdpJob.Id
