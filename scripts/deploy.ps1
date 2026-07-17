# ============================================================
# PROJECT: SCYLLA // Full Deployment Script (PowerShell)
# Installs all dependencies, builds C++ core, starts all services.
# Run from project root: .\scripts\deploy.ps1
# ============================================================

param(
    [switch]$SkipPython,
    [switch]$SkipCpp,
    [switch]$SkipVendors,
    [switch]$DevMode   # Dev mode: serve frontend via Python http.server (no C++ build required)
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path $PSScriptRoot -Parent

Write-Host @"

  ____  ____   ___     _ _____ ____ _____      ____   ______   ____  _     _       _    
 |  _ \|  _ \ / _ \   | | ____/ ___|_   _|    / ___| / ___\ \ / /  | |   | |     / \   
 | |_) | |_) | | | |  | |  _|| |     | |      \___ \| |    \ V / | |   | |    / _ \  
 |  __/|  _ <| |_| |  | | |__| |___  | |       ___) | |___  | |  | |___| |___/ ___ \ 
 |_|   |_| \_\___/  _/ |_____\____| |_|      |____/ \____| |_|  |_____|_____/_/   \_\
                    |__/                                                                 
 PROJECT: SCYLLA // DEPLOY SCRIPT v1.0
"@ -ForegroundColor Cyan

# ────────────────────────────────────────────────────────────
# Step 1: Python environment + dependencies
# ────────────────────────────────────────────────────────────
if (-not $SkipPython) {
    Write-Host "`n[1/4] Setting up Python backend..." -ForegroundColor Yellow

    $VenvPath = "$ProjectRoot\backend\.venv"
    if (-not (Test-Path $VenvPath)) {
        Write-Host "  Creating virtual environment..."
        python -m venv $VenvPath
    }

    $PipExe = "$VenvPath\Scripts\pip.exe"
    $PythonExe = "$VenvPath\Scripts\python.exe"

    Write-Host "  Installing Python dependencies..."
    & $PipExe install --quiet --upgrade pip
    & $PipExe install --quiet -r "$ProjectRoot\backend\requirements.txt"

    Write-Host "  [OK] Python environment ready." -ForegroundColor Green
} else {
    $VenvPath = "$ProjectRoot\backend\.venv"
    $PythonExe = "$VenvPath\Scripts\python.exe"
    Write-Host "[1/4] Skipping Python setup." -ForegroundColor DarkGray
}

# ────────────────────────────────────────────────────────────
# Step 2: Fetch C++ vendor headers
# ────────────────────────────────────────────────────────────
if (-not $SkipVendors -and -not $DevMode) {
    Write-Host "`n[2/4] Fetching C++ vendor headers..." -ForegroundColor Yellow
    & "$PSScriptRoot\fetch_vendors.ps1"
    Write-Host "  [OK] Vendors ready." -ForegroundColor Green
} else {
    Write-Host "[2/4] Skipping vendor fetch." -ForegroundColor DarkGray
}

# ────────────────────────────────────────────────────────────
# Step 3: Build C++ Core
# ────────────────────────────────────────────────────────────
if (-not $SkipCpp -and -not $DevMode) {
    Write-Host "`n[3/4] Building C++ core engine..." -ForegroundColor Yellow

    $CppDir   = "$ProjectRoot\cpp_core"
    $BuildDir = "$CppDir\build"

    if (-not (Test-Path $BuildDir)) {
        New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
    }

    Set-Location $BuildDir
    Write-Host "  Running CMake configure..."
    cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
    Write-Host "  Running CMake build..."
    cmake --build . --config Release --parallel
    Set-Location $ProjectRoot

    $ExePath = "$BuildDir\Release\scylla_core.exe"
    if (Test-Path $ExePath) {
        Write-Host "  [OK] C++ core built: $ExePath" -ForegroundColor Green
    } else {
        Write-Host "  [WARN] Build may have failed. Check output above." -ForegroundColor Red
        Write-Host "  Continuing in DEV MODE (Python HTTP server)..." -ForegroundColor Yellow
        $DevMode = $true
    }
} else {
    Write-Host "[3/4] Skipping C++ build." -ForegroundColor DarkGray
}

# ────────────────────────────────────────────────────────────
# Step 4: Launch all services
# ────────────────────────────────────────────────────────────
Write-Host "`n[4/4] Starting services..." -ForegroundColor Yellow

# Launch Python ODP backend on port 6900
$OdpJob = Start-Process -FilePath "$VenvPath\Scripts\python.exe" `
    -ArgumentList "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "6900", "--log-level", "info" `
    -WorkingDirectory "$ProjectRoot\backend" `
    -PassThru -WindowStyle Normal
Write-Host "  [OK] Python ODP Gateway started (PID: $($OdpJob.Id)) on http://127.0.0.1:6900" -ForegroundColor Green

Start-Sleep -Seconds 3

if ($DevMode) {
    # Dev fallback: serve frontend with Python http.server
    $FrontendJob = Start-Process -FilePath "python" `
        -ArgumentList "-m", "http.server", "8080" `
        -WorkingDirectory "$ProjectRoot\frontend" `
        -PassThru -WindowStyle Normal
    Write-Host "  [DEV] Frontend served at http://127.0.0.1:8080 (Python http.server)" -ForegroundColor Cyan
    Write-Host "  [NOTE] In DEV MODE, the frontend calls go directly to the Python ODP on port 6900." -ForegroundColor DarkYellow
    Write-Host "         Edit app.js API_BASE to 'http://127.0.0.1:6900' for direct dev access." -ForegroundColor DarkYellow
} else {
    # Production: launch C++ Crow server
    $CppExePath = "$ProjectRoot\cpp_core\build\Release\scylla_core.exe"
    $CppJob = Start-Process -FilePath $CppExePath `
        -WorkingDirectory "$ProjectRoot\cpp_core" `
        -PassThru -WindowStyle Normal
    Write-Host "  [OK] C++ Core Engine started (PID: $($CppJob.Id)) on http://127.0.0.1:8080" -ForegroundColor Green
}

Write-Host @"

  ╔══════════════════════════════════════════════════╗
  ║   PROJECT: SCYLLA // ONLINE                      ║
  ║                                                  ║
  ║   ODP Gateway:  http://127.0.0.1:6900            ║
  ║   C++ Core:     http://127.0.0.1:8080            ║
  ║   Dashboard:    http://127.0.0.1:8080/           ║
  ║                                                  ║
  ║   Press Ctrl+C in each window to stop.           ║
  ╚══════════════════════════════════════════════════╝
"@ -ForegroundColor Cyan
