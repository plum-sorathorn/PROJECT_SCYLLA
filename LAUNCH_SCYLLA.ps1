$Host.UI.RawUI.WindowTitle = "PROJECT: SCYLLA // TERMINAL LAUNCHER"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   PROJECT: SCYLLA // TERMINAL" -ForegroundColor Cyan
Write-Host "   Hybrid Options Whale Scanner - Single-Click Launcher" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# -- Safely resolve root path regardless of spaces or ampersands ------
$ROOT = $PSScriptRoot
$VENV_PY = Join-Path $ROOT "backend\.venv\Scripts\python.exe"
$BACKEND_DIR = Join-Path $ROOT "backend"
$CPP_EXE = Join-Path $ROOT "cpp_core\build\Release\scylla_core.exe"
$FRONTEND = Join-Path $ROOT "frontend\index.html"
$APP_JS = Join-Path $ROOT "frontend\app.js"

Write-Host "[ROOT] $ROOT"
Write-Host ""

# -- Verify Python Environment ---------------------------------------
if (!(Get-Command python -ErrorAction SilentlyContinue) -and !(Test-Path $VENV_PY)) {
    Write-Host "[ERROR] Python not found on system PATH." -ForegroundColor Red
    Write-Host "        Install Python 3.11+ from https://python.org"
    Write-Host "        Make sure to check 'Add Python to PATH' during install."
    Read-Host "Press Enter to exit..."
    Exit
}

# -- First-Time Dependencies Deployment -----------------------------
if (!(Test-Path $VENV_PY)) {
    Write-Host "[SETUP] First run detected. Creating Python virtual environment..." -ForegroundColor Yellow
    Write-Host "        This will take 1-3 minutes. Please wait..."
    Write-Host ""
    
    Start-Process python -ArgumentList "-m venv `"$BACKEND_DIR\.venv`"" -Wait -NoNewWindow
    if (!(Test-Path $VENV_PY)) {
        Write-Host "[ERROR] Failed to create virtual environment." -ForegroundColor Red
        Read-Host "Press Enter to exit..."
        Exit
    }
    
    Write-Host "[SETUP] Installing packages (fastapi, uvicorn, yfinance, pandas, numpy, scipy)..." -ForegroundColor Yellow
    $PIP = Join-Path $BACKEND_DIR ".venv\Scripts\pip.exe"
    Start-Process $PIP -ArgumentList "install --quiet --upgrade pip" -Wait -NoNewWindow
    Start-Process $PIP -ArgumentList "install fastapi uvicorn[standard] yfinance pandas numpy scipy" -Wait -NoNewWindow
    Write-Host "[OK] First-time setup complete." -ForegroundColor Green
    Write-Host ""
}

# -- Port Cleansing --------------------------------------------------
Write-Host "[*] Releasing conflicting network ports (6900, 8080)..."
foreach ($port in @(6900, 8080)) {
    $proc = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($proc) {
        Stop-Process -Id $proc.OwningProcess -Force -ErrorAction SilentlyContinue
        # Wait up to 5 seconds for the port to be fully released by the OS
        for ($j = 1; $j -le 10; $j++) {
            Start-Sleep -Milliseconds 500
            if (!(Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue)) {
                break
            }
        }
    }
}
Start-Sleep -Seconds 1

# -- Launch Python ODP Backend ---------------------------------------
Write-Host "[1/3] Starting Python ODP Gateway on port 6900..."
$LaunchArgs = @("-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "6900", "--log-level", "info")
Start-Process $VENV_PY -ArgumentList $LaunchArgs -WorkingDirectory $BACKEND_DIR -WindowStyle Normal

# Wait for binding check loop
Write-Host "[*] Waiting for ODP Gateway server connection..."
$ODP_OK = $false
for ($i = 1; $i -le 10; $i++) {
    Start-Sleep -Seconds 1
    if (Get-NetTCPConnection -LocalPort 6900 -State Listen -ErrorAction SilentlyContinue) {
        $ODP_OK = $true
        break
    }
}

if (!$ODP_OK) {
    Write-Host "[WARN] ODP Gateway port binding is delayed. Check background shell for exceptions." -ForegroundColor Yellow
} else {
    Write-Host "[OK] ODP Gateway running on http://127.0.0.1:6900" -ForegroundColor Green
}

# -- Patch client configuration asset javascript endpoints ------------
if (Test-Path $APP_JS) {
    $Content = Get-Content $APP_JS -Raw
    if (Test-Path $CPP_EXE) {
        $Content = $Content -replace '127\.0\.0\.1:6900', '127.0.0.1:8080'
    } else {
        $Content = $Content -replace '127\.0\.0\.1:8080', '127.0.0.1:6900'
    }
    Set-Content $APP_JS $Content -NoNewline
}

# -- Auto-Build C++ Core Engine if missing ----------------------------
if (!(Test-Path $CPP_EXE) -and (Get-Command cmake -ErrorAction SilentlyContinue)) {
    Write-Host "[SETUP] C++ Core engine binary not found. Building native C++ Crow engine..." -ForegroundColor Yellow
    Write-Host "        Executing CMake Release build via MSVC..."
    $BUILD_DIR = Join-Path $ROOT "cpp_core\build"
    if (!(Test-Path $BUILD_DIR)) { New-Item -ItemType Directory -Path $BUILD_DIR -Force | Out-Null }
    
    # Run CMake configuration & parallel build
    Start-Process cmake -ArgumentList ".. -G `"Visual Studio 18 2026`" -A x64 -DCMAKE_BUILD_TYPE=Release" -WorkingDirectory $BUILD_DIR -Wait -NoNewWindow -ErrorAction SilentlyContinue
    if (!(Test-Path $CPP_EXE)) {
        # Fallback to VS 2022 generator if VS 2026 generator syntax varies
        Start-Process cmake -ArgumentList ".. -G `"Visual Studio 17 2022`" -A x64 -DCMAKE_BUILD_TYPE=Release" -WorkingDirectory $BUILD_DIR -Wait -NoNewWindow -ErrorAction SilentlyContinue
    }
    Start-Process cmake -ArgumentList "--build . --config Release --parallel" -WorkingDirectory $BUILD_DIR -Wait -NoNewWindow -ErrorAction SilentlyContinue
}

# -- Launch UI / Processing Nodes ------------------------------------
if (Test-Path $CPP_EXE) {
    Write-Host "[2/3] Native C++ LightGBM Engine binary active. Initializing port 8080..." -ForegroundColor Cyan
    Start-Process $CPP_EXE -WorkingDirectory (Join-Path $ROOT "cpp_core") -WindowStyle Normal
    Start-Sleep -Seconds 2
    Write-Host "[3/3] Launching Cyberpunk Web Interface..."
    Start-Process "http://127.0.0.1:8080/"
} else {
    Write-Host "[2/3] DEV MODE - C++ core unbuilt. Serving frontend directly via Python ODP layer." -ForegroundColor Yellow
    Write-Host "[3/3] Launching Web Interface on http://127.0.0.1:6900/..."
    Start-Process "http://127.0.0.1:6900/"
}

# -- Print System Status Block ---------------------------------------
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "   SCYLLA HYBRID 3-TIER ENGINE IS ONLINE" -ForegroundColor Green
Write-Host "   (No NPM/Node build required — Native HTML/JS + C++ Crow)" -ForegroundColor DarkGray
Write-Host ""
if (Test-Path $CPP_EXE) {
    Write-Host "   Python ODP Gateway : http://127.0.0.1:6900"
    Write-Host "   C++ Native Engine  : http://127.0.0.1:8080"
    Write-Host "   Dashboard Terminal : http://127.0.0.1:8080/"
} else {
    Write-Host "   Python ODP Gateway : http://127.0.0.1:6900"
    Write-Host "   Dashboard Terminal : http://127.0.0.1:6900/"
}
Write-Host ""
Write-Host "   Termination sequence: Close background application shells manually."
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""