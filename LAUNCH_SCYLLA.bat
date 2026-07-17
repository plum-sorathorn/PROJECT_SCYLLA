@echo off
title PROJECT: SCYLLA // TERMINAL LAUNCHER
color 0B

echo.
echo  ============================================================
echo   PROJECT: SCYLLA // TERMINAL
echo   Hybrid Options Whale Scanner - Single-Click Launcher
echo  ============================================================
echo.

:: ── Set project root to the folder containing this .bat ──────────
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "VENV_PY=%ROOT%\backend\.venv\Scripts\python.exe"
set "BACKEND_DIR=%ROOT%\backend"
set "CPP_EXE=%ROOT%\cpp_core\build\Release\scylla_core.exe"
set "FRONTEND=%ROOT%\frontend\index.html"

echo  [ROOT] %ROOT%
echo.

:: ── Check Python is installed ─────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found on PATH.
    echo          Install Python 3.11+ from https://python.org
    echo          Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

:: ── First-time setup: create venv and install deps ───────────────
if not exist "%VENV_PY%" (
    echo  [SETUP] First run detected. Installing Python environment...
    echo  [SETUP] This will take 1-3 minutes. Please wait...
    echo.
    python -m venv "%BACKEND_DIR%\.venv"
    if errorlevel 1 (
        echo  [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo  [SETUP] Installing packages (fastapi, uvicorn, yfinance, pandas, numpy, scipy)...
    "%BACKEND_DIR%\.venv\Scripts\pip.exe" install --quiet --upgrade pip
    "%BACKEND_DIR%\.venv\Scripts\pip.exe" install fastapi uvicorn[standard] yfinance pandas numpy scipy
    if errorlevel 1 (
        echo  [ERROR] Package installation failed. Check your internet connection.
        pause
        exit /b 1
    )
    echo  [OK] First-time setup complete.
    echo.
)

:: ── Kill any stale processes on our ports ─────────────────────────
echo  [*] Releasing ports 6900 and 8080...
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":6900 "') do (
    taskkill /PID %%a /F >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":8080 "') do (
    taskkill /PID %%a /F >nul 2>&1
)
timeout /t 1 /nobreak >nul

:: ── Write a helper launcher script for the ODP backend ───────────
:: (Avoids the cd-ordering bug — the helper cd's first, then runs uvicorn)
echo @echo off > "%TEMP%\scylla_odp_launch.bat"
echo cd /d "%BACKEND_DIR%" >> "%TEMP%\scylla_odp_launch.bat"
echo "%VENV_PY%" -m uvicorn main:app --host 127.0.0.1 --port 6900 --log-level info >> "%TEMP%\scylla_odp_launch.bat"
echo pause >> "%TEMP%\scylla_odp_launch.bat"

:: ── Launch Python ODP backend ─────────────────────────────────────
echo  [1/3] Starting Python ODP Gateway on port 6900...
start "SCYLLA // ODP :6900" "%TEMP%\scylla_odp_launch.bat"

:: Wait for Python to bind the port (up to 10 seconds)
echo  [*] Waiting for ODP to come online...
set "ODP_OK=0"
for /l %%i in (1,1,10) do (
    timeout /t 1 /nobreak >nul
    netstat -ano 2>nul | findstr ":6900 " >nul 2>&1
    if not errorlevel 1 (
        set "ODP_OK=1"
        goto :odp_ready
    )
)
:odp_ready

if "%ODP_OK%"=="0" (
    echo.
    echo  [WARN] ODP did not bind to port 6900 within 10 seconds.
    echo         Check the ODP console window for errors.
    echo         Common fix: run this launcher as Administrator.
    echo.
)

echo  [OK] ODP Gateway running on http://127.0.0.1:6900

:: ── Patch app.js API_BASE to point to correct port ───────────────
if exist "%CPP_EXE%" (
    :: Production: frontend goes through C++ core on 8080
    powershell -NoProfile -Command "$f='%ROOT%\frontend\app.js'; $c=Get-Content $f -Raw; if($c -match '127\.0\.0\.1:6900'){$c=$c -replace '127\.0\.0\.1:6900','127.0.0.1:8080'; Set-Content $f $c}"
) else (
    :: Dev mode: frontend calls Python directly on 6900
    powershell -NoProfile -Command "$f='%ROOT%\frontend\app.js'; $c=Get-Content $f -Raw; if($c -match '127\.0\.0\.1:8080'){$c=$c -replace '127\.0\.0\.1:8080','127.0.0.1:6900'; Set-Content $f $c}"
)

:: ── Launch frontend (C++ or dev) ─────────────────────────────────
if exist "%CPP_EXE%" (
    echo  [2/3] C++ Core found. Launching on port 8080...
    echo @echo off > "%TEMP%\scylla_cpp_launch.bat"
    echo cd /d "%ROOT%\cpp_core" >> "%TEMP%\scylla_cpp_launch.bat"
    echo "%CPP_EXE%" >> "%TEMP%\scylla_cpp_launch.bat"
    echo pause >> "%TEMP%\scylla_cpp_launch.bat"
    start "SCYLLA // C++ Core :8080" "%TEMP%\scylla_cpp_launch.bat"
    timeout /t 3 /nobreak >nul
    echo  [3/3] Opening dashboard at http://127.0.0.1:8080/
    start "" "http://127.0.0.1:8080/"
) else (
    echo  [2/3] DEV MODE - No C++ build found. Frontend calls ODP directly.
    echo  [3/3] Opening frontend\index.html in browser...
    start "" "%FRONTEND%"
)

:: ── Summary ───────────────────────────────────────────────────────
echo.
echo  ============================================================
echo   SCYLLA IS ONLINE
echo.
if exist "%CPP_EXE%" (
    echo   ODP Gateway : http://127.0.0.1:6900
    echo   C++ Core    : http://127.0.0.1:8080
    echo   Dashboard   : http://127.0.0.1:8080/
    echo   API Docs    : http://127.0.0.1:6900/docs
) else (
    echo   ODP Gateway : http://127.0.0.1:6900  [check this window for errors]
    echo   Dashboard   : frontend\index.html opened in your browser
    echo   API Docs    : http://127.0.0.1:6900/docs
    echo.
    echo   NO API KEYS REQUIRED - yfinance is 100%% free
)
echo.
echo   To stop: close the "SCYLLA // ODP" console window.
echo  ============================================================
echo.
pause
