@echo off
:: -- Get the directory containing this batch file --
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT%\.."

:: -- Execute the PowerShell script safely, bypassing execution policy flags --
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0LAUNCH_SCYLLA.ps1"

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] SCYLLA launcher exited with error code %errorlevel%.
    pause
)