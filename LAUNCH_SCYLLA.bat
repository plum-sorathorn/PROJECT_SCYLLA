@echo off
:: -- Get the directory containing this batch file --
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

:: -- Execute the PowerShell script safely, bypassing execution policy flags --
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\LAUNCH_SCYLLA.ps1"

exit /b %errorlevel%