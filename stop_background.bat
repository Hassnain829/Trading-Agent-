@echo off
REM Double-click to stop the background AI hedge fund server.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_background.ps1" %*
echo.
echo This window closes in 20 seconds (press any key to close now).
timeout /t 20 > nul 2>&1
