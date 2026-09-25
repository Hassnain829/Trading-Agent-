@echo off
REM Double-click to start the AI hedge fund server in the background.
REM The server keeps running after this window closes; see server.log for output.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_background.ps1" %*
echo.
echo This window closes in 20 seconds (press any key to close now).
timeout /t 20 > nul 2>&1
