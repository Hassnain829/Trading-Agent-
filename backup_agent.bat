@echo off
REM Double-click to back up the agent's learning data (safe while the bot is running).
REM Zips go to C:\TRBOT-backups; see backup_agent.ps1 for options and how to restore.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0backup_agent.ps1" %*
echo.
echo This window closes in 15 seconds (press any key to close now).
timeout /t 15 > nul 2>&1
