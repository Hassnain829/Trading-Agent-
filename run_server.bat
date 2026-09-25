@echo off
setlocal EnableExtensions
REM ===========================================================================
REM  Runs the AI hedge fund server with the project's virtual environment and
REM  appends all output to server.log. If the server crashes it is restarted
REM  after 15 seconds so the engine keeps running 24/7.
REM  A clean shutdown (exit 0) or "already running" (exit 3) ends the loop.
REM  Stop it with stop_background.bat.
REM ===========================================================================
cd /d "%~dp0"
set "PYTHONUNBUFFERED=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

if not exist ".venv\Scripts\python.exe" (
    >> server.log echo [SYSTEM] %date% %time% .venv\Scripts\python.exe not found. Create the virtual environment first - see requirements.txt.
    exit /b 1
)

:run
>> server.log echo [SYSTEM] %date% %time% Launching main.py
".venv\Scripts\python.exe" "%~dp0main.py" >> server.log 2>&1
set "EXITCODE=%ERRORLEVEL%"
>> server.log echo [SYSTEM] %date% %time% main.py exited with code %EXITCODE%
if "%EXITCODE%"=="0" exit /b 0
if "%EXITCODE%"=="3" exit /b 3
>> server.log echo [SYSTEM] %date% %time% Restarting in 15 seconds
REM ping is used as a sleep because "timeout" fails without an interactive console.
ping -n 16 127.0.0.1 > nul
goto run
