<#
.SYNOPSIS
    Starts the AI hedge fund server as a detached background process.

.DESCRIPTION
    Spawns "cmd.exe /c run_server.bat" through WMI (Win32_Process.Create). The new
    process is parented to WmiPrvSE.exe instead of this console, so closing the
    terminal, VS Code or the PowerShell window does not stop the engine.
    All output is appended to server.log. Stop it with stop_background.ps1.
#>
[CmdletBinding()]
param(
    [int]$Port = 0,
    # Startup waits for MT5 initialize (up to 60 s) and login (up to 60 s) before the port opens.
    [int]$StartupTimeoutSeconds = 150
)

$ErrorActionPreference = 'Stop'
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BatchFile = Join-Path $ProjectDir 'run_server.bat'
$Python = Join-Path $ProjectDir '.venv\Scripts\python.exe'
$MainPy = Join-Path $ProjectDir 'main.py'
$EnvFile = Join-Path $ProjectDir '.env'
$LogFile = Join-Path $ProjectDir 'server.log'

function Get-ConfiguredPort {
    if (Test-Path $EnvFile) {
        $match = Select-String -Path $EnvFile -Pattern '^\s*PORT\s*=\s*(\d+)' | Select-Object -First 1
        if ($match) { return [int]$match.Matches[0].Groups[1].Value }
    }
    return 8000
}

function Test-Listening([int]$LocalPort) {
    return [bool](Get-NetTCPConnection -LocalPort $LocalPort -State Listen -ErrorAction SilentlyContinue)
}

if ($Port -le 0) { $Port = Get-ConfiguredPort }

if (-not (Test-Path $Python)) {
    Write-Host "[SYSTEM] Virtual environment not found at $Python" -ForegroundColor Red
    Write-Host "         Create it with:  py -3.13 -m venv .venv ; .venv\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
}
if (-not (Test-Path $BatchFile)) {
    Write-Host "[SYSTEM] run_server.bat not found in $ProjectDir" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $EnvFile)) {
    Write-Warning ".env not found: copy .env.example to .env and fill in your DeepSeek and MT5 credentials."
}

# Never start a second engine: two engines would trade the same account twice.
$existing = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -and $_.CommandLine.IndexOf($MainPy, [StringComparison]::OrdinalIgnoreCase) -ge 0
}
if ($existing) {
    $ids = ($existing | ForEach-Object { $_.ProcessId }) -join ', '
    Write-Host "[SYSTEM] Server is already running (PID $ids). Dashboard: http://127.0.0.1:$Port" -ForegroundColor Yellow
    exit 0
}
if (Test-Listening $Port) {
    $owner = (Get-NetTCPConnection -LocalPort $Port -State Listen | Select-Object -First 1).OwningProcess
    Write-Host "[SYSTEM] Port $Port is already in use by PID $owner. Stop that process or change PORT in .env." -ForegroundColor Red
    exit 1
}

$startup = New-CimInstance -Namespace root/cimv2 -ClassName Win32_ProcessStartup -ClientOnly -Property @{
    ShowWindow = [uint16]0   # SW_HIDE: no console window
}
$commandLine = "cmd.exe /c `"`"$BatchFile`"`""
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine               = $commandLine
    CurrentDirectory          = $ProjectDir
    ProcessStartupInformation = $startup
}
if ($result.ReturnValue -ne 0) {
    Write-Host "[SYSTEM] Win32_Process.Create failed with code $($result.ReturnValue)" -ForegroundColor Red
    exit 1
}

Write-Host "[SYSTEM] Server launched in the background (cmd.exe PID $($result.ProcessId), parent WmiPrvSE.exe)." -ForegroundColor Green
Write-Host "[SYSTEM] Waiting for the dashboard on port $Port ..."

$deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 1
    if (Test-Listening $Port) {
        Write-Host "[SYSTEM] Online. Dashboard: http://127.0.0.1:$Port" -ForegroundColor Green
        Write-Host "[SYSTEM] Logs: $LogFile"
        exit 0
    }
}

Write-Warning ("The server did not open port $Port within $StartupTimeoutSeconds s (it may still be waiting " +
    "for the MT5 terminal). Last lines of server.log:")
if (Test-Path $LogFile) { Get-Content $LogFile -Tail 25 }
exit 1
