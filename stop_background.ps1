<#
.SYNOPSIS
    Stops the background AI hedge fund server.

.DESCRIPTION
    1. Ends the run_server.bat wrapper (cmd.exe) so its restart loop cannot relaunch.
    2. Asks the server to shut down gracefully via POST /api/shutdown, which stops
       the engine and closes the MT5 bridge cleanly.
    3. Waits for the python processes to exit, then force-terminates any python
       process still running this project's main.py or listening on the port.
    Open positions are not touched: they stay on the broker with their SL/TP.
#>
[CmdletBinding()]
param(
    [int]$Port = 0,
    [int]$GraceSeconds = 15
)

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BatchFile = Join-Path $ProjectDir 'run_server.bat'
$MainPy = Join-Path $ProjectDir 'main.py'
$EnvFile = Join-Path $ProjectDir '.env'

function Get-ConfiguredPort {
    if (Test-Path $EnvFile) {
        $match = Select-String -Path $EnvFile -Pattern '^\s*PORT\s*=\s*(\d+)' | Select-Object -First 1
        if ($match) { return [int]$match.Matches[0].Groups[1].Value }
    }
    return 8000
}

function Test-Contains([string]$Text, [string]$Needle) {
    return $Text -and $Text.IndexOf($Needle, [StringComparison]::OrdinalIgnoreCase) -ge 0
}

function Get-Wrappers {
    Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" -ErrorAction SilentlyContinue |
        Where-Object { Test-Contains $_.CommandLine $BatchFile }
}

function Get-ServerPythons {
    $byCommandLine = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like 'python*.exe' -and (Test-Contains $_.CommandLine $MainPy) }
    $ids = @($byCommandLine | ForEach-Object { [int]$_.ProcessId })
    $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($procId in $listeners) {
        $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if ($proc -and $proc.ProcessName -like 'python*' -and $ids -notcontains [int]$procId) { $ids += [int]$procId }
    }
    return $ids | Where-Object { $_ } | Sort-Object -Unique
}

if ($Port -le 0) { $Port = Get-ConfiguredPort }
$stoppedAnything = $false

# 1. Stop the restart loop first.
foreach ($wrapper in @(Get-Wrappers)) {
    Write-Host "[SYSTEM] Stopping run_server.bat wrapper (cmd.exe PID $($wrapper.ProcessId))"
    Stop-Process -Id $wrapper.ProcessId -Force -ErrorAction SilentlyContinue
    $stoppedAnything = $true
}

# 2. Graceful shutdown request.
$pythons = @(Get-ServerPythons)
if ($pythons.Count -gt 0) {
    $stoppedAnything = $true
    try {
        Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$Port/api/shutdown" -TimeoutSec 5 | Out-Null
        Write-Host "[SYSTEM] Graceful shutdown requested; waiting up to $GraceSeconds s ..."
    } catch {
        Write-Host "[SYSTEM] Graceful shutdown endpoint unavailable ($($_.Exception.Message)); terminating directly."
    }

    # 3. Wait, then force-terminate whatever is left.
    $deadline = (Get-Date).AddSeconds($GraceSeconds)
    while ((Get-Date) -lt $deadline -and @(Get-ServerPythons).Count -gt 0) {
        Start-Sleep -Milliseconds 500
    }
    foreach ($procId in @(Get-ServerPythons)) {
        Write-Host "[SYSTEM] Force-terminating python PID $procId"
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    }
}

# A late restart by a wrapper that was mid-loop is caught here.
foreach ($wrapper in @(Get-Wrappers)) {
    Stop-Process -Id $wrapper.ProcessId -Force -ErrorAction SilentlyContinue
}

Start-Sleep -Milliseconds 500
if (@(Get-ServerPythons).Count -gt 0) {
    Write-Host "[SYSTEM] Some server processes are still running; re-run as Administrator." -ForegroundColor Red
    exit 1
}
if ($stoppedAnything) {
    Write-Host "[SYSTEM] AI hedge fund server stopped. Open positions remain protected by their broker-side SL/TP." -ForegroundColor Green
} else {
    Write-Host "[SYSTEM] No running server found for $ProjectDir." -ForegroundColor Yellow
}
exit 0
