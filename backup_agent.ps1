<#
  Backs up the agent's learning state into a timestamped zip:
    data\agent\*            experience.jsonl (what the agent learns from) + exploration trades
    shadow_trades.json      blocked/skipped setups still being followed
    memory.json             real trades
    new_rules.json, settings.json, risk_state.json, .env
  Safe to run while the bot is running: the bot writes its JSON files atomically, and every file is first
  copied to a staging folder so the zip never holds a live file open.

  Restore: stop the bot, extract the zip into the project folder (overwrite), start the bot.

  Optional off-VPS copy: if <Destination>\data-repo is a git clone (e.g. a PRIVATE GitHub repo made only for
  this data), the files (without .env) are copied there, committed and pushed.

  Usage: backup_agent.bat [-Destination C:\TRBOT-backups] [-Keep 48] [-IncludeJournal]
#>
param(
    [string]$Destination = "C:\TRBOT-backups",
    [int]$Keep = 48,
    [switch]$IncludeJournal
)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$files = @("memory.json", "memory.pending.jsonl", "shadow_trades.json", "new_rules.json",
           "settings.json", "risk_state.json", ".env")
$dirs = @("data\agent")
if ($IncludeJournal) { $dirs += "data\journal" }

New-Item -ItemType Directory -Force $Destination | Out-Null
$logFile = Join-Path $Destination "backup.log"
function Log([string]$message) {
    $line = "{0:yyyy-MM-dd HH:mm:ss} {1}" -f (Get-Date), $message
    Write-Host $line
    Add-Content -Path $logFile -Value $line -Encoding utf8
}

# The bot may be replacing a file at the exact moment we read it, so retry briefly.
function Copy-WithRetry([string]$source, [string]$target) {
    New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
    for ($attempt = 1; $attempt -le 10; $attempt++) {
        try { Copy-Item -LiteralPath $source -Destination $target -Force; return }
        catch { if ($attempt -eq 10) { throw }; Start-Sleep -Milliseconds 300 }
    }
}

$stamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$staging = Join-Path $env:TEMP "trbot-backup-$stamp"
try {
    foreach ($name in $files) {
        $source = Join-Path $root $name
        if (Test-Path -LiteralPath $source) { Copy-WithRetry $source (Join-Path $staging $name) }
    }
    foreach ($dir in $dirs) {
        $sourceDir = Join-Path $root $dir
        if (-not (Test-Path -LiteralPath $sourceDir)) { continue }
        Get-ChildItem -LiteralPath $sourceDir -Recurse -File | Where-Object { $_.Name -notlike "*.tmp" } | ForEach-Object {
            Copy-WithRetry $_.FullName (Join-Path $staging $_.FullName.Substring($root.Length + 1))
        }
    }

    $experience = Join-Path $staging "data\agent\experience.jsonl"
    $rewards = if (Test-Path $experience) { @(Get-Content $experience).Count } else { 0 }
    $zip = Join-Path $Destination "agent-backup_$stamp.zip"
    Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $zip -CompressionLevel Optimal
    Log ("Saved {0} ({1:N0} KB, {2} agent rewards)" -f (Split-Path $zip -Leaf), ((Get-Item $zip).Length / 1KB), $rewards)

    # Keep only the newest $Keep zips.
    Get-ChildItem -Path $Destination -Filter "agent-backup_*.zip" | Sort-Object Name -Descending |
        Select-Object -Skip $Keep | ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force; Log "Removed old $($_.Name)" }

    # Optional: push to a separate private git repo (never the code repo, never .env).
    $repo = Join-Path $Destination "data-repo"
    if (Test-Path (Join-Path $repo ".git")) {
        Get-ChildItem -LiteralPath $staging -Recurse -File | Where-Object { $_.Name -ne ".env" } | ForEach-Object {
            Copy-WithRetry $_.FullName (Join-Path $repo $_.FullName.Substring($staging.Length + 1))
        }
        git -C $repo add -A
        git -C $repo diff --cached --quiet
        if ($LASTEXITCODE -eq 0) {
            Log "data-repo: no changes to push"
        } else {
            git -C $repo commit -q -m "Agent backup $stamp ($rewards rewards)"
            git -C $repo push -q
            if ($LASTEXITCODE -eq 0) { Log "data-repo: pushed" } else { Log "data-repo: PUSH FAILED (exit $LASTEXITCODE)" }
        }
    }
} catch {
    Log "BACKUP FAILED: $_"
    exit 1
} finally {
    if (Test-Path $staging) { Remove-Item -LiteralPath $staging -Recurse -Force }
}
