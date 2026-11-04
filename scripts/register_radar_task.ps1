# Registers the Windows Task Scheduler job for the news radar collector (shadow mode).
#   TradingBot-Radar  every 30 minutes, every day  ->  uv run python scripts/radar_collect.py
#   Output is appended to data\logs\radar.log
#
# Run from the repo root in PowerShell:   .\scripts\register_radar_task.ps1
# Remove later with:                     .\scripts\register_radar_task.ps1 -Unregister
param(
    [switch]$Unregister
)

$Repo = Split-Path -Parent $PSScriptRoot
$Name = "TradingBot-Radar"

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $Name -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Removed $Name."
    return
}

$uv = (Get-Command uv -ErrorAction Stop).Source
$log = Join-Path $Repo "data\logs\radar.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null

$cmd = "Set-Location '$Repo'; & '$uv' run python scripts/radar_collect.py *>> '$log'"
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -WindowStyle Hidden -Command `"$cmd`""
# Start at the next half hour, repeat every 30 minutes indefinitely (every day).
$now = Get-Date
$start = $now.Date.AddHours($now.Hour).AddMinutes([Math]::Ceiling(($now.Minute + 1) / 30) * 30)
$trigger = New-ScheduledTaskTrigger -Once -At $start -RepetitionInterval (New-TimeSpan -Minutes 30)
# Collector is cheap: no wake-from-sleep (missed half-hours are fine), catch up when available,
# never run two at once, and kill a stuck run after 10 minutes (a normal run takes < 3).
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
# S4U = run whether or not you are logged in, without storing your password. Requires an elevated
# (Run as administrator) PowerShell. Falls back to "only when logged in" if not elevated.
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
$logon = if ($isAdmin) { "S4U" } else { "Interactive" }
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType $logon -RunLevel Limited
Register-ScheduledTask -TaskName $Name -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
if (-not $isAdmin) { Write-Warning "Not elevated: $Name will only run while you are logged in. Re-run as administrator for unattended runs." }
Write-Host ("Registered {0} every 30 min starting {1:yyyy-MM-dd HH:mm} local; log: {2}" -f $Name, $start, $log)
