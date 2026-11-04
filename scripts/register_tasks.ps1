# Registers Windows Task Scheduler jobs for the trading bot (weekdays only).
# Times are given in US Eastern and converted to this PC's local time zone.
#   analyze   17:15 ET  (after the close; agents form views, targets saved)
#   execute   09:45 ET  (after the open; orders submitted)
#   reconcile 16:15 ET  (record fills and end-of-day equity)
#   watch     10:00-15:30 ET every 30 min (intraday: refresh orders, intraday equity, existing stops)
#
# Profile smallcap (bot B, docs/PROFILE_B_SMALLCAP.md) registers separate tasks:
#   TradingBotB-Analyze 17:30, TradingBotB-Execute 10:00, TradingBotB-Reconcile 16:20 ET (weekdays),
#   TradingBotB-Screen Sunday 20:00 ET. Logs go to data\logs\smallcap\.
#
# Frozen v1 controls (spec 2026-11-04 Phase 0; simulated book, no broker orders), run with -Mode control:
#   -Profile control:          TradingBotCtl-Analyze 19:00, -Execute 09:50, -Reconcile 16:25 ET (weekdays)
#   -Profile smallcap_control: TradingBotCtlB-Analyze 19:20, -Execute 10:05, -Reconcile 16:30 ET (weekdays)
#   Initialize each book once first: uv run python -m bot.run_daily --profile control --phase reconcile --mode control --init-book
#
# Run from the repo root in PowerShell:   .\scripts\register_tasks.ps1 -Mode paper
#                                         .\scripts\register_tasks.ps1 -Mode paper -Profile smallcap
# Remove later with:                     .\scripts\register_tasks.ps1 -Unregister [-Profile smallcap]
param(
    [ValidateSet("dry-run", "paper", "live", "control")] [string]$Mode = "paper",
    [ValidateSet("default", "smallcap", "control", "smallcap_control")] [string]$Profile = "default",
    [switch]$Unregister
)

$Repo = Split-Path -Parent $PSScriptRoot
$Weekdays = @("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
if ($Profile -eq "smallcap") {
    $LogDir = "data\logs\smallcap"
    $jobs = @(
        @{ Name = "TradingBotB-Analyze";   Phase = "analyze";   H = 17; M = 30; Days = $Weekdays },
        @{ Name = "TradingBotB-Execute";   Phase = "execute";   H = 10; M = 0;  Days = $Weekdays },
        @{ Name = "TradingBotB-Reconcile"; Phase = "reconcile"; H = 16; M = 20; Days = $Weekdays },
        @{ Name = "TradingBotB-Screen";    Phase = "screen";    H = 20; M = 0;  Days = @("Sunday") },
        # intraday watch: every 30 min 10:15-15:45 ET (offset from the 10:00 execute so they never race)
        @{ Name = "TradingBotB-Watch";     Phase = "watch";     H = 10; M = 15; Days = $Weekdays; RepeatHours = 5.5 },
        # weekly memory curation (Phase 2): staged observations -> lessons
        @{ Name = "TradingBotB-Curate";    Phase = "curate";    H = 18; M = 45; Days = @("Sunday") }
    )
} elseif ($Profile -eq "control") {
    $LogDir = "data\logs\control"
    $jobs = @(
        @{ Name = "TradingBotCtl-Analyze";   Phase = "analyze";   H = 19; M = 0;  Days = $Weekdays },
        @{ Name = "TradingBotCtl-Execute";   Phase = "execute";   H = 9;  M = 50; Days = $Weekdays },
        @{ Name = "TradingBotCtl-Reconcile"; Phase = "reconcile"; H = 16; M = 25; Days = $Weekdays }
    )
} elseif ($Profile -eq "smallcap_control") {
    $LogDir = "data\logs\smallcap_control"
    $jobs = @(
        @{ Name = "TradingBotCtlB-Analyze";   Phase = "analyze";   H = 19; M = 20; Days = $Weekdays },
        @{ Name = "TradingBotCtlB-Execute";   Phase = "execute";   H = 10; M = 5;  Days = $Weekdays },
        @{ Name = "TradingBotCtlB-Reconcile"; Phase = "reconcile"; H = 16; M = 30; Days = $Weekdays }
    )
} else {
    $LogDir = "data\logs"
    $jobs = @(
        @{ Name = "TradingBot-Analyze";   Phase = "analyze";   H = 17; M = 15; Days = $Weekdays },
        @{ Name = "TradingBot-Execute";   Phase = "execute";   H = 9;  M = 45; Days = $Weekdays },
        @{ Name = "TradingBot-Reconcile"; Phase = "reconcile"; H = 16; M = 15; Days = $Weekdays },
        # intraday watch: every 30 min 10:00-15:30 ET
        @{ Name = "TradingBot-Watch";     Phase = "watch";     H = 10; M = 0;  Days = $Weekdays; RepeatHours = 5.5 },
        # weekly memory curation (Phase 2): staged observations -> lessons
        @{ Name = "TradingBot-Curate";    Phase = "curate";    H = 18; M = 30; Days = @("Sunday") }
    )
}
$Names = $jobs | ForEach-Object { $_.Name }

if ($Unregister) {
    foreach ($n in $Names) { Unregister-ScheduledTask -TaskName $n -Confirm:$false -ErrorAction SilentlyContinue }
    Write-Host "Removed trading bot tasks (profile $Profile)."
    return
}
if (($Profile -like "*control") -ne ($Mode -eq "control")) {
    throw "Control profiles run only with -Mode control, and -Mode control only with a control profile."
}
if ($Mode -eq "live") {
    Write-Warning "Live mode asks for approval on every execute run until execution.live_auto_execute is true; a scheduled run cannot answer that prompt, so it will not trade."
}

$uv = (Get-Command uv -ErrorAction Stop).Source
$et = [TimeZoneInfo]::FindSystemTimeZoneById("Eastern Standard Time")

function LocalTimeFromET([int]$h, [int]$m) {
    $today = [DateTime]::Today
    $etTime = New-Object DateTime($today.Year, $today.Month, $today.Day, $h, $m, 0, [DateTimeKind]::Unspecified)
    return [TimeZoneInfo]::ConvertTime($etTime, $et, [TimeZoneInfo]::Local)
}

foreach ($j in $jobs) {
    $log = Join-Path $Repo "$LogDir\$($j.Phase).log"
    New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
    if ($j.Phase -eq "screen") {
        $run = "python scripts/screen.py --profile smallcap"
    } elseif ($Profile -ne "default") {
        $run = "python -m bot.run_daily --profile $Profile --phase $($j.Phase) --mode $Mode"
    } else {
        $run = "python -m bot.run_daily --phase $($j.Phase) --mode $Mode"
    }
    $cmd = "Set-Location '$Repo'; & '$uv' run $run *>> '$log'"
    # conhost --headless: no console window flashes up on every run (matches the tasks as deployed)
    $action = New-ScheduledTaskAction -Execute "C:\Windows\System32\conhost.exe" `
        -Argument "--headless powershell.exe -NoProfile -WindowStyle Hidden -Command `"$cmd`""
    $at = LocalTimeFromET $j.H $j.M
    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $j.Days -At $at
    if ($j.RepeatHours) {   # every 30 minutes for RepeatHours (last run at start + RepeatHours)
        $rep = New-ScheduledTaskTrigger -Once -At $at -RepetitionInterval (New-TimeSpan -Minutes 30) `
            -RepetitionDuration (New-TimeSpan -Minutes ([int]($j.RepeatHours * 60) + 1))
        $trigger.Repetition = $rep.Repetition
    }
    # Unattended: wake the PC from sleep, catch up if a run was missed, retry on failure (e.g. Wi-Fi not up yet),
    # and run on battery.
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5) `
        -ExecutionTimeLimit (New-TimeSpan -Hours 1)
    # S4U = run whether or not you are logged in, without storing your password. Requires an elevated
    # (Run as administrator) PowerShell. Falls back to "only when logged in" if not elevated.
    $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
    $logon = if ($isAdmin) { "S4U" } else { "Interactive" }
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType $logon -RunLevel Limited
    Register-ScheduledTask -TaskName $j.Name -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
    if (-not $isAdmin) { Write-Warning "Not elevated: $($j.Name) will only run while you are logged in. Re-run as administrator for unattended runs." }
    Write-Host ("Registered {0,-22} {1:HH:mm} local ({2:00}:{3:00} ET)  mode={4} profile={5}" -f $j.Name, $at, $j.H, $j.M, $Mode, $Profile)
}
Write-Host "Note: ET/local offset is computed today; re-run after a daylight-saving change if your zone differs from ET."
