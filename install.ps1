# ══════════════════════════════════════════════════════════════════════════
#  install.ps1 — Windows installer
#
#  Replaces install.bat + run_hidden.bat. Deliberate changes:
#
#   * Scheduled Task instead of a hidden VBS dropped into Startup. A VBS
#     launcher that hides a webcam process is the exact signature EDR products
#     flag as stalkerware, and it made the tool impossible to stop cleanly.
#   * A venv, so the install cannot break the system Python.
#   * Autostart is asked for, not assumed.
#   * The watchdog gets its own task. It existed as a file before and was never
#     scheduled by anything, so it protected nothing on Windows.
#   * Stops by PID via the task, not `taskkill /f /im pythonw.exe`, which killed
#     unrelated Python processes and did not even match the launcher.
#
#  Run:  powershell -ExecutionPolicy Bypass -File install.ps1
# ══════════════════════════════════════════════════════════════════════════
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $Here ".venv"

Write-Host "lock-on-absence installer"
Write-Host "  repo: $Here"

$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command python3 -ErrorAction SilentlyContinue }
if (-not $py) { throw "Python not found on PATH. Install Python 3.10+ from python.org." }

$ver = & $py.Source -c "import sys; print('%d.%d' % sys.version_info[:2])"
Write-Host "  python: $ver"
& $py.Source -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"
if ($LASTEXITCODE -ne 0) { throw "Python 3.10+ required (found $ver)" }

Write-Host "==> creating venv at $Venv"
& $py.Source -m venv $Venv
$VPy      = Join-Path $Venv "Scripts\python.exe"
$Agent    = Join-Path $Venv "Scripts\lock-on-absence.exe"
$Enroll   = Join-Path $Venv "Scripts\lock-on-absence-enroll.exe"
$Watchdog = Join-Path $Venv "Scripts\lock-on-absence-watchdog.exe"

& $VPy -m pip install --quiet --upgrade pip
Write-Host "==> installing package"
& $VPy -m pip install --quiet -e $Here
if (-not (Test-Path $Agent)) { throw "entry point missing after install" }
Write-Host ("==> installed: " + (& $Agent --version))

if (-not (Test-Path (Join-Path $Here "face_model.yml"))) {
    Write-Host ""
    Write-Host "No face model found. The agent refuses to start without one, on purpose:" -ForegroundColor Yellow
    Write-Host "--any-face mode lets ANY face keep your screen unlocked." -ForegroundColor Yellow
    Write-Host "Enroll with:  $Enroll" -ForegroundColor Yellow
    Write-Host ""
}

$ans = Read-Host "Enable autostart at logon (agent + watchdog)? [y/N]"
if ($ans -notmatch '^[Yy]') {
    Write-Host "==> autostart NOT enabled. Run manually:"
    Write-Host "      $Agent --delay 10"
    Write-Host "      $Watchdog --heartbeat `"$Here\watchdog_heartbeat.txt`""
    exit 0
}

# Hidden=false on purpose: you should be able to see that it is running.
$agentAction = New-ScheduledTaskAction -Execute $Agent `
    -Argument "--delay 10 --mode security" -WorkingDirectory $Here
$logon = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName "LockOnAbsence" -Action $agentAction `
    -Trigger $logon -Settings $settings -RunLevel Limited -Force | Out-Null

$wdAction = New-ScheduledTaskAction -Execute $Watchdog `
    -Argument "--heartbeat `"$Here\watchdog_heartbeat.txt`" --max-age 120 --once" `
    -WorkingDirectory $Here
$wdRepeat = New-ScheduledTaskTrigger -Once -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName "LockOnAbsenceWatchdog" -Action $wdAction `
    -Trigger @($logon, $wdRepeat) -RunLevel Limited -Force | Out-Null

Write-Host ""
Write-Host "==> autostart ENABLED (Scheduled Tasks: LockOnAbsence, LockOnAbsenceWatchdog)"
Write-Host "    status:  Get-ScheduledTask LockOnAbsence*"
Write-Host "    stop:    Stop-ScheduledTask -TaskName LockOnAbsence"
Write-Host "    remove:  Unregister-ScheduledTask -TaskName LockOnAbsence,LockOnAbsenceWatchdog -Confirm:`$false"
