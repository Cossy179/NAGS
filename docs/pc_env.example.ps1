# Paths and helpers for docs/PC_TASKS.md, for Windows PowerShell 5.1 or PowerShell 7.
#
# Copy this file to runs\pc_env.ps1 in the NAGS-round3 worktree, replace the paths on the lines marked EDIT
# with this machine's real ones, and load it at the start of every PowerShell block (a new shell knows none
# of these variables):
#
#     Invoke-Expression (Get-Content -Raw runs\pc_env.ps1)
#
# (unlike running the .ps1, this works whatever the script execution policy is). It stops with an error if a
# path does not exist, and moves to the worktree.

$ROOT  = 'C:\path\to\NAGS-round3'                     # EDIT: this worktree (git worktree add, section 0)
$PY    = 'C:\path\to\NAGS\.venv\Scripts\python.exe'   # EDIT: the CUDA venv's python.exe
$DATA8 = 'D:\NAGS\data'                               # EDIT: network 8's round directory (read only)
$R9    = 'D:\NAGS\r9'                                 # EDIT: section 6's data; section 6a creates it

foreach ($p in $ROOT, $PY) {
    if (-not (Test-Path -LiteralPath $p)) { throw "pc_env.ps1: $p does not exist (edit the paths at its top)" }
}
Set-Location -LiteralPath $ROOT
$T = [Environment]::ProcessorCount   # self-play threads: every logical processor (6 on a Ryzen 5 3500X)
$C = $T - 1                          # games played at once in a match
$ENG = Join-Path $ROOT 'build\Release\nags_enhanced.exe'
$DG = Join-Path $ROOT 'build\Release\nags_datagen.exe'
$NET8 = Join-Path $ROOT 'nets\nags.nnue'
$OPENINGS = 'tools\openings\nags_balanced.epd'
$FT = @()          # EDIT after section 9: @('--ft-impl', 'fused') if it adopted the fused path, else @()
$TB345 = ''        # EDIT if present: the directory of the 3-4-5 Syzygy tables (sections 10 and 12)
# Section 6's shards, named one by one: PowerShell passes a wildcard to a program unexpanded.
$SHARDS = @(foreach ($shardSeed in 9101..9108) { Join-Path $R9 "selfplay_$shardSeed.bin" })

# One command line for a .cmd file: every argument in double quotes, so that paths with spaces stay whole,
# and % doubled (a .cmd file expands %...% even inside quotes). An argument containing a double quote or
# ending in a backslash would not reach the program as written, so it is refused.
function ConvertTo-CmdLine([object[]]$Argv) {
    (@($Argv) | ForEach-Object {
        $a = [string]$_
        if ($a.Contains('"') -or $a.EndsWith('\')) { throw "ConvertTo-CmdLine: unsupported argument: $a" }
        '"' + $a.Replace('%', '%%') + '"'
    }) -join ' '
}

# A long job, detached from this session: writes runs\NAME.cmd (the command lines, run one after the other in
# the worktree, their output appended to runs\NAME.log) and starts it outside the Claude app. A NAME that
# exists is refused; Resume-Detached NAME starts the same .cmd again, which continues the job (every command
# in these tasks resumes where it stopped, or refuses to redo finished work).
#
# The job stops at the first command that exits with a code other than 0 (logging which one and its code;
# for tools\sprt.py that includes H0 = 1 and undecided = 2, which end a match anyway), so a later command
# never runs on the output of a failed one. Only one copy of a job runs at a time: the .cmd holds
# runs\NAME.lock open while it runs, a second copy (a double launch, or the .cmd opened twice) cannot open
# it and exits at once (code 3, noted in runs\NAME.refused.log), and Windows releases the lock however the job ends: finished,
# failed, window closed, killed or the PC restarted, so no stale lock is left to clear.
function Start-Detached([string]$Name, [string[]]$Lines) {
    if ($Name -notmatch '^[A-Za-z0-9_.-]+$') { throw "Start-Detached: job name '$Name' must be letters, digits, _ . -" }
    $cmd = Join-Path $ROOT "runs\$Name.cmd"
    if (Test-Path -LiteralPath $cmd) { throw "$cmd exists: Resume-Detached $Name continues that job" }
    New-Item -ItemType Directory -Force -Path (Join-Path $ROOT 'runs') | Out-Null
    $log = Join-Path $ROOT "runs\$Name.log"
    $lock = Join-Path $ROOT "runs\$Name.lock"
    $refused = Join-Path $ROOT "runs\$Name.refused.log"
    $Lines = @($Lines)
    # Redirections are written first (>>"log" echo ...): "echo ... code 7>> log" would redirect handle 7.
    $body = @(
        '@echo off', "cd /d `"$ROOT`"",
        'set "NAGS_JOB_STARTED="',
        "9>`"$lock`" call :job",
        'set "NAGS_JOB_RC=%errorlevel%"',
        # The running copy holds runs\NAME.log open while a command writes to it, so a refusal goes to its own
        # file, and stays on screen for a few seconds.
        "if not defined NAGS_JOB_STARTED (echo [$Name] not started: another copy is running, runs\$Name.lock is in use& >>`"$refused`" echo [$Name] not started %DATE% %TIME%: another copy was running& timeout /t 10 >nul 2>&1& exit /b 3)",
        'exit /b %NAGS_JOB_RC%',
        ':job',
        'set "NAGS_JOB_STARTED=1"',
        ">>`"$log`" echo [$Name] started %DATE% %TIME%"
    )
    for ($i = 0; $i -lt $Lines.Count; $i++) {
        $body += "$($Lines[$i]) >> `"$log`" 2>&1"
        $body += "if not `"%errorlevel%`"==`"0`" (>>`"$log`" echo [$Name] command $($i + 1) of $($Lines.Count) exited with code %errorlevel%: stopped, the commands after it did not run& exit /b %errorlevel%)"
    }
    $body += ">>`"$log`" echo [$Name] all $($Lines.Count) commands finished %DATE% %TIME%"
    $body += 'exit /b 0'
    Set-Content -LiteralPath $cmd -Value $body -Encoding Oem
    Resume-Detached $Name
}

# Whether job NAME is running now (its .cmd holds runs\NAME.lock open).
function Test-DetachedRunning([string]$Name) {
    $lock = Join-Path $ROOT "runs\$Name.lock"
    if (-not (Test-Path -LiteralPath $lock)) { return $false }
    try { ([IO.File]::Open($lock, 'Open', 'ReadWrite', 'None')).Close(); return $false } catch { return $true }
}

# The .cmd is opened through explorer.exe, which hands it to the desktop's own Explorer: the job then runs
# outside the Claude app's Windows job object. A process started directly from the app's shell (Start-Process)
# stays in that job and ends when the app updates itself (docs/PC_RESULTS.md, section 4). A job that is
# running is refused here; the .cmd's own lock refuses a second copy started any other way.
function Resume-Detached([string]$Name) {
    $cmd = Join-Path $ROOT "runs\$Name.cmd"
    if (-not (Test-Path -LiteralPath $cmd)) { throw "$cmd does not exist: Start-Detached $Name writes it" }
    if (Test-DetachedRunning $Name) { throw "$Name is running (runs\$Name.lock is in use): wait for it, or stop it first" }
    Start-Process -FilePath 'explorer.exe' -ArgumentList "`"$cmd`""
}

# A match: tools\sprt.py with these arguments, its games in runs\NAME.pgn and --resume runs\NAME.log (the log
# it appends to), so that starting it again (Resume-Detached NAME) continues it, or reports its decision.
function Start-Match([string]$Name, [object[]]$SprtArgs) {
    $argv = @($PY, 'tools\sprt.py') + @($SprtArgs) + @('--pgnout', "runs\$Name.pgn", '--resume', "runs\$Name.log")
    Start-Detached $Name (ConvertTo-CmdLine $argv)
}
