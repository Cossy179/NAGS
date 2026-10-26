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
# Section 6's shards, named one by one: PowerShell passes a wildcard to a program unexpanded.
$SHARDS = @(foreach ($s in 9101..9108) { Join-Path $R9 "selfplay_$s.bin" })

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
# the worktree, their output appended to runs\NAME.log) and starts it in its own minimised window. A NAME that
# exists is refused; Resume-Detached NAME starts the same .cmd again, which continues the job (every command
# in these tasks resumes where it stopped, or refuses to redo finished work).
function Start-Detached([string]$Name, [string[]]$Lines) {
    $cmd = Join-Path $ROOT "runs\$Name.cmd"
    if (Test-Path -LiteralPath $cmd) { throw "$cmd exists: Resume-Detached $Name continues that job" }
    New-Item -ItemType Directory -Force -Path (Join-Path $ROOT 'runs') | Out-Null
    $log = Join-Path $ROOT "runs\$Name.log"
    $body = @('@echo off', "cd /d `"$ROOT`"") + @($Lines | ForEach-Object { "$_ >> `"$log`" 2>&1" })
    Set-Content -LiteralPath $cmd -Value $body -Encoding Oem
    Resume-Detached $Name
}

function Resume-Detached([string]$Name) {
    Start-Process -FilePath (Join-Path $ROOT "runs\$Name.cmd") -WorkingDirectory $ROOT -WindowStyle Minimized
}

# A match: tools\sprt.py with these arguments, its games in runs\NAME.pgn and --resume runs\NAME.log (the log
# it appends to), so that starting it again (Resume-Detached NAME) continues it, or reports its decision.
function Start-Match([string]$Name, [object[]]$SprtArgs) {
    $argv = @($PY, 'tools\sprt.py') + @($SprtArgs) + @('--pgnout', "runs\$Name.pgn", '--resume', "runs\$Name.log")
    Start-Detached $Name (ConvertTo-CmdLine $argv)
}
