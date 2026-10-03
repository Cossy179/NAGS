@echo off
REM NAGS training pipeline runner (Windows).
REM
REM Thin wrapper around training_pipeline.py, which starts and stops the GNN and
REM meta-learner services itself, plays the self-play / evaluation games and
REM promotes models. Extra arguments are passed through, e.g.
REM   run_training.bat selfplay --games 20
REM Set PYTHON to choose the interpreter (default: the "py -3" launcher, else python).

setlocal
cd /d "%~dp0"

if not defined PYTHON (
    where py >nul 2>&1 && (set "PYTHON=py -3") || (set "PYTHON=python")
)

set "STEP=%~1"
if "%STEP%"=="" set "STEP=full"
if /i "%STEP%"=="help" goto usage
if /i "%STEP%"=="-h" goto usage
if /i "%STEP%"=="--help" goto usage
for %%s in (parse supervised selfplay ppo evaluate meta full) do if /i "%STEP%"=="%%s" goto valid_step
echo Unknown step: %STEP% 1>&2
call :print_usage
exit /b 1

:valid_step
set "ARGS="
shift
:collect_args
if "%~1"=="" goto args_done
set ARGS=%ARGS% %1
shift
goto collect_args
:args_done

%PYTHON% -c "import torch, torch_geometric, chess" >nul 2>&1
if errorlevel 1 (
    echo ERROR: missing Python dependencies. Install with: %PYTHON% -m pip install -r requirements.txt 1>&2
    exit /b 1
)

if /i "%STEP%"=="selfplay" goto check_engine
if /i "%STEP%"=="evaluate" goto check_engine
if /i "%STEP%"=="full" goto check_engine
goto run

:check_engine
if exist "build\Release\nags.exe" goto run
if exist "build\nags.exe" goto run
if exist "build\Debug\nags.exe" goto run
echo ERROR: NAGS engine not found. Build it with: cmake -B build ^&^& cmake --build build --config Release 1>&2
exit /b 1

:run
if not exist logs mkdir logs
echo Running step '%STEP%' (log: logs\training.log)
%PYTHON% training_pipeline.py --step %STEP%%ARGS%
exit /b %ERRORLEVEL%

:usage
call :print_usage
exit /b 0

:print_usage
echo Usage: run_training.bat [step] [pipeline options]
echo Steps: parse, supervised, selfplay, ppo, evaluate, meta, full (default: full)
echo Options are passed to training_pipeline.py (see: %PYTHON% training_pipeline.py --help)
exit /b 0
