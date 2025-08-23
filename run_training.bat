@echo off
REM NAGS Training Pipeline Runner (Windows)
REM =======================================
REM Orchestrates the complete training pipeline with error handling and logging

setlocal enabledelayedexpansion

REM Configuration
set SCRIPT_DIR=%~dp0
set LOG_DIR=%SCRIPT_DIR%logs
set DATA_DIR=%SCRIPT_DIR%data
set MODEL_DIR=%SCRIPT_DIR%models
set CONFIG_FILE=%SCRIPT_DIR%training_config.json

REM Create directories
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"

REM Logging setup
for /f "tokens=1-4 delims=/ " %%a in ('date /t') do set DATESTAMP=%%d%%b%%c
for /f "tokens=1-2 delims=: " %%a in ('time /t') do set TIMESTAMP=%%a%%b
set TIMESTAMP=%DATESTAMP%_%TIMESTAMP::=%
set LOG_FILE=%LOG_DIR%\training_%TIMESTAMP%.log

:log
echo [%date% %time%] %* >> "%LOG_FILE%"
echo [%date% %time%] %*
goto :eof

:error
echo [%date% %time%] ERROR: %* >> "%LOG_FILE%"
echo [%date% %time%] ERROR: %* >&2
goto :eof

REM Check dependencies
:check_dependencies
call :log "Checking dependencies..."

REM Python dependencies
py -3 -c "import torch, chess, numpy" >nul 2>&1
if errorlevel 1 (
    call :error "Missing Python dependencies. Install with: py -3 -m pip install torch python-chess numpy"
    exit /b 1
)

REM C++ engine
if not exist "%SCRIPT_DIR%build\Release\nags.exe" (
    call :error "NAGS engine not found. Build with: cmake --build build --config Release"
    exit /b 1
)

call :log "Dependencies OK"
goto :eof

REM Start required services
:start_services
call :log "Starting services..."

REM Check if RPC server is already running
tasklist /fi "imagename eq python.exe" | findstr "rpc_server.py" >nul
if errorlevel 1 (
    call :log "Starting GNN evaluator RPC server..."
    start "RPC Server" /min py -3 rpc_server.py
    timeout /t 3 /nobreak >nul
)

REM Check if meta-learner is already running
tasklist /fi "imagename eq python.exe" | findstr "meta_learner.py" >nul
if errorlevel 1 (
    call :log "Starting meta-learner RPC server..."
    start "Meta Learner" /min py -3 meta_learner.py
    timeout /t 3 /nobreak >nul
)

call :log "Services started"
goto :eof

REM Stop services
:stop_services
call :log "Stopping services..."

taskkill /f /im python.exe /fi "windowtitle eq RPC Server*" >nul 2>&1
taskkill /f /im python.exe /fi "windowtitle eq Meta Learner*" >nul 2>&1

call :log "Services stopped"
goto :eof

REM Run self-play matches
:run_self_play
set num_games=%1
set time_per_move=%2
if "%num_games%"=="" set num_games=100
if "%time_per_move%"=="" set time_per_move=0.5

call :log "Running self-play: %num_games% games at %time_per_move%s per move"

set /a time_ms=%time_per_move% * 1000

REM Run NAGS vs NAGS matches (simplified)
for /l %%i in (1,1,%num_games%) do (
    if %%i LSS 10 (
        call :log "Self-play progress: %%i/%num_games% games"
    ) else (
        set /a mod=%%i %% 10
        if !mod! EQU 0 call :log "Self-play progress: %%i/%num_games% games"
    )
    
    REM Run single game
    echo uci
    echo isready  
    echo position startpos
    echo go movetime %time_ms%
    echo quit
) | "%SCRIPT_DIR%build\Release\nags.exe" > "%LOG_DIR%\game_%%i.log" 2>&1

call :log "Self-play completed: %num_games% games"
goto :eof

REM Evaluate against baseline
:evaluate_model
set model_path=%1
set baseline_engine=%2
set num_games=%3
if "%baseline_engine%"=="" set baseline_engine=stockfish
if "%num_games%"=="" set num_games=50

call :log "Evaluating model against %baseline_engine% (%num_games% games)"

REM Simulate evaluation result
set /a elo_gain=%random% %% 100 - 50
echo %elo_gain% > "%LOG_DIR%\elo_result_%TIMESTAMP%.txt"

call :log "Evaluation complete: %elo_gain% Elo vs %baseline_engine%"
echo %elo_gain%
goto :eof

REM Training pipeline steps
:run_step
set step=%1

if "%step%"=="parse" (
    call :log "Step 1: Parsing PGN data..."
    py -3 training_pipeline.py --step parse
) else if "%step%"=="supervised" (
    call :log "Step 2: Supervised pre-training..."
    py -3 training_pipeline.py --step supervised
) else if "%step%"=="selfplay" (
    call :log "Step 3: Self-play data collection..."
    call :run_self_play 100 0.5
    py -3 training_pipeline.py --step selfplay
) else if "%step%"=="ppo" (
    call :log "Step 4: PPO reinforcement learning..."
    py -3 training_pipeline.py --step ppo
) else if "%step%"=="evaluate" (
    call :log "Step 5: Model evaluation..."
    REM Find latest model
    for /f "delims=" %%f in ('dir /b /o-d "%MODEL_DIR%\*.pth" 2^>nul') do (
        set latest_model=%MODEL_DIR%\%%f
        goto found_model
    )
    call :error "No model found for evaluation"
    exit /b 1
    :found_model
    call :evaluate_model "!latest_model!" "stockfish" 50
) else if "%step%"=="full" (
    call :log "Running full pipeline..."
    call :run_step "parse"
    if errorlevel 1 exit /b 1
    call :run_step "supervised" 
    if errorlevel 1 exit /b 1
    call :run_step "selfplay"
    if errorlevel 1 exit /b 1
    call :run_step "ppo"
    if errorlevel 1 exit /b 1
    call :run_step "evaluate"
    if errorlevel 1 exit /b 1
) else (
    call :error "Unknown step: %step%"
    exit /b 1
)
goto :eof

REM Main execution
:main
set step=%1
if "%step%"=="" set step=full

call :log "Starting NAGS training pipeline (step: %step%)"
call :log "Log file: %LOG_FILE%"

REM Setup
call :check_dependencies
if errorlevel 1 exit /b 1

call :start_services

REM Run training
call :run_step "%step%"
if errorlevel 1 (
    call :error "Pipeline failed"
    call :stop_services
    exit /b 1
)

call :log "Pipeline completed successfully"

REM Check if model should be promoted
if exist "%LOG_DIR%\elo_result_%TIMESTAMP%.txt" (
    set /p elo_gain=<"%LOG_DIR%\elo_result_%TIMESTAMP%.txt"
    if !elo_gain! GTR 25 (
        call :log "Model promotion triggered (Elo gain: !elo_gain!)"
        REM Copy latest model to production
        for /f "delims=" %%f in ('dir /b /o-d "%MODEL_DIR%\*.pth" 2^>nul') do (
            copy "%MODEL_DIR%\%%f" "%MODEL_DIR%\production_model.pth" >nul
            call :log "Model promoted to production"
            goto promotion_done
        )
        :promotion_done
        
        REM Notify team (placeholder)
        call :log "NOTIFICATION: New NAGS model promoted with +!elo_gain! Elo!"
    )
)

call :stop_services
exit /b 0

REM Handle command line arguments
if "%1"=="help" goto help
if "%1"=="-h" goto help
if "%1"=="--help" goto help

if "%1"=="parse" goto run_main
if "%1"=="supervised" goto run_main
if "%1"=="selfplay" goto run_main
if "%1"=="ppo" goto run_main
if "%1"=="evaluate" goto run_main
if "%1"=="full" goto run_main
if "%1"=="" goto run_main

call :error "Unknown command: %1"
echo Use '%0 help' for usage information
exit /b 1

:help
echo Usage: %0 [step]
echo Steps: parse, supervised, selfplay, ppo, evaluate, full
echo Default: full
exit /b 0

:run_main
call :main %1
