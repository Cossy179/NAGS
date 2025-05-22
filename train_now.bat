@echo off
echo ========================================
echo NAGS Chess Engine - Quick Training Start
echo ========================================
echo.

REM Check if Python is installed
where python >nul 2>&1
if %errorlevel% neq 0 (
    where py >nul 2>&1
    if %errorlevel% neq 0 (
        echo ERROR: Python is not installed!
        echo Please install Python 3.8+ from https://www.python.org/downloads/
        echo Make sure to check "Add Python to PATH" during installation
        pause
        exit /b 1
    )
    set PYTHON=py
) else (
    set PYTHON=python
)

echo Python found: %PYTHON%

REM Create virtual environment if needed
if not exist "venv" (
    echo Creating Python virtual environment...
    %PYTHON% -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install minimal dependencies
echo Installing dependencies...
pip install --upgrade pip
pip install numpy torch python-chess tqdm

REM Check if PGN file exists
if not exist "AJ-CORR-PGN-000.pgn" (
    echo.
    echo WARNING: AJ-CORR-PGN-000.pgn not found!
    echo ========================================
    echo.
    echo To start training, you need to place your PGN file in this directory.
    echo The file should be named: AJ-CORR-PGN-000.pgn
    echo.
    echo Once you have the file, run this script again.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Starting Supervised Training
echo ========================================
echo.

REM Start supervised training
%PYTHON% scripts\train_supervised.py --pgn_path AJ-CORR-PGN-000.pgn --num_epochs 10 --batch_size 32

echo.
echo ========================================
echo Training Started!
echo ========================================
echo.
echo The model is now training on your PGN data.
echo This will create a neural network model for the chess engine.
echo.
echo Training progress will be displayed above.
echo Model checkpoints will be saved in the 'models' directory.
echo.
pause 