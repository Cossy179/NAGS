@echo off
echo ====================================
echo NAGS Chess Engine - Quick Start
echo ====================================

REM Check if Python is installed
py --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Check if CMake is installed
cmake --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: CMake is not installed or not in PATH
    echo Please install CMake from https://cmake.org/download/
    pause
    exit /b 1
)

echo All prerequisites found!
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating Python virtual environment...
    py -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install Python dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

REM Create build directory if it doesn't exist
if not exist "build" (
    echo Creating build directory...
    mkdir build
)

cd build

REM Configure with CMake
echo Configuring with CMake...
cmake .. -DCMAKE_BUILD_TYPE=Release

REM Build the executables
echo Building C++ executables...
cmake --build . --config Release

cd ..

REM Check if PGN file exists
if not exist "AJ-CORR-PGN-000.pgn" (
    echo WARNING: AJ-CORR-PGN-000.pgn not found
    echo You'll need this file for training. Place it in the root directory.
    echo You can still test the engine without training.
    echo.
)

echo ====================================
echo Setup Complete!
echo ====================================
echo.
echo Your executables are now available:
echo - Engine: build\bin\Release\nags_v1.exe
echo - Trainer: build\bin\Release\nags_train.exe
echo.
echo To start training (if PGN file is available):
echo .\build\bin\Release\nags_train.exe --pgn_path AJ-CORR-PGN-000.pgn --save_dir models
echo.
echo To test the engine directly:
echo .\build\bin\Release\nags_v1.exe
echo.
pause 