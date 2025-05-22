@echo off
echo Building minimal NAGS chess engine...

REM Create build directory
if not exist "build_minimal" mkdir build_minimal
cd build_minimal

REM Simple CMake configuration for minimal build
echo cmake_minimum_required(VERSION 3.14) > CMakeLists.txt
echo project(NAGS_Minimal) >> CMakeLists.txt
echo set(CMAKE_CXX_STANDARD 17) >> CMakeLists.txt
echo add_executable(nags_minimal ../src/main.cpp ../src/core/engine.cpp) >> CMakeLists.txt
echo target_include_directories(nags_minimal PRIVATE ../src/core) >> CMakeLists.txt

REM Build
cmake . -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

cd ..
echo.
echo Minimal build complete!
echo Executable: build_minimal\Release\nags_minimal.exe
pause 