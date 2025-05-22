cmake_minimum_required(VERSION 3.14)
project(NAGS VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build" FORCE)
endif()

# Enable optimizations for Release build
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Find packages
find_package(Threads REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Add subdirectories
add_subdirectory(src/core)
add_subdirectory(src/uci)
add_subdirectory(src/meta)
add_subdirectory(src/nn)
add_subdirectory(src/gnn)
add_subdirectory(src/mcts)
add_subdirectory(src/training)

# Build the main executable
add_executable(nags_v1 src/main.cpp)
target_link_libraries(nags_v1 PRIVATE core uci meta nn gnn mcts Threads::Threads)

# Build the training executable
add_executable(nags_train src/training/train_main.cpp)
target_link_libraries(nags_train PRIVATE core meta nn gnn mcts training Threads::Threads)

# Install targets
install(TARGETS nags_v1 nags_train DESTINATION bin)

# Training configuration options
option(ENABLE_CUDA "Enable CUDA support for neural network training" OFF)
if(ENABLE_CUDA)
  find_package(CUDA REQUIRED)
  add_definitions(-DUSE_CUDA)
endif()

# Testing
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif() 